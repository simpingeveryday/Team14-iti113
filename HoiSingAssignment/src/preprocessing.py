"""
Shared preprocessing definitions for the fair-advertised-rate regression pipeline.

WHY THIS IS A MODULE AND NOT NOTEBOOK CODE
------------------------------------------
The fitted pipeline is pickled. Unpickling requires the *class definitions* to be
importable at load time, in the training container, in the inference container and
in any notebook that reloads the artifact. Classes defined in a notebook cell
pickle as ``__main__.ClassName`` and fail to load anywhere else. Every transformer
therefore lives here, and this file ships inside ``source_dir`` for both the
training job and the endpoint.

This module is the single definition of the feature contract. Notebook 01 derived
it procedurally and froze the fitted constants to JSON; here the same logic becomes
fit/transform objects so that one serialized artifact carries both the rules and
the constants.
"""

from __future__ import annotations

import ast
import json
import re

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

# --------------------------------------------------------------------------
# Policy constants — identical to Notebook 01 §0. Changing any of these is a
# feature-contract change and must bump PREPROCESSOR_VERSION.
# --------------------------------------------------------------------------
PREPROCESSOR_VERSION = "1.0.0"

SNAPSHOT_DATE = pd.Timestamp("2021-03-01")
TOP_K_AMENITIES = 30
COORD_DECIMALS = 2
COUNT_CAP_QUANTILE = 0.99
PRICE_CAP_QUANTILE = 0.995
MIN_NIGHTS_CAP = 365
MAX_NIGHTS_CAP = 1125
BEDROOMS_CAP = 16

CITY_CENTERS = {
    "Paris": (48.8530, 2.3499),
    "New York": (40.7580, -73.9855),
    "Sydney": (-33.8568, 151.2153),
    "Rome": (41.8986, 12.4769),
    "Rio de Janeiro": (-22.9711, -43.1822),
    "Istanbul": (41.0054, 28.9768),
    "Mexico City": (19.4326, -99.1332),
    "Bangkok": (13.7460, 100.5340),
    "Cape Town": (-33.9221, 18.4231),
    "Hong Kong": (22.2819, 114.1582),
}

CURRENCY = {
    "Paris": "EUR", "Rome": "EUR", "New York": "USD", "Sydney": "AUD",
    "Rio de Janeiro": "BRL", "Istanbul": "TRY", "Mexico City": "MXN",
    "Bangkok": "THB", "Cape Town": "ZAR", "Hong Kong": "HKD",
}

RESPONSE_TIME_LEVELS = [
    "within an hour", "within a few hours", "within a day",
    "a few days or more", "no history",
]

REVIEW_COLS = [
    "review_scores_rating", "review_scores_accuracy", "review_scores_cleanliness",
    "review_scores_checkin", "review_scores_communication",
    "review_scores_location", "review_scores_value",
]

TF_COLS = ["host_is_superhost", "host_has_profile_pic",
           "host_identity_verified", "instant_bookable"]

# The serving contract. Anything absent from a scoring payload is filled with NaN
# and handled by the missing-flag + imputation path — the endpoint never 400s
# because a host left the review block empty.
RAW_INPUT_COLUMNS = [
    "city", "neighbourhood", "latitude", "longitude",
    "property_type", "room_type", "accommodates", "bedrooms", "amenities",
    "minimum_nights", "maximum_nights", "instant_bookable",
    "host_since", "host_response_time", "host_response_rate",
    "host_acceptance_rate", "host_is_superhost", "host_total_listings_count",
    "host_has_profile_pic", "host_identity_verified",
] + REVIEW_COLS

# Without these a prediction is not meaningful, so they are the only hard requirement.
REQUIRED_RAW_COLUMNS = ["city", "room_type", "accommodates", "latitude", "longitude"]

DASH_QUALIFIER = re.compile(r"\s+[\u2013\u2014-]\s+.*$")
AMENITY_ALIASES = {
    "fast wifi": "wifi", "free wifi": "wifi", "hdtv": "tv",
    "washer in unit": "washer", "dryer in unit": "dryer",
}


# --------------------------------------------------------------------------
# Deterministic helpers (identical results on any sample -> safe pre-split)
# --------------------------------------------------------------------------
def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in km. Naive degree-Euclidean is anisotropic and
    wrong by ~40% between Paris and Hong Kong latitudes."""
    r = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(np.asarray(lat2, dtype="float64") - np.asarray(lat1, dtype="float64"))
    dlam = np.radians(np.asarray(lon2, dtype="float64") - np.asarray(lon1, dtype="float64"))
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlam / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def parse_amenities(raw):
    """JSON-ish list -> canonical sorted set. Strips vendor qualifiers so
    'Wifi - 1000 Mbps' and 'Fast wifi' both collapse onto 'wifi'."""
    if isinstance(raw, (list, tuple, set)):
        items = list(raw)
    elif isinstance(raw, str) and raw.strip() not in ("", "[]"):
        txt = raw.replace("\xa0", " ")
        try:
            items = json.loads(txt)
        except Exception:
            try:
                items = ast.literal_eval(txt)
            except Exception:
                return []
    else:
        return []
    out = set()
    for it in items:
        a = str(it).replace("\xa0", " ").strip().lower()
        a = DASH_QUALIFIER.sub("", a).strip()
        if a:
            out.add(AMENITY_ALIASES.get(a, a))
    return sorted(out)


def property_group(pt):
    """144 raw strings -> 7 auditable buckets. Rule-based, not frequency-based:
    frequency buckets are a statistic and would drift on every retrain."""
    t = str(pt).lower()
    if any(k in t for k in ("hotel", "hostel", "resort")):
        return "hotel_hostel"
    if any(k in t for k in ("bed and breakfast", "guesthouse", "guest suite")):
        return "bnb_guesthouse"
    if any(k in t for k in ("boat", "camper", "tent", "castle", "treehouse", "yurt",
                            "tiny house", "island", "cave", "dome", "windmill",
                            "lighthouse", "barn", "farm stay", "earth house", "hut",
                            "tipi", "igloo", "ryokan", "riad", "casa particular",
                            "minsu", "pension", "cycladic", "dammuso", "trullo",
                            "shepherd")):
        return "unique_stay"
    if any(k in t for k in ("apartment", "condominium", "loft", "serviced")):
        return "apartment_condo"
    if any(k in t for k in ("house", "townhouse", "villa", "cottage", "bungalow",
                            "cabin", "chalet")):
        return "house_villa"
    if "private room" in t or "shared room" in t or "room in" in t:
        return "room_other_dwelling"
    return "other"


PROPERTY_GROUP_LEVELS = ["apartment_condo", "bnb_guesthouse", "hotel_hostel",
                         "house_villa", "other", "room_other_dwelling", "unique_stay"]


def amenity_slug(a):
    return "amen_" + re.sub(r"[^a-z0-9]+", "_", a).strip("_")


def coerce_raw_frame(records):
    """Accept a dict, a list of dicts or a DataFrame and return a frame with the
    full raw schema present. Absent optional fields become NaN rather than errors:
    a host pricing a brand-new listing has no reviews and no response history."""
    if isinstance(records, pd.DataFrame):
        df = records.copy()
    else:
        if isinstance(records, dict):
            records = [records]
        df = pd.DataFrame(list(records))
    missing_required = [c for c in REQUIRED_RAW_COLUMNS if c not in df.columns]
    if missing_required:
        raise ValueError(f"Missing required fields: {missing_required}. "
                         f"Required: {REQUIRED_RAW_COLUMNS}")
    for col in RAW_INPUT_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
    return df[RAW_INPUT_COLUMNS]


# --------------------------------------------------------------------------
# Stage 1 — deterministic, row-wise feature construction (Notebook 01 §4.1, §5)
# --------------------------------------------------------------------------
class DeterministicFeatureBuilder(BaseEstimator, TransformerMixin):
    """Fixed rules only: no statistic is estimated here, so this stage behaves
    identically on one listing at an endpoint and on 223k rows in training.

    Also enforces the Tier-2 domain caps. Applying them at serve time matters:
    a payload with minimum_nights=99999 must be clipped exactly as the training
    rows were, or the model is asked to extrapolate off the edge of its support.
    """

    def __init__(self, snapshot_date=SNAPSHOT_DATE, coord_decimals=COORD_DECIMALS):
        self.snapshot_date = snapshot_date
        self.coord_decimals = coord_decimals

    def fit(self, X, y=None):
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        df = coerce_raw_frame(X).copy()

        # --- dtype normalisation ------------------------------------------
        for c in TF_COLS:
            df[c] = df[c].map({"t": 1, "f": 0, True: 1, False: 0, 1: 1, 0: 0})
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df["host_since"] = pd.to_datetime(df["host_since"], errors="coerce")
        for c in (["latitude", "longitude", "accommodates", "bedrooms",
                   "minimum_nights", "maximum_nights", "host_response_rate",
                   "host_acceptance_rate", "host_total_listings_count"] + REVIEW_COLS):
            df[c] = pd.to_numeric(df[c], errors="coerce")

        # The whole host block is missing together when the upstream join failed.
        df["host_info_missing"] = df["host_since"].isna().astype(int)
        for c in TF_COLS[:3]:
            df[c] = df[c].fillna(0).astype(int)
        df["instant_bookable"] = df["instant_bookable"].fillna(0).astype(int)
        df["host_total_listings_count"] = df["host_total_listings_count"].fillna(1)

        # --- Tier-2 fixed domain caps (winsorise, never delete) -------------
        df["minimum_nights"] = df["minimum_nights"].fillna(1).clip(lower=1, upper=MIN_NIGHTS_CAP)
        df["maximum_nights"] = df["maximum_nights"].fillna(MAX_NIGHTS_CAP).clip(
            lower=1, upper=MAX_NIGHTS_CAP)
        df["bedrooms"] = df["bedrooms"].clip(upper=BEDROOMS_CAP)

        # --- §5.1 spatial: derive, then destroy the exact coordinates -------
        clat = df["city"].map({c: v[0] for c, v in CITY_CENTERS.items()})
        clon = df["city"].map({c: v[1] for c, v in CITY_CENTERS.items()})
        df["dist_center_km"] = haversine_km(df["latitude"], df["longitude"],
                                            clat, clon).round(3)
        df["lat_coarse"] = df["latitude"].round(self.coord_decimals)
        df["lon_coarse"] = df["longitude"].round(self.coord_decimals)
        df = df.drop(columns=["latitude", "longitude"])

        # --- §5.2 amenities -------------------------------------------------
        df["amen_list"] = df["amenities"].map(parse_amenities)
        df["amenity_count"] = df["amen_list"].map(len)
        df = df.drop(columns=["amenities"])

        # --- §5.3 host track record ------------------------------------------
        df["host_tenure_days"] = (self.snapshot_date - df["host_since"]).dt.days
        df = df.drop(columns=["host_since"])
        df["host_response_rate_missing"] = df["host_response_rate"].isna().astype(int)
        df["host_acceptance_rate_missing"] = df["host_acceptance_rate"].isna().astype(int)
        rt = df["host_response_time"].astype("object").fillna("no history")
        df["host_response_time"] = rt.where(rt.isin(RESPONSE_TIME_LEVELS), "no history")

        # --- §5.4 capacity ----------------------------------------------------
        df["bedrooms_missing"] = df["bedrooms"].isna().astype(int)
        df["bedrooms"] = df["bedrooms"].fillna(
            np.ceil(df["accommodates"] / 2).clip(lower=1))
        df["persons_per_bedroom"] = (df["accommodates"] / df["bedrooms"].clip(lower=1)).round(3)

        # --- §5.5 / §5.6 property group and stay-length regime ----------------
        df["property_group"] = df["property_type"].map(property_group)
        df = df.drop(columns=["property_type"])
        df["is_monthly_min"] = (df["minimum_nights"] >= 28).astype(int)
        df["log_minimum_nights"] = np.log1p(df["minimum_nights"])
        df["has_review_scores"] = df["review_scores_rating"].notna().astype(int)
        return df


# --------------------------------------------------------------------------
# Stage 2 — train-fitted statistics (Notebook 01 §6.2-§6.6)
# --------------------------------------------------------------------------
class TrainFittedEncoder(BaseEstimator, TransformerMixin):
    """Every quantity here is estimated from the training split and frozen.

    Unseen categories, unseen neighbourhoods and unseen amenity strings all have
    a defined landing place (all-zero one-hot, 0.0 frequency, rare_amenity_count),
    so vocabulary drift degrades accuracy gracefully instead of raising at 3am.
    """

    def __init__(self, top_k_amenities=TOP_K_AMENITIES,
                 count_cap_quantile=COUNT_CAP_QUANTILE):
        self.top_k_amenities = top_k_amenities
        self.count_cap_quantile = count_cap_quantile

    IMPUTE_COLS = REVIEW_COLS + ["host_response_rate", "host_acceptance_rate",
                                 "host_tenure_days"]

    def fit(self, X, y=None):
        df = X
        self.imputation_values_ = {c: float(df[c].median()) for c in self.IMPUTE_COLS}
        # Guard against an all-NaN column in a small retraining slice.
        for c, v in self.imputation_values_.items():
            if not np.isfinite(v):
                self.imputation_values_[c] = 0.0

        counts = {}
        for lst in df["amen_list"]:
            for a in lst:
                counts[a] = counts.get(a, 0) + 1
        self.amenity_vocab_ = [a for a, _ in sorted(counts.items(),
                                                    key=lambda kv: (-kv[1], kv[0]))
                               ][: self.top_k_amenities]
        self.amenity_flag_cols_ = [amenity_slug(a) for a in self.amenity_vocab_]

        key = df["city"].astype(str) + "||" + df["neighbourhood"].fillna("unknown").astype(str)
        self.neighbourhood_freq_ = key.value_counts(normalize=True).to_dict()

        self.host_count_cap_ = float(df["host_total_listings_count"].quantile(
            self.count_cap_quantile))

        self.onehot_levels_ = {
            "city": sorted(CITY_CENTERS.keys()),
            "room_type": sorted(df["room_type"].dropna().unique().tolist()),
            "property_group": PROPERTY_GROUP_LEVELS,
            "host_response_time": RESPONSE_TIME_LEVELS,
        }
        self.feature_columns_ = self._assemble(df).columns.tolist()
        return self

    # -- shared by fit and transform so the column set cannot diverge --------
    def _assemble(self, df):
        out = df.copy()
        for c, v in self.imputation_values_.items():
            out[c] = out[c].fillna(v)

        amen_sets = out["amen_list"].map(set)
        vocab_set = set(self.amenity_vocab_)
        for a in self.amenity_vocab_:
            out[amenity_slug(a)] = amen_sets.map(lambda s, a=a: int(a in s))
        out["rare_amenity_count"] = out["amenity_count"] - amen_sets.map(
            lambda s: len(s & vocab_set))
        out = out.drop(columns=["amen_list"])

        key = out["city"].astype(str) + "||" + out["neighbourhood"].fillna("unknown").astype(str)
        out["neighbourhood_freq"] = key.map(self.neighbourhood_freq_).fillna(0.0)

        out["log_host_listings"] = np.log1p(
            out["host_total_listings_count"].clip(upper=self.host_count_cap_))

        pieces = []
        for col, levels in self.onehot_levels_.items():
            cat = pd.Categorical(out[col], categories=levels)
            d = pd.get_dummies(cat, prefix=col).astype(int)
            d.index = out.index
            pieces.append(d)
        dummies = pd.concat(pieces, axis=1)
        dummies.columns = [re.sub(r"\W+", "_", c).strip("_") for c in dummies.columns]

        cont = CONTINUOUS_FEATURES
        binary = BINARY_FEATURES
        X = pd.concat([out[cont + binary + self.amenity_flag_cols_], dummies], axis=1)
        return X

    def transform(self, X):
        out = self._assemble(X)
        # Column parity is the contract Notebook 02 and the endpoint both rely on.
        out = out.reindex(columns=self.feature_columns_, fill_value=0)
        return out.astype("float64")

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.feature_columns_, dtype=object)


CONTINUOUS_FEATURES = (
    ["accommodates", "bedrooms", "persons_per_bedroom", "dist_center_km",
     "lat_coarse", "lon_coarse", "amenity_count", "rare_amenity_count",
     "neighbourhood_freq", "host_tenure_days", "host_response_rate",
     "host_acceptance_rate", "log_host_listings", "log_minimum_nights",
     "maximum_nights"] + REVIEW_COLS
)
BINARY_FEATURES = [
    "host_is_superhost", "host_has_profile_pic", "host_identity_verified",
    "instant_bookable", "host_info_missing", "host_response_rate_missing",
    "host_acceptance_rate_missing", "bedrooms_missing", "has_review_scores",
    "is_monthly_min",
]


# --------------------------------------------------------------------------
# Stage 3 — scaling that preserves the DataFrame
# --------------------------------------------------------------------------
class FrameScaler(BaseEstimator, TransformerMixin):
    """StandardScaler on the continuous block only, returning a DataFrame.

    A ColumnTransformer would work but reorders columns and degrades names to
    ``num__x0``. Feature names have to survive to the SHAP analysis and the model
    card, and the endpoint's column order has to stay stable, so the frame is kept.
    Binary and one-hot columns are left at 0/1 — standardising indicators buys
    nothing and destroys interpretability.
    """

    def __init__(self, columns=None):
        self.columns = columns

    def fit(self, X, y=None):
        self.columns_ = list(self.columns if self.columns is not None else X.columns)
        self.columns_ = [c for c in self.columns_ if c in X.columns]
        block = X[self.columns_].astype("float64")
        self.mean_ = block.mean().values
        scale = block.std(ddof=0).values
        self.scale_ = np.where(scale > 0, scale, 1.0)  # constant column -> no-op
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def transform(self, X):
        out = X.copy()
        out[self.columns_] = (out[self.columns_].values - self.mean_) / self.scale_
        return out

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.feature_names_in_, dtype=object)


# --------------------------------------------------------------------------
# Target side — kept out of the feature Pipeline on purpose
# --------------------------------------------------------------------------
class PriceTargetTransformer:
    """log1p target with per-city winsorisation fitted on train.

    Deliberately asymmetric, and the asymmetry is the point:

    * ``fit_transform`` winsorises at the per-city train p99.5 so a handful of
      six-figure villas cannot dominate a squared-error fit.
    * ``inverse_transform`` is a plain ``expm1`` — capping is a training-time
      decision about the loss surface, not a claim about reality, so predictions
      are never clipped back down.

    It is not a Pipeline stage because the caps are keyed by city, which lives in
    X, not y. Encoding that as a sklearn stage would smuggle a feature into the
    target transform; keeping it explicit keeps it auditable.
    """

    def __init__(self, cap_quantile=PRICE_CAP_QUANTILE):
        self.cap_quantile = cap_quantile

    def fit(self, price, city):
        s = pd.Series(np.asarray(price, dtype="float64"))
        c = pd.Series(np.asarray(city)).reset_index(drop=True)
        self.price_caps_ = s.groupby(c).quantile(self.cap_quantile).to_dict()
        self.global_cap_ = float(s.quantile(self.cap_quantile))
        return self

    def transform(self, price, city):
        s = pd.Series(np.asarray(price, dtype="float64")).reset_index(drop=True)
        caps = pd.Series(np.asarray(city)).reset_index(drop=True).map(
            self.price_caps_).fillna(self.global_cap_)
        return np.log1p(np.minimum(s.values, caps.values))

    def fit_transform(self, price, city):
        return self.fit(price, city).transform(price, city)

    @staticmethod
    def inverse_transform(pred_log):
        return np.expm1(np.asarray(pred_log, dtype="float64"))


# --------------------------------------------------------------------------
# Factories — one definition, reused by Studio, the training job and serving
# --------------------------------------------------------------------------
def build_preprocessor():
    """The reusable, common data pipeline. Raw listing fields in, model-ready
    numeric matrix out. Fitted once, serialized, and never re-implemented."""
    return Pipeline([
        ("deterministic", DeterministicFeatureBuilder()),
        ("encode", TrainFittedEncoder()),
        ("scale", FrameScaler(columns=CONTINUOUS_FEATURES)),
    ])


def build_regressor(model_family, params=None):
    """Regression estimators only. This project has no classification anywhere."""
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.linear_model import Ridge

    params = dict(params or {})
    if model_family == "ridge":
        params.setdefault("alpha", 1.0)
        return Ridge(**params)
    if model_family == "hgb":
        params.setdefault("learning_rate", 0.10)
        params.setdefault("max_leaf_nodes", 127)
        params.setdefault("max_iter", 300)
        params.setdefault("min_samples_leaf", 50)
        params.setdefault("l2_regularization", 1.0)
        params.setdefault("early_stopping", False)
        params.setdefault("random_state", 42)
        return HistGradientBoostingRegressor(**params)
    raise ValueError(f"Unknown model_family: {model_family!r} (expected 'ridge' or 'hgb')")


def build_pipeline(model_family, params=None):
    """End-to-end artifact: raw listing -> preprocessing -> regressor -> log price.

    Serving loads exactly this object, so train-serve skew is structurally
    impossible: there is no second implementation of the transforms to drift.
    """
    return Pipeline([
        ("preprocess", build_preprocessor()),
        ("model", build_regressor(model_family, params)),
    ])
