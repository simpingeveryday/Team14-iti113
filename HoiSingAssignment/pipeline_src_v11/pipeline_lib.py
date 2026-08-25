"""
pipeline_lib — the single source of truth for the Airbnb nightly-price regression pipeline.

Consolidates Notebook 01 (EDA & feature engineering) into production estimators.
Consumed by: 03_production_pipeline.ipynb, preprocess.py (SageMaker Processing),
train.py (SageMaker Training), inference.py (Serverless endpoint).
Regression only: continuous advertised nightly `price`.
"""
import ast
import hashlib
import json
import re

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

__version__ = "1.0.0"

# ---------------------------------------------------------------- constants
SNAPSHOT_DATE = pd.Timestamp("2021-03-01")   # data as-of date (max host_since = 2021-02-26)
R_EARTH_KM = 6371.0
TOP_K_AMENITIES, COORD_DECIMALS = 30, 2
PRICE_CAP_QUANTILE, COUNT_CAP_QUANTILE = 0.995, 0.99
MIN_NIGHTS_CAP, MAX_NIGHTS_CAP, BEDROOMS_CAP = 365, 1125, 16
SPARSITY_THRESHOLD = 0.80

CITY_CENTERS = {  # fixed external landmarks — constants, not statistics
    "Paris": (48.8530, 2.3499), "New York": (40.7580, -73.9855),
    "Sydney": (-33.8568, 151.2153), "Rome": (41.8986, 12.4769),
    "Rio de Janeiro": (-22.9711, -43.1822), "Istanbul": (41.0054, 28.9768),
    "Mexico City": (19.4326, -99.1332), "Bangkok": (13.7460, 100.5340),
    "Cape Town": (-33.9221, 18.4231), "Hong Kong": (22.2819, 114.1582),
}
CURRENCY = {"Paris": "EUR", "Rome": "EUR", "New York": "USD", "Sydney": "AUD",
            "Rio de Janeiro": "BRL", "Istanbul": "TRY", "Mexico City": "MXN",
            "Bangkok": "THB", "Cape Town": "ZAR", "Hong Kong": "HKD"}

RESPONSE_TIME_LEVELS = ["within an hour", "within a few hours", "within a day",
                        "a few days or more", "no history"]
REVIEW_COLS = ["review_scores_rating", "review_scores_accuracy", "review_scores_cleanliness",
               "review_scores_checkin", "review_scores_communication",
               "review_scores_location", "review_scores_value"]
TF_COLS = ["host_is_superhost", "host_has_profile_pic", "host_identity_verified", "instant_bookable"]
HOST_BLOCK = ["host_since", "host_is_superhost", "host_has_profile_pic",
              "host_identity_verified", "host_total_listings_count"]
PII_DROP = ["name", "host_location", "host_id"]

DASH_QUALIFIER = re.compile(r"\s+[\u2013\u2014-]\s+.*$")
ALIASES = {"fast wifi": "wifi", "free wifi": "wifi", "hdtv": "tv",
           "washer in unit": "washer", "dryer in unit": "dryer"}

# The exact raw fields a serving request must provide (price/listing_id excluded).
SERVING_COLUMNS = [
    "city", "latitude", "longitude", "neighbourhood", "property_type", "room_type",
    "accommodates", "bedrooms", "amenities", "minimum_nights", "maximum_nights",
    "instant_bookable", "host_since", "host_is_superhost", "host_has_profile_pic",
    "host_identity_verified", "host_total_listings_count", "host_response_time",
    "host_response_rate", "host_acceptance_rate",
] + REVIEW_COLS

CONTINUOUS_FEATURES = (["accommodates", "bedrooms", "persons_per_bedroom", "dist_center_km",
                        "lat_coarse", "lon_coarse", "amenity_count", "rare_amenity_count",
                        "neighbourhood_freq", "host_tenure_days", "host_response_rate",
                        "host_acceptance_rate", "log_host_listings", "log_minimum_nights",
                        "maximum_nights"] + REVIEW_COLS)
BINARY_FEATURES = ["host_is_superhost", "host_has_profile_pic", "host_identity_verified",
                   "instant_bookable", "host_info_missing", "host_response_rate_missing",
                   "host_acceptance_rate_missing", "bedrooms_missing", "has_review_scores",
                   "is_monthly_min"]


# ---------------------------------------------------------------- helpers
def haversine_km(lat1, lon1, lat2, lon2):
    p = np.pi / 180.0
    a = (np.sin((lat2 - lat1) * p / 2) ** 2
         + np.cos(lat1 * p) * np.cos(lat2 * p) * np.sin((lon2 - lon1) * p / 2) ** 2)
    return 2 * R_EARTH_KM * np.arcsin(np.sqrt(a))


def parse_amenities(raw):
    """Canonicalise the amenities field (JSON-ish string or list) into a sorted set."""
    if isinstance(raw, (list, tuple, set)):
        items = list(raw)
    else:
        if not isinstance(raw, str) or raw.strip() in ("", "[]"):
            return []
        txt = raw.replace("\xa0", " ")
        try:
            items = json.loads(txt)
        except Exception:
            try:
                items = ast.literal_eval(txt)
            except Exception:
                return []
    out = set()
    for it in items:
        a = str(it).replace("\xa0", " ").strip().lower()
        a = DASH_QUALIFIER.sub("", a).strip()
        out.add(ALIASES.get(a, a))
    return sorted(out)


def property_group(pt):
    t = str(pt).lower()
    if any(k in t for k in ("hotel", "hostel", "resort")):                        return "hotel_hostel"
    if any(k in t for k in ("bed and breakfast", "guesthouse", "guest suite")):   return "bnb_guesthouse"
    if any(k in t for k in ("boat", "camper", "tent", "castle", "treehouse", "yurt",
                            "tiny house", "island", "cave", "dome", "windmill", "lighthouse",
                            "barn", "farm stay", "earth house", "hut", "tipi", "igloo",
                            "ryokan", "riad", "casa particular", "minsu", "pension",
                            "cycladic", "dammuso", "trullo", "shepherd")):        return "unique_stay"
    if any(k in t for k in ("apartment", "condominium", "loft", "serviced")):     return "apartment_condo"
    if any(k in t for k in ("house", "townhouse", "villa", "cottage", "bungalow",
                            "cabin", "chalet")):                                  return "house_villa"
    if "private room" in t or "shared room" in t or "room in" in t:               return "room_other_dwelling"
    return "other"


def _to_binary(s):
    """Robust t/f | true/false | 0/1 -> 0/1 with NaN preserved."""
    mapped = s.map({"t": 1, "f": 0, True: 1, False: 0, "true": 1, "false": 0})
    return mapped.fillna(pd.to_numeric(s, errors="coerce"))


def _slug(a):
    return "amen_" + re.sub(r"[^a-z0-9]+", "_", a).strip("_")


def get_data_version(path):
    """Content-addressed data version: sha256 prefix + byte size."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return f"sha256:{h.hexdigest()[:12]}"


# ---------------------------------------------------------------- batch cleaning
def clean_listings(df_raw):
    """Notebook 01 Step 4 — auditable row-level cleaning (batch path only; serving never drops rows).

    Order preserved from Notebook 01: near-duplicate removal (needs name/host_id),
    then PII drop, then Tier-1 validity removals, then the sparsity rule.
    Tier-2 value caps are applied in ListingFeatureEngineer so that the batch and
    serving paths share one implementation (clipping commutes with row removal).
    """
    df = df_raw.copy()
    audit = []
    assert df["listing_id"].duplicated().sum() == 0, "primary key violated"
    assert df.duplicated().sum() == 0, "full-row duplicates present"

    n0 = len(df)
    near_dup = (df.duplicated(subset=["host_id", "latitude", "longitude", "name"], keep="first")
                & df["name"].notna())
    df = df[~near_dup].copy()
    audit.append({"step": "near-duplicate removal", "rows_removed": n0 - len(df),
                  "note": "same host_id + exact coords + identical name -> kept first"})

    n1 = len(df)
    df = df.drop(columns=[c for c in PII_DROP if c in df.columns])
    audit.append({"step": "PII drop", "rows_removed": 0, "note": f"dropped {PII_DROP}"})

    df = df[df["price"] > 0]
    audit.append({"step": "Tier-1 price validity", "rows_removed": n1 - len(df),
                  "note": "price <= 0 is not an advertised rate"})
    n2 = len(df)
    df = df[df["accommodates"] > 0]
    audit.append({"step": "Tier-1 capacity validity", "rows_removed": n2 - len(df),
                  "note": "accommodates <= 0 is not rentable"})

    protected = set(SERVING_COLUMNS) | {"price", "listing_id"}   # the contract is never droppable
    sparse = [c for c in df.columns
              if c not in protected and df[c].isna().mean() > SPARSITY_THRESHOLD]
    df = df.drop(columns=sparse)
    audit.append({"step": "sparsity drop", "rows_removed": 0,
                  "note": f"columns > {SPARSITY_THRESHOLD:.0%} missing dropped: {sparse}"})
    return df, audit


# ---------------------------------------------------------------- stateless FE
class ListingFeatureEngineer(BaseEstimator, TransformerMixin):
    """Notebook 01 Step 5 — deterministic, row-wise transforms. Stateless by design:
    identical output for a 280k-row batch or a single serving request."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()

        # typing + host block (Notebook 01 Step 4.1)
        df["host_since"] = pd.to_datetime(df["host_since"], errors="coerce")
        for c in TF_COLS:
            df[c] = _to_binary(df[c])
        df["host_info_missing"] = df[HOST_BLOCK].isna().all(axis=1).astype(int)
        for c in ["host_is_superhost", "host_has_profile_pic", "host_identity_verified"]:
            df[c] = df[c].fillna(0).astype(int)
        df["host_total_listings_count"] = df["host_total_listings_count"].fillna(0)

        # Tier-2 domain caps (winsorise, never delete)
        df["minimum_nights"] = pd.to_numeric(df["minimum_nights"], errors="coerce").clip(upper=MIN_NIGHTS_CAP)
        df["maximum_nights"] = pd.to_numeric(df["maximum_nights"], errors="coerce").clip(upper=MAX_NIGHTS_CAP)
        df["bedrooms"] = pd.to_numeric(df["bedrooms"], errors="coerce").clip(upper=BEDROOMS_CAP)

        # Step 5.1 spatial: derive from exact coords, then destroy them (geo-privacy)
        clat = df["city"].map({c: v[0] for c, v in CITY_CENTERS.items()})
        clon = df["city"].map({c: v[1] for c, v in CITY_CENTERS.items()})
        df["dist_center_km"] = haversine_km(df["latitude"], df["longitude"], clat, clon).round(3)
        df["lat_coarse"] = df["latitude"].round(COORD_DECIMALS)
        df["lon_coarse"] = df["longitude"].round(COORD_DECIMALS)
        df = df.drop(columns=["latitude", "longitude"])

        # Step 5.2 amenities
        df["amen_list"] = df["amenities"].map(parse_amenities)
        df["amenity_count"] = df["amen_list"].str.len()
        df = df.drop(columns=["amenities"])

        # Step 5.3 host
        df["host_tenure_days"] = (SNAPSHOT_DATE - df["host_since"]).dt.days
        df = df.drop(columns=["host_since"])
        df["host_response_rate_missing"] = df["host_response_rate"].isna().astype(int)
        df["host_acceptance_rate_missing"] = df["host_acceptance_rate"].isna().astype(int)
        df["host_response_time"] = df["host_response_time"].fillna("no history")

        # Step 5.4 capacity
        df["bedrooms_missing"] = df["bedrooms"].isna().astype(int)
        df["bedrooms"] = df["bedrooms"].fillna(np.ceil(df["accommodates"] / 2).clip(lower=1))
        df["persons_per_bedroom"] = (df["accommodates"] / df["bedrooms"].clip(lower=1)).round(3)

        # Step 5.5 property grouping (144 raw strings -> 7 governed buckets)
        df["property_group"] = df["property_type"].map(property_group)
        df = df.drop(columns=["property_type"])

        # Step 5.6 stay-length regime
        df["is_monthly_min"] = (df["minimum_nights"] >= 28).astype(int)
        df["log_minimum_nights"] = np.log1p(df["minimum_nights"])
        return df


# ---------------------------------------------------------------- fitted encoders
class AirbnbPreprocessor(BaseEstimator, TransformerMixin):
    """Notebook 01 Step 6 — every transform that is a *statistic*, fitted on TRAIN ONLY:
    imputation medians, amenity vocabulary, neighbourhood frequencies, host-count cap,
    frozen one-hot levels, standard scaler. transform() emits the frozen 88-column matrix."""

    IMPUTE_COLS = REVIEW_COLS + ["host_response_rate", "host_acceptance_rate", "host_tenure_days"]

    def fit(self, X, y=None):
        df = X
        self.imputation_values_ = {}
        for c in self.IMPUTE_COLS:
            med = df[c].median()
            # A NaN median means the column was entirely missing in the fit sample —
            # impossible on the full training split, but guarded so degenerate
            # samples (unit tests, tiny batches) fit loudly-sane instead of crashing.
            self.imputation_values_[c] = float(med) if pd.notna(med) else 0.0

        from collections import Counter
        counter = Counter()
        for lst in df["amen_list"]:
            counter.update(lst)
        self.amenity_vocab_ = [a for a, _ in counter.most_common(TOP_K_AMENITIES)]

        nbhd_key = df["city"] + "||" + df["neighbourhood"].fillna("unknown")
        self.neighbourhood_freq_ = nbhd_key.value_counts(normalize=True).to_dict()

        self.host_count_cap_ = float(df["host_total_listings_count"].quantile(COUNT_CAP_QUANTILE))
        self.onehot_levels_ = {
            "city": sorted(df["city"].unique()),
            "room_type": sorted(df["room_type"].unique()),
            "property_group": sorted(df["property_group"].unique()),
            "host_response_time": RESPONSE_TIME_LEVELS,
        }
        unscaled = self._assemble(df)
        self.scaler_ = StandardScaler().fit(unscaled[CONTINUOUS_FEATURES])
        self.feature_columns_ = unscaled.columns.tolist()
        return self

    def _assemble(self, X):
        df = X.copy()
        df["has_review_scores"] = df["review_scores_rating"].notna().astype(int)
        for c, v in self.imputation_values_.items():
            df[c] = df[c].fillna(v)

        amen_sets = df["amen_list"].map(set)
        amen_flags = {}
        for a in self.amenity_vocab_:
            amen_flags[_slug(a)] = amen_sets.map(lambda s, a=a: int(a in s))
        amen_df = pd.DataFrame(amen_flags, index=df.index)
        in_vocab = amen_sets.map(lambda s: len(s & set(self.amenity_vocab_)))
        df["rare_amenity_count"] = df["amenity_count"] - in_vocab

        nbhd_key = df["city"] + "||" + df["neighbourhood"].fillna("unknown")
        df["neighbourhood_freq"] = nbhd_key.map(self.neighbourhood_freq_).fillna(0.0)

        df["log_host_listings"] = np.log1p(df["host_total_listings_count"].clip(upper=self.host_count_cap_))

        pieces = []
        for col, levels in self.onehot_levels_.items():
            cat = pd.Categorical(df[col], categories=levels)
            d = pd.get_dummies(cat, prefix=col).astype(int)
            d.index = df.index
            pieces.append(d)
        onehot = pd.concat(pieces, axis=1)
        onehot.columns = [re.sub(r"\W+", "_", c).strip("_") for c in onehot.columns]

        out = pd.concat([df[CONTINUOUS_FEATURES + BINARY_FEATURES], amen_df, onehot], axis=1)
        return out

    def transform(self, X):
        out = self._assemble(X)
        out[CONTINUOUS_FEATURES] = self.scaler_.transform(out[CONTINUOUS_FEATURES])
        out = out[self.feature_columns_]
        assert out.isna().sum().sum() == 0, "NaNs after preprocessing"
        return out.astype(np.float32)   # float32 parity with the Notebook 02 experiments


# ---------------------------------------------------------------- label hygiene
def fit_price_caps(prices, cities):
    """Tier-3 target winsorisation: per-city train p99.5 caps (training labels only)."""
    return pd.Series(prices).groupby(pd.Series(cities).values).quantile(PRICE_CAP_QUANTILE).to_dict()


def apply_price_caps(prices, cities, caps):
    cap = pd.Series(cities).map(caps).values
    return np.minimum(np.asarray(prices, dtype=float), cap)


# ---------------------------------------------------------------- model pipeline
def build_model_pipeline(hparams, random_state=42):
    """The full deployable object: raw cleaned rows in, PRICE (local currency) out."""
    hgb = HistGradientBoostingRegressor(early_stopping=False, random_state=random_state, **hparams)
    return Pipeline([
        ("engineer", ListingFeatureEngineer()),
        ("prep", AirbnbPreprocessor()),
        ("model", TransformedTargetRegressor(regressor=hgb, func=np.log1p,
                                             inverse_func=np.expm1, check_inverse=False)),
    ])


# ---------------------------------------------------------------- evaluation
def evaluate_predictions(price_pred, price_raw, cities, price_caps):
    """The Notebook 02 metric suite: pooled log-scale MAE/RMSE/R2 + pooled MAPE,
    plus the per-city table in local currency. log targets use the same per-city
    winsorised definition as training (caps fitted on train)."""
    price_raw = np.asarray(price_raw, dtype=float)
    pred_log = np.log1p(np.asarray(price_pred, dtype=float))
    y_log = np.log1p(apply_price_caps(price_raw, cities, price_caps))
    pooled = {
        "test_mae_log": round(mean_absolute_error(y_log, pred_log), 4),
        "test_rmse_log": round(float(np.sqrt(mean_squared_error(y_log, pred_log))), 4),
        "test_r2_log": round(r2_score(y_log, pred_log), 4),
        "test_mape": round(float(np.mean(np.abs(price_pred - price_raw) / price_raw)) * 100, 2),
    }
    rows = []
    cities = pd.Series(list(cities))
    for c in sorted(cities.unique()):
        m = (cities == c).values
        rows.append({"city": c, "currency": CURRENCY[c], "n": int(m.sum()),
                     "mae_ccy": round(mean_absolute_error(price_raw[m], price_pred[m]), 1),
                     "mape_pct": round(float(np.mean(np.abs(price_pred[m] - price_raw[m]) / price_raw[m])) * 100, 1),
                     "r2_log": round(r2_score(y_log[m], pred_log[m]), 3)})
    return pooled, pd.DataFrame(rows).set_index("city")
