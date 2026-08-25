"""Shared feature engineering for the Airbnb nightly-price regression model.

This module is the single source of truth for feature logic. It is imported
unchanged by:
  * the development notebook (local experimentation),
  * the SageMaker Processing job (`preprocess.py`),
  * the SageMaker training job (`train.py`),
  * the inference handler (`inference.py`).

Because there is exactly one definition, training/serving skew cannot be
introduced without editing this file, which changes every path at once.

Rationale for each transformation is documented in Notebook 01, sections 4-5.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import KFold

# ----------------------------------------------------------------------------
# Versioned modelling assumptions
# ----------------------------------------------------------------------------
FX_VERSION = "fx_2021Q1_v1"

# Local currency -> USD, approximate mid-market rates for 2021 Q1, matching the
# vintage of the listings snapshot. This is an ANALYST ASSUMPTION, not data.
FX_RATES = {
    "Paris": 1.1800, "Rome": 1.1800, "New York": 1.0000, "Sydney": 0.7500,
    "Hong Kong": 0.1287, "Istanbul": 0.1200, "Rio de Janeiro": 0.1900,
    "Bangkok": 0.0316, "Mexico City": 0.0500, "Cape Town": 0.0680,
}

CITY_CENTRE = {
    "Paris": (48.8566, 2.3522), "New York": (40.7580, -73.9855),
    "Bangkok": (13.7563, 100.5018), "Rio de Janeiro": (-22.9068, -43.1729),
    "Sydney": (-33.8688, 151.2093), "Istanbul": (41.0082, 28.9784),
    "Rome": (41.9028, 12.4964), "Hong Kong": (22.3193, 114.1694),
    "Mexico City": (19.4326, -99.1332), "Cape Town": (-33.9249, 18.4241),
}

AMENITY_FLAGS = {
    "pool": "Pool", "aircon": "Air conditioning", "dishwasher": "Dishwasher",
    "elevator": "Elevator", "free_parking": "Free parking on premises",
    "washer": "Washer", "dryer": "Dryer", "gym": "Gym", "hot_tub": "Hot tub",
    "workspace": "Dedicated workspace", "self_checkin": "Self check-in",
    "bathtub": "Bathtub", "bbq": "BBQ grill", "breakfast": "Breakfast",
    "tv": "TV", "wifi": "Wifi", "kitchen": "Kitchen", "heating": "Heating",
    "patio": "Patio or balcony", "crib": "Crib",
    "longterm": "Long term stays allowed", "smoke_alarm": "Smoke alarm",
    "pets": "Pets allowed",
}
LUXURY_SET = ["pool", "hot_tub", "gym", "dishwasher", "bathtub",
              "bbq", "elevator", "free_parking", "aircon"]

SUB_SCORES = ["review_scores_accuracy", "review_scores_cleanliness",
              "review_scores_checkin", "review_scores_communication",
              "review_scores_location", "review_scores_value"]
REVIEW_COLS = SUB_SCORES + ["review_scores_rating"]

RESP_ORDER = {"within an hour": 1, "within a few hours": 2,
              "within a day": 3, "a few days or more": 4}

SNAPSHOT_DATE = pd.Timestamp("2021-03-01")
TRIM_LOW, TRIM_HIGH = 0.005, 0.995

CATEGORICAL = ["city", "room_type", "property_grouped"]


def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in km. See Notebook 01 section 5.2."""
    R = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = p2 - p1
    dlam = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlam / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def parse_amenities(series: pd.Series) -> pd.DataFrame:
    """One pass over the JSON amenities column -> count + targeted flags."""
    raw = series.fillna("[]").values
    n = len(raw)
    counts = np.zeros(n, dtype="int16")
    flags = {k: np.zeros(n, dtype="int8") for k in AMENITY_FLAGS}
    for i, v in enumerate(raw):
        try:
            items = set(json.loads(v)) if isinstance(v, str) else set()
        except (ValueError, TypeError):
            items = set()
        counts[i] = len(items)
        for key, label in AMENITY_FLAGS.items():
            if label in items:
                flags[key][i] = 1
    out = pd.DataFrame({"n_amenities": counts}, index=series.index)
    for key, arr in flags.items():
        out[f"am_{key}"] = arr
    return out


class SmoothedTargetEncoder(BaseEstimator, TransformerMixin):
    """Empirical-Bayes target encoder with out-of-fold training values.

    mu_g = (n_g * ybar_g + k * ybar) / (n_g + k)

    `k` is the number of prior pseudo-observations added to every group, and
    the group size at which the estimate sits halfway between the group mean
    and the global prior. See Notebook 01 section 7.
    """

    def __init__(self, column: str = "neighbourhood", k: float = 20.0,
                 n_folds: int = 5, random_state: int = 42):
        self.column = column
        self.k = k
        self.n_folds = n_folds
        self.random_state = random_state

    def _fit_map(self, keys: pd.Series, target: pd.Series, prior: float) -> dict:
        agg = (pd.DataFrame({"k": keys.values, "y": np.asarray(target)})
               .groupby("k")["y"].agg(["mean", "count"]))
        blended = (agg["mean"] * agg["count"] + prior * self.k) / (agg["count"] + self.k)
        return blended.to_dict()

    def fit(self, X: pd.DataFrame, y=None):
        keys = X[self.column].astype(str)
        self.prior_ = float(np.mean(y))
        self.mapping_ = self._fit_map(keys, y, self.prior_)
        return self

    def transform(self, X: pd.DataFrame) -> pd.Series:
        keys = X[self.column].astype(str)
        return keys.map(self.mapping_).astype(float).fillna(self.prior_)

    def fit_transform_oof(self, X: pd.DataFrame, y) -> pd.Series:
        """Fit the full mapping but return leakage-free out-of-fold values."""
        self.fit(X, y)
        keys = X[self.column].astype(str)
        y = pd.Series(np.asarray(y), index=X.index)
        oof = pd.Series(np.nan, index=X.index, dtype=float)
        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
        for tr_idx, va_idx in kf.split(X):
            prior = float(y.iloc[tr_idx].mean())
            mapping = self._fit_map(keys.iloc[tr_idx], y.iloc[tr_idx], prior)
            oof.iloc[va_idx] = keys.iloc[va_idx].map(mapping).astype(float).fillna(prior)
        return oof.fillna(self.prior_)

    def to_dict(self) -> dict:
        return {"column": self.column, "k": self.k,
                "prior": self.prior_, "mapping": self.mapping_}


class AirbnbFeatureEngineer(BaseEstimator, TransformerMixin):
    """Cleaning + feature engineering, with all statistics learned on fit.

    Parameters
    ----------
    include_review_gaps : bool
        `value_gap` and `location_premium_signal` are measured AFTER the price
        was set, so they are downstream of the target. They are safe for an
        estimation model but not for counterfactual "what should I charge?"
        use. Setting this False produces the counterfactual-safe variant,
        which costs about 0.001 R2 (Notebook 02 section 9.6).
    """

    def __init__(self, include_review_gaps: bool = True):
        self.include_review_gaps = include_review_gaps

    # ---------------- fit: learn every statistic from TRAINING data only ----
    def fit(self, X: pd.DataFrame, y=None):
        df = X.copy()
        df = self._add_price_usd(df)

        # Applicability boundary, learned per city.
        if "price_usd" in df.columns and df["price_usd"].notna().any():
            pos = df[df["price_usd"] > 0]
            self.trim_bounds_ = pos.groupby("city")["price_usd"].agg(
                lo=lambda s: s.quantile(TRIM_LOW),
                hi=lambda s: s.quantile(TRIM_HIGH)).to_dict("index")
        else:
            self.trim_bounds_ = {}

        # Imputation statistics.
        self.bedrooms_median_ = (df.groupby(["city", "accommodates"])["bedrooms"]
                                 .median().to_dict())
        self.bedrooms_global_ = float(df["bedrooms"].median()) if df["bedrooms"].notna().any() else 1.0
        self.review_medians_city_ = {
            c: df.groupby("city")[c].median().to_dict() for c in REVIEW_COLS}
        self.review_medians_global_ = {
            c: float(df[c].median()) if df[c].notna().any() else 0.0 for c in REVIEW_COLS}
        self.response_rate_median_ = float(df["host_response_rate"].median()) \
            if df["host_response_rate"].notna().any() else 1.0
        self.acceptance_rate_median_ = float(df["host_acceptance_rate"].median()) \
            if df["host_acceptance_rate"].notna().any() else 1.0
        self.tenure_median_ = float(
            (SNAPSHOT_DATE - pd.to_datetime(df["host_since"], errors="coerce")).dt.days.median())

        # Categorical vocabulary.
        self.top_property_types_ = list(
            df["property_type"].value_counts().head(15).index)

        self.feature_names_ = None
        return self

    # ---------------- transform: apply learned statistics --------------------
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        df = self._add_price_usd(df)

        # --- missingness indicators BEFORE imputation (they are the feature) ---
        df["has_reviews"] = df["review_scores_rating"].notna().astype("int8")
        df["has_response_data"] = df["host_response_rate"].notna().astype("int8")
        df["bedrooms_missing"] = df["bedrooms"].isna().astype("int8")

        # --- sentinel repair ---
        df["minimum_nights"] = pd.to_numeric(
            df["minimum_nights"], errors="coerce").fillna(1).clip(1, 365)
        df["maximum_nights"] = pd.to_numeric(
            df["maximum_nights"], errors="coerce").fillna(1125).clip(1, 1125)
        df["accommodates"] = pd.to_numeric(
            df["accommodates"], errors="coerce").fillna(2).clip(lower=1)

        # --- bedrooms: group-conditional imputation ---
        keys = list(zip(df["city"], df["accommodates"]))
        grp = pd.Series([self.bedrooms_median_.get(k, np.nan) for k in keys], index=df.index)
        df["bedrooms"] = (df["bedrooms"].fillna(grp)
                          .fillna(self.bedrooms_global_).clip(1, 20))

        # --- review scores: city median then global ---
        for c in REVIEW_COLS:
            city_map = self.review_medians_city_.get(c, {})
            df[c] = df[c].fillna(df["city"].map(city_map))
            df[c] = df[c].fillna(self.review_medians_global_.get(c, 0.0))

        df["host_response_rate"] = df["host_response_rate"].fillna(self.response_rate_median_)
        df["host_acceptance_rate"] = df["host_acceptance_rate"].fillna(self.acceptance_rate_median_)
        df["host_total_listings_count"] = pd.to_numeric(
            df["host_total_listings_count"], errors="coerce").fillna(1.0)

        for c in ["host_is_superhost", "host_has_profile_pic", "host_identity_verified",
                  "instant_bookable"]:
            df[c] = (df[c].fillna("f").astype(str).str.lower() == "t").astype("int8")

        # --- geography ---
        clat = df["city"].map(lambda c: CITY_CENTRE.get(c, (0.0, 0.0))[0])
        clon = df["city"].map(lambda c: CITY_CENTRE.get(c, (0.0, 0.0))[1])
        df["dist_centre_km"] = haversine_km(df["latitude"].values, df["longitude"].values,
                                            clat.values, clon.values)
        df["log_dist_centre"] = np.log1p(df["dist_centre_km"])

        # --- amenities ---
        am = parse_amenities(df["amenities"])
        for c in am.columns:
            df[c] = am[c]
        df["luxury_score"] = df[[f"am_{k}" for k in LUXURY_SET]].sum(axis=1).astype("int8")

        # --- capacity density ---
        df["persons_per_bedroom"] = (df["accommodates"] / df["bedrooms"]).clip(0.25, 8)
        df["is_studio"] = ((df["bedrooms"] == 1) & (df["accommodates"] <= 2)).astype("int8")
        df["is_large_group"] = (df["accommodates"] >= 7).astype("int8")

        # --- host ---
        tenure = (SNAPSHOT_DATE - pd.to_datetime(df["host_since"], errors="coerce")).dt.days
        df["host_tenure_days"] = tenure.fillna(self.tenure_median_).clip(lower=0)
        df["log_host_listings"] = np.log1p(df["host_total_listings_count"])
        df["is_professional_host"] = (df["host_total_listings_count"] > 5).astype("int8")
        df["response_speed"] = df["host_response_time"].map(RESP_ORDER).fillna(5).astype("int8")

        # --- reviews ---
        df["review_composite"] = df[SUB_SCORES].mean(axis=1)
        df["value_gap"] = df["review_scores_value"] - df["review_scores_rating"] / 10.0
        df["location_premium_signal"] = df["review_scores_location"] - df["review_composite"]

        # --- stay policy ---
        df["log_min_nights"] = np.log1p(df["minimum_nights"])
        df["is_monthly_only"] = (df["minimum_nights"] >= 30).astype("int8")
        df["booking_window"] = (df["maximum_nights"] - df["minimum_nights"]).clip(lower=0)

        # --- categorical consolidation ---
        df["property_grouped"] = np.where(
            df["property_type"].isin(self.top_property_types_), df["property_type"], "Other")

        out = df[self.output_columns()].copy()
        self.feature_names_ = list(out.columns)
        return out

    # ---------------- helpers ------------------------------------------------
    @staticmethod
    def _add_price_usd(df: pd.DataFrame) -> pd.DataFrame:
        if "price" in df.columns:
            df["price_usd"] = pd.to_numeric(df["price"], errors="coerce") * df["city"].map(FX_RATES)
        return df

    def output_columns(self) -> list:
        cols = [
            "accommodates", "bedrooms", "persons_per_bedroom", "is_studio", "is_large_group",
            "latitude", "longitude", "dist_centre_km", "log_dist_centre",
            "n_amenities", "luxury_score", *[f"am_{k}" for k in AMENITY_FLAGS],
            "host_tenure_days", "log_host_listings", "is_professional_host",
            "host_is_superhost", "host_identity_verified", "host_has_profile_pic",
            "host_response_rate", "host_acceptance_rate", "response_speed",
            "review_scores_rating", "review_composite",
            "review_scores_cleanliness", "review_scores_location",
            "log_min_nights", "is_monthly_only", "booking_window", "instant_bookable",
            "has_reviews", "has_response_data", "bedrooms_missing",
            *CATEGORICAL,
        ]
        if self.include_review_gaps:
            cols += ["value_gap", "location_premium_signal"]
        return cols

    def in_applicability_range(self, df: pd.DataFrame) -> pd.Series:
        """True where price_usd falls inside the city range the model was fitted on."""
        if "price_usd" not in df.columns or not self.trim_bounds_:
            return pd.Series(True, index=df.index)
        lo = df["city"].map(lambda c: self.trim_bounds_.get(c, {}).get("lo", -np.inf))
        hi = df["city"].map(lambda c: self.trim_bounds_.get(c, {}).get("hi", np.inf))
        return (df["price_usd"] >= lo) & (df["price_usd"] <= hi)


def build_target(df: pd.DataFrame) -> pd.Series:
    """log(price_usd) — the regression target. See Notebook 01 section 5.1."""
    price_usd = pd.to_numeric(df["price"], errors="coerce") * df["city"].map(FX_RATES)
    return np.log(price_usd)
