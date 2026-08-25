#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Host / customer prototype on top of the three ITI113 Airbnb models
==================================================================

Two audiences, three models, one shared listing store.

    PRICE     regression, SageMaker serverless. Suggests a nightly rate.
    BOOKING   classification, SageMaker serverless. Instant-booking probability.
              Takes price as a FEATURE, so its answer is conditional on the rate.
    RECSYS    not an endpoint. Test_API_Gradio.ipynb IS the model: a filter and a
              weighted sort over a corpus the other two endpoints pre-scored.
              Reachable over HTTP when that notebook runs, in-process otherwise.

    HOST      pricing assistant. One button returns the suggested rate, a SHAP
              attribution of what drives it against a typical listing in the same
              city, and the instant-booking probability as a friction indicator.
              Then set a price and save. Writes are authenticated.
    CUSTOMER  discovery assistant. Search the corpus, then pick a listing and get
              like-for-like alternatives -- close not only on attributes but on
              modelled fair price and booking friction, which is what you want when
              the one you liked is booked or over budget. Also runs either model
              directly, and browses saved listings with a *fairness verdict*.
              Read only, no login.

The explanation is real KernelSHAP, computed against the endpoint already deployed:
the price contract accepts a list of records, so every coalition fits in one or two
invocations and no pipeline change is needed. `--self-test` checks each attribution
against the closed form for an additive model, coef_i * (x_i - E[x_i]), and checks
that baseline plus attributions equals the prediction.

Why this is a separate file from the notebook demo
--------------------------------------------------
The notebook front-end is stateless: it posts fields and renders responses, and
holds nothing. This app holds data, which brings three concerns the notebook does
not have and should not inherit -- persistence, authentication, and staleness
(a verdict computed against champion v7 is wrong once v8 ships). Keeping them
apart keeps the notebook's claim honest and its failure modes simple.

Run
---
    python host_customer_app.py --role host       # authenticated, writes  :7861
    python host_customer_app.py --role customer   # public, read only      :7862
    python host_customer_app.py --role both       # single-process demo    :7859
    python host_customer_app.py --self-test       # no UI, no AWS needed

    HOST_USERS='alice:pw1,bob:pw2' python host_customer_app.py --role host
    python host_customer_app.py --recsys-backend inline   # ignore the notebook API

Explanation cost is coalitions x background rows, sent in batches of 1000. Roughly
1.5k records for a typical listing. EXPLAIN_BACKGROUND_N and EXPLAIN_NSAMPLES trade
accuracy against endpoint traffic; without `shap` installed it degrades to
leave-one-out occlusion in a single call, labelled as such in the UI.

Recommender backends
--------------------
--recsys-backend api      POST to Test_API_Gradio.ipynb's FastAPI server, exactly as
                          API_Test_Client.ipynb does. Requires that cell to be running.
                   inline same filter and sort in-process against the same CSV.
                   auto   (default) api, falling back to inline.

Inline is not merely a fallback. The FastAPI cell fits its MinMaxScaler once,
globally, over all ~280k rows before any filtering; notebook 05's own evaluation
functions fit it per candidate set. Global scaling flattens the value term to
nothing, so the served ranking was driven entirely by instant_book_prob and put an
overpriced listing at rank 1. `--self-test` reproduces both behaviours. Use
RECSYS_SCALING=global to serve the notebook's exact numbers for comparison.

Trust boundary
--------------
Gradio's auth is per-application, not per-tab. `--role both` puts the host and
customer views in one process behind one login, which is fine for a demo and is
NOT a security boundary: anyone who can reach the app can click the Host tab.
For anything resembling real use, run two processes -- `--role host` behind auth
on a private port, `--role customer` public -- so the boundary is enforced by the
process, not by a tab label.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import boto3
import gradio as gr

# ============================================================================
# 1. CONFIGURATION
# ============================================================================

TEAM_ID = os.environ.get("TEAM_ID", "team14")
STUDENT_ID = os.environ.get("STUDENT_ID", "s1402")
COURSE = "ITI113"
REGION = os.environ.get("AWS_REGION", "ap-southeast-1")

PRICE_ENDPOINT = os.environ.get("PRICE_ENDPOINT", f"iti113-{TEAM_ID}-airbnb-price-sls")
BOOKING_ENDPOINT = os.environ.get("BOOKING_ENDPOINT", f"iti113-{TEAM_ID}-airbnb-instant-booking")

# The recommender is not an endpoint. Test_API_Gradio.ipynb IS the model: a pandas
# filter-and-rank over a pre-scored CSV, wrapped in FastAPI so API_Test_Client.ipynb
# could reach it from a second notebook. Two backends, same signature:
#
#   api     POST the SearchQuery to that running notebook, exactly as the client does
#   inline  do the same work in-process against the CSV, no second server
#   auto    try api, fall back to inline (default -- the notebook is often not running)
#
# Inline is not just a fallback. The API fits its MinMaxScaler once, globally, over
# all ~280k rows at import; the eval functions in notebook 05 fit it per candidate
# set. Global scaling flattens the value term to nothing (see RECSYS_SCALING below),
# so inline is also the only mode where w_value does anything.
RECSYS_API_URL = os.environ.get("RECSYS_API_URL", "http://localhost:8000/recommend")
RECSYS_BACKEND = os.environ.get("RECSYS_BACKEND", "auto").lower()
RECSYS_SCALING = os.environ.get("RECSYS_SCALING", "candidates").lower()   # candidates | global

BUCKET = os.environ.get("BUCKET", "nyp-26s1-iti113")
PREFIX = f"iti113/{TEAM_ID}/data/airbnb-listings"
PROCESSED_PREFIX = f"{PREFIX}/processed"
ARTIFACTS_PREFIX = f"{PREFIX}/artifacts"

# Notebook 05's output: raw Listings.csv joined to pred_price and instant_book_prob,
# which is the recommender's entire brain.
RECSYS_PREFIX = f"iti113/{TEAM_ID}/data/airbnb-recommendation"
RECSYS_CSV = os.environ.get(
    "RECSYS_CSV", f"s3://{BUCKET}/{RECSYS_PREFIX}/processed/master_recsys_dataset.csv")

# The shared listing store. S3 first so it survives a Studio restart and is
# visible to a customer app running in another process; local file as fallback.
STORE_KEY = f"iti113/{TEAM_ID}/apps/listing-store/listings.json"
STORE_LOCAL = os.environ.get("STORE_LOCAL", "listing_store.json")
USE_S3 = os.environ.get("STORE_BACKEND", "s3").lower() == "s3"

# Demo credentials. Real deployments read these from Secrets Manager, not env.
HOST_USERS = os.environ.get("HOST_USERS", "host:iti113")

sm = boto3.client("sagemaker", region_name=REGION)
runtime = boto3.client("sagemaker-runtime", region_name=REGION)
s3 = boto3.client("s3", region_name=REGION)


# ============================================================================
# 2. THE SERVING CONTRACT
# ============================================================================
# pipeline_lib is authoritative when it is importable; mirrored otherwise so this
# app runs on a laptop with nothing else checked out.
# ============================================================================

for _p in ("src", ".", "../src"):
    if os.path.isdir(_p) and os.path.abspath(_p) not in sys.path:
        sys.path.insert(0, os.path.abspath(_p))

try:
    from pipeline_lib import (CITY_CENTERS, CURRENCY, RESPONSE_TIME_LEVELS, REVIEW_COLS,
                              SERVING_COLUMNS, SNAPSHOT_DATE, property_group)
    SNAPSHOT_DATE = date(SNAPSHOT_DATE.year, SNAPSHOT_DATE.month, SNAPSHOT_DATE.day)
    CONTRACT_SOURCE = "pipeline_lib"
except Exception:
    CONTRACT_SOURCE = "mirrored"
    SNAPSHOT_DATE = date(2021, 3, 1)
    CITY_CENTERS = {
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
    SERVING_COLUMNS = [
        "city", "latitude", "longitude", "neighbourhood", "property_type", "room_type",
        "accommodates", "bedrooms", "amenities", "minimum_nights", "maximum_nights",
        "instant_bookable", "host_since", "host_is_superhost", "host_has_profile_pic",
        "host_identity_verified", "host_total_listings_count", "host_response_time",
        "host_response_rate", "host_acceptance_rate",
    ] + REVIEW_COLS

    def property_group(pt):
        t = str(pt).lower()
        if any(k in t for k in ("hotel", "hostel", "resort")):                      return "hotel_hostel"
        if any(k in t for k in ("bed and breakfast", "guesthouse", "guest suite")): return "bnb_guesthouse"
        if any(k in t for k in ("boat", "camper", "tent", "castle", "treehouse",
                                "yurt", "tiny house", "island", "cave")):          return "unique_stay"
        if any(k in t for k in ("apartment", "condominium", "loft", "serviced")):  return "apartment_condo"
        if any(k in t for k in ("house", "townhouse", "villa", "cottage",
                                "bungalow", "cabin", "chalet")):                   return "house_villa"
        if "private room" in t or "shared room" in t or "room in" in t:            return "room_other_dwelling"
        return "other"

CITIES = sorted(CITY_CENTERS)
R_EARTH_KM = 6371.0

FALLBACK_AMENITIES = [
    "wifi", "kitchen", "air conditioning", "heating", "washer", "dryer", "tv", "essentials",
    "hangers", "iron", "shampoo", "hair dryer", "hot water", "refrigerator", "microwave",
    "coffee maker", "cooking basics", "oven", "stove", "dishes and silverware", "elevator",
    "free parking on premises", "pool", "gym", "smoke alarm", "carbon monoxide alarm",
    "fire extinguisher", "first aid kit", "long term stays allowed", "dedicated workspace",
]
FALLBACK_ROOM_TYPES = ["Entire place", "Private room", "Shared room", "Hotel room"]
FALLBACK_PROPERTY_TYPES = [
    "Entire apartment", "Entire condominium", "Entire house", "Entire loft",
    "Entire townhouse", "Entire villa", "Private room in apartment", "Private room in house",
    "Private room in bed and breakfast", "Room in hotel", "Shared room in hostel", "Boat",
]
FALLBACK_REVIEW_DEFAULTS = {
    "review_scores_rating": 95.0, "review_scores_accuracy": 10.0,
    "review_scores_cleanliness": 9.0, "review_scores_checkin": 10.0,
    "review_scores_communication": 10.0, "review_scores_location": 10.0,
    "review_scores_value": 9.0,
}
FALLBACK_PER_CITY = {
    "Bangkok": 37.4, "Cape Town": 34.2, "Hong Kong": 34.4, "Istanbul": 39.0,
    "Mexico City": 33.5, "New York": 29.8, "Paris": 27.6, "Rio de Janeiro": 44.6,
    "Rome": 33.6, "Sydney": 32.6,
}

ARTIFACTS: dict[str, Any] = {}
EVAL_REPORT: dict[str, Any] = {}


def _load_json(filename, prefixes):
    for folder in ("artifacts", "processed", ".", "../artifacts"):
        p = Path(folder, filename)
        if p.is_file():
            try:
                return json.loads(p.read_text())
            except Exception:
                pass
    for prefix in prefixes:
        try:
            return json.loads(s3.get_object(Bucket=BUCKET, Key=f"{prefix}/{filename}")["Body"].read())
        except Exception:
            continue
    return None


def load_fitted_state():
    global ARTIFACTS, EVAL_REPORT
    ARTIFACTS = _load_json("preprocessing_artifacts.json",
                           [PROCESSED_PREFIX, ARTIFACTS_PREFIX]) or {}
    EVAL_REPORT = _load_json("evaluation_report.json", [ARTIFACTS_PREFIX]) or {}


def amenity_choices():
    return ARTIFACTS.get("amenity_vocab") or FALLBACK_AMENITIES


def room_type_choices():
    return ARTIFACTS.get("onehot_levels", {}).get("room_type") or FALLBACK_ROOM_TYPES


def response_time_choices():
    return ARTIFACTS.get("onehot_levels", {}).get("host_response_time") or list(RESPONSE_TIME_LEVELS)


def known_neighbourhoods(city):
    freq = ARTIFACTS.get("neighbourhood_freq") or {}
    return sorted({k.split("||", 1)[1] for k in freq if k.split("||", 1)[0] == city})


def review_defaults():
    med = ARTIFACTS.get("imputation_values") or {}
    return {c: float(med.get(c, FALLBACK_REVIEW_DEFAULTS[c])) for c in REVIEW_COLS}


def review_slider_max(col, default_value):
    if col != "review_scores_rating":
        return 10.0
    return 100.0 if default_value > 10 else (10.0 if default_value > 5 else 5.0)


def city_mape(city) -> float | None:
    for row in EVAL_REPORT.get("per_city", []) or []:
        if row.get("city") == city:
            return float(row["mape_pct"])
    return FALLBACK_PER_CITY.get(city)


# ============================================================================
# 3. THE SHARED LISTING STORE
# ============================================================================
# gr.State is per session, so it cannot carry data between two audiences. The
# data has to leave the process. Two backends, same interface.
#
# Concurrency: last write wins. The local backend writes through os.replace so a
# reader never sees a half-written file; the S3 backend relies on put_object
# being atomic per object. Neither does read-modify-write locking, so two hosts
# editing the SAME listing id within the same second can lose one edit. That is
# acceptable for a prototype and is the first thing to fix if this grows up
# (S3 conditional writes on ETag, or DynamoDB with a conditional expression).
# ============================================================================

def store_read() -> dict:
    if USE_S3:
        try:
            return json.loads(s3.get_object(Bucket=BUCKET, Key=STORE_KEY)["Body"].read())
        except Exception:
            return {}
    if not os.path.exists(STORE_LOCAL):
        return {}
    try:
        return json.loads(Path(STORE_LOCAL).read_text())
    except Exception:
        return {}


def store_write(data: dict) -> None:
    body = json.dumps(data, indent=2, sort_keys=True)
    if USE_S3:
        s3.put_object(Bucket=BUCKET, Key=STORE_KEY, Body=body.encode(),
                      ContentType="application/json")
        return
    tmp = STORE_LOCAL + ".tmp"
    Path(tmp).write_text(body)
    os.replace(tmp, STORE_LOCAL)          # atomic: no reader sees a partial file


def store_backend_name() -> str:
    return f"s3://{BUCKET}/{STORE_KEY}" if USE_S3 else os.path.abspath(STORE_LOCAL)


# ============================================================================
# 4. THE PRIVACY SPLIT
# ============================================================================
# The pipeline already treats coordinates as sensitive: ListingFeatureEngineer
# derives dist_center_km, coarsens lat/lon to 2 decimals, and discards the exact
# values so precise GPS never reaches a stored artifact. Publishing exact
# coordinates to customers would undo that on the way back out.
#
# So the customer view is an ALLOWLIST, not a denylist. A field is invisible
# unless it is named here, which means adding a field to the host form cannot
# silently expose it. HOST_PRIVATE is then asserted as a second, redundant check.
# ============================================================================

# The split mirrors what a real listing page publishes, so it is defensible rather
# than arbitrary: response rate and response time are public on Airbnb, acceptance
# rate is host-dashboard only, and the exact pin is withheld until booking.
CUSTOMER_VISIBLE = {
    "city", "neighbourhood", "property_type", "room_type", "accommodates", "bedrooms",
    "amenities", "minimum_nights", "maximum_nights", "instant_bookable",
    "host_is_superhost", "host_identity_verified", "host_has_profile_pic",
    "host_response_time", "host_response_rate", *REVIEW_COLS,
}

HOST_PRIVATE = {
    "latitude",                   # exact GPS -- the whole point of the coarsening
    "longitude",
    "host_since",                 # exact join date; a listing page shows the year at most
    "host_acceptance_rate",       # host dashboard only, and the more sensitive of the two rates
    "host_total_listings_count",  # portfolio size
}

assert not (CUSTOMER_VISIBLE & HOST_PRIVATE), "a field cannot be both public and private"
assert CUSTOMER_VISIBLE | HOST_PRIVATE >= set(SERVING_COLUMNS), \
    f"unclassified contract fields: {set(SERVING_COLUMNS) - CUSTOMER_VISIBLE - HOST_PRIVATE}"


def haversine_km(lat1, lon1, lat2, lon2) -> float:
    p = math.pi / 180.0
    a = (math.sin((lat2 - lat1) * p / 2) ** 2
         + math.cos(lat1 * p) * math.cos(lat2 * p) * math.sin((lon2 - lon1) * p / 2) ** 2)
    return 2 * R_EARTH_KM * math.asin(math.sqrt(a))


def public_view(record: dict) -> dict:
    """Everything a customer may see, and nothing else."""
    payload = record["payload"]
    view = {k: v for k, v in payload.items() if k in CUSTOMER_VISIBLE}

    # Location, at the resolution the model itself uses.
    city = payload.get("city")
    if city in CITY_CENTERS and payload.get("latitude") is not None:
        clat, clon = CITY_CENTERS[city]
        view["dist_center_km"] = round(
            haversine_km(payload["latitude"], payload["longitude"], clat, clon), 2)

    view["asking_price"] = record["asking_price"]
    view["currency"] = CURRENCY.get(city, "")

    leaked = HOST_PRIVATE & set(view)
    if leaked:                     # belt and braces: allowlist should make this unreachable
        raise AssertionError(f"customer view leaked host-private fields: {sorted(leaked)}")
    return view


# ============================================================================
# 5. THE ENDPOINT, AND WHICH MODEL ANSWERED
# ============================================================================
# A saved verdict is only meaningful next to the model that produced it. Stamp
# every record with the deployed model package so the customer view can say
# "this was computed against an older model" instead of quietly showing a stale
# number after a redeploy.
# ============================================================================

# Keyed BY ENDPOINT. This was a flat dict while there was only one model, which
# made model_lineage() ignore its own endpoint_name argument: asking for the
# booking endpoint after the price endpoint returned the price endpoint's stamp,
# silently, and stamped it onto saved records. One model hid the bug; two expose it.
_LINEAGE_CACHE: dict[str, dict] = {}


def model_lineage(endpoint_name: str = PRICE_ENDPOINT, refresh: bool = False) -> dict:
    # Always hand back a COPY. Callers persist this into the store, and a cached
    # dict returned by reference would let a later refresh silently rewrite the
    # stamp on every record already saved -- defeating the whole point of it.
    cached = _LINEAGE_CACHE.get(endpoint_name)
    if cached and not refresh:
        return dict(cached)
    out = {"endpoint": endpoint_name, "model_package": None, "model": None, "status": "Unknown"}
    try:
        desc = sm.describe_endpoint(EndpointName=endpoint_name)
        out["status"] = desc["EndpointStatus"]
        cfg = sm.describe_endpoint_config(EndpointConfigName=desc["EndpointConfigName"])
        variant = cfg["ProductionVariants"][0]
        out["model"] = variant.get("ModelName")
        model = sm.describe_model(ModelName=out["model"])
        containers = model.get("Containers") or [model.get("PrimaryContainer", {})]
        out["model_package"] = containers[0].get("ModelPackageName")
    except Exception as e:
        out["error"] = f"{e.__class__.__name__}: {e}"
    _LINEAGE_CACHE[endpoint_name] = dict(out)
    return dict(out)


def invoke_price(payload: dict) -> tuple[dict, float]:
    t0 = time.perf_counter()
    resp = runtime.invoke_endpoint(EndpointName=PRICE_ENDPOINT, ContentType="application/json",
                                   Body=json.dumps(payload))
    records = json.loads(resp["Body"].read())
    return records[0], (time.perf_counter() - t0) * 1000.0


def invoke_booking(payload: dict) -> tuple[dict, float]:
    """Instant-booking classifier. Different envelope from the price endpoint.

    inference.py's _predict_fn accepts a bare dict, {"data": dict}, {"data": list}
    or a bare list; the price endpoint's inference.py takes the list form only.
    Sending {"data": [payload]} keeps both readable rather than relying on the
    classifier's more forgiving parser to paper over the difference.
    """
    t0 = time.perf_counter()
    resp = runtime.invoke_endpoint(EndpointName=BOOKING_ENDPOINT, ContentType="application/json",
                                   Accept="application/json", Body=json.dumps({"data": [payload]}))
    records = json.loads(resp["Body"].read())
    return records[0], (time.perf_counter() - t0) * 1000.0


def invoke_both(price_payload: dict, booking_payload: dict) -> dict:
    """Fan out to both endpoints at once.

    Serverless scales to zero and a cold start runs to ~60 s, so doing these in
    sequence makes the first click of the day take two minutes. Partial failure is
    the normal case here, not the edge case: each side is reported independently so
    one cold endpoint cannot blank out the other's answer.
    """
    from concurrent.futures import ThreadPoolExecutor
    out: dict[str, Any] = {}
    with ThreadPoolExecutor(max_workers=2) as pool:
        jobs = {"price": pool.submit(invoke_price, price_payload),
                "booking": pool.submit(invoke_booking, booking_payload)}
        for name, fut in jobs.items():
            try:
                out[name], out[f"{name}_ms"] = fut.result()
            except Exception as e:
                out[f"{name}_error"] = e
    return out


def explain_error(exc: Exception) -> str:
    name, msg = exc.__class__.__name__, str(exc)
    hint = ""
    if "ValidationException" in name + msg:
        hint = f"No endpoint `{PRICE_ENDPOINT}` in `{REGION}`."
    elif "AccessDenied" in name + msg:
        hint = "This role cannot invoke the endpoint, or cannot write to the store prefix."
    elif "ModelError" in name + msg:
        hint = "The endpoint rejected the payload. Compare it against the 27 contract fields."
    elif "Throttling" in name + msg:
        hint = "Serverless endpoint at max_concurrency=5. Try again shortly."
    elif "NoCredentials" in name or "ExpiredToken" in msg:
        hint = "AWS credentials missing or expired."
    elif "ReadTimeout" in name or "timed out" in msg.lower():
        hint = "Serverless cold start, up to ~60 s on the first call. Send it again."
    return f"**{name}**\n\n```\n{msg}\n```\n\n{hint}".rstrip()


# ============================================================================
# 6. THE FAIRNESS VERDICT
# ============================================================================
# What a customer wants is not "what should this cost" -- they can already see the
# price -- but "is this price reasonable for a listing like this". Same model,
# different question, and the model's own error band decides when the gap is
# worth mentioning. Without the band you end up accusing hosts of overcharging on
# noise: Paris is +/-27.6%, so a listing 20% above the suggestion is unremarkable.
# ============================================================================

def fairness_verdict(asking: float, suggested: float, mape_pct: float | None) -> dict:
    gap_pct = (asking - suggested) / suggested * 100.0
    band = mape_pct if mape_pct is not None else 0.0

    if mape_pct is None:
        tone, headline = "unknown", "No error estimate available for this city"
    elif abs(gap_pct) <= band:
        tone, headline = "normal", "In line with comparable listings"
    elif gap_pct > band * 2:
        tone, headline = "high", "Well above comparable listings"
    elif gap_pct > band:
        tone, headline = "high", "Above comparable listings"
    elif gap_pct < -band * 2:
        tone, headline = "low", "Well below comparable listings"
    else:
        tone, headline = "low", "Below comparable listings"

    return {"gap_pct": gap_pct, "band_pct": band, "tone": tone, "headline": headline}


def verdict_is_stale(record: dict) -> str | None:
    """Was this verdict computed against a model that is no longer deployed?"""
    saved = (record.get("model") or {}).get("model_package")
    if not saved:
        return "This listing was saved without a model stamp, so it cannot be checked."
    current = model_lineage().get("model_package")
    if current and saved != current:
        return (f"Computed against `{saved.rsplit('/', 1)[-1]}`, but the endpoint now serves "
                f"`{current.rsplit('/', 1)[-1]}`. The host should re-save to refresh it.")
    return None


# ============================================================================
# 7. HOST SIDE -- FORM, PAYLOAD, SAVE
# ============================================================================

HOST_FIELDS = [
    "listing_id", "asking_price",
    "city", "neighbourhood", "latitude", "longitude",
    "property_type", "room_type", "accommodates", "bedrooms",
    "amenities_selected", "amenities_extra",
    "minimum_nights", "maximum_nights", "instant_bookable",
    "host_since", "host_is_superhost", "host_has_profile_pic", "host_identity_verified",
    "host_total_listings_count", "host_response_time", "host_response_rate",
    "host_acceptance_rate", "has_reviews", *REVIEW_COLS,
]


def _tf(flag) -> str:
    return "t" if bool(flag) else "f"


def _amenities(selected, extra) -> list[str]:
    items = list(selected or [])
    for token in re.split(r"[,;\n]", extra or ""):
        if token.strip():
            items.append(token.strip())
    return sorted({a.strip().lower() for a in items if a.strip()})


def build_payload(ui: dict) -> dict:
    has_reviews = bool(ui.get("has_reviews", True))
    payload = {
        "city": ui["city"],
        "latitude": float(ui["latitude"]),
        "longitude": float(ui["longitude"]),
        "neighbourhood": (ui.get("neighbourhood") or "").strip() or None,
        "property_type": (ui.get("property_type") or "").strip() or "Other",
        "room_type": ui["room_type"],
        "accommodates": int(ui["accommodates"]),
        "bedrooms": int(ui["bedrooms"]),
        "amenities": json.dumps(_amenities(ui.get("amenities_selected"), ui.get("amenities_extra"))),
        "minimum_nights": int(ui["minimum_nights"]),
        "maximum_nights": int(ui["maximum_nights"]),
        "instant_bookable": _tf(ui.get("instant_bookable")),
        "host_since": (ui.get("host_since") or "").strip() or None,
        "host_is_superhost": _tf(ui.get("host_is_superhost")),
        "host_has_profile_pic": _tf(ui.get("host_has_profile_pic")),
        "host_identity_verified": _tf(ui.get("host_identity_verified")),
        "host_total_listings_count": int(ui.get("host_total_listings_count") or 0),
        "host_response_time": ui.get("host_response_time") or None,
        "host_response_rate": float(ui["host_response_rate"]),
        "host_acceptance_rate": float(ui["host_acceptance_rate"]),
    }
    for col in REVIEW_COLS:
        payload[col] = float(ui[col]) if has_reviews else None
    return {k: payload[k] for k in SERVING_COLUMNS}


# The classifier's contract is NOT SERVING_COLUMNS. Three differences, each of which
# silently degrades the prediction rather than raising, so none are optional:
#
#   price      REQUIRED here, absent there. create_features() builds
#              log_price_city_zscore from it, so the classifier answers a
#              conditional question -- P(instant book | this listing AT THIS PRICE).
#              That is what makes the two-price comparison on the host tab possible.
#   booleans   ints, not 't'/'f'. Notebook 05 maps them before batch scoring; the
#              't' string would one-hot as an unseen category and quietly vanish.
#   dropped    neighbourhood is not in the classifier's feature schema.
#
# Scales follow Listings.csv, which is what both models trained on: response and
# acceptance rates 0-100, review_scores_rating 0-100, the other six 0-10. The test
# payload in notebook 03 cell 39 uses 0-1 and 0-5 instead, and room_type
# "Entire home/apt" which does not occur in Listings.csv at all -- it survives as an
# unseen OHE category. Do not copy those values as a reference.
BOOKING_REQUIRED = ["city", "amenities", "host_total_listings_count", "price"]

BOOKING_COLUMNS = [
    "city", "latitude", "longitude", "property_type", "room_type",
    "accommodates", "bedrooms", "amenities", "price",
    "minimum_nights", "maximum_nights",
    "host_since", "host_is_superhost", "host_has_profile_pic", "host_identity_verified",
    "host_total_listings_count", "host_response_time", "host_response_rate",
    "host_acceptance_rate",
] + REVIEW_COLS


def _int_flag(flag) -> int:
    return 1 if bool(flag) else 0


def build_booking_payload(ui: dict, price: float | None = None) -> dict:
    """Classifier payload. `price` overrides the asking price for counterfactuals."""
    has_reviews = bool(ui.get("has_reviews", True))
    p = float(price if price is not None else ui["asking_price"])
    payload = {
        "city": ui["city"],
        "latitude": float(ui["latitude"]),
        "longitude": float(ui["longitude"]),
        "property_type": (ui.get("property_type") or "").strip() or "Other",
        "room_type": ui["room_type"],
        "accommodates": int(ui["accommodates"]),
        "bedrooms": int(ui["bedrooms"]),
        "amenities": json.dumps(_amenities(ui.get("amenities_selected"), ui.get("amenities_extra"))),
        "price": p,
        "minimum_nights": int(ui["minimum_nights"]),
        "maximum_nights": int(ui["maximum_nights"]),
        "host_since": (ui.get("host_since") or "").strip() or None,
        "host_is_superhost": _int_flag(ui.get("host_is_superhost")),
        "host_has_profile_pic": _int_flag(ui.get("host_has_profile_pic")),
        "host_identity_verified": _int_flag(ui.get("host_identity_verified")),
        "host_total_listings_count": int(ui.get("host_total_listings_count") or 0),
        "host_response_time": ui.get("host_response_time") or None,
        "host_response_rate": float(ui["host_response_rate"]),
        "host_acceptance_rate": float(ui["host_acceptance_rate"]),
    }
    for col in REVIEW_COLS:
        payload[col] = float(ui[col]) if has_reviews else None
    missing = [c for c in BOOKING_REQUIRED if payload.get(c) is None]
    if missing:
        raise ValueError(f"classifier requires: {', '.join(missing)}")
    return {k: payload[k] for k in BOOKING_COLUMNS}


def booking_verdict(prob: float, threshold: float = 0.50) -> dict:
    """Frame the classifier as a peer benchmark, not as a prediction of the host's own choice.

    `instant_bookable` is an INPUT to the price model and the TARGET of this one, so
    "we predict whether you enable instant booking" is incoherent next to a checkbox
    where the host just told us. Read as a peer rate it is coherent and useful: how
    many listings like this one, at this price, offer instant booking.

    No +/- band here. MAPE gives the regression a defensible tolerance; the honest
    equivalent for a classifier is calibration, and presenting a probability next to
    a percentage error band would be a category error.
    """
    if prob is None:
        return {"tone": "unknown", "headline": "No probability returned"}
    if prob >= 0.75:
        return {"tone": "normal", "headline": "Most comparable listings offer instant booking"}
    if prob >= threshold:
        return {"tone": "normal", "headline": "Comparable listings lean towards instant booking"}
    if prob >= 0.25:
        return {"tone": "low", "headline": "Comparable listings lean against instant booking"}
    return {"tone": "low", "headline": "Few comparable listings offer instant booking"}


LISTING_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{2,31}$")


def validate_host(ui: dict, payload: dict) -> list[str]:
    problems = []
    lid = (ui.get("listing_id") or "").strip()
    if not LISTING_ID_RE.match(lid):
        problems.append("Listing ID must be 3-32 characters: letters, digits, hyphen, underscore.")
    if not ui.get("asking_price") or float(ui["asking_price"]) <= 0:
        problems.append("Asking price must be greater than zero.")
    if payload["city"] not in CITY_CENTERS:
        problems.append(f"City must be one of: {', '.join(CITIES)}.")
    if payload["accommodates"] < 1:
        problems.append("Accommodates must be at least 1.")
    if payload["minimum_nights"] > payload["maximum_nights"]:
        problems.append("Minimum nights is greater than maximum nights.")
    for k in SERVING_COLUMNS:
        if k not in payload:
            problems.append(f"Missing contract field: {k}.")
    return problems


def current_user(request) -> str | None:
    """The authenticated username, or None when the app runs without auth.

    Returning None rather than "anonymous" matters: an unauthenticated save
    creates an UNOWNED listing that anyone can later edit or delete. Inventing a
    pseudo-user would instead strand the record the moment auth is switched on.
    """
    return getattr(request, "username", None) or None


def save_listing(listing_id: str, payload: dict, asking_price: float,
                 prediction: dict | None, username: str | None) -> dict:
    store = store_read()
    existing = store.get(listing_id)
    if existing and existing.get("owner") not in (None, username):
        raise PermissionError(ownership_message(listing_id, existing.get("owner"), username))

    store[listing_id] = {
        "payload": payload,
        "asking_price": float(asking_price),
        "prediction": prediction,
        "model": model_lineage(),
        "owner": existing.get("owner") if existing else username,
        "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "updated_by": username or "unauthenticated",
    }
    store_write(store)
    return store[listing_id]


def ownership_message(listing_id: str, owner: str | None, username: str | None) -> str:
    """Name both sides. An error that hides the actual owner cannot be acted on."""
    you = f"`{username}`" if username else "not signed in (no auth on this app)"
    return (f"`{listing_id}` is owned by `{owner}`, and you are {you}. "
            f"Only the owner can change it.\n\n"
            f"To inspect or repair the store from the command line:\n"
            f"```\n"
            f"python host_customer_app.py --list-store\n"
            f"python host_customer_app.py --set-owner {listing_id}:{username or 'YOURNAME'}\n"
            f"python host_customer_app.py --reset-store\n"
            f"```")


def delete_listing(listing_id: str, username: str | None) -> str:
    store = store_read()
    rec = store.get(listing_id)
    if rec is None:
        return f"No listing `{listing_id}`."
    if rec.get("owner") not in (None, username):
        return ownership_message(listing_id, rec.get("owner"), username)
    del store[listing_id]
    store_write(store)
    return f"Deleted `{listing_id}`."


# ============================================================================
# 7B. THE RECOMMENDER
# ============================================================================
# Lifted from Test_API_Gradio.ipynb (the ranker) and API_Test_Client.ipynb (the
# request shape). Those two notebooks are the whole model: there is no artifact and
# no endpoint behind them, only a filter and a weighted sort over a CSV that
# notebook 05 pre-scored through the other two endpoints. So this section does not
# import from 05 -- it reads 05's output.
#
# ONE DELIBERATE DIVERGENCE FROM THE NOTEBOOK, and it changes the results.
#
# The FastAPI cell fits its MinMaxScaler once at import, over all ~280k rows and all
# ten cities, before any filtering. compare_recsys() and evaluate_ndcg() in notebook
# 05 fit it per candidate set instead. Global min/max are set by price outliers
# across every city, so after filtering to one city every candidate lands inside a
# sliver of the range and the value term contributes nothing. The API_Test_Client
# output shows it: at w_value=0.5 the top three Paris results have deal_values of
# -20.16, +25.19 and +9.82 -- a 45-unit spread -- and ml_scores of 0.9835, 0.9832
# and 0.9831. Under 0.001 apart. Ranking was 100% instant_book_prob, and rank #1 was
# an OVERPRICED listing (EUR 85 asking against EUR 64.84 predicted).
#
# So RECSYS_SCALING defaults to "candidates". Set it to "global" to reproduce the
# notebook's behaviour exactly; the self test asserts the difference is real.
# It also means the NDCG in notebook 05 describes candidate-set scaling and never
# described what the API actually served.
# ============================================================================

# The privacy allowlist, applied at load. usecols= is a stronger guarantee than
# filtering on the way out: a column that is never read cannot leak through a
# display bug, and the pre-scored CSV is raw Listings.csv, so every HOST_PRIVATE
# field is sitting in that object. `name` is also withheld -- it is free text and
# hosts put cross streets in it.
RECSYS_COLUMNS = [
    "listing_id", "city", "neighbourhood", "property_type", "room_type",
    "accommodates", "bedrooms", "minimum_nights", "price",
    "instant_bookable", "pred_price", "instant_book_prob",
    "amenities",          # ~80 MB of the corpus, and worth it: without it the price
]                         # explanation can never attribute anything to amenities

assert not (set(RECSYS_COLUMNS) & HOST_PRIVATE), \
    f"recommender would read host-private columns: {sorted(set(RECSYS_COLUMNS) & HOST_PRIVATE)}"

_RECSYS_DF: Any = None
_RECSYS_ERROR: str | None = None


def recsys_load(force: bool = False):
    """Load the pre-scored corpus once. Projected to RECSYS_COLUMNS at read time."""
    global _RECSYS_DF, _RECSYS_ERROR
    if _RECSYS_DF is not None and not force:
        return _RECSYS_DF
    try:
        import pandas as pd
        df = pd.read_csv(RECSYS_CSV, usecols=lambda c: c in set(RECSYS_COLUMNS),
                         encoding="utf-8", encoding_errors="replace", low_memory=False)
        for col in ("price", "pred_price", "instant_book_prob", "accommodates"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=["city", "price", "pred_price", "instant_book_prob"])
        _RECSYS_DF, _RECSYS_ERROR = df, None
    except Exception as e:
        _RECSYS_DF, _RECSYS_ERROR = None, f"{e.__class__.__name__}: {e}"
    return _RECSYS_DF


def recsys_cities() -> list[str]:
    """Cities the corpus actually contains, not the ten the price model knows."""
    if RECSYS_BACKEND == "api":
        return CITIES            # no corpus to consult; don't pull it just for a dropdown
    df = recsys_load()
    if df is None or df.empty:
        return CITIES
    return sorted(df["city"].dropna().astype(str).unique())


def _minmax(values):
    lo, hi = float(values.min()), float(values.max())
    if hi <= lo:                       # every candidate identical: no signal to scale
        return values * 0.0 + 0.5
    return (values - lo) / (hi - lo)


def recsys_rank(city: str, min_guests: int = 2, max_price: float = 150.0,
                w_value: float = 0.5, w_convenience: float = 0.5,
                top_n: int = 10) -> tuple[list[dict], str]:
    """In-process ranking. Same three steps as the notebook: filter, score, sort."""
    df = recsys_load()
    if df is None:
        raise RuntimeError(_RECSYS_ERROR or "corpus not loaded")

    if RECSYS_SCALING == "global":
        pool = df.copy()
        pool["scaled_value_score"] = _minmax(pool["pred_price"] - pool["price"])
        candidates = pool[(pool["city"].str.lower() == city.lower())
                          & (pool["accommodates"] >= min_guests)
                          & (pool["price"] <= max_price)].copy()
    else:
        candidates = df[(df["city"].str.lower() == city.lower())
                        & (df["accommodates"] >= min_guests)
                        & (df["price"] <= max_price)].copy()
        if not candidates.empty:
            candidates["scaled_value_score"] = _minmax(
                candidates["pred_price"] - candidates["price"])

    if candidates.empty:
        return [], f"No listings match those filters in {city}."

    candidates["ml_score"] = (w_value * candidates["scaled_value_score"]
                              + w_convenience * candidates["instant_book_prob"])
    ranked = candidates.sort_values("ml_score", ascending=False).head(int(top_n))
    return ranked.to_dict(orient="records"), ""


def recsys_api_rank(city: str, min_guests: int, max_price: float,
                    w_value: float, w_convenience: float,
                    top_n: int = 10) -> tuple[list[dict], str]:
    """POST to the running notebook server, exactly as API_Test_Client.ipynb does.

    SearchQuery has no top_n and the handler hardcodes head(10), so top_n is a
    ceiling in this mode, never a floor.
    """
    import requests
    r = requests.post(RECSYS_API_URL, timeout=20, json={
        "city": city, "min_guests": int(min_guests), "max_price": float(max_price),
        "w_value": float(w_value), "w_convenience": float(w_convenience)})
    r.raise_for_status()
    body = r.json()
    rows = body.get("results", [])[:int(top_n)]
    return rows, body.get("message", "") if not rows else ""


def recommend(city: str, min_guests: int = 2, max_price: float = 150.0,
              w_value: float = 0.5, w_convenience: float = 0.5,
              top_n: int = 10) -> tuple[list[dict], str, str]:
    """Returns (rows, message, backend_actually_used)."""
    args = (city, min_guests, max_price, w_value, w_convenience, top_n)
    if RECSYS_BACKEND in ("api", "auto"):
        try:
            rows, msg = recsys_api_rank(*args)
            return rows, msg, "api"
        except Exception as e:
            if RECSYS_BACKEND == "api":
                return [], f"Recommender API at {RECSYS_API_URL} is unreachable: {e}", "api"
    rows, msg = recsys_rank(*args)
    return rows, msg, "inline"


def recsys_backend_name() -> str:
    if RECSYS_BACKEND == "api":
        return f"api {RECSYS_API_URL}"
    df = recsys_load()
    rows = "unavailable" if df is None else f"{len(df):,} listings"
    where = f"inline {RECSYS_CSV.rsplit('/', 1)[-1]} ({rows}, {RECSYS_SCALING} scaling)"
    return f"auto: {RECSYS_API_URL} then {where}" if RECSYS_BACKEND == "auto" else where


# ============================================================================
# 7C. WHY THAT PRICE -- SHAP OVER THE LIVE ENDPOINT
# ============================================================================
# The price endpoint returns a number, not attributions, and redeploying its
# inference.py to add them is a pipeline change this app cannot make. But the
# contract takes a LIST of records (notebook 05 batches 1000 at a time), and
# KernelSHAP is model-agnostic -- it only needs to evaluate coalitions. So the whole
# explanation fits in one or two invocations of the endpoint already deployed.
#
# WHAT THE BASELINE IS, because it decides what the numbers mean. Background rows
# are real listings from the SAME city, so the explanation reads "against a typical
# Paris listing, your second bedroom is worth +18 EUR". City and location therefore
# carry no attribution by construction -- they are folded into the baseline. That is
# the useful framing for a host, who cannot move the flat.
#
# Groups, not columns. Latitude and longitude are meaningless apart, so they move as
# one feature. Any set of payload keys can be grouped this way.
#
# If shap is not installed the fallback is leave-one-out occlusion: replace one
# group at a time with its background median and measure the move. Cheaper (K+1
# records, one call) and honest, but it ignores interactions, so it is labelled as
# what it is rather than as SHAP.
# ============================================================================

EXPLAIN_GROUPS: list[tuple[str, tuple[str, ...]]] = [
    ("Location", ("latitude", "longitude")),
    ("Neighbourhood", ("neighbourhood",)),
    ("Room type", ("room_type",)),
    ("Property type", ("property_type",)),
    ("Sleeps", ("accommodates",)),
    ("Bedrooms", ("bedrooms",)),
    ("Amenities", ("amenities",)),
    ("Minimum nights", ("minimum_nights",)),
    ("Instant bookable", ("instant_bookable",)),
    ("Superhost", ("host_is_superhost",)),
    ("Overall rating", ("review_scores_rating",)),
    ("Response time", ("host_response_time",)),
]

EXPLAIN_BACKGROUND_N = int(os.environ.get("EXPLAIN_BACKGROUND_N", "12"))
EXPLAIN_NSAMPLES = int(os.environ.get("EXPLAIN_NSAMPLES", "256"))
EXPLAIN_CHUNK = 1000         # records per call; notebook 05 batches at this size


def _background_payloads(payload: dict) -> list[dict]:
    """Reference listings in the same city, expressed in the price contract.

    Corpus rows only carry RECSYS_COLUMNS, so the fields it does not have are held
    at the explicand's own values. A field that never varies gets zero attribution,
    which is the correct answer for a field we have no comparison for -- silently
    inventing variation would manufacture attribution out of nothing.
    """
    out: list[dict] = []
    df = recsys_load()
    if df is not None and not df.empty:
        peers = df[df["city"].astype(str).str.lower() == str(payload["city"]).lower()]
        if not peers.empty:
            peers = peers.sample(n=min(EXPLAIN_BACKGROUND_N, len(peers)), random_state=0)
            for _, row in peers.iterrows():
                bg = dict(payload)
                for col in ("neighbourhood", "property_type", "room_type",
                            "accommodates", "bedrooms", "minimum_nights", "amenities"):
                    if col in row and row[col] is not None and str(row[col]) != "nan":
                        bg[col] = (int(row[col]) if col in
                                   ("accommodates", "bedrooms", "minimum_nights")
                                   else str(row[col]))
                # Corpus amenities are already the raw Listings.csv string, which is
                # exactly the shape the endpoint's parser expects. Pass it through.
                bg["instant_bookable"] = "t" if int(row.get("instant_bookable", 0) or 0) else "f"
                out.append(bg)
    if not out:
        # No corpus. Vary the handful of fields we can justify from fitted state.
        med = IMPUTE_MEDIANS.get("review_scores_rating")
        for acc, bed, rt in [(2, 1, "Entire place"), (4, 2, "Entire place"),
                             (2, 1, "Private room"), (6, 3, "Entire place"),
                             (1, 1, "Private room"), (3, 1, "Entire place")]:
            bg = dict(payload, accommodates=acc, bedrooms=bed, room_type=rt)
            if med is not None:
                bg["review_scores_rating"] = float(med)
            out.append(bg)
        for flag in ("t", "f"):
            out.append(dict(payload, instant_bookable=flag))
    return out


def _varying_groups(payload: dict, background: list[dict]):
    """Only explain what the background actually varies. Returns (labels, codebooks)."""
    labels, books = [], []
    for label, keys in EXPLAIN_GROUPS:
        vals = [tuple(json.dumps(b.get(k), sort_keys=True) for k in keys) for b in background]
        mine = tuple(json.dumps(payload.get(k), sort_keys=True) for k in keys)
        distinct = [mine] + [v for v in dict.fromkeys(vals) if v != mine]
        if len(distinct) > 1:
            labels.append((label, keys))
            books.append(distinct)
    return labels, books


def _decode(codes, labels, books, payload: dict) -> list[dict]:
    rows = []
    for code_row in codes:
        rec = dict(payload)
        for k, (_, keys) in enumerate(labels):
            chosen = books[k][int(round(float(code_row[k])))]
            for key, raw in zip(keys, chosen):
                rec[key] = json.loads(raw)
        rows.append({c: rec[c] for c in SERVING_COLUMNS})
    return rows


def _batch_prices(records: list[dict]) -> "Any":
    import numpy as np
    prices = []
    for i in range(0, len(records), EXPLAIN_CHUNK):
        chunk = records[i:i + EXPLAIN_CHUNK]
        resp = runtime.invoke_endpoint(EndpointName=PRICE_ENDPOINT,
                                       ContentType="application/json",
                                       Body=json.dumps(chunk))
        prices += [r["suggested_nightly_price"] for r in json.loads(resp["Body"].read())]
    return np.array(prices, dtype=float)


def explain_price(payload: dict) -> dict:
    """Attribute the suggested price to listing features. One or two endpoint calls."""
    import numpy as np
    background = _background_payloads(payload)
    labels, books = _varying_groups(payload, background)
    if not labels:
        return {"method": "none", "reason": "no comparable listings to explain against"}

    bg_codes = np.array(
        [[next((i for i, v in enumerate(books[k])
                if v == tuple(json.dumps(b.get(key), sort_keys=True) for key in keys)), 0)
          for k, (_, keys) in enumerate(labels)] for b in background], dtype=float)
    mine_codes = np.zeros((1, len(labels)))      # the explicand is always index 0

    baseline = float(_batch_prices(_decode(bg_codes, labels, books, payload)).mean())

    try:
        import shap
        f = lambda M: _batch_prices(_decode(M, labels, books, payload))
        explainer = shap.KernelExplainer(f, bg_codes)
        vals = np.array(explainer.shap_values(
            mine_codes, nsamples=EXPLAIN_NSAMPLES, silent=True)).reshape(-1)
        method = "shap"
    except Exception:
        # Leave-one-out. K+1 records in a single call.
        import numpy as np
        med = np.median(bg_codes, axis=0)
        probe = np.tile(mine_codes, (len(labels) + 1, 1))
        for k in range(len(labels)):
            probe[k + 1, k] = med[k]
        prices = _batch_prices(_decode(probe, labels, books, payload))
        vals = prices[0] - prices[1:]
        method = "occlusion"

    drivers = sorted(({"label": lbl, "value": float(v)}
                      for (lbl, _), v in zip(labels, vals)),
                     key=lambda d: abs(d["value"]), reverse=True)
    explained = {lbl for lbl, _ in labels}
    return {"method": method, "baseline": baseline, "drivers": drivers,
            "n_background": len(background),
            "not_compared": [lbl for lbl, _ in EXPLAIN_GROUPS if lbl not in explained]}


# ============================================================================
# 7D. LIKE FOR LIKE -- SIMILARITY OVER THE SCORED CORPUS
# ============================================================================
# The guest journey is "this one is booked / over budget, what else?", which is an
# item-to-item query. The notebook's /recommend cannot serve it: SearchQuery carries
# filters and weights, with no seed listing, so this section is inline only and
# RECSYS_BACKEND does not apply to it.
#
# Similar along three axes the guest actually cares about, each weighted:
#   attributes  size, layout, stay rules, room and property type, neighbourhood
#   fair price  the regression's estimate, NOT the asking price -- two listings
#               asking the same can be worth very different amounts
#   friction    the classifier's instant-booking probability
#
# Each axis is normalised to 0-1 across the candidate set before weighting, so the
# sliders trade off comparable quantities instead of raw units.
# ============================================================================

SIM_NUMERIC = ["accommodates", "bedrooms", "minimum_nights"]


def corpus_listing(listing_id: str) -> dict | None:
    df = recsys_load()
    if df is None or df.empty:
        return None
    hit = df[df["listing_id"].astype(str) == str(listing_id).strip()]
    return None if hit.empty else hit.iloc[0].to_dict()


def _unit(series):
    lo, hi = float(series.min()), float(series.max())
    return series * 0.0 if hi <= lo else (series - lo) / (hi - lo)


def similar_listings(listing_id: str, top_n: int = 10,
                     w_attrs: float = 0.4, w_price: float = 0.3,
                     w_friction: float = 0.3,
                     max_price: float | None = None) -> tuple[list[dict], str, dict | None]:
    df = recsys_load()
    if df is None:
        return [], f"Corpus unavailable. {_RECSYS_ERROR or ''}", None
    seed = corpus_listing(listing_id)
    if seed is None:
        return [], f"No listing `{listing_id}` in the corpus.", None

    cand = df[(df["city"] == seed["city"])
              & (df["listing_id"].astype(str) != str(seed["listing_id"]))].copy()
    if max_price is not None:
        cand = cand[cand["price"] <= float(max_price)]
    if cand.empty:
        return [], "No alternatives in that city under those constraints.", seed

    import numpy as np
    # Attributes: z-scored numeric gap plus a flat penalty per categorical mismatch,
    # so "same room type" is worth about one standard deviation of size difference.
    attr = np.zeros(len(cand))
    for col in SIM_NUMERIC:
        sd = float(cand[col].std()) or 1.0
        attr += (np.abs(cand[col].astype(float) - float(seed[col])) / sd) ** 2
    attr = np.sqrt(attr)
    for col in ("room_type", "property_type", "neighbourhood"):
        if col in cand.columns:
            attr += (cand[col].astype(str) != str(seed[col])).astype(float)
    cand["d_attrs"] = _unit(pd_series(cand, attr))
    cand["d_price"] = _unit((cand["pred_price"] - float(seed["pred_price"])).abs())
    cand["d_friction"] = _unit(
        (cand["instant_book_prob"] - float(seed["instant_book_prob"])).abs())

    total = (w_attrs + w_price + w_friction) or 1.0
    cand["distance"] = (w_attrs * cand["d_attrs"] + w_price * cand["d_price"]
                        + w_friction * cand["d_friction"]) / total
    cand["similarity"] = 1.0 - cand["distance"]
    ranked = cand.sort_values("distance").head(int(top_n))
    return ranked.to_dict(orient="records"), "", seed


def pd_series(df, array):
    import pandas as pd
    return pd.Series(array, index=df.index)


# ============================================================================
# 8. UI
# ============================================================================

# Injected as a <style> component rather than Blocks(css=...) or launch(css=...).
# Gradio 4/5 take css on the Blocks constructor; Gradio 6 moved it to launch() and
# TabbedInterface never accepted it at all. A <style> component works on all three.
CSS = """
.card {padding: 20px 22px; border: 1px solid var(--border-color-primary);
       border-radius: 10px; background: var(--background-fill-secondary);}
.card .eyebrow {font-size: .78rem; letter-spacing: .09em; text-transform: uppercase;
       opacity: .65; margin-bottom: 6px;}
.card .big {font-size: 2.4rem; font-weight: 650; line-height: 1.15;
       font-variant-numeric: tabular-nums;}
.card .sub {font-size: 1.05rem; font-weight: 500; opacity: .75; margin-top: 2px;}
.card .band {margin-top: 12px; padding: 10px 12px; border-radius: 8px;
       background: var(--background-fill-primary); font-size: .9rem; line-height: 1.5;}
.card .meta {margin-top: 12px; font-size: .78rem; opacity: .6;}
.card .stale {margin-top: 10px; padding: 9px 11px; border-radius: 8px; font-size: .85rem;
       border: 1px dashed var(--border-color-primary);}
.tone-normal {border-left: 4px solid #16a34a; padding-left: 14px;}
.tone-high {border-left: 4px solid #ea580c; padding-left: 14px;}
.tone-low {border-left: 4px solid #2563eb; padding-left: 14px;}
.tone-unknown {border-left: 4px solid var(--border-color-primary); padding-left: 14px;}
"""


def inject_css() -> None:
    gr.HTML(f"<style>{CSS}</style>")


def host_preset() -> dict:
    d = review_defaults()
    vocab = set(amenity_choices())
    amen = ["wifi", "kitchen", "heating", "washer", "essentials", "hot water"]
    return {
        "listing_id": "PAR-001", "asking_price": 90.0,
        "city": "Paris", "neighbourhood": "Buttes-Montmartre",
        "latitude": 48.8867, "longitude": 2.3431,
        "property_type": "Entire apartment", "room_type": "Entire place",
        "accommodates": 4, "bedrooms": 2,
        "amenities_selected": [a for a in amen if a in vocab],
        "amenities_extra": ", ".join(a for a in amen if a not in vocab),
        "minimum_nights": 2, "maximum_nights": 90, "instant_bookable": False,
        "host_since": "2016-04-21", "host_is_superhost": True,
        "host_has_profile_pic": True, "host_identity_verified": True,
        "host_total_listings_count": 3, "host_response_time": "within an hour",
        "host_response_rate": 95.0, "host_acceptance_rate": 88.0,
        "has_reviews": True, **{c: d[c] for c in REVIEW_COLS},
    }


def _suggestion_card(rec_pred, ms=None):
    price = rec_pred["suggested_nightly_price"]
    ccy = rec_pred["currency"]
    mape = city_mape(rec_pred.get("city"))
    band = ""
    if mape:
        lo, hi = price * (1 - mape / 100), price * (1 + mape / 100)
        band = (f"<div class='band'>Typical error for {rec_pred['city']} is "
                f"<strong>&plusmn;{mape:.1f}%</strong>, so treat this as roughly "
                f"<strong>{lo:,.0f}&ndash;{hi:,.0f} {ccy}</strong>.</div>")
    meta = f"<div class='meta'>{ms:,.0f} ms &middot; {PRICE_ENDPOINT}</div>" if ms else ""
    return (f"<div class='card'><div class='eyebrow'>Model suggestion</div>"
            f"<div class='big'>{price:,.2f} <span class='sub'>{ccy}</span></div>{band}{meta}</div>")


def _drivers_card(exp: dict, ccy: str) -> str:
    if exp.get("method") == "none":
        return ""
    note = ("Shapley values against the same-city baseline below, from "
            f"{exp['n_background']} comparable listings."
            if exp["method"] == "shap" else
            "Leave-one-out contributions: shap is not installed, so these ignore "
            "interactions between features.")
    rows = ""
    top = [d for d in exp["drivers"] if abs(d["value"]) >= 0.5][:8]
    if not top:
        return (f"<div class='card'><div class='eyebrow'>Price drivers</div>"
                f"<div class='band'>Nothing moves this listing more than 0.5 {ccy} away "
                f"from a typical listing in the same city.</div></div>")
    span = max(abs(d["value"]) for d in top) or 1.0
    for d in top:
        v = d["value"]
        pct, sign = abs(v) / span * 100.0, "+" if v >= 0 else "&minus;"
        colour = "#16a34a" if v >= 0 else "#2563eb"
        rows += (f"<div style='display:flex;align-items:center;gap:10px;margin:5px 0'>"
                 f"<div style='width:36%;font-size:.86rem'>{d['label']}</div>"
                 f"<div style='flex:1;background:var(--background-fill-primary);"
                 f"border-radius:4px;height:14px'>"
                 f"<div style='width:{pct:.0f}%;height:14px;border-radius:4px;"
                 f"background:{colour}'></div></div>"
                 f"<div style='width:22%;text-align:right;font-variant-numeric:tabular-nums;"
                 f"font-size:.86rem'>{sign}{abs(v):,.0f} {ccy}</div></div>")
    absent = ""
    if exp.get("not_compared"):
        absent = ("<br>Not compared, because the reference listings do not vary on them: "
                  + ", ".join(exp["not_compared"]).lower() + ".")
    return (f"<div class='card'><div class='eyebrow'>Why that price</div>{rows}"
            f"<div class='band'>Baseline, a typical listing in this city: "
            f"<strong>{exp['baseline']:,.0f} {ccy}</strong>. Bars are the move from "
            f"there to yours.</div><div class='meta'>{note}{absent}</div></div>")


def host_suggest(*values):
    """One button, three answers: the price, why, and the booking friction."""
    ui = dict(zip(HOST_FIELDS, values))
    try:
        payload = build_payload(ui)
        booking_payload = build_booking_payload(ui)
    except (TypeError, ValueError) as e:
        return f"<div class='card'>Cannot build the request: {e}</div>", "", "", "", None
    problems = validate_host(ui, payload)
    if problems:
        bullets = "".join(f"<li>{p}</li>" for p in problems)
        return (f"<div class='card'><div class='eyebrow'>Fix these</div><ul>{bullets}</ul></div>",
                "", "", json.dumps(payload, indent=2), None)

    out = invoke_both(payload, booking_payload)
    if "price_error" in out:
        return (f"<div class='card'>{explain_error(out['price_error'])}</div>", "", "",
                json.dumps(payload, indent=2), None)
    pred, ms = out["price"], out["price_ms"]
    ccy = pred.get("currency", "")

    # Friction is an indicator beside the price, not a separate errand. A blank card
    # if the classifier is cold: it must not take the price answer down with it.
    friction = ""
    if "booking" in out:
        friction = _booking_card(out["booking"], out.get("booking_ms"),
                                 booking_payload["price"], ccy)
    elif "booking_error" in out:
        friction = (f"<div class='card tone-unknown'><div class='eyebrow'>Instant booking"
                    f"</div><div class='band'>{explain_error(out['booking_error'])}"
                    f"</div></div>")

    try:
        drivers = _drivers_card(explain_price(payload), ccy)
    except Exception as e:
        drivers = (f"<div class='card'><div class='eyebrow'>Why that price</div>"
                   f"<div class='band'>Attribution failed: {e.__class__.__name__}: {e}"
                   f"</div></div>")
    return (_suggestion_card(pred, ms), drivers, friction,
            json.dumps(payload, indent=2), pred)


def host_save(request: gr.Request, pred_state, *values):
    # Returns (status markdown, refreshed "your listings" table).
    #
    # `request` MUST be the first parameter. gradio.helpers.special_args builds its
    # positional list by walking the signature and breaking at the first
    # non-positional parameter, so a keyword-only `request` sitting after *values
    # is never reached and silently arrives as None -- which previously made every
    # listing save as "anonymous" while deletes ran as the logged-in user.
    ui = dict(zip(HOST_FIELDS, values))
    username = current_user(request)
    try:
        payload = build_payload(ui)
    except (TypeError, ValueError) as e:
        return f"Cannot build the request: {e}", host_my_listings(request)
    problems = validate_host(ui, payload)
    if problems:
        return ("Fix these first:\n" + "\n".join(f"- {p}" for p in problems),
                host_my_listings(request))

    pred = pred_state
    if pred is None:
        try:
            pred, _ = invoke_price(payload)
        except Exception as e:
            return (f"Could not get a model suggestion before saving.\n\n{explain_error(e)}",
                    host_my_listings(request))

    lid = ui["listing_id"].strip()
    try:
        rec = save_listing(lid, payload, float(ui["asking_price"]), pred, username)
    except PermissionError as e:
        return str(e), host_my_listings(request)
    except Exception as e:
        return f"Save failed.\n\n{explain_error(e)}", host_my_listings(request)

    gap = fairness_verdict(rec["asking_price"], pred["suggested_nightly_price"],
                           city_mape(payload["city"]))
    return (f"Saved **{lid}** at {rec['updated_at']} as `{username}`.\n\n"
            f"Asking {rec['asking_price']:,.2f} {pred['currency']} against a model suggestion of "
            f"{pred['suggested_nightly_price']:,.2f} ({gap['gap_pct']:+.0f}%). "
            f"Customers will see: *{gap['headline']}*.\n\n"
            f"Store: `{store_backend_name()}`"), host_my_listings(request)


def host_delete(listing_id, request: gr.Request = None):
    return delete_listing((listing_id or "").strip(), current_user(request))


def host_my_listings(request: gr.Request = None):
    # Show EVERY listing with its owner, not just your own. Hiding records you
    # cannot edit is how one becomes unreachable from the UI with no explanation.
    username = current_user(request)
    store = store_read()
    if not store:
        return "No listings saved yet."
    rows = ["| listing | owner | asking | model | gap | updated |",
            "|---|---|---:|---:|---:|---|"]
    for lid, rec in sorted(store.items()):
        pred = rec.get("prediction") or {}
        sug = pred.get("suggested_nightly_price")
        gap = f"{(rec['asking_price'] - sug) / sug * 100:+.0f}%" if sug else "-"
        owner = rec.get("owner")
        if owner is None:
            who = "_unowned_"
        elif owner == username:
            who = f"`{owner}` (you)"
        else:
            who = f"`{owner}` &mdash; read only"
        rows.append(f"| `{lid}` | {who} | {rec['asking_price']:,.0f} | "
                    f"{sug:,.0f} | {gap} | {rec['updated_at'][:16]} |" if sug else
                    f"| `{lid}` | {who} | {rec['asking_price']:,.0f} | - | - | "
                    f"{rec['updated_at'][:16]} |")
    signed = f"Signed in as `{username}`." if username else "Not signed in; saves will be unowned."
    return signed + "\n\n" + "\n".join(rows)


def customer_render(listing_id):
    if not listing_id:
        return "<div class='card'><div class='eyebrow'>Pick a listing</div></div>", "{}"
    rec = store_read().get(listing_id)
    if rec is None:
        return ("<div class='card'><div class='eyebrow'>Not found</div>"
                "<div class='meta'>That listing may have been removed. Refresh the list.</div>"
                "</div>", "{}")

    view = public_view(rec)
    pred = rec.get("prediction") or {}
    sug = pred.get("suggested_nightly_price")
    ccy = view.get("currency") or pred.get("currency", "")
    asking = rec["asking_price"]

    if not sug:
        card = (f"<div class='card'><div class='eyebrow'>{listing_id} &middot; {view.get('city','')}"
                f"</div><div class='big'>{asking:,.2f} <span class='sub'>{ccy}</span></div>"
                f"<div class='band'>No model comparison was stored for this listing.</div></div>")
        return card, json.dumps(view, indent=2)

    v = fairness_verdict(asking, sug, city_mape(view.get("city")))
    stale = verdict_is_stale(rec)
    stale_html = f"<div class='stale'>{stale}</div>" if stale else ""
    dist = (f" &middot; {view['dist_center_km']:.1f} km from the centre"
            if "dist_center_km" in view else "")

    card = (
        f"<div class='card tone-{v['tone']}'>"
        f"<div class='eyebrow'>{listing_id} &middot; {view.get('city','')}"
        f"{dist}</div>"
        f"<div class='big'>{asking:,.2f} <span class='sub'>{ccy} per night</span></div>"
        f"<div class='band'><strong>{v['headline']}.</strong> "
        f"The model expects about {sug:,.0f} {ccy} for a listing with these characteristics, "
        f"so this is <strong>{v['gap_pct']:+.0f}%</strong> against that. "
        f"Its typical error for {view.get('city','this city')} is &plusmn;{v['band_pct']:.0f}%, "
        f"which is the threshold used above.</div>"
        f"{stale_html}"
        f"<div class='meta'>Updated {rec['updated_at'][:16]} &middot; "
        f"exact address is not published</div>"
        "</div>"
    )
    return card, json.dumps(view, indent=2)


_HOST_LOAD_HOOKS: list = []


def listing_form(h: dict, preset: dict, rdef: dict, *,
                 price_label: str = "Nightly price",
                 show_listing_id: bool = True) -> dict:
    """Build the 27-field contract form into the open container, populating `h`.

    Shared by every tab that needs a listing: the host's price form and both
    customer prediction tabs. Written once so that a field added for one model
    cannot go missing for another -- the same reasoning behind making
    CUSTOMER_VISIBLE an allowlist rather than a denylist.

    ON THE PRIVACY SPLIT. On the customer tabs the person types their OWN
    coordinates, so HOST_PRIVATE protects nothing there. That layer governs what is
    published about a listing some host saved, and it still does that job on the
    Browse tab. Here it is inert by construction, not bypassed.
    """
    with gr.Row():
        h["listing_id"] = gr.Textbox(
            label="Listing ID", value=preset["listing_id"], visible=show_listing_id,
            info="3-32 chars. Used as the store key.")
        h["asking_price"] = gr.Number(label=price_label, value=preset["asking_price"],
                                      minimum=1)
    with gr.Tab("Listing"):
        with gr.Row():
            h["city"] = gr.Dropdown(label="City", choices=CITIES, value=preset["city"])
            h["neighbourhood"] = gr.Dropdown(
                label="Neighbourhood", choices=known_neighbourhoods(preset["city"]) or [],
                value=preset["neighbourhood"], allow_custom_value=True)
        with gr.Row():
            h["latitude"] = gr.Number(label="Latitude (never published)",
                                      value=preset["latitude"])
            h["longitude"] = gr.Number(label="Longitude (never published)",
                                       value=preset["longitude"])
        gr.Markdown("<small>Customers see distance from the city centre, rounded to "
                    "100 m. The exact point stays in the host store.</small>")
        with gr.Row():
            h["property_type"] = gr.Dropdown(
                label="Property type", choices=FALLBACK_PROPERTY_TYPES,
                value=preset["property_type"], allow_custom_value=True)
            h["room_type"] = gr.Dropdown(label="Room type", choices=room_type_choices(),
                                         value=preset["room_type"])
        with gr.Row():
            h["accommodates"] = gr.Number(label="Accommodates", precision=0, minimum=1,
                                          value=preset["accommodates"])
            h["bedrooms"] = gr.Number(label="Bedrooms", precision=0, minimum=0,
                                      value=preset["bedrooms"])
    with gr.Tab("Amenities"):
        h["amenities_selected"] = gr.CheckboxGroup(
            label="Amenities", choices=amenity_choices(),
            value=preset["amenities_selected"])
        h["amenities_extra"] = gr.Textbox(label="Other amenities", lines=2,
                                          value=preset["amenities_extra"])
    with gr.Tab("Booking rules"):
        with gr.Row():
            h["minimum_nights"] = gr.Number(label="Minimum nights", precision=0,
                                            minimum=1, value=preset["minimum_nights"])
            h["maximum_nights"] = gr.Number(label="Maximum nights", precision=0,
                                            minimum=1, value=preset["maximum_nights"])
        h["instant_bookable"] = gr.Checkbox(label="Instant bookable",
                                            value=preset["instant_bookable"])
    with gr.Tab("You"):
        with gr.Row():
            h["host_since"] = gr.Textbox(label="Host since", value=preset["host_since"],
                                         placeholder="YYYY-MM-DD")
            h["host_total_listings_count"] = gr.Number(
                label="Your total listings", precision=0, minimum=0,
                value=preset["host_total_listings_count"])
        with gr.Row():
            h["host_is_superhost"] = gr.Checkbox(label="Superhost",
                                                 value=preset["host_is_superhost"])
            h["host_has_profile_pic"] = gr.Checkbox(label="Profile picture",
                                                    value=preset["host_has_profile_pic"])
            h["host_identity_verified"] = gr.Checkbox(
                label="Identity verified", value=preset["host_identity_verified"])
        h["host_response_time"] = gr.Dropdown(
            label="Response time", choices=response_time_choices(),
            value=preset["host_response_time"])
        h["host_response_rate"] = gr.Slider(label="Response rate (%)", minimum=0,
                                            maximum=100, step=1,
                                            value=preset["host_response_rate"])
        h["host_acceptance_rate"] = gr.Slider(label="Acceptance rate (%)", minimum=0,
                                              maximum=100, step=1,
                                              value=preset["host_acceptance_rate"])
    with gr.Tab("Reviews"):
        h["has_reviews"] = gr.Checkbox(label="This listing has reviews", value=True)
        for col in REVIEW_COLS:
            h[col] = gr.Slider(
                label=col.replace("review_scores_", "").replace("_", " ").title(),
                minimum=0, maximum=review_slider_max(col, rdef[col]),
                step=1 if col == "review_scores_rating" else 0.1, value=rdef[col])
    return h


def host_body() -> None:
    """Render the host view into whatever Blocks context is currently open.

    Kept separate from build_host_ui() so the same components can be placed
    inside a tab of a combined app without nesting one Blocks inside another.
    """
    rdef = review_defaults()
    preset = host_preset()
    gr.Markdown(f"""
        # Host - price your listing
        `{PRICE_ENDPOINT}` &middot; store `{store_backend_name()}`

        One button, three answers: the suggested nightly rate, what is driving it against a
        typical listing in the same city, and how much booking friction to expect at your
        asking price. Then set your price and save.

        Five fields never reach customers: your exact coordinates, your join date, your
        acceptance rate and your total listing count.
    """)
    h: dict[str, Any] = {}
    with gr.Row():
        with gr.Column(scale=3):
            listing_form(h, preset, rdef, price_label="Your asking price")

        with gr.Column(scale=2):
            suggest_btn = gr.Button("Get model suggestion", variant="secondary", size="lg")
            save_btn = gr.Button("Save listing", variant="primary", size="lg")
            suggestion = gr.HTML("<div class='card'><div class='eyebrow'>No suggestion yet"
                                 "</div><div class='meta'>The first call after an idle period "
                                 "is slow: the endpoint scales to zero.</div></div>")
            drivers_view = gr.HTML()
            friction_view = gr.HTML()
            save_status = gr.Markdown()
            pred_state = gr.State(None)
            with gr.Accordion("Request JSON (27 fields)", open=False):
                request_view = gr.Code(language="json", label="POST body")
            with gr.Accordion("Your listings", open=False):
                mine = gr.Markdown()
                gr.Button("Refresh", size="sm").click(host_my_listings, None, mine)
            with gr.Accordion("Delete a listing", open=False):
                del_id = gr.Textbox(label="Listing ID to delete")
                del_out = gr.Markdown()
                gr.Button("Delete", variant="stop", size="sm").click(
                    host_delete, del_id, del_out)

    inputs = [h[k] for k in HOST_FIELDS]
    suggest_btn.click(host_suggest, inputs,
                      [suggestion, drivers_view, friction_view, request_view, pred_state])
    save_btn.click(host_save, [pred_state, *inputs], [save_status, mine])

    h["city"].change(
        lambda c: (gr.update(choices=known_neighbourhoods(c),
                             value=(known_neighbourhoods(c) or [None])[0]),
                   gr.update(value=CITY_CENTERS.get(c, (0, 0))[0]),
                   gr.update(value=CITY_CENTERS.get(c, (0, 0))[1])),
        h["city"], [h["neighbourhood"], h["latitude"], h["longitude"]])
    _HOST_LOAD_HOOKS.append((host_my_listings, None, mine))


def build_host_ui() -> gr.Blocks:
    with gr.Blocks(title=f"{COURSE} {TEAM_ID} - host") as demo:
        inject_css()
        host_tabs()
        for fn, i, o in _HOST_LOAD_HOOKS:
            demo.load(fn, i, o)
        _HOST_LOAD_HOOKS.clear()
    return demo


def customer_body() -> None:
    """Render the customer view into the currently open Blocks context."""
    gr.Markdown("""
        # Is this listing fairly priced?

        Pick a listing to see how its asking price compares with what the model expects for a
        listing with these characteristics. The comparison uses the model's own measured error
        for that city, so a small gap is reported as normal rather than as overpricing.
    """)
    with gr.Row():
        with gr.Column(scale=2):
            picker = gr.Dropdown(label="Listing", choices=sorted(store_read()), value=None)
            gr.Button("Refresh list", size="sm").click(
                lambda: gr.update(choices=sorted(store_read())), None, picker)
            card = gr.HTML("<div class='card'><div class='eyebrow'>Pick a listing</div></div>")
        with gr.Column(scale=1):
            with gr.Accordion("Everything published about this listing", open=False):
                detail = gr.Code(language="json", label="Public view")
            gr.Markdown("<small>Withheld: the exact coordinates, the host's join date, "
                        "their acceptance rate and how many listings they run.</small>")

    picker.change(customer_render, picker, [card, detail])
    timer = gr.Timer(5)
    timer.tick(lambda: gr.update(choices=sorted(store_read())), None, picker)


def _booking_card(pred, ms=None, price=None, ccy="", alt=None):
    prob = pred.get("probability_class_1")
    if prob is None:
        return ("<div class='card tone-unknown'><div class='eyebrow'>Instant booking</div>"
                "<div class='band'>The endpoint returned no probability for this listing."
                "</div></div>")
    v = booking_verdict(prob)
    at = f" at {price:,.0f} {ccy}" if price else ""
    # The counterfactual is the point of feeding price to a classifier that takes it
    # as a feature. One probability is a number; two are a lever the host can pull.
    delta = ""
    if alt and alt.get("probability_class_1") is not None:
        ap, aprob = alt["price"], alt["probability_class_1"]
        arrow = "rises" if aprob > prob else "falls"
        delta = (f"<div class='band'>Priced at <strong>{ap:,.0f} {ccy}</strong> instead, "
                 f"the peer rate {arrow} to <strong>{aprob * 100:.0f}%</strong>.</div>")
    meta = f"<div class='meta'>{ms:,.0f} ms &middot; {BOOKING_ENDPOINT}</div>" if ms else ""
    return (f"<div class='card tone-{v['tone']}'><div class='eyebrow'>Instant booking{at}</div>"
            f"<div class='big'>{prob * 100:.0f}<span class='sub'>%</span></div>"
            f"<div class='band'><strong>{v['headline']}.</strong> This is the share of "
            f"comparable listings that offer instant booking, not a prediction of what you "
            f"should choose.</div>{delta}{meta}</div>")


def predict_price_only(*values):
    """Regression tab. Same contract and same card as the host side, minus the store."""
    ui = dict(zip(HOST_FIELDS, values))
    try:
        payload = build_payload(ui)
    except (TypeError, ValueError) as e:
        return f"<div class='card'>Cannot build the request: {e}</div>", ""
    problems = [p for p in validate_host(ui, payload) if "Listing ID" not in p]
    if problems:
        bullets = "".join(f"<li>{p}</li>" for p in problems)
        return (f"<div class='card'><div class='eyebrow'>Fix these</div><ul>{bullets}</ul></div>",
                json.dumps(payload, indent=2))
    try:
        pred, ms = invoke_price(payload)
    except Exception as e:
        return f"<div class='card'>{explain_error(e)}</div>", json.dumps(payload, indent=2)
    return _suggestion_card(pred, ms), json.dumps(payload, indent=2)


def predict_booking_only(*values):
    """Classification tab. Scores the asking price, then the model's suggestion."""
    ui = dict(zip(HOST_FIELDS, values))
    try:
        payload = build_booking_payload(ui)
    except (TypeError, ValueError) as e:
        return f"<div class='card'>Cannot build the request: {e}</div>", ""
    problems = [p for p in validate_host(ui, payload | {"neighbourhood": None})
                if "Listing ID" not in p and "contract field" not in p]
    if problems:
        bullets = "".join(f"<li>{p}</li>" for p in problems)
        return (f"<div class='card'><div class='eyebrow'>Fix these</div><ul>{bullets}</ul></div>",
                json.dumps(payload, indent=2))

    ccy = CURRENCY.get(payload["city"], "")
    # Round one: the classification we were asked for, and the price we need for the
    # counterfactual, at the same time. Two cold serverless starts in sequence is a
    # two-minute wait for one button.
    out = invoke_both(build_payload(ui), payload)
    if "booking_error" in out:
        return (f"<div class='card'>{explain_error(out['booking_error'])}</div>",
                json.dumps(payload, indent=2))
    pred, ms = out["booking"], out["booking_ms"]

    alt = None
    try:                          # round two. A bonus, never a hard failure.
        sug = out["price"]["suggested_nightly_price"]
        if abs(sug - payload["price"]) > 0.01:
            alt_pred, _ = invoke_booking(build_booking_payload(ui, price=sug))
            alt = {"price": sug, "probability_class_1": alt_pred.get("probability_class_1")}
    except Exception:
        pass
    return (_booking_card(pred, ms, payload["price"], ccy, alt),
            json.dumps(payload, indent=2))


def _recsys_rows(city, min_guests, max_price, w_value, w_convenience, top_n):
    try:
        rows, msg, backend = recommend(city, min_guests, max_price,
                                       w_value, w_convenience, top_n)
    except Exception as e:
        return [], f"Recommender unavailable. {e.__class__.__name__}: {e}"
    if not rows:
        return [], msg or "No listings match those filters."
    table = []
    for i, r in enumerate(rows, 1):
        price, pred = float(r["price"]), float(r["pred_price"])
        table.append([i, r["listing_id"], int(r["accommodates"]),
                      round(price, 2), round(pred, 2), round(pred - price, 2),
                      f"{float(r['instant_book_prob']) * 100:.1f}%",
                      round(float(r["ml_score"]), 4)])
    ccy = CURRENCY.get(city, "")
    note = (f"{len(table)} of the corpus, ranked by `{backend}`. "
            f"**Under value** is model price minus asking price in {ccy}: positive means "
            f"the model thinks the listing is worth more than it charges.")
    return table, note


def _similar_rows(listing_id, top_n, w_attrs, w_price, w_friction, budget):
    if not str(listing_id or "").strip():
        return [], "Enter a listing ID from the search results above."
    try:
        cap = float(budget) if budget not in (None, "", 0) else None
        rows, msg, seed = similar_listings(str(listing_id), int(top_n), float(w_attrs),
                                           float(w_price), float(w_friction), cap)
    except Exception as e:
        return [], f"Could not find alternatives. {e.__class__.__name__}: {e}"
    if not rows:
        return [], msg
    table = []
    for i, r in enumerate(rows, 1):
        table.append([i, str(r["listing_id"]), f"{r['similarity'] * 100:.0f}%",
                      int(r["accommodates"]), int(r["bedrooms"]), str(r["room_type"]),
                      str(r.get("neighbourhood", "")), round(float(r["price"]), 2),
                      round(float(r["pred_price"]), 2),
                      f"{float(r['instant_book_prob']) * 100:.0f}%"])
    ccy = CURRENCY.get(seed["city"], "")
    note = (f"Alternatives to **{seed['listing_id']}** in {seed['city']} "
            f"({seed['accommodates']} guests, {seed['room_type']}, "
            f"{float(seed['price']):,.0f} {ccy} asking, "
            f"{float(seed['pred_price']):,.0f} {ccy} modelled, "
            f"{float(seed['instant_book_prob']) * 100:.0f}% instant book).")
    return table, note


def recsys_body(title: str = "Find a stay") -> None:
    """The recommender, lifted from Test_API_Gradio.ipynb. Shared by both roles."""
    cities = recsys_cities()
    gr.Markdown(f"""
        # {title}
        `{recsys_backend_name()}`

        Two signals, and you choose the mix. **Value** is how far under the price model's
        estimate a listing sits. **Convenience** is the classifier's instant-booking
        probability. The sliders are the `w_value` and `w_convenience` weights from the
        API's `SearchQuery`.
    """)
    with gr.Row():
        with gr.Column(scale=1):
            r_city = gr.Dropdown(label="City", choices=cities,
                                 value=("Paris" if "Paris" in cities else
                                        (cities[0] if cities else "Paris")))
            r_guests = gr.Number(label="Guests", value=2, precision=0, minimum=1)
            r_max = gr.Number(label="Max nightly price", value=150.0, minimum=1)
            r_wv = gr.Slider(label="Weight: value", minimum=0, maximum=1, step=0.05, value=0.5)
            r_wc = gr.Slider(label="Weight: convenience", minimum=0, maximum=1,
                             step=0.05, value=0.5)
            r_n = gr.Slider(label="Results", minimum=1, maximum=25, step=1, value=10)
            r_btn = gr.Button("Search", variant="primary", size="lg")
        with gr.Column(scale=3):
            r_note = gr.Markdown()
            r_table = gr.Dataframe(
                headers=["#", "listing", "sleeps", "price", "model price",
                         "under value", "instant book", "score"],
                datatype=["number", "str", "number", "number", "number",
                          "number", "str", "number"],
                interactive=False, wrap=True)

    # Keep the two weights summing to 1, as the notebook's examples always do.
    r_wv.change(lambda v: gr.update(value=round(1 - v, 2)), r_wv, r_wc)
    r_wc.change(lambda v: gr.update(value=round(1 - v, 2)), r_wc, r_wv)
    r_btn.click(_recsys_rows, [r_city, r_guests, r_max, r_wv, r_wc, r_n],
                [r_table, r_note])

    # ---- step two: like for like -------------------------------------------
    # The guest journey proper. Search above is how you FIND a listing; this is
    # what happens when it turns out to be booked or over budget.
    gr.Markdown("""
        ---
        ## More like this
        Paste a listing ID from the table above. Alternatives are drawn from the same
        city and ranked on three axes you control: **attributes** (size, layout, stay
        rules, room and property type, neighbourhood), **fair price** -- the regression's
        estimate rather than the asking price, so two listings charging the same but
        worth different amounts are not treated as alike -- and **booking friction**,
        the classifier's instant-booking probability.
    """)
    with gr.Row():
        with gr.Column(scale=1):
            s_id = gr.Textbox(label="Listing ID you like",
                              placeholder="e.g. 42702649")
            s_budget = gr.Number(label="Cap alternatives at (blank = no cap)", value=None)
            s_wa = gr.Slider(label="Weight: attributes", minimum=0, maximum=1,
                             step=0.05, value=0.4)
            s_wp = gr.Slider(label="Weight: fair price", minimum=0, maximum=1,
                             step=0.05, value=0.3)
            s_wf = gr.Slider(label="Weight: booking friction", minimum=0, maximum=1,
                             step=0.05, value=0.3)
            s_n = gr.Slider(label="Alternatives", minimum=1, maximum=25, step=1, value=8)
            s_btn = gr.Button("Find like-for-like", variant="primary", size="lg")
        with gr.Column(scale=3):
            s_note = gr.Markdown()
            s_table = gr.Dataframe(
                headers=["#", "listing", "similarity", "sleeps", "beds", "room type",
                         "neighbourhood", "price", "model price", "instant book"],
                datatype=["number", "str", "str", "number", "number", "str",
                          "str", "number", "number", "str"],
                interactive=False, wrap=True)
    s_btn.click(_similar_rows, [s_id, s_n, s_wa, s_wp, s_wf, s_budget],
                [s_table, s_note])


def prediction_body(kind: str) -> None:
    """One model, one form, one button. `kind` is "price" or "booking"."""
    rdef, preset = review_defaults(), host_preset()
    if kind == "price":
        gr.Markdown(f"""
            # What should this cost?
            `{PRICE_ENDPOINT}`

            Describe a listing and the regression model returns a suggested nightly rate,
            with the error it typically makes in that city.
        """)
        price_label, btn_label = "Nightly price you have in mind", "Estimate the price"
    else:
        gr.Markdown(f"""
            # Would a listing like this book instantly?
            `{BOOKING_ENDPOINT}`

            The classifier takes **price** as a feature, so this is a conditional answer:
            the instant-booking rate among comparable listings *at the price you set*.
            Change the price and the answer moves.
        """)
        price_label, btn_label = "Nightly price", "Estimate instant booking"

    h: dict[str, Any] = {}
    with gr.Row():
        with gr.Column(scale=3):
            listing_form(h, preset, rdef, price_label=price_label, show_listing_id=False)
        with gr.Column(scale=2):
            btn = gr.Button(btn_label, variant="primary", size="lg")
            card = gr.HTML("<div class='card'><div class='eyebrow'>No estimate yet</div>"
                           "<div class='meta'>The first call after an idle period is slow: "
                           "the endpoint scales to zero.</div></div>")
            n = len(SERVING_COLUMNS) if kind == "price" else len(BOOKING_COLUMNS)
            with gr.Accordion(f"Request JSON ({n} fields)", open=False):
                req = gr.Code(language="json", label="POST body")

    inputs = [h[k] for k in HOST_FIELDS]
    btn.click(predict_price_only if kind == "price" else predict_booking_only,
              inputs, [card, req])
    h["city"].change(
        lambda c: (gr.update(choices=known_neighbourhoods(c),
                             value=(known_neighbourhoods(c) or [None])[0]),
                   gr.update(value=CITY_CENTERS.get(c, (0, 0))[0]),
                   gr.update(value=CITY_CENTERS.get(c, (0, 0))[1])),
        h["city"], [h["neighbourhood"], h["latitude"], h["longitude"]])


def customer_tabs() -> None:
    """Three models, one per tab, plus the saved-listing browse.

    Separate tabs rather than one page because the three take different inputs: the
    two predictors want a full listing described, the recommender wants search
    filters. Sharing a page would mean a form where most fields are inert.
    """
    with gr.Tabs():
        with gr.Tab("Price"):
            prediction_body("price")
        with gr.Tab("Instant booking"):
            prediction_body("booking")
        with gr.Tab("Find a stay"):
            recsys_body()
        with gr.Tab("Browse saved listings"):
            customer_body()


def host_tabs() -> None:
    with gr.Tabs():
        with gr.Tab("Price & save"):
            host_body()
        with gr.Tab("Instant booking"):
            prediction_body("booking")
        with gr.Tab("Your competitive set"):
            recsys_body("Where a listing like yours would rank")


def build_customer_ui() -> gr.Blocks:
    with gr.Blocks(title=f"{COURSE} {TEAM_ID} - browse") as demo:
        inject_css()
        customer_tabs()
    return demo


def build_both_ui() -> gr.Blocks:
    """Host and customer in one Blocks. Not a trust boundary -- see the module docstring."""
    with gr.Blocks(title=f"{COURSE} {TEAM_ID} - listing app (demo mode)") as demo:
        inject_css()
        gr.Markdown(f"# {COURSE} {TEAM_ID} - listing app (demo mode)")
        with gr.Tabs():
            with gr.Tab("Host"):
                host_tabs()
            with gr.Tab("Customer"):
                customer_tabs()
        for fn, i, o in _HOST_LOAD_HOOKS:
            demo.load(fn, i, o)
        _HOST_LOAD_HOOKS.clear()
    return demo


# ============================================================================
# 9. SELF TEST + ENTRY POINT
# ============================================================================

def self_test() -> int:
    """No UI, no AWS. Exercises the store, the privacy split and the verdict."""
    global USE_S3, STORE_LOCAL
    USE_S3 = False
    STORE_LOCAL = "._selftest_store.json"
    if os.path.exists(STORE_LOCAL):
        os.remove(STORE_LOCAL)
    load_fitted_state()

    ui = host_preset()
    payload = build_payload(ui)
    assert set(payload) == set(SERVING_COLUMNS), set(payload) ^ set(SERVING_COLUMNS)
    assert not validate_host(ui, payload), validate_host(ui, payload)
    print(f"contract        : {len(payload)} fields, validation clean")

    pred = {"suggested_nightly_price": 70.9, "currency": "EUR", "city": "Paris", "log_price": 4.2752}
    _LINEAGE_CACHE[PRICE_ENDPOINT] = {"endpoint": PRICE_ENDPOINT,
                                      "model_package": "arn:.../models/8",
                                      "model": "m-2026", "status": "InService"}
    rec = save_listing("PAR-001", payload, 90.0, pred, "alice")
    print(f"store           : wrote PAR-001 to {store_backend_name()}")

    view = public_view(rec)
    leaked = HOST_PRIVATE & set(view)
    assert not leaked, leaked
    assert "dist_center_km" in view and "latitude" not in view
    print(f"privacy         : {len(view)} public fields, {len(HOST_PRIVATE)} withheld, "
          f"distance {view['dist_center_km']} km published instead of coordinates")

    # The allowlist must hold even if someone adds a private field to the payload.
    rec2 = dict(rec, payload=dict(payload, host_secret_note="do not publish"))
    assert "host_secret_note" not in public_view(rec2)
    print("privacy         : an unclassified new field is withheld by default (allowlist)")

    for asking, expect in [(70.0, "normal"), (90.0, "normal"), (160.0, "high"), (30.0, "low")]:
        v = fairness_verdict(asking, 70.9, 27.6)
        assert v["tone"] == expect, (asking, v)
        print(f"verdict         : asking {asking:>6,.0f} -> {v['gap_pct']:+6.0f}% -> {v['headline']}")

    assert verdict_is_stale(rec) is None
    assert rec["model"] is not _LINEAGE_CACHE[PRICE_ENDPOINT], "stamp aliases the live cache"
    _LINEAGE_CACHE[PRICE_ENDPOINT]["model_package"] = "arn:.../models/9"   # simulate a redeploy
    assert rec["model"]["model_package"] == "arn:.../models/8", "stamp mutated under us"
    assert "re-save" in (verdict_is_stale(rec) or "")
    print("staleness       : stamp is immutable; a redeploy is detected and surfaced")

    try:
        save_listing("PAR-001", payload, 99.0, pred, "mallory")
        raise SystemExit("ownership check failed")
    except PermissionError as e:
        assert "owned by `alice`" in str(e) and "you are `mallory`" in str(e), str(e)
        print("ownership       : refusal names the owner AND the current user")

    msg = delete_listing("PAR-001", "mallory")
    assert "--set-owner" in msg and "--list-store" in msg
    print("recovery        : the refusal tells you how to repair it from the CLI")

    # No auth -> unowned, not a phantom "anonymous" owner that strands the record.
    unowned = save_listing("PAR-002", payload, 60.0, pred, None)
    assert unowned["owner"] is None
    assert "Deleted" in delete_listing("PAR-002", "anyone")
    print("no auth         : saves are unowned and stay deletable by anyone")

    # Reassign, then delete as the new owner.
    st = store_read(); st["PAR-001"]["owner"] = "bob"; store_write(st)
    assert "Deleted" in delete_listing("PAR-001", "bob")
    assert store_read() == {}
    print("delete          : owner-only, store empty again")

    # ---- second model: the classifier's contract is not the regression's ----
    bp = build_booking_payload(ui)
    assert set(bp) == set(BOOKING_COLUMNS), set(bp) ^ set(BOOKING_COLUMNS)
    assert bp["price"] == ui["asking_price"], "classifier must receive the asking price"
    assert "price" not in payload, "the regression contract must not carry price"
    assert "neighbourhood" not in bp, "not in the classifier's feature schema"
    assert bp["host_is_superhost"] == 1 and payload["host_is_superhost"] == "t", \
        "booleans are ints for the classifier and 't'/'f' for the regression"
    assert build_booking_payload(ui, price=42.0)["price"] == 42.0, "counterfactual price"
    print(f"booking         : {len(bp)} fields, price carried, flags coerced to int")

    for prob, expect in [(0.91, "normal"), (0.60, "normal"), (0.31, "low"), (0.04, "low")]:
        assert booking_verdict(prob)["tone"] == expect, (prob, booking_verdict(prob))
    print("booking         : peer-rate framing, no MAPE band on a classifier")

    # ---- the lineage cache must not hand model 2 model 1's stamp ----
    _LINEAGE_CACHE.clear()
    _LINEAGE_CACHE[PRICE_ENDPOINT] = {"endpoint": PRICE_ENDPOINT, "model_package": "arn:.../8"}
    _LINEAGE_CACHE[BOOKING_ENDPOINT] = {"endpoint": BOOKING_ENDPOINT,
                                        "model_package": "arn:.../clf-3"}
    assert model_lineage(PRICE_ENDPOINT)["model_package"] == "arn:.../8"
    assert model_lineage(BOOKING_ENDPOINT)["model_package"] == "arn:.../clf-3", \
        "cache is not keyed by endpoint: model 2 received model 1's stamp"
    print("lineage         : stamps are per endpoint, not shared across models")

    # ---- recommender: the ranker, and the scaling bug it shipped with ----
    assert not (set(RECSYS_COLUMNS) & HOST_PRIVATE)
    print(f"recsys privacy  : reads {len(RECSYS_COLUMNS)} columns, none host-private")

    import pandas as pd
    # The API_Test_Client result, reduced. Two Paris listings 45 apart in value with
    # near-identical convenience -- "a" overpriced, "b" the real bargain -- plus two
    # out-of-city price outliers that set the global min/max, which is what the raw
    # corpus has and what makes the notebook's global fit degenerate.
    _RECSYS_DF = pd.DataFrame({
        "listing_id": ["a", "b", "outlier_hi", "outlier_lo"],
        "city": ["Paris", "Paris", "Rio de Janeiro", "Rio de Janeiro"],
        "neighbourhood": ["x", "y", "z", "z"], "property_type": ["p"] * 4,
        "room_type": ["Entire place"] * 4, "accommodates": [4, 4, 4, 4],
        "bedrooms": [1, 1, 1, 1], "minimum_nights": [1, 1, 1, 1],
        "price": [85.0, 98.0, 1.0, 50000.0], "instant_bookable": [1, 1, 1, 1],
        "pred_price": [64.84, 123.19, 50000.0, 1.0],
        "instant_book_prob": [0.994, 0.993, 0.5, 0.5],
    })
    globals()["_RECSYS_DF"] = _RECSYS_DF
    rows, _ = recsys_rank("Paris", 2, 150.0, w_value=0.5, w_convenience=0.5, top_n=5)
    assert [r["listing_id"] for r in rows] == ["b", "a"], \
        "candidate-set scaling must rank the underpriced listing first"
    spread = abs(rows[0]["ml_score"] - rows[1]["ml_score"])
    assert spread > 0.2, f"value term is not contributing (spread {spread:.4f})"
    print(f"recsys ranking  : underpriced listing ranks first, score spread {spread:.3f}")

    global RECSYS_SCALING
    RECSYS_SCALING = "global"
    rows_g, _ = recsys_rank("Paris", 2, 150.0, w_value=0.5, w_convenience=0.5, top_n=5)
    spread_g = abs(rows_g[0]["ml_score"] - rows_g[1]["ml_score"])
    assert spread_g < 0.01, "global scaling should have flattened the value term"
    assert rows_g[0]["listing_id"] == "a", "global scaling ranks the OVERPRICED listing first"
    print(f"recsys scaling  : notebook's global fit collapses the spread to {spread_g:.4f} "
          f"and puts the overpriced listing first -- the API_Test_Client result")
    RECSYS_SCALING = "candidates"
    globals()["_RECSYS_DF"] = None

    # ---- host journey: attribution against a known additive model ----------
    import types
    import numpy as np
    real_runtime = globals()["runtime"]
    calls = {"n": 0, "records": 0}

    def _fake(EndpointName, ContentType, Body, Accept=None):
        recs = json.loads(Body)
        recs = recs if isinstance(recs, list) else [recs]
        calls["n"] += 1
        calls["records"] += len(recs)
        out = [{"suggested_nightly_price": round(
                    40 + 15 * r["accommodates"] + 25 * r["bedrooms"]
                    + (30 if r["room_type"] == "Entire place" else 0)
                    - 2 * r["minimum_nights"], 2),
                "currency": "EUR", "city": r["city"], "log_price": 0.0} for r in recs]
        return {"Body": types.SimpleNamespace(read=lambda: json.dumps(out).encode())}

    rng = np.random.default_rng(0)
    n = 300
    _RECSYS_DF = pd.DataFrame({
        "listing_id": [str(1000 + i) for i in range(n)], "city": ["Paris"] * n,
        "neighbourhood": rng.choice(["Louvre", "Marais"], n),
        "property_type": ["Entire apartment"] * n,
        "room_type": rng.choice(["Entire place", "Private room"], n),
        "accommodates": rng.integers(1, 7, n), "bedrooms": rng.integers(1, 4, n),
        "minimum_nights": rng.integers(1, 5, n), "price": rng.uniform(40, 300, n).round(2),
        "amenities": ['["Wifi", "Kitchen"]'] * n,
        "instant_bookable": rng.integers(0, 2, n),
        "pred_price": rng.uniform(40, 300, n).round(2),
        "instant_book_prob": rng.uniform(0, 1, n).round(3)})
    globals()["_RECSYS_DF"] = _RECSYS_DF
    globals()["runtime"] = types.SimpleNamespace(invoke_endpoint=_fake)
    try:
        exp = explain_price(payload)
        # For an additive model the Shapley value of feature i is exactly
        # coef_i * (x_i - E[x_i]) over the background. So the mock has a closed-form
        # answer to check against, rather than a ranking that shifts with sampling.
        bg = _background_payloads(payload)
        got = {d["label"]: d["value"] for d in exp["drivers"]}
        for label, key, coef in [("Sleeps", "accommodates", 15.0),
                                 ("Bedrooms", "bedrooms", 25.0),
                                 ("Minimum nights", "minimum_nights", -2.0)]:
            want = coef * (payload[key] - sum(b[key] for b in bg) / len(bg))
            assert abs(got[label] - want) < 0.5, f"{label}: {got[label]:.2f} != {want:.2f}"
        share = sum(1 for b in bg if b["room_type"] == "Entire place") / len(bg)
        want_rt = 30.0 * ((payload["room_type"] == "Entire place") - share)
        assert abs(got["Room type"] - want_rt) < 0.5, got["Room type"]
        assert abs(got.get("Neighbourhood", 0.0)) < 0.5, "a feature the model ignores"
        total = exp["baseline"] + sum(d["value"] for d in exp["drivers"])
        assert abs(total - _batch_prices([payload])[0]) < 0.5, \
            f"SHAP efficiency violated: {total} vs the prediction"
        assert "Location" in exp["not_compared"], "must declare what it could not compare"
        print(f"explanation     : {exp['method']}, every attribution matches the "
              f"closed form, baseline+sum == prediction "
              f"({calls['records']} records in {calls['n']} calls)")

        # ---- guest journey: like for like -------------------------------------
        rows, _, seed = similar_listings("1000", top_n=5)
        assert len(rows) == 5 and all(str(r["listing_id"]) != "1000" for r in rows)
        assert all(r["city"] == seed["city"] for r in rows), "alternatives must stay in city"
        by_attrs = similar_listings("1000", 5, w_attrs=1, w_price=0, w_friction=0)[0]
        by_fric = similar_listings("1000", 5, w_attrs=0, w_price=0, w_friction=1)[0]
        assert [r["listing_id"] for r in by_attrs] != [r["listing_id"] for r in by_fric], \
            "the weights do not change the ranking"
        assert max(abs(r["instant_book_prob"] - seed["instant_book_prob"])
                   for r in by_fric) < 0.02, "friction-only must pick nearest friction"
        assert all(r["price"] <= 100 for r in similar_listings("1000", 5, max_price=100)[0])
        assert similar_listings("nope")[0] == [], "unknown seed must not raise"
        print("similarity      : same city, weights steer the ranking, budget cap honoured")
    finally:
        globals()["runtime"] = real_runtime
        globals()["_RECSYS_DF"] = None

    os.path.exists(STORE_LOCAL) and os.remove(STORE_LOCAL)
    print("\nSELF TEST PASSED")
    return 0


def cmd_list_store() -> int:
    store = store_read()
    print(f"store: {store_backend_name()}")
    if not store:
        print("(empty)")
        return 0
    print(f"{len(store)} listing(s)\n")
    print(f"  {'listing':<16s} {'owner':<16s} {'asking':>10s}  {'model pkg':<22s} updated")
    for lid, rec in sorted(store.items()):
        owner = rec.get("owner") or "(unowned)"
        pkg = ((rec.get("model") or {}).get("model_package") or "-").rsplit("/", 1)[-1]
        print(f"  {lid:<16s} {owner:<16s} {rec['asking_price']:>10,.2f}  {pkg:<22s} "
              f"{rec['updated_at'][:16]}")
    return 0


def cmd_set_owner(spec: str) -> int:
    if ":" not in spec:
        print("Expected --set-owner LISTING_ID:USERNAME (use ':' with an empty name to unown).")
        return 1
    listing_id, _, new_owner = spec.partition(":")
    listing_id, new_owner = listing_id.strip(), new_owner.strip() or None
    store = store_read()
    if listing_id not in store:
        print(f"No listing '{listing_id}'. Run --list-store to see what is there.")
        return 1
    was = store[listing_id].get("owner")
    store[listing_id]["owner"] = new_owner
    store_write(store)
    print(f"{listing_id}: owner {was!r} -> {new_owner!r}")
    return 0


def cmd_reset_store(assume_yes: bool) -> int:
    store = store_read()
    if not store:
        print("Store is already empty.")
        return 0
    print(f"This deletes {len(store)} listing(s) from {store_backend_name()}.")
    if not assume_yes:
        if input("Type DELETE to confirm: ").strip() != "DELETE":
            print("Nothing deleted.")
            return 1
    store_write({})
    print("Store cleared.")
    return 0


def parse_users(spec: str):
    pairs = [p.split(":", 1) for p in spec.split(",") if ":" in p]
    return [(u.strip(), p.strip()) for u, p in pairs]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--role", choices=["host", "customer", "both"], default="both")
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--local-store", action="store_true",
                        help="use a local JSON file instead of S3")
    parser.add_argument("--recsys-backend", choices=["auto", "api", "inline"], default=None,
                        help="where the recommender runs (default auto)")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--list-store", action="store_true",
                        help="print every listing with its owner, then exit")
    parser.add_argument("--set-owner", metavar="ID:USER",
                        help="reassign ownership, e.g. PAR-001:host ('PAR-001:' to unown)")
    parser.add_argument("--reset-store", action="store_true",
                        help="delete every listing, then exit")
    parser.add_argument("--yes", action="store_true", help="skip the reset confirmation")
    args = parser.parse_args(argv)

    global USE_S3, RECSYS_BACKEND
    if args.local_store:
        USE_S3 = False
    if args.recsys_backend:
        RECSYS_BACKEND = args.recsys_backend

    if args.self_test:
        return self_test()
    if args.list_store:
        return cmd_list_store()
    if args.set_owner:
        return cmd_set_owner(args.set_owner)
    if args.reset_store:
        return cmd_reset_store(args.yes)

    load_fitted_state()
    price_lin = model_lineage(PRICE_ENDPOINT)
    book_lin = model_lineage(BOOKING_ENDPOINT)
    print(f"{COURSE} {TEAM_ID} / {STUDENT_ID} | role={args.role} | region={REGION}")
    print(f"price    : {PRICE_ENDPOINT} ({price_lin.get('status')}) "
          f"{len(SERVING_COLUMNS)} fields from {CONTRACT_SOURCE}")
    print(f"           {price_lin.get('model_package') or 'model package unknown'}")
    print(f"booking  : {BOOKING_ENDPOINT} ({book_lin.get('status')}) "
          f"{len(BOOKING_COLUMNS)} fields")
    print(f"           {book_lin.get('model_package') or 'model package unknown'}")
    print(f"recsys   : {recsys_backend_name()}")
    if _RECSYS_ERROR and RECSYS_BACKEND != "api":
        print(f"           corpus did not load: {_RECSYS_ERROR}")
    print(f"store    : {store_backend_name()} ({len(store_read())} listings)")

    # The corpus was scored by whatever endpoint notebook 05 pointed at, and it
    # carries no stamp of its own -- so pred_price can silently belong to a model the
    # price tab no longer serves. Same staleness the store records guard against,
    # minus the guard.
    print("           note: corpus pred_price is unstamped; re-run notebook 05 after "
          "any price redeploy")

    users = parse_users(HOST_USERS)
    if args.role == "customer":
        demo, port, auth = build_customer_ui(), args.port or 7862, None
    elif args.role == "host":
        demo, port, auth = build_host_ui(), args.port or 7861, users
    else:
        print("\nDEMO MODE: host and customer in one process behind one login.")
        print("This is NOT a trust boundary -- anyone who can reach the app can open the")
        print("Host tab. Use --role host and --role customer as separate processes for a")
        print("real split.\n")
        demo = build_both_ui()
        port, auth = args.port or 7859, users

    if auth:
        print(f"auth     : enabled for {[u for u, _ in auth]}")
    else:
        print("auth     : none (read-only customer view)")

    demo.launch(server_name="0.0.0.0", server_port=port, share=True,
                auth=auth, show_error=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
