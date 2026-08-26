#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Three models, two journeys, one decision-support system
=======================================================

    1. a fair nightly price for any listing        PRICE    regression endpoint
    2. whether it is instantly bookable            BOOKING  classification endpoint
    3. listings ranked using both of those         RECSYS   POST /recommend, served here

HOST JOURNEY -- pricing assistant
    Enter a listing, or select one to start from, and get back the suggested price,
    which of its attributes carry that price (SHAP, computed against the endpoint),
    and how easily a listing like it books at that rate. Then save it, which puts it
    in the pool the guest journey ranks.

GUEST JOURNEY -- discovery assistant
    POST /recommend: city, min_guests, max_price, w_value, w_convenience. The ranking
    is the two model outputs combined --

        ml_score = w_value x (pred_price - price, scaled) + w_convenience x P(instant book)

    -- so the value term is the price model and the convenience term is the classifier.
    Neither weight is decoration: see RECOMMEND_SCALING, which decides whether the
    first of them reaches the ranking at all.

Both journeys are plain functions returning plain dicts, so the UI only renders them
and the API only adapts them to HTTP:

    price_listing(ui)                       -> the whole host answer, JSON-ready
    api_recommend(city, guests, ...)        -> the whole guest answer, JSON-ready

Run
---
    python host_guest_app.py --role host        # pricing assistant, authenticated  :7861
    python host_guest_app.py --role guest       # discovery assistant, public       :7862
    python host_guest_app.py --role both        # single-process demo               :7859
    python host_guest_app.py --with-api         # UI + API in one process, over HTTP
    python host_guest_app.py --self-test        # no UI, no AWS, no gradio, no boto3

    python host_guest_app.py --serve-api        # the recommender API, nothing else :8000
    python host_guest_app.py --serve-api --api-test    # start it, exercise it, exit
    python host_guest_app.py --recommend Paris --scaling candidates
    python host_guest_app.py --price-json listing.json # host journey, JSON
    python host_guest_app.py --list-store              # what hosts have saved
    python host_guest_app.py --rescore-store           # score listings saved by v1

Where the API came from
-----------------------
Test_API_Gradio.ipynb and API_Test_Client.ipynb are now `--serve-api` and `--api-test`
in this file. Two notebooks meant two copies of the 280k-row corpus in memory and two
copies of the ranking code, and that second copy is how the served ranking and
notebook 05's evaluation came to disagree about scaling. /recommend keeps its original
contract exactly: same five request fields, same seven response fields.

RECOMMEND_SCALING, and why it is not a tuning knob
--------------------------------------------------
deal_value = pred_price - price has to be scaled to [0,1] before it can be added to a
probability. 

So the Utility Lift and the NDCG@10 in the report measured candidate-set scaling while
the API served global. Under global, prices spanning ten currencies put a single
city's candidates inside a sliver of the range: every listing scales to nearly the
same number and the ranking is instant_book_prob alone at any w_value below 1. That is
why the served API returned an overpriced listing (EUR 85 against a EUR 64.84 model
value) ahead of a EUR 25 bargain at rank 1.

The default is therefore `candidates`, which is what was evaluated.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import sys
import time
from datetime import date, datetime, timezone
from itertools import combinations
from math import factorial
from pathlib import Path
from typing import Any, Callable, Iterable

# ============================================================================
# 1. CONFIGURATION
# ============================================================================

TEAM_ID = os.environ.get("TEAM_ID", "team14")
STUDENT_ID = os.environ.get("STUDENT_ID", "s1402")
COURSE = "ITI113"
REGION = os.environ.get("AWS_REGION", "ap-southeast-1")

PRICE_ENDPOINT = os.environ.get("PRICE_ENDPOINT", f"iti113-{TEAM_ID}-airbnb-price-sls")
BOOKING_ENDPOINT = os.environ.get("BOOKING_ENDPOINT", f"iti113-{TEAM_ID}-airbnb-instant-booking")

BUCKET = os.environ.get("BUCKET", "nyp-26s1-iti113")
PREFIX = f"iti113/{TEAM_ID}/data/airbnb-listings"
PROCESSED_PREFIX = f"{PREFIX}/processed"
ARTIFACTS_PREFIX = f"{PREFIX}/artifacts"

# Notebook 05's output: raw Listings.csv joined to pred_price and instant_book_prob.
# The guest journey is a similarity search over this, so it is the corpus, not a cache.
CORPUS_PREFIX = f"iti113/{TEAM_ID}/data/airbnb-recommendation"
CORPUS_CSV = os.environ.get(
    "CORPUS_CSV", f"s3://{BUCKET}/{CORPUS_PREFIX}/processed/master_recsys_dataset.csv")

# SHAP budget. 7 groups is 2**7 = 128 coalitions, which is exact. The switch to
# permutation sampling only fires if someone adds an eighth group and does not raise
# this -- it is a guard rail, not a normal path.
SHAP_MAX_CALLS = int(os.environ.get("SHAP_MAX_CALLS", "160"))
SHAP_PERMUTATIONS = int(os.environ.get("SHAP_PERMUTATIONS", "48"))
SHAP_BATCH = os.environ.get("SHAP_BATCH", "1").lower() not in ("0", "false", "no")
SHAP_CHUNK = int(os.environ.get("SHAP_CHUNK", "64"))

# Where the guest query is ranked.
#
#   api     POST it to the FastAPI server in Test_API_Gradio.ipynb, which serves this
#           module's own similar_listings() over HTTP
#   inline  rank in this process
#   auto    (default) try the API, fall back to inline -- the notebook cell is often
#           not running, and a demo that dies because a second notebook is closed is
#           worse than one that quietly does the same work locally and says so
#
# The notebook's original /recommend cannot serve this journey: its SearchQuery is
# city + guests + budget + two weights, with no seed listing, so "more like this one"
# is not expressible in it. Rather than bend the guest journey to fit the old
# contract, the notebook gained a /similar endpoint that imports THIS module and
# calls the same function the inline path calls. One ranker, two transports: the
# alternative is a second copy of the algorithm in a notebook, drifting from this one
# -- which is exactly how the served ranking and notebook 05's evaluation ended up
# disagreeing about scaling in the first place.
RECSYS_API_URL = os.environ.get("RECSYS_API_URL", "http://127.0.0.1:8000")
RECSYS_BACKEND = os.environ.get("RECSYS_BACKEND", "auto").lower()   # api | inline | auto
API_HOST = os.environ.get("API_HOST", "0.0.0.0")
API_PORT = int(os.environ.get("API_PORT", "8000"))

# WHERE /recommend's VALUE TERM GETS ITS MIN AND MAX.
#
#   candidates  fitted over the listings actually being ranked, after filtering.
#               DEFAULT, because it is what notebook 05 EVALUATED. compare_recsys()
#               (cell 19, the Utility Lift) and evaluate_ndcg() (cell 21, NDCG@10)
#               both filter to `candidates` first and only then call
#               scaler.fit_transform(candidates[['deal_value']]).
#   global      fitted over every row in every city, before any filtering. What
#               Test_API_Gradio.ipynb served, and what notebook 05's own engine cell
#               (cell 14) used. Set RECOMMEND_SCALING=global or pass --scaling global
#               to reproduce the existing client output exactly.
#
# Prices span ten currencies, so the global range is set by outliers in other cities; a Paris
# candidate set then occupies a sliver of it, every listing scales to nearly the same
# number, and the ranking is instant_book_prob alone at any w_value below 1. The
# reported NDCG and Utility Lift therefore described an algorithm that was never
# served -- which is why the API returned an overpriced listing (EUR 85 against a
# EUR 64.84 model value) ahead of a EUR 25 bargain at rank 1.
#
# Both are reproduced exactly by the self test, against literal transcriptions of both
# notebooks, so neither is inherited by accident.
RECOMMEND_SCALING = os.environ.get("RECOMMEND_SCALING", "candidates").lower()
# Captured at import so the self test can catch the default being flipped back without
# the evaluation mismatch being re-argued.
_SCALING_DEFAULT = RECOMMEND_SCALING

# The shared listing store. S3 first, so a saved listing survives a Studio restart
# and is visible to a guest app running in a different process; a local file
# otherwise. This is the only thing either journey writes.
STORE_KEY = f"iti113/{TEAM_ID}/apps/listing-store/listings.json"
STORE_LOCAL = os.environ.get("STORE_LOCAL", "listing_store.json")
USE_S3 = os.environ.get("STORE_BACKEND", "s3").lower() == "s3"

# Demo credentials for the host dashboard. Real deployments read these from Secrets
# Manager. Saving is a write and saved listings carry an owner, so this is what
# stops one host overwriting another's listing.
HOST_USERS = os.environ.get("HOST_USERS", "host:iti113")

_CLIENTS: dict[str, Any] = {}


def aws(service: str):
    """boto3 clients, built on first use.

    Module-level clients made `--self-test` require boto3 installed and a region
    configured, which contradicts the one thing that mode promises. Building them
    lazily keeps every offline path -- the self test, `--similar`, the SHAP maths,
    the similarity maths -- importable on a machine with neither boto3 nor gradio.
    """
    if service not in _CLIENTS:
        import boto3
        _CLIENTS[service] = boto3.client(service, region_name=REGION)
    return _CLIENTS[service]


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
                              SERVING_COLUMNS, SNAPSHOT_DATE)
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
            return json.loads(
                aws("s3").get_object(Bucket=BUCKET, Key=f"{prefix}/{filename}")["Body"].read())
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


def modal_neighbourhood(city):
    """The most common neighbourhood in a city, used as the SHAP reference location."""
    freq = ARTIFACTS.get("neighbourhood_freq") or {}
    here = {k.split("||", 1)[1]: v for k, v in freq.items() if k.split("||", 1)[0] == city}
    return max(here, key=here.get) if here else None


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


def haversine_km(lat1, lon1, lat2, lon2) -> float:
    p = math.pi / 180.0
    a = (math.sin((lat2 - lat1) * p / 2) ** 2
         + math.cos(lat1 * p) * math.cos(lat2 * p) * math.sin((lon2 - lon1) * p / 2) ** 2)
    return 2 * R_EARTH_KM * math.asin(math.sqrt(a))


# ============================================================================
# 3. THE PUBLICATION ALLOWLIST
# ============================================================================
# The pipeline already treats coordinates as sensitive: ListingFeatureEngineer
# derives dist_center_km, coarsens lat/lon to 2 decimals and discards the exact
# values, so precise GPS never reaches a fitted artifact. The guest journey must not
# undo that on the way back out.
#
# So publication is an ALLOWLIST. A field is invisible to guests unless it is named
# here, which means adding a field to the host form cannot silently expose it.
# HOST_PRIVATE is then asserted as a second, redundant check, and asserted AGAIN
# against the corpus projection in section 9 -- the corpus is raw Listings.csv plus
# two score columns, so every private field is sitting in that object.
# ============================================================================

GUEST_VISIBLE = {
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

assert not (GUEST_VISIBLE & HOST_PRIVATE), "a field cannot be both published and private"
assert GUEST_VISIBLE | HOST_PRIVATE >= set(SERVING_COLUMNS), \
    f"unclassified contract fields: {set(SERVING_COLUMNS) - GUEST_VISIBLE - HOST_PRIVATE}"


def public_view(payload: dict, asking_price: float | None = None) -> dict:
    """Everything a guest may see of a listing, and nothing else."""
    view = {k: v for k, v in payload.items() if k in GUEST_VISIBLE}
    city = payload.get("city")
    if city in CITY_CENTERS and payload.get("latitude") is not None:
        clat, clon = CITY_CENTERS[city]
        view["dist_center_km"] = round(
            haversine_km(float(payload["latitude"]), float(payload["longitude"]), clat, clon), 2)
    if asking_price is not None:
        view["price"] = float(asking_price)
    view["currency"] = CURRENCY.get(city, "")
    leaked = HOST_PRIVATE & set(view)
    if leaked:                     # belt and braces: the allowlist should make this unreachable
        raise AssertionError(f"guest view leaked host-private fields: {sorted(leaked)}")
    return view


# ============================================================================
# 4. THE ENDPOINTS, AND WHICH MODEL ANSWERED
# ============================================================================
# Keyed BY ENDPOINT. This was a flat dict while there was only one model, which made
# model_lineage() ignore its own argument: asking for the booking endpoint after the
# price endpoint returned the price endpoint's stamp, silently. One model hid the
# bug; two expose it. Three would have hidden it again behind a plausible answer.
# ============================================================================

_LINEAGE_CACHE: dict[str, dict] = {}


def model_lineage(endpoint_name: str = PRICE_ENDPOINT, refresh: bool = False) -> dict:
    # Always hand back a COPY. Callers embed this in responses that get logged and
    # compared; a cached dict returned by reference would let a later refresh rewrite
    # the stamp on answers that were already given.
    cached = _LINEAGE_CACHE.get(endpoint_name)
    if cached and not refresh:
        return dict(cached)
    out = {"endpoint": endpoint_name, "model_package": None, "model": None, "status": "Unknown"}
    try:
        sm = aws("sagemaker")
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
    resp = aws("sagemaker-runtime").invoke_endpoint(
        EndpointName=PRICE_ENDPOINT, ContentType="application/json", Body=json.dumps(payload))
    records = json.loads(resp["Body"].read())
    return records[0], (time.perf_counter() - t0) * 1000.0


def invoke_price_batch(payloads: list[dict]) -> list[dict]:
    """Score many listings in as few round trips as possible.

    This is what makes exact SHAP affordable. 128 coalitions scored one at a time
    against a serverless endpoint capped at max_concurrency=5 is a slow minute; sent
    as one JSON array it is one request.

    The fallback is not defensive noise. The single-payload form above is the one
    that is known to work in this project, so if the array form comes back the wrong
    length -- an inference.py that quietly takes the first row, an older container --
    the whole explanation would be built from wrong numbers with nothing raising.
    Length is checked, and a mismatch falls back to the form that is known good.
    """
    if not payloads:
        return []
    if SHAP_BATCH:
        try:
            out: list[dict] = []
            for i in range(0, len(payloads), max(1, SHAP_CHUNK)):
                chunk = payloads[i:i + max(1, SHAP_CHUNK)]
                resp = aws("sagemaker-runtime").invoke_endpoint(
                    EndpointName=PRICE_ENDPOINT, ContentType="application/json",
                    Body=json.dumps(chunk))
                records = json.loads(resp["Body"].read())
                if not isinstance(records, list) or len(records) != len(chunk):
                    raise ValueError(
                        f"batch of {len(chunk)} returned {len(records) if isinstance(records, list) else type(records).__name__}")
                out.extend(records)
            return out
        except Exception:
            pass                                    # fall through to the known-good form
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=4) as pool:   # 4, not 5: leave headroom
        return [r for r, _ in pool.map(invoke_price, payloads)]


def invoke_booking(payload: dict) -> tuple[dict, float]:
    """Instant-booking classifier. Different envelope from the price endpoint.

    inference.py's _predict_fn accepts a bare dict, {"data": dict}, {"data": list} or
    a bare list; the price endpoint's takes the bare form only. Sending
    {"data": [payload]} keeps the difference visible rather than relying on the
    classifier's more forgiving parser to paper over it.
    """
    t0 = time.perf_counter()
    resp = aws("sagemaker-runtime").invoke_endpoint(
        EndpointName=BOOKING_ENDPOINT, ContentType="application/json",
        Accept="application/json", Body=json.dumps({"data": [payload]}))
    records = json.loads(resp["Body"].read())
    return records[0], (time.perf_counter() - t0) * 1000.0


def invoke_parallel(jobs: dict[str, Callable[[], Any]]) -> dict:
    """Run independent endpoint calls at once, reporting failures per job.

    Serverless scales to zero and a cold start runs to ~60 s, so sequencing two calls
    makes the first click of the day take two minutes. Partial failure is the normal
    case here rather than the edge case, so one cold endpoint must not blank out the
    other's answer.
    """
    from concurrent.futures import ThreadPoolExecutor
    out: dict[str, Any] = {}
    with ThreadPoolExecutor(max_workers=max(2, len(jobs))) as pool:
        futures = {name: pool.submit(fn) for name, fn in jobs.items()}
        for name, fut in futures.items():
            try:
                out[name] = fut.result()
            except Exception as e:
                out[f"{name}_error"] = e
    return out


def explain_error(exc: Exception) -> str:
    """Say what broke and what to do about it. Errors don't apologise and don't shrug."""
    name, msg = exc.__class__.__name__, str(exc)
    hint = ""
    if "ValidationException" in name + msg:
        hint = f"No endpoint `{PRICE_ENDPOINT}` in `{REGION}`. Check the name and the region."
    elif "AccessDenied" in name + msg:
        hint = "This role cannot invoke the endpoint. Check the execution role's policy."
    elif "ModelError" in name + msg:
        hint = "The endpoint rejected the payload. Compare it against the contract fields below."
    elif "Throttling" in name + msg:
        hint = "Serverless endpoint at max_concurrency=5. Send it again in a moment."
    elif "NoCredentials" in name or "ExpiredToken" in msg:
        hint = "AWS credentials are missing or expired. Refresh them and retry."
    elif "ReadTimeout" in name or "timed out" in msg.lower():
        hint = "Cold start, up to ~60 s on the first call of the day. Send it again."
    return f"**{name}**\n\n```\n{msg}\n```\n\n{hint}".rstrip()


# ============================================================================
# 5. THE LISTING STORE
# ============================================================================
# The join between the two journeys. A host prices a listing and saves it; a guest
# can then seed a search from it AND find it among the results, because a saved
# listing enters the same similarity space as the corpus.
#
# WHAT IS SAVED, AND WHY THAT SET. A saved record has to carry both model outputs,
# not just the price: pred_price and instant_book_prob are two of the three
# similarity dimensions, so a listing saved without a booking score has no position
# in that space. Such records are stored anyway -- losing the host's work because one
# endpoint was cold would be worse -- but they are held back from the guest catalogue
# and the host is told why, rather than saving successfully and never appearing.
#
# MIXED MODEL VERSIONS. The corpus was scored by whatever endpoint notebook 05
# pointed at and carries no stamp. Saved listings are scored live and do carry one.
# So a saved pred_price and a corpus pred_price can come from different models, and
# pred_price is a ranking dimension -- a systematic shift between the two would tilt
# every comparison. Nothing here can detect drift in the corpus, so the honest move
# is to stamp what we can, flag a saved record whose stamp no longer matches the
# deployed model, and say so where the two are mixed.
#
# CONCURRENCY. Last write wins. The local backend writes through os.replace so a
# reader never sees a half-written file; the S3 backend relies on put_object being
# atomic per object. Neither does read-modify-write locking, so two hosts editing the
# SAME id within the same second can lose one edit. Acceptable for a prototype, and
# the first thing to fix if it grows up: S3 conditional writes on ETag, or DynamoDB
# with a conditional expression.
# ============================================================================

LISTING_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{2,31}$")

# The store key is unchanged from the previous app, so this reads records that app
# wrote. Its shape was different: the price lived in a nested `prediction` dict, there
# was no instant-booking probability at all, and `model` was one flat lineage stamp
# for the price endpoint. Same key plus a changed schema and no migration is how a
# working app greets you with a KeyError on the records you already had.
#
# Upgrading happens on READ, so every caller downstream sees exactly one shape and
# none of them need to know two. It is non-destructive -- the upgrade is in memory
# until something writes -- and `--migrate-store` makes it permanent in one pass.
STORE_SCHEMA = 2


def _upgrade_record(rec: dict) -> dict:
    """Bring one stored record up to the current schema, in memory."""
    if not isinstance(rec, dict):
        return {}
    if rec.get("schema") == STORE_SCHEMA:
        return rec
    out = dict(rec)
    payload = out.get("payload") or {}
    pred = out.pop("prediction", None) or {}            # v1 nested the price in here
    if out.get("suggested_price") is None:
        sp = pred.get("suggested_nightly_price")
        out["suggested_price"] = float(sp) if sp is not None else None
    if not out.get("currency"):
        out["currency"] = pred.get("currency") or CURRENCY.get(payload.get("city"), "")
    # v1 predates the classifier being part of a saved listing, so there is no
    # probability to recover and inventing one would be worse than not having it.
    # None is truthful, and keeps the record out of the guest catalogue until it is
    # re-scored -- which --rescore-store can do from the stored payload alone.
    out.setdefault("instant_book_prob", None)
    out.setdefault("instant_book_prob_at", None)
    out.setdefault("drivers", [])
    model = out.get("model")
    if isinstance(model, dict) and "price" not in model and "booking" not in model:
        out["model"] = {"price": model or None, "booking": None}
    elif not isinstance(model, dict):
        out["model"] = {"price": None, "booking": None}
    ask = out.get("asking_price")
    out["asking_price"] = float(ask) if ask not in (None, "") else out.get("suggested_price")
    out.setdefault("owner", None)
    out.setdefault("updated_at", "")
    out.setdefault("updated_by", "unknown")
    out["schema"] = STORE_SCHEMA
    return out


def store_read() -> dict:
    raw: Any = {}
    if USE_S3:
        try:
            raw = json.loads(aws("s3").get_object(Bucket=BUCKET, Key=STORE_KEY)["Body"].read())
        except Exception:
            raw = {}
    elif os.path.exists(STORE_LOCAL):
        try:
            raw = json.loads(Path(STORE_LOCAL).read_text())
        except Exception:
            raw = {}
    if not isinstance(raw, dict):
        return {}
    return {k: _upgrade_record(v) for k, v in raw.items()}


def store_write(data: dict) -> None:
    body = json.dumps(data, indent=2, sort_keys=True)
    if USE_S3:
        aws("s3").put_object(Bucket=BUCKET, Key=STORE_KEY, Body=body.encode(),
                             ContentType="application/json")
        return
    tmp = STORE_LOCAL + ".tmp"
    Path(tmp).write_text(body)
    os.replace(tmp, STORE_LOCAL)          # atomic: no reader sees a partial file


def store_backend_name() -> str:
    return f"s3://{BUCKET}/{STORE_KEY}" if USE_S3 else os.path.abspath(STORE_LOCAL)


def current_user(request) -> str | None:
    """The authenticated username, or None when the app runs without auth.

    None rather than "anonymous" on purpose: an unauthenticated save creates an
    UNOWNED listing that anyone can later edit or delete, which is the correct
    behaviour for a public demo. Inventing a pseudo-user would instead strand the
    record the moment auth is switched on.
    """
    return getattr(request, "username", None) or None


def ownership_message(listing_id: str, owner: str | None, username: str | None) -> str:
    """Name both sides. A refusal that hides the actual owner cannot be acted on."""
    you = f"`{username}`" if username else "not signed in (this app has no auth)"
    return (f"`{listing_id}` belongs to `{owner}`, and you are {you}. "
            f"Only the owner can change it.\n\n"
            f"To inspect or repair the store from the command line:\n"
            f"```\n"
            f"python host_guest_app.py --list-store\n"
            f"python host_guest_app.py --set-owner {listing_id}:{username or 'YOURNAME'}\n"
            f"python host_guest_app.py --reset-store\n"
            f"```")


def save_listing(listing_id: str, resp: dict, username: str | None) -> dict:
    """Persist a priced listing. `resp` is a host-journey response.

    Everything a guest sees is derived from that response rather than re-entered, so
    the published price and the price the model actually produced cannot drift apart.
    """
    if not LISTING_ID_RE.match(listing_id or ""):
        raise ValueError("Listing ID must be 3-32 characters: letters, digits, "
                         "hyphen, underscore.")
    if not resp.get("ok") or resp.get("suggested_nightly_price") is None:
        raise ValueError("Price the listing before saving it.")

    store = store_read()
    existing = store.get(listing_id)
    if existing and existing.get("owner") not in (None, username):
        raise PermissionError(ownership_message(listing_id, existing.get("owner"), username))

    suggested = float(resp["suggested_nightly_price"])
    # No asking price means the host is publishing at the model's number. Recording
    # the suggestion as the asking price keeps `price` meaning one thing everywhere:
    # what a guest would pay.
    asking = resp.get("asking_price")
    asking = float(asking) if asking not in (None, "") else suggested

    # AT THE ASKING PRICE, not at the suggestion. The classifier takes price as a
    # feature, so a probability means nothing without the price it was scored at, and
    # the corpus holds each listing's probability at the price it actually charges.
    # Storing the reading taken at the model's suggestion would put one listing's
    # shelf price against another's hypothetical -- and friction is a ranking
    # dimension, so that error would tilt every comparison a guest makes.
    friction = resp.get("friction_at_asking") or resp.get("friction") or {}
    prob = friction.get("instant_book_prob")

    store[listing_id] = {
        "payload": resp["request"],
        "asking_price": asking,
        "suggested_price": suggested,
        "instant_book_prob": float(prob) if prob is not None else None,
        "instant_book_prob_at": friction.get("at_price"),
        "currency": resp.get("currency", ""),
        "schema": STORE_SCHEMA,
        "drivers": (resp.get("explanation") or {}).get("contributions", [])[:3],
        "model": {"price": model_lineage(PRICE_ENDPOINT),
                  "booking": model_lineage(BOOKING_ENDPOINT)},
        "owner": existing.get("owner") if existing else username,
        "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "updated_by": username or "unauthenticated",
    }
    store_write(store)
    return store[listing_id]


def delete_listing(listing_id: str, username: str | None) -> str:
    store = store_read()
    rec = store.get(listing_id)
    if rec is None:
        return f"No listing `{listing_id}`."
    if rec.get("owner") not in (None, username):
        return ownership_message(listing_id, rec.get("owner"), username)
    del store[listing_id]
    store_write(store)
    return f"Deleted `{listing_id}`. It will drop out of guest results on the next search."


def record_is_stale(rec: dict) -> str | None:
    """Were this record's scores produced by a model that is no longer deployed?"""
    saved = ((rec.get("model") or {}).get("price") or {}).get("model_package")
    if not saved:
        return None
    current = model_lineage(PRICE_ENDPOINT).get("model_package")
    if current and saved != current:
        return (f"scored by {saved.rsplit('/', 1)[-1]}, endpoint now serves "
                f"{current.rsplit('/', 1)[-1]} -- re-price and save again")
    return None


def store_rows(store: dict | None = None) -> list[dict]:
    """Saved listings, projected to the corpus schema so they can be searched.

    This is the publication allowlist applied a third time. The store holds all 27
    contract fields, coordinates and acceptance rate included; the guest catalogue
    must receive only the columns the corpus itself exposes. Built by naming columns
    explicitly and then asserted, because this is the one place where host-private
    data and a guest-facing surface are in the same function.
    """
    rows = []
    for lid, rec in (store_read() if store is None else store).items():
        p = rec.get("payload") or {}
        if (rec.get("suggested_price") is None or rec.get("instant_book_prob") is None
                or rec.get("asking_price") is None or not p.get("city")):
            continue                       # no position in the similarity space

        row = {
            "listing_id": str(lid),
            "city": p.get("city"),
            "neighbourhood": p.get("neighbourhood") or "",
            "property_type": p.get("property_type") or "",
            "room_type": p.get("room_type") or "",
            "accommodates": int(p.get("accommodates") or 1),
            "bedrooms": int(p.get("bedrooms") or 0),
            "minimum_nights": int(p.get("minimum_nights") or 1),
            "price": float(rec["asking_price"]),
            "instant_bookable": 1 if p.get("instant_bookable") == "t" else 0,
            "pred_price": float(rec["suggested_price"]),
            "instant_book_prob": float(rec["instant_book_prob"]),
            "source": "saved",
        }
        leaked = HOST_PRIVATE & set(row)
        if leaked:
            raise AssertionError(f"the guest catalogue leaked host-private fields: {sorted(leaked)}")
        rows.append(row)
    return rows


# ============================================================================
# 6. THE EXPLANATION
# ============================================================================
# Shapley values over attribute GROUPS, with the endpoint as the value function.
#
# WHY GROUPS. Two reasons, one practical and one about the product. 27 features is
# 2**27 coalitions, so exact attribution is out and any answer would be a sampled
# approximation with error bars. And a host cannot act on `review_scores_checkin`
# alone -- "your reviews are worth +9 EUR" is advice, "checkin score is worth +0.4"
# is noise. Seven groups is 128 coalitions: exact, in one batched request.
#
# WHY CITY IS HELD FIXED. Swapping Paris for Bangkok inside a coalition changes the
# currency, and averaging EUR against THB produces a number in no unit at all. The
# reference is therefore a typical listing IN THE SAME CITY, so every attribution
# reads "against a typical Paris listing" -- which is also the only comparison the
# host can do anything about. City is not unexplained by oversight; it is the axis
# the explanation is conditioned on, and section 5's assert makes leaving a field
# out of both the groups and the fixed set impossible.
#
# WHAT THIS IS, PRECISELY. Exact Shapley values against a SINGLE reference point,
# i.e. interventional/baseline SHAP with |background| = 1, computed on the price
# scale. It is not TreeSHAP against the training distribution, and the numbers move
# if the reference moves -- which is why the reference is published in the response
# and shown in the UI. An attribution without its baseline is not an explanation.
# ============================================================================

EXPLAIN_GROUPS: dict[str, list[str]] = {
    "Location":      ["latitude", "longitude", "neighbourhood"],
    "Property type": ["property_type", "room_type"],
    "Capacity":      ["accommodates", "bedrooms"],
    "Amenities":     ["amenities"],
    "Booking rules": ["minimum_nights", "maximum_nights", "instant_bookable"],
    "Host profile":  ["host_since", "host_is_superhost", "host_has_profile_pic",
                      "host_identity_verified", "host_total_listings_count",
                      "host_response_time", "host_response_rate", "host_acceptance_rate"],
    "Reviews":       list(REVIEW_COLS),
}
EXPLAIN_FIXED = {"city"}

_GROUPED = [c for cols in EXPLAIN_GROUPS.values() for c in cols]
assert len(_GROUPED) == len(set(_GROUPED)), "a field appears in two explanation groups"
assert not (set(_GROUPED) & EXPLAIN_FIXED), "a field cannot be both explained and held fixed"
assert set(_GROUPED) | EXPLAIN_FIXED == set(SERVING_COLUMNS), (
    f"unexplained contract fields: {set(SERVING_COLUMNS) - set(_GROUPED) - EXPLAIN_FIXED}")

# The reference listing. Medians come from the fitted preprocessing artifacts where
# they exist -- those are the training medians, not invented numbers -- and fall back
# to these documented values when the artifact is unavailable, so the app still runs
# on a laptop with nothing checked out. Every fallback here is a plausible mode of
# the Listings.csv data, and each one is stated in the response so nothing is taken
# on trust.
BASELINE_NUMERIC = {
    "accommodates": 3, "bedrooms": 1, "minimum_nights": 2, "maximum_nights": 365,
    "host_total_listings_count": 2, "host_response_rate": 100.0, "host_acceptance_rate": 90.0,
}
BASELINE_AMENITIES = ["wifi", "kitchen", "essentials", "heating", "hot water",
                      "hangers", "tv", "washer", "air conditioning", "smoke alarm"]


def baseline_listing(city: str) -> dict:
    """A typical listing in `city`, in serving-contract shape."""
    rdef = review_defaults()
    med = ARTIFACTS.get("imputation_values") or {}

    def num(col):
        try:
            return float(med[col])
        except (KeyError, TypeError, ValueError):
            return float(BASELINE_NUMERIC[col])

    lat, lon = CITY_CENTERS.get(city, (0.0, 0.0))
    rooms = room_type_choices()
    vocab = amenity_choices()
    payload = {
        "city": city,
        "latitude": lat, "longitude": lon,          # the city centre, i.e. dist_center_km = 0
        "neighbourhood": modal_neighbourhood(city),
        "property_type": "Entire apartment",
        "room_type": "Entire place" if "Entire place" in rooms else rooms[0],
        "accommodates": int(num("accommodates")),
        "bedrooms": int(num("bedrooms")),
        "amenities": json.dumps(sorted(a for a in BASELINE_AMENITIES if a in set(vocab))
                                or BASELINE_AMENITIES[:6]),
        "minimum_nights": int(num("minimum_nights")),
        "maximum_nights": int(num("maximum_nights")),
        "instant_bookable": "f",
        "host_since": f"{SNAPSHOT_DATE.year - 5:04d}-01-01",
        "host_is_superhost": "f",
        "host_has_profile_pic": "t",
        "host_identity_verified": "t",
        "host_total_listings_count": int(num("host_total_listings_count")),
        "host_response_time": response_time_choices()[0],
        "host_response_rate": num("host_response_rate"),
        "host_acceptance_rate": num("host_acceptance_rate"),
        **{c: rdef[c] for c in REVIEW_COLS},
    }
    return {k: payload[k] for k in SERVING_COLUMNS}


def _coalition(x: dict, base: dict, names: list[str], gdef: dict[str, list[str]],
               present: Iterable[int]) -> dict:
    """`base`, with the groups in `present` taken from `x`.

    `gdef` is passed rather than read from the module global on purpose: explain_price
    takes a `groups` argument, and looking the columns up in EXPLAIN_GROUPS regardless
    would have quietly explained the default grouping while reporting the caller's
    names -- wrong numbers under right labels, which is the worst failure an
    explanation can have.
    """
    out = dict(base)
    present = set(present)
    for i, name in enumerate(names):
        if i in present:
            for col in gdef[name]:
                out[col] = x[col]
    return {k: out[k] for k in SERVING_COLUMNS}


def shapley_exact(value: dict[frozenset, float], n: int) -> list[float]:
    """phi_i = sum over S not containing i of |S|!(n-|S|-1)!/n! * [v(S+i) - v(S)]."""
    phi = [0.0] * n
    for i in range(n):
        others = [j for j in range(n) if j != i]
        for k in range(len(others) + 1):
            w = factorial(k) * factorial(n - k - 1) / factorial(n)
            for subset in combinations(others, k):
                s = frozenset(subset)
                phi[i] += w * (value[s | {i}] - value[s])
    return phi


def shapley_sampled(x, base, names, gdef, score, n_perm, seed=0):
    """Permutation sampling, for when someone adds an eighth group.

    Each permutation walks from the empty coalition to the full one, so its marginal
    contributions telescope to exactly v(N) - v(empty). Efficiency therefore holds
    for every single permutation and so for the mean of any number of them: the
    estimate is noisy per group but never fails to add up, which is the property the
    UI depends on.
    """
    rng = random.Random(seed)
    n = len(names)
    memo: dict[frozenset, float] = {}

    def v(mask: frozenset) -> float:
        if mask not in memo:
            memo[mask] = score([_coalition(x, base, names, gdef, mask)])[0]
        return memo[mask]

    totals = [0.0] * n
    order = list(range(n))
    for _ in range(n_perm):
        rng.shuffle(order)
        cur, prev = frozenset(), v(frozenset())
        for i in order:
            cur = cur | {i}
            nxt = v(cur)
            totals[i] += nxt - prev
            prev = nxt
    return [t / n_perm for t in totals], len(memo)


def _fmt_value(col: str, v: Any) -> str:
    if col == "amenities":
        try:
            return f"{len(json.loads(v))} amenities"
        except Exception:
            return "amenities"
    if col in ("latitude", "longitude"):
        return f"{float(v):.3f}"
    if v is None:
        return "not given"
    if isinstance(v, float):
        return f"{v:g}"
    if col.startswith("host_") and v in ("t", "f"):
        return "yes" if v == "t" else "no"
    if col == "instant_bookable":
        return "on" if v == "t" else "off"
    return str(v)


# Mechanical de-prefixing turned host_since into "since" and
# review_scores_checkin into "checkin", which reads like a typo in the one place the
# host is being asked to trust a number. Named where the automatic version is poor.
FIELD_LABELS = {
    "host_since": "hosting since", "host_is_superhost": "superhost",
    "host_has_profile_pic": "profile picture", "host_identity_verified": "identity verified",
    "host_total_listings_count": "listings run", "host_response_time": "response time",
    "host_response_rate": "response rate", "host_acceptance_rate": "acceptance rate",
    "review_scores_rating": "overall rating", "review_scores_checkin": "check-in score",
    "review_scores_value": "value score", "minimum_nights": "minimum stay",
    "maximum_nights": "maximum stay", "instant_bookable": "instant booking",
    "accommodates": "sleeps",
}


def _field_label(col: str) -> str:
    return FIELD_LABELS.get(col) or (col.replace("host_", "")
                                     .replace("review_scores_", "")
                                     .replace("_", " "))


def _group_detail(group: str, x: dict, base: dict, gdef: dict[str, list[str]],
                  max_items: int = 3) -> str:
    """What actually differs between this listing and the reference, for this group.

    The attribution number is meaningless without it: "+18 EUR from Capacity" is only
    actionable once you can see it means 6 guests against a typical 3.
    """
    if group == "Location" and x.get("city") in CITY_CENTERS:
        clat, clon = CITY_CENTERS[x["city"]]
        d = haversine_km(float(x["latitude"]), float(x["longitude"]), clat, clon)
        where = f"{d:.1f} km from the centre"
        if x.get("neighbourhood") and x["neighbourhood"] != base.get("neighbourhood"):
            where += f", {x['neighbourhood']}"
        return where
    if group == "Amenities":
        try:
            a, b = len(json.loads(x["amenities"])), len(json.loads(base["amenities"]))
        except Exception:
            return "amenity list differs"
        return f"{a} amenities against a typical {b}"
    bits = []
    for col in gdef[group]:
        if str(x.get(col)) != str(base.get(col)):
            bits.append(f"{_field_label(col)} {_fmt_value(col, x.get(col))} "
                        f"vs {_fmt_value(col, base.get(col))}")
        if len(bits) >= max_items:
            break
    return ", ".join(bits) or "same as the reference listing"


def explain_price(payload: dict, score_batch: Callable[[list[dict]], list[float]] | None = None,
                  groups: dict[str, list[str]] | None = None) -> dict:
    """Attribute the suggested price to attribute groups.

    `score_batch` maps a list of listings to a list of prices. It defaults to the
    price endpoint and is injected by the self test, which is the whole reason the
    Shapley axioms can be checked without AWS.
    """
    gdef = groups or EXPLAIN_GROUPS
    names = list(gdef.keys())
    n = len(names)
    base = baseline_listing(payload["city"])
    x = {k: payload[k] for k in SERVING_COLUMNS}

    if score_batch is None:
        def score_batch(rows):
            return [float(r["suggested_nightly_price"]) for r in invoke_price_batch(rows)]

    t0 = time.perf_counter()
    exact = 2 ** n <= SHAP_MAX_CALLS
    if exact:
        masks = [frozenset(s) for k in range(n + 1) for s in combinations(range(n), k)]
        prices = score_batch([_coalition(x, base, names, gdef, m) for m in masks])
        value = dict(zip(masks, prices))
        phi = shapley_exact(value, n)
        f_x, f_base, calls = value[frozenset(range(n))], value[frozenset()], len(masks)
        method = f"exact Shapley, {calls} coalitions"
    else:
        phi, calls = shapley_sampled(x, base, names, gdef, score_batch, SHAP_PERMUTATIONS)
        f_base = score_batch([_coalition(x, base, names, gdef, [])])[0]
        f_x = score_batch([_coalition(x, base, names, gdef, range(n))])[0]
        method = f"permutation sampling, {SHAP_PERMUTATIONS} permutations, {calls} coalitions"

    contributions = [{"group": nm, "value": round(v, 4),
                      "detail": _group_detail(nm, x, base, gdef)}
                     for nm, v in zip(names, phi)]
    contributions.sort(key=lambda c: abs(c["value"]), reverse=True)

    return {
        "suggested_nightly_price": round(f_x, 2),
        "reference_price": round(f_base, 2),
        "currency": CURRENCY.get(payload["city"], ""),
        "city": payload["city"],
        "contributions": contributions,
        # Efficiency: the parts must add up to the whole. Exact Shapley makes this
        # zero to floating point; it is reported rather than assumed so a batching
        # bug that silently misaligns requests and responses shows up as a number
        # instead of as a plausible-looking waterfall.
        "residual": round(f_x - f_base - sum(phi), 6),
        "method": method,
        "endpoint_calls": calls,
        "elapsed_ms": round((time.perf_counter() - t0) * 1000.0, 1),
        "reference_listing": base,
    }


# ============================================================================
# 7. BOOKING FRICTION
# ============================================================================
# The classifier's target is `instant_bookable`, which is also an INPUT to the price
# model. So "we predict whether you will enable instant booking" is incoherent next
# to a checkbox where the host just told us. Read as a peer rate it is both coherent
# and useful: among comparable listings AT THIS PRICE, how many can be booked without
# waiting for the host to approve.
#
# That is what makes it a friction reading. A guest filtering for instant book never
# sees the rest, so a listing in a segment that mostly offers it and does not is
# carrying friction its neighbours do not. And because price is a feature, the
# reading moves when the price moves -- which is why the host journey scores it twice.
#
# No +/- band here. MAPE gives the regression a defensible tolerance; the honest
# equivalent for a classifier is calibration, and putting a percentage error band
# next to a probability would be a category error.
# ============================================================================

def friction_reading(prob: float | None) -> dict:
    if prob is None:
        return {"level": "unknown", "tone": "unknown",
                "headline": "The endpoint returned no probability"}
    if prob >= 0.75:
        return {"level": "low", "tone": "normal",
                "headline": "Most comparable listings can be booked instantly"}
    if prob >= 0.50:
        return {"level": "moderate", "tone": "normal",
                "headline": "Comparable listings lean towards instant booking"}
    if prob >= 0.25:
        return {"level": "raised", "tone": "low",
                "headline": "Comparable listings lean against instant booking"}
    return {"level": "high", "tone": "low",
            "headline": "Few comparable listings can be booked instantly"}


# ============================================================================
# 8. HOST JOURNEY
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


def _int_flag(flag) -> int:
    return 1 if bool(flag) else 0


def _amenities(selected, extra) -> list[str]:
    items = list(selected or [])
    for token in re.split(r"[,;\n]", extra or ""):
        if token.strip():
            items.append(token.strip())
    return sorted({a.strip().lower() for a in items if a.strip()})


def build_payload(ui: dict) -> dict:
    """The regression contract. Note there is no price in it -- that is the target."""
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
# degrades the prediction silently rather than raising, so none of them are optional:
#
#   price      REQUIRED here, absent there. create_features() builds
#              log_price_city_zscore from it, so the answer is conditional --
#              P(instant book | this listing AT THIS PRICE). That conditioning is
#              exactly what makes the friction reading a lever instead of a label.
#   booleans   ints, not 't'/'f'. Notebook 05 maps them before batch scoring; the
#              't' string would one-hot as an unseen category and quietly vanish.
#   dropped    neighbourhood is not in the classifier's feature schema.
#
# Scales follow Listings.csv, which is what both models trained on: response and
# acceptance rates 0-100, review_scores_rating 0-100, the other six 0-10. The test
# payload in notebook 03 cell 39 uses 0-1 and 0-5 instead, and a room_type that does
# not occur in Listings.csv at all. Do not copy those values as a reference.
BOOKING_REQUIRED = ["city", "amenities", "host_total_listings_count", "price"]

BOOKING_COLUMNS = [
    "city", "latitude", "longitude", "property_type", "room_type",
    "accommodates", "bedrooms", "amenities", "price",
    "minimum_nights", "maximum_nights",
    "host_since", "host_is_superhost", "host_has_profile_pic", "host_identity_verified",
    "host_total_listings_count", "host_response_time", "host_response_rate",
    "host_acceptance_rate",
] + REVIEW_COLS


def booking_payload_from_serving(payload: dict, price: float) -> dict:
    """The classifier contract, derived from the regression contract.

    These were two independent readings of the same form, which meant a fix applied
    to one could quietly miss the other. Deriving one from the other makes that
    impossible -- and it is what lets a listing saved months ago be re-scored from
    its stored payload, with no form to re-type.
    """
    out = dict(payload)
    out["price"] = float(price)
    for col in ("host_is_superhost", "host_has_profile_pic", "host_identity_verified"):
        out[col] = _int_flag(out.get(col) == "t")
    missing = [c for c in BOOKING_REQUIRED if out.get(c) is None]
    if missing:
        raise ValueError(f"the classifier requires: {', '.join(missing)}")
    # The projection drops neighbourhood, which is not in the classifier's schema, and
    # instant_bookable, which is this model's TARGET.
    return {k: out[k] for k in BOOKING_COLUMNS}


def build_booking_payload(ui: dict, price: float) -> dict:
    return booking_payload_from_serving(build_payload(ui), price)


def validate_host(ui: dict, payload: dict) -> list[str]:
    problems = []
    ask = ui.get("asking_price")
    if ask not in (None, "") and float(ask) <= 0:
        problems.append("Asking price must be greater than zero, or left blank.")
    if payload["city"] not in CITY_CENTERS:
        problems.append(f"City must be one of: {', '.join(CITIES)}.")
    if payload["accommodates"] < 1:
        problems.append("Accommodates must be at least 1.")
    if payload["bedrooms"] < 0:
        problems.append("Bedrooms cannot be negative.")
    if payload["minimum_nights"] > payload["maximum_nights"]:
        problems.append("Minimum nights is greater than maximum nights.")
    for k in SERVING_COLUMNS:
        if k not in payload:
            problems.append(f"Missing contract field: {k}.")
    return problems


def price_listing(ui: dict, explain: bool = True,
                  score_batch: Callable[[list[dict]], list[float]] | None = None,
                  booking_fn: Callable[[dict], dict] | None = None) -> dict:
    """The whole host journey, as one JSON-ready dict.

    Order is forced by the models, not by taste: the classifier takes price as a
    feature, so there is no friction reading until there is a price. The two booking
    calls that follow are independent of each other and go out together.

    The two callables are injection points for the self test and for anyone wiring
    this into a host-dashboard service with its own transport.
    """
    payload = build_payload(ui)
    problems = validate_host(ui, payload)
    if problems:
        return {"ok": False, "problems": problems, "request": payload}

    city = payload["city"]
    ccy = CURRENCY.get(city, "")
    asking = ui.get("asking_price")
    asking = float(asking) if asking not in (None, "") else None

    out: dict[str, Any] = {"ok": True, "city": city, "currency": ccy, "request": payload,
                           "asking_price": asking, "mape_pct": city_mape(city)}

    # 1. the price, and if asked, the explanation -- which scores f(x) as one of its
    #    coalitions, so explaining costs no extra call for the headline number.
    if explain:
        try:
            out["explanation"] = explain_price(payload, score_batch=score_batch)
            out["suggested_nightly_price"] = out["explanation"]["suggested_nightly_price"]
        except Exception as e:
            out["explanation_error"] = f"{e.__class__.__name__}: {e}"
    if "suggested_nightly_price" not in out:
        try:
            if score_batch is not None:
                out["suggested_nightly_price"] = round(score_batch([payload])[0], 2)
            else:
                pred, ms = invoke_price(payload)
                out["suggested_nightly_price"] = round(float(pred["suggested_nightly_price"]), 2)
                out["price_ms"] = round(ms, 1)
        except Exception as e:
            out["ok"] = False
            out["price_error"] = f"{e.__class__.__name__}: {e}"
            return out

    suggested = out["suggested_nightly_price"]
    if out["mape_pct"]:
        m = out["mape_pct"] / 100.0
        out["price_range"] = [round(suggested * (1 - m), 2), round(suggested * (1 + m), 2)]

    # 2. booking friction, at the suggested price and at the host's own, together.
    def _book(p):
        if booking_fn is not None:
            return booking_fn(build_booking_payload(ui, p))
        rec, _ = invoke_booking(build_booking_payload(ui, p))
        return rec

    jobs = {"suggested": lambda: _book(suggested)}
    if asking is not None and abs(asking - suggested) > 0.01:
        jobs["asking"] = lambda: _book(asking)

    booked = invoke_parallel(jobs)
    if "suggested_error" in booked:
        out["friction_error"] = explain_error(booked["suggested_error"])
        return out

    prob = booked["suggested"].get("probability_class_1")
    prob = float(prob) if prob is not None else None
    out["friction"] = {"at_price": suggested, "instant_book_prob": prob, **friction_reading(prob)}
    if "asking" in booked and booked["asking"].get("probability_class_1") is not None:
        ap = float(booked["asking"]["probability_class_1"])
        out["friction_at_asking"] = {"at_price": asking, "instant_book_prob": ap,
                                     **friction_reading(ap)}
    return out


# ============================================================================
# 9. GUEST JOURNEY
# ============================================================================
# Item-to-item similarity over the corpus notebook 05 pre-scored: raw Listings.csv
# joined to pred_price (the regression's fair price) and instant_book_prob (the
# classifier's friction reading).
#
# WHAT "ENHANCED" MEANS HERE, AND HOW IT IS CHECKED. A content-based recommender
# using the attribute columns alone is a reasonable baseline and this returns it too,
# on every query, so the enhancement is visible as rank movement rather than asserted
# in a slide. The two model outputs are SIMILARITY DIMENSIONS, not a re-ranking
# objective: a listing is a good alternative if it is worth about the same and books
# about as easily, which is not the same claim as "it is a better deal". The deal gap
# is displayed because guests want it, and it is kept out of the ranking because
# quietly optimising for it would make every result an outlier of the price model
# rather than a substitute for the listing the guest asked about.
#
# WHY THAT IS THE RIGHT SHAPE FOR "OVER BUDGET". Similarity on the MODELLED price
# with a filter on the ASKING price is what finds the same tier of listing for less:
# stay near pred_price, cap price, and the results are things the model rates like
# the original that happen to charge less than it. Ranking on cheapness would just
# return the cheapest listings in the city.
# ============================================================================

# The publication allowlist, applied with usecols at read time. Never reading a
# column is a stronger guarantee than filtering it on the way out: a column that was
# not read cannot leak through a display bug, and this CSV is raw Listings.csv, so
# every HOST_PRIVATE field is sitting in that object. `name` is withheld too -- it is
# free text and hosts put cross streets in it.
CORPUS_COLUMNS = [
    "listing_id", "city", "neighbourhood", "property_type", "room_type",
    "accommodates", "bedrooms", "minimum_nights", "price",
    "instant_bookable", "pred_price", "instant_book_prob",
]
assert not (set(CORPUS_COLUMNS) & HOST_PRIVATE), \
    f"the corpus would read host-private columns: {sorted(set(CORPUS_COLUMNS) & HOST_PRIVATE)}"

# Numeric and categorical columns that need cleaning at load. Not a feature set:
# /recommend filters on accommodates and price and ranks on the two model outputs.
SIM_NUMERIC = ["accommodates", "bedrooms", "minimum_nights"]
SIM_CATEGORICAL = ["room_type", "property_type", "neighbourhood"]

_CORPUS: Any = None
_CORPUS_ERROR: str | None = None
# Bumped on every successful load. The catalogue caches a merge of the corpus and the
# store, so keying that cache on the store alone would hand back a frame built from a
# corpus that has since been reloaded -- rare, silent, and impossible to spot in the
# results.
_CORPUS_VERSION = 0
# Rows the loader dropped as unscorable. Reported because it is the ONE place the pool
# can differ from the notebook's: the notebook never drops anything, so a row with a
# valid price but no instant_book_prob still contributes to its global min and max
# while contributing nothing to any ranking (its ml_score is NaN and sorts last).
# If this is 0, the global scaling here is arithmetically identical to the notebook's.
_CORPUS_DROPPED = 0


def corpus_load(force: bool = False):
    global _CORPUS, _CORPUS_ERROR, _CORPUS_VERSION, _CORPUS_DROPPED
    if _CORPUS is not None and not force:
        return _CORPUS
    try:
        import pandas as pd
        df = pd.read_csv(CORPUS_CSV, usecols=lambda c: c in set(CORPUS_COLUMNS),
                         encoding="utf-8", encoding_errors="replace", low_memory=False)
        for col in ("price", "pred_price", "instant_book_prob", "accommodates",
                    "bedrooms", "minimum_nights"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        before = len(df)
        df = df.dropna(subset=["city", "price", "pred_price", "instant_book_prob"])
        _CORPUS_DROPPED = before - len(df)
        missing = [c for c in ["listing_id", "city", "price", "pred_price",
                               "instant_book_prob"] + SIM_NUMERIC + SIM_CATEGORICAL
                   if c not in df.columns]
        if missing:
            raise ValueError(f"corpus is missing required columns: {', '.join(missing)}")
        df["listing_id"] = df["listing_id"].astype(str)
        # NaN in a similarity dimension poisons the whole distance vector: every
        # candidate scores NaN, the sort returns them in file order, and the ranking
        # looks like a ranking. Fill from the column median instead, once, at load.
        for col in SIM_NUMERIC:
            if df[col].isna().any():
                df[col] = df[col].fillna(df[col].median())
        for col in SIM_CATEGORICAL:
            df[col] = df[col].fillna("").astype(str)
        df["source"] = "corpus"
        _CORPUS, _CORPUS_ERROR = df, None
        _CORPUS_VERSION += 1
    except Exception as e:
        msg = f"{e.__class__.__name__}: {e}"
        # pandas reads s3:// through s3fs, which is not a boto3 dependency and is
        # routinely absent. Saying so beats making the reader work it out.
        if str(CORPUS_CSV).startswith("s3://") and "s3fs" in msg:
            msg += ("  -- run `pip install s3fs`, or point --corpus at a local copy "
                    "of master_recsys_dataset.csv.")
        _CORPUS, _CORPUS_ERROR = None, msg
    return _CORPUS


# The searchable catalogue is the corpus plus whatever hosts have saved. Rebuilt only
# when the store changes: concatenating a 280k-row frame on every keystroke would be
# the slowest thing in the app, and saves are rare.
_CATALOGUE_CACHE: dict[str, Any] = {}


def _store_signature(store: dict) -> str:
    return json.dumps(sorted((k, v.get("updated_at", "")) for k, v in store.items()))


def catalogue(include_saved: bool = True):
    """Corpus rows and saved rows in one frame, tagged by `source`.

    Saved listings are searchable, not merely selectable. A host who saves a listing
    and never appears in anyone's results has not been given a channel to guests,
    which is the entire point of saving it.

    Either half may be missing and the catalogue still works: with no corpus the
    guest journey runs over saved listings alone, which is what makes the app
    demonstrable before the pre-scored CSV is reachable.
    """
    import pandas as pd
    corpus = corpus_load()
    store = store_read() if include_saved else {}
    key = f"{include_saved}|{_CORPUS_VERSION}|{_store_signature(store)}"
    if key in _CATALOGUE_CACHE:
        return _CATALOGUE_CACHE[key]
    rows = store_rows(store)

    frames = []
    if corpus is not None and not corpus.empty:
        frames.append(corpus)
    if rows:
        frames.append(pd.DataFrame(rows))
    if not frames:
        out = None
    elif len(frames) == 1:
        out = frames[0]
    else:
        # Saved ids win a collision with the corpus: the host is the authority on
        # their own listing, and a silent duplicate would let one listing occupy two
        # ranks in the same result set.
        merged = pd.concat(frames, ignore_index=True)
        out = merged.drop_duplicates(subset="listing_id", keep="last").reset_index(drop=True)
    _CATALOGUE_CACHE.clear()               # one entry: the store only moves forward
    _CATALOGUE_CACHE[key] = out
    return out


def catalogue_cities() -> list[str]:
    df = catalogue()
    if df is None or df.empty:
        return CITIES
    return sorted(df["city"].dropna().astype(str).unique())


def corpus_status() -> str:
    df = corpus_load()
    where = CORPUS_CSV.rsplit("/", 1)[-1]
    saved = len(store_rows())
    if df is None:
        return f"corpus unavailable ({where}): {_CORPUS_ERROR}"
    base = f"{len(df):,} listings from {where}"
    if _CORPUS_DROPPED:
        base += f" ({_CORPUS_DROPPED:,} unscorable rows dropped)"
    return f"{base} + {saved} saved by hosts" if saved else base


def catalogue_browse(city: str, min_guests: int = 1, max_price: float | None = None,
                     room_type: str | None = None, limit: int = 20,
                     include_saved: bool = True, seed: int = 0) -> list[dict]:
    """A small sample to pick a seed listing from.

    Sampled, not head(): the CSV arrives grouped by city and roughly by id, so the
    first 20 rows of Paris are 20 near-identical listings from one arrondissement and
    the picker looks broken. Saved listings are exempt from the sampling and always
    shown -- there are few of them and a host who saved one expects to see it.
    """
    df = catalogue(include_saved)
    if df is None:
        return []
    m = df["city"].str.lower() == str(city).lower()
    if min_guests:
        m &= df["accommodates"] >= int(min_guests)
    if max_price:
        m &= df["price"] <= float(max_price)
    if room_type and room_type != "Any":
        m &= df["room_type"] == room_type
    hits = df[m]
    if hits.empty:
        return []
    saved = hits[hits["source"] == "saved"]
    rest = hits[hits["source"] != "saved"]
    if len(rest) > max(0, limit - len(saved)):
        rest = rest.sample(n=max(0, limit - len(saved)), random_state=seed)
    import pandas as pd
    return pd.concat([saved, rest]).sort_values("price").to_dict(orient="records")


def catalogue_get(listing_id: str, include_saved: bool = True) -> dict | None:
    df = catalogue(include_saved)
    if df is None:
        return None
    hit = df[df["listing_id"] == str(listing_id).strip()]
    return None if hit.empty else hit.iloc[0].to_dict()


# ----------------------------------------------------------------------------
# 9B. THE RECOMMENDER TRANSPORT
# ----------------------------------------------------------------------------
# urllib rather than requests: the api backend should not add a dependency the rest
# of the app does not have, and the self test can then stand up a real HTTP server and
# exercise the wire format without installing anything.
# ----------------------------------------------------------------------------

def _jsonable(v):
    """Coerce one pandas/numpy value into something json.dumps will accept.

    Rows come off a DataFrame, so numeric fields can be numpy scalars, and a missing
    bedroom count is NaN -- which json.dumps writes as bare `NaN`, invalid JSON that
    pydantic on the other end rejects with a parse error naming neither the field nor
    the row.
    """
    if v is None:
        return None
    if isinstance(v, (str, bool)):
        return v
    item = getattr(v, "item", None)          # numpy scalar -> python scalar
    if callable(item):
        try:
            v = v.item()
        except Exception:
            return str(v)
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return None
    if isinstance(v, (int, float, str, bool)):
        return v
    return str(v)


def _row_for_wire(row: dict) -> dict:
    return {k: _jsonable(v) for k, v in row.items()}


def _post_json(url: str, payload: dict, timeout: float = 30.0) -> dict:
    import urllib.error
    import urllib.request
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        # The body is where the reason lives. A bare "HTTP Error 422: Unprocessable
        # Entity" names neither the field nor the rule, and urllib discards the body
        # unless it is read here, before the handle closes.
        detail = ""
        try:
            detail = e.read().decode("utf-8")[:600]
        except Exception:
            pass
        raise RuntimeError(f"HTTP {e.code} from {url}: {detail or e.reason}") from None


def _get_json(url: str, params: dict | None = None, timeout: float = 30.0) -> dict:
    import urllib.parse
    import urllib.request
    if params:
        clean = {k: v for k, v in params.items() if v is not None}
        url = f"{url}?{urllib.parse.urlencode(clean)}"
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def _api(path: str) -> str:
    return f"{RECSYS_API_URL.rstrip('/')}/{path.lstrip('/')}"


def _api_unreachable(e: Exception) -> str:
    return (f"No recommender API answered at {RECSYS_API_URL} "
            f"({e.__class__.__name__}: {e}).\n\n"
            f"Start one in the same process with `--with-api`, or separately with "
            f"`--serve-api`, or rank in process with `--recsys-backend inline`.")


def recsys_backend_name() -> str:
    if RECSYS_BACKEND == "api":
        return f"api {_api('/recommend')}"
    if RECSYS_BACKEND == "auto":
        return f"auto: {_api('/recommend')}, then in process"
    return f"in process, {corpus_status()}"


def recsys_cities() -> list[str]:
    """Cities for the pickers, without pulling a 280k-row CSV in api mode."""
    saved = sorted({r["city"] for r in store_rows() if r.get("city")})
    if RECSYS_BACKEND == "api":
        try:
            return sorted(set(_get_json(_api("/health"))["cities"]) | set(saved)) or CITIES
        except Exception:
            return sorted(set(CITIES) | set(saved))
    return sorted(set(catalogue_cities()) | set(saved))


def recsys_recommend(city: str, min_guests: int = 2, max_price: float = 150.0,
                     w_value: float = 0.5, w_convenience: float = 0.5,
                     only_saved: bool = False) -> dict:
    """The ORIGINAL filter-and-rank query -- SearchQuery, unchanged -- over the wire.

    city, min_guests, max_price, w_value, w_convenience. Exactly the five fields the
    notebook client posts, so what the guest tab sends and what API_Test_Client.ipynb
    sent are the same request.
    """
    q = {"city": city, "min_guests": int(min_guests), "max_price": float(max_price),
         "w_value": float(w_value), "w_convenience": float(w_convenience),
         "only_saved": bool(only_saved)}
    api_error = None
    if RECSYS_BACKEND in ("api", "auto"):
        try:
            out = _post_json(_api("/recommend"), q)
            out["backend"] = "api"
            return out
        except Exception as e:
            if RECSYS_BACKEND == "api":
                return {"results": [], "count": 0, "backend": "api",
                        "message": _api_unreachable(e)}
            api_error = f"{e.__class__.__name__}: {e}"
    out = api_recommend(**q)
    out["backend"] = "inline"
    if api_error:
        out["backend_note"] = (f"Ranked in this process. Nothing is serving the API on "
                               f"{RECSYS_API_URL} -- start the app with `--with-api` to "
                               f"route this over HTTP instead.")
    return out


# ----------------------------------------------------------------------------
# 9C. THE RECOMMENDER API -- SERVER SIDE
# ----------------------------------------------------------------------------
# This was Test_API_Gradio.ipynb: a second notebook that had to be running, holding a
# second copy of the 280k-row corpus in memory, with its own copy of the ranking code.
# Folding it in here removes all three problems -- one process, one corpus, one
# ranker -- and removes the failure mode where the API and the app quietly answer
# differently because someone edited one of them.
#
# The five handlers below are plain functions returning plain dicts. FastAPI and the
# stdlib server are both thin adapters over them, which is the same discipline the
# client side follows: the transport is a detail, the logic has one home.
#
# NO RECURSION. These handlers call similar_listings() and catalogue() directly, never
# recsys_similar(). With --with-api the app serves this API and consumes it over HTTP
# in the same process, so a handler that went back through the client dispatcher would
# call itself until the socket ran out.
# ----------------------------------------------------------------------------

_GLOBAL_SPAN: dict[int, tuple] = {}


def _value_score(frame, candidates, scoring_pool=None):
    """The value term: how far under the model's estimate a listing sits, scaled to [0,1].

    deal_value = pred_price - price. Positive means the price model thinks the listing
    is worth more than it charges. Scaling to [0,1] is what makes it addable to
    instant_book_prob, which is already a probability -- and WHERE the min and max are
    taken from decides whether the addition means anything.

    `scoring_pool` is the set a candidate-set fit is taken over. It defaults to the
    candidates themselves and is passed explicitly by only_saved, which is a
    VISIBILITY filter: narrowing what is displayed must not change what anything
    scores. Without this, restricting to a host's own listings would refit the scaler
    over that handful of rows and show them a rank and a score that exist nowhere
    else -- and the bug would only appear under candidate-set scaling, i.e. only after
    the default changed.
    """
    import numpy as np
    pool = candidates if scoring_pool is None else scoring_pool
    deal = (candidates["pred_price"].to_numpy(dtype=float)
            - candidates["price"].to_numpy(dtype=float))
    if RECOMMEND_SCALING == "global":
        key = _CORPUS_VERSION
        if key not in _GLOBAL_SPAN:
            allv = (frame["pred_price"].to_numpy(dtype=float)
                    - frame["price"].to_numpy(dtype=float))
            _GLOBAL_SPAN.clear()
            _GLOBAL_SPAN[key] = (float(np.nanmin(allv)), float(np.nanmax(allv)))
        lo, hi = _GLOBAL_SPAN[key]
    else:
        span = (pool["pred_price"].to_numpy(dtype=float)
                - pool["price"].to_numpy(dtype=float))
        lo, hi = float(np.nanmin(span)), float(np.nanmax(span))
    if hi <= lo:                      # every candidate identical: no signal to scale
        return np.full(len(deal), 0.5)
    return np.clip((deal - lo) / (hi - lo), 0.0, 1.0)


def _api_pool():
    """What /recommend ranks: the pre-scored corpus plus everything hosts have saved.

    This is the host-to-guest link now that there is no seed listing to hand across:
    a host prices a listing, saves it, and it enters the pool this endpoint ranks.
    """
    frame = catalogue(include_saved=True)
    if frame is None:
        raise RuntimeError("Nothing to search: no corpus loaded and no saved listings.")
    if "source" not in frame.columns:
        frame = frame.assign(source="corpus")
    return frame


def api_health() -> dict:
    frame = catalogue(include_saved=True)
    saved = sum(1 for _ in store_rows())
    return {
        "ok": frame is not None,
        "listings": 0 if frame is None else int(len(frame)),
        "saved_listings": saved,
        "cities": [] if frame is None else sorted(frame["city"].dropna().astype(str).unique()),
        "ranker": "host_guest_app.api_recommend",
        "value_scaling": RECOMMEND_SCALING,
        "corpus": corpus_status(),
        "unscorable_rows_dropped": _CORPUS_DROPPED,
        "endpoints": ["/recommend", "/health"],
    }


def api_recommend(city: str, min_guests: int = 2, max_price: float = 150.0,
                  w_value: float = 0.5, w_convenience: float = 0.5,
                  only_saved: bool = False) -> dict:
    """The original filter-and-rank. Same contract, same maths, same response shape."""
    frame = _api_pool()
    candidates = frame[(frame["city"].str.lower() == str(city).lower())
                       & (frame["accommodates"] >= int(min_guests))
                       & (frame["price"] <= float(max_price))].copy()
    # Held before only_saved narrows anything: this is the set the value term scales
    # over, so a host's listing scores the same whether or not the filter is on.
    scoring_pool = candidates
    if only_saved:
        candidates = candidates[candidates["source"] == "saved"]
    if candidates.empty:
        where = "saved in this app" if only_saved else f"in {city}"
        return {"results": [], "count": 0,
                "message": f"No listings match your criteria {where}."}
    candidates["scaled_value_score"] = _value_score(frame, candidates, scoring_pool)
    candidates["ml_score"] = (float(w_value) * candidates["scaled_value_score"]
                              + float(w_convenience) * candidates["instant_book_prob"])
    # Stable on both sorts, so the comparison below is about the value term and not
    # about how quicksort happened to break a tie.
    ranked = candidates.sort_values(by="ml_score", ascending=False, kind="stable").head(10)
    cols = ["listing_id", "city", "accommodates", "price", "pred_price",
            "instant_book_prob", "ml_score"]
    out = {"results": [_row_for_wire(r) for r in ranked[cols].to_dict(orient="records")],
           "count": int(len(ranked))}

    # Report the flattening instead of leaving it to be found in the numbers. The test
    # is not a threshold on the spread -- that would be an arbitrary line -- but the
    # claim itself: did the value term change the order at all? If ranking by
    # instant_book_prob alone produces the same list, w_value bought nothing.
    if float(w_value) > 0:
        prob_only = candidates.sort_values(
            by="instant_book_prob", ascending=False, kind="stable").head(10)
        spread = float(ranked["scaled_value_score"].max()
                       - ranked["scaled_value_score"].min())
        if ranked["listing_id"].tolist() == prob_only["listing_id"].tolist():
            out["warning"] = (
                f"w_value={w_value} changed nothing: ranking on instant_book_prob alone "
                f"returns this same order. The value term spans only {spread:.5f} across "
                f"these results because RECOMMEND_SCALING=global fits its min and max "
                f"over every row in every city before filtering, so one city's candidates "
                f"land inside a sliver of that range. RECOMMEND_SCALING=candidates fits "
                f"it over the listings actually being ranked.")
    out["value_scaling"] = RECOMMEND_SCALING
    return out


def _fastapi_app():
    """FastAPI adapter. Gives /docs, which is worth having in a demo.

    The models are defined here rather than at module scope so that pydantic stays
    an optional dependency -- and are then published into module globals, which is
    not decoration. Under `from __future__ import annotations` every annotation is a
    string, and FastAPI resolves a route's hints with the function's __globals__,
    i.e. this module. A model that exists only as a local of this function therefore
    does not resolve, and FastAPI silently falls back to treating the body parameter
    as a QUERY parameter: every POST comes back 422 with a complaint about a missing
    field named `q`. Publishing them makes the hint resolvable.
    """
    from fastapi import FastAPI
    from pydantic import BaseModel

    class SearchQuery(BaseModel):
        city: str
        min_guests: int = 2
        max_price: float = 150.0
        w_value: float = 0.5
        w_convenience: float = 0.5
        # Added since the original contract, with a default that leaves a five-field
        # request behaving exactly as it did. Restricts the ranking to listings hosts
        # saved through this app, which is the only reliable way to see one: a new
        # listing rarely cracks the top 10 of a whole city on merit alone.
        only_saved: bool = False

    # Published into module globals deliberately. Under `from __future__ import
    # annotations` every annotation is a string, and FastAPI resolves a route's hints
    # against the function's __globals__ -- this module. A model that exists only as a
    # local of this function does not resolve, and FastAPI then silently treats the
    # body parameter as a QUERY parameter: every POST comes back 422 complaining that
    # a field named `q` is missing. The self test asserts the route kept a body field.
    globals()["SearchQuery"] = SearchQuery

    api = FastAPI(title="ITI113 Airbnb Recommendation API",
                  description="Filter and rank listings on price value and instant booking.")

    @api.post("/recommend")
    def _recommend(q: SearchQuery):
        return api_recommend(q.city, q.min_guests, q.max_price, q.w_value,
                             q.w_convenience, q.only_saved)

    @api.get("/health")
    def _health():
        return api_health()

    return api


def _stdlib_server(host: str = API_HOST, port: int = API_PORT):
    """Same five routes with no third-party dependency.

    Not a toy: --serve-api has to work on a machine where fastapi is not installed,
    and the self test needs a real socket without installing anything to get one.
    """
    import http.server
    import urllib.parse

    class Handler(http.server.BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):
            pass

        def _send(self, obj, code=200):
            body = json.dumps(obj).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            try:
                q = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
                route = urllib.parse.urlparse(self.path).path
                if route == "/recommend":
                    self._send(api_recommend(**q))
                else:
                    self._send({"detail": f"No route {route}"}, 404)
            except KeyError as e:
                self._send({"detail": str(e)}, 404)
            except Exception as e:
                self._send({"detail": f"{e.__class__.__name__}: {e}"}, 400)

        def do_GET(self):
            try:
                parts = urllib.parse.urlparse(self.path)
                if parts.path == "/health":
                    self._send(api_health())
                else:
                    self._send({"detail": f"No route {parts.path}"}, 404)
            except KeyError as e:
                self._send({"detail": str(e)}, 404)
            except Exception as e:
                self._send({"detail": f"{e.__class__.__name__}: {e}"}, 400)

    return http.server.ThreadingHTTPServer((host, port), Handler)


def _wait_for_api(port: int, timeout: float = 30.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            _get_json(f"http://127.0.0.1:{port}/health", timeout=2)
            return True
        except Exception:
            time.sleep(0.25)
    return False


def serve_api(host: str = API_HOST, port: int = API_PORT, background: bool = False) -> str:
    """Start the recommender API. Returns a description of what is serving it."""
    try:
        import uvicorn
        server = uvicorn.Server(uvicorn.Config(_fastapi_app(), host=host, port=port,
                                               log_level="warning"))
        runner = server.run
        name = f"fastapi/uvicorn on http://{host}:{port} (docs at /docs)"
    except ImportError:
        srv = _stdlib_server(host, port)
        runner = srv.serve_forever
        name = (f"http.server on http://{host}:{port} "
                f"(pip install fastapi uvicorn for /docs)")
    if background:
        import threading
        threading.Thread(target=runner, daemon=True).start()
        if not _wait_for_api(port):
            raise RuntimeError(f"the API did not come up on port {port}")
        return name
    print(f"Recommender API: {name}")
    for route in api_health()["endpoints"]:
        print(f"    {route}")
    print("Serving. Ctrl-C to stop.")
    runner()
    return name


# ============================================================================
# 10. UI
# ============================================================================
# gradio is imported on first use, not at module scope, so `--self-test`,
# `--similar` and `--price-json` run on a machine that has neither gradio nor boto3
# installed. The self test claims "no UI, no AWS"; a module-level import made that
# claim false at import time, before any of it could be checked.
# ============================================================================

from html import escape as esc

gr: Any = None

# Populated by the bodies, drained by the builders: some panels need one render at
# page load and the demo object that fires it does not exist until the Blocks closes.
_LOAD_HOOKS: list = []


def _gradio():
    global gr
    if gr is None:
        import gradio
        gr = gradio
        _bind_request_annotations()
    return gr


# Injected as a <style> component rather than through Blocks(css=...) or
# launch(css=...): Gradio 4/5 take css on the constructor, Gradio 6 moved it to
# launch(). A <style> component works on all three.
CSS = """
.card {padding: 20px 22px; border: 1px solid var(--border-color-primary);
       border-radius: 10px; background: var(--background-fill-secondary); margin-bottom: 12px;}
.card .eyebrow {font-size: .78rem; letter-spacing: .09em; text-transform: uppercase;
       opacity: .65; margin-bottom: 6px;}
.card .big {font-size: 2.4rem; font-weight: 650; line-height: 1.15;
       font-variant-numeric: tabular-nums;}
.card .sub {font-size: 1.05rem; font-weight: 500; opacity: .75;}
.card .band {margin-top: 12px; padding: 10px 12px; border-radius: 8px;
       background: var(--background-fill-primary); font-size: .9rem; line-height: 1.5;}
.card .meta {margin-top: 12px; font-size: .78rem; opacity: .6;}
.tone-normal {border-left: 4px solid #16a34a; padding-left: 14px;}
.tone-high {border-left: 4px solid #ea580c; padding-left: 14px;}
.tone-low {border-left: 4px solid #2563eb; padding-left: 14px;}
.tone-unknown {border-left: 4px solid var(--border-color-primary); padding-left: 14px;}

/* Driver breakdown. The centre line is the reference listing; bars run right when
   an attribute adds to the price and left when it takes away. */
.wf {margin-top: 14px; display: flex; flex-direction: column; gap: 9px;}
.wf-row {display: grid; grid-template-columns: minmax(96px, 1.1fr) 2.2fr minmax(74px, auto);
       gap: 12px; align-items: center;}
.wf-name {font-size: .9rem; font-weight: 600; line-height: 1.25;}
.wf-detail {font-size: .74rem; opacity: .6; line-height: 1.3; margin-top: 2px;}
.wf-track {position: relative; height: 20px; border-radius: 5px;
       background: var(--background-fill-primary);}
.wf-track::before {content: ""; position: absolute; left: 50%; top: 0; bottom: 0;
       width: 1px; background: var(--border-color-primary);}
.wf-bar {position: absolute; top: 4px; height: 12px; border-radius: 3px;}
.wf-bar.pos {left: 50%; background: #ea580c;}
.wf-bar.neg {right: 50%; background: #2563eb;}
.wf-val {text-align: right; font-size: .9rem; font-variant-numeric: tabular-nums;
       font-weight: 600;}
.wf-val.pos {color: #ea580c;} .wf-val.neg {color: #2563eb;}
.wf-foot {margin-top: 14px; font-size: .78rem; opacity: .6; line-height: 1.5;}

/* Friction meter: the probability, with the reading it maps to. */
.gauge {margin-top: 10px; height: 10px; border-radius: 5px;
       background: var(--background-fill-primary); position: relative; overflow: hidden;}
.gauge > span {position: absolute; left: 0; top: 0; bottom: 0; border-radius: 5px;}
.pill {display: inline-block; padding: 2px 9px; border-radius: 999px; font-size: .74rem;
       letter-spacing: .04em; text-transform: uppercase; border: 1px solid var(--border-color-primary);}
"""


def inject_css() -> None:
    gr.HTML(f"<style>{CSS}</style>")


def _empty(eyebrow: str, note: str = "") -> str:
    return (f"<div class='card'><div class='eyebrow'>{esc(eyebrow)}</div>"
            f"<div class='meta'>{esc(note)}</div></div>")


def _problems_card(problems: list[str]) -> str:
    items = "".join(f"<li>{esc(p)}</li>" for p in problems)
    return f"<div class='card'><div class='eyebrow'>Fix these first</div><ul>{items}</ul></div>"


def host_preset() -> dict:
    d = review_defaults()
    vocab = set(amenity_choices())
    amen = ["wifi", "kitchen", "heating", "washer", "essentials", "hot water"]
    return {
        "listing_id": "PAR-001", "asking_price": None,
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


def listing_form(h: dict, preset: dict, rdef: dict) -> dict:
    """The 27-field contract as a form, populating `h` in place."""
    with gr.Row():
        h["listing_id"] = gr.Textbox(
            label="Listing ID", value=preset["listing_id"],
            info="3-32 characters. Only needed to save; pricing works without it.")
        h["asking_price"] = gr.Number(
            label="Your asking price (optional)", value=preset["asking_price"], minimum=0,
            info="Blank publishes at the suggested price. Fill it in to compare booking "
                 "friction at your price against the suggestion.")
    with gr.Tab("Listing"):
        with gr.Row():
            h["city"] = gr.Dropdown(label="City", choices=CITIES, value=preset["city"])
            h["neighbourhood"] = gr.Dropdown(
                label="Neighbourhood", choices=known_neighbourhoods(preset["city"]) or [],
                value=preset["neighbourhood"], allow_custom_value=True)
        with gr.Row():
            h["latitude"] = gr.Number(label="Latitude", value=preset["latitude"])
            h["longitude"] = gr.Number(label="Longitude", value=preset["longitude"])
        gr.Markdown("<small>Coordinates stay on this side. The model uses distance from "
                    "the city centre and guests are shown that, rounded to 100 m.</small>")
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
            label="Amenities", choices=amenity_choices(), value=preset["amenities_selected"])
        h["amenities_extra"] = gr.Textbox(label="Other amenities", lines=2,
                                          value=preset["amenities_extra"])
    with gr.Tab("Booking rules"):
        with gr.Row():
            h["minimum_nights"] = gr.Number(label="Minimum nights", precision=0, minimum=1,
                                            value=preset["minimum_nights"])
            h["maximum_nights"] = gr.Number(label="Maximum nights", precision=0, minimum=1,
                                            value=preset["maximum_nights"])
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
            h["host_identity_verified"] = gr.Checkbox(label="Identity verified",
                                                      value=preset["host_identity_verified"])
        h["host_response_time"] = gr.Dropdown(label="Response time",
                                              choices=response_time_choices(),
                                              value=preset["host_response_time"])
        h["host_response_rate"] = gr.Slider(label="Response rate (%)", minimum=0, maximum=100,
                                            step=1, value=preset["host_response_rate"])
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


# ---------------------------------------------------------------- host renderers

def _price_card(resp: dict) -> str:
    price, ccy = resp["suggested_nightly_price"], resp["currency"]
    band = ""
    if resp.get("price_range"):
        lo, hi = resp["price_range"]
        band = (f"<div class='band'>The model's typical error in {esc(resp['city'])} is "
                f"<strong>&plusmn;{resp['mape_pct']:.1f}%</strong>, so read this as roughly "
                f"<strong>{lo:,.0f}&ndash;{hi:,.0f} {esc(ccy)}</strong> a night.</div>")
    ask = ""
    if resp.get("asking_price"):
        gap = (resp["asking_price"] - price) / price * 100.0
        ask = (f"<div class='band'>You are asking <strong>{resp['asking_price']:,.0f} "
               f"{esc(ccy)}</strong>, which is <strong>{gap:+.0f}%</strong> against the "
               f"suggestion.</div>")
    return (f"<div class='card'><div class='eyebrow'>Suggested nightly price</div>"
            f"<div class='big'>{price:,.2f} <span class='sub'>{esc(ccy)}</span></div>"
            f"{band}{ask}<div class='meta'>{esc(PRICE_ENDPOINT)}</div></div>")


def _friction_card(resp: dict) -> str:
    if resp.get("friction_error"):
        return (f"<div class='card tone-unknown'><div class='eyebrow'>Booking friction</div>"
                f"<div class='band'>{resp['friction_error']}</div></div>")
    f = resp.get("friction")
    if not f or f.get("instant_book_prob") is None:
        return _empty("Booking friction", "No probability came back for this listing.")
    prob, ccy = f["instant_book_prob"], resp["currency"]
    colour = {"normal": "#16a34a", "low": "#2563eb"}.get(f["tone"], "#94a3b8")
    alt = ""
    a = resp.get("friction_at_asking")
    if a and a.get("instant_book_prob") is not None:
        move = "rises to" if a["instant_book_prob"] > prob else "falls to"
        alt = (f"<div class='band'>At your asking price of "
               f"<strong>{a['at_price']:,.0f} {esc(ccy)}</strong> the peer rate {move} "
               f"<strong>{a['instant_book_prob'] * 100:.0f}%</strong> "
               f"&mdash; {esc(a['headline'].lower())}.</div>")
    return (f"<div class='card tone-{f['tone']}'>"
            f"<div class='eyebrow'>Booking friction at {f['at_price']:,.0f} {esc(ccy)}</div>"
            f"<div class='big'>{prob * 100:.0f}<span class='sub'>%</span> "
            f"<span class='pill'>{esc(f['level'])} friction</span></div>"
            f"<div class='gauge'><span style='width:{prob * 100:.0f}%;background:{colour}'></span></div>"
            f"<div class='band'><strong>{esc(f['headline'])}.</strong> This is the share of "
            f"comparable listings at this price that a guest can book without waiting for "
            f"approval &mdash; not a prediction of what you should choose.</div>"
            f"{alt}<div class='meta'>{esc(BOOKING_ENDPOINT)} &middot; price is a feature of "
            f"this model, so the reading moves with it</div></div>")


def _explain_card(ex: dict) -> str:
    ccy = ex["currency"]
    rows, widest = [], max((abs(c["value"]) for c in ex["contributions"]), default=1.0) or 1.0
    for c in ex["contributions"]:
        v = c["value"]
        cls = "pos" if v >= 0 else "neg"
        width = min(50.0, abs(v) / widest * 50.0)
        rows.append(
            f"<div class='wf-row'>"
            f"<div><div class='wf-name'>{esc(c['group'])}</div>"
            f"<div class='wf-detail'>{esc(c['detail'])}</div></div>"
            f"<div class='wf-track'><div class='wf-bar {cls}' style='width:{width:.1f}%'></div></div>"
            f"<div class='wf-val {cls}'>{v:+,.1f}</div></div>")
    resid = ""
    if abs(ex["residual"]) > 0.01:
        resid = (f"<br>Unattributed remainder: {ex['residual']:+,.2f} {esc(ccy)}. "
                 f"Exact Shapley leaves none, so anything here means the batch of "
                 f"coalitions and the batch of prices came back misaligned.")
    return (
        f"<div class='card'><div class='eyebrow'>What drives this price</div>"
        f"<div class='band'>A typical {esc(ex['city'])} listing prices at "
        f"<strong>{ex['reference_price']:,.0f} {esc(ccy)}</strong>. Yours prices at "
        f"<strong>{ex['suggested_nightly_price']:,.0f}</strong>. The "
        f"<strong>{ex['suggested_nightly_price'] - ex['reference_price']:+,.0f} {esc(ccy)}</strong> "
        f"between them is split below, by how much each group of attributes moves the "
        f"model on its own and in combination with the others.</div>"
        f"<div class='wf'>{''.join(rows)}</div>"
        f"<div class='wf-foot'>{esc(ex['method'])} &middot; {ex['endpoint_calls']} endpoint "
        f"calls &middot; {ex['elapsed_ms']:,.0f} ms. City is held fixed: the reference is a "
        f"typical listing in {esc(ex['city'])}, not an average across all ten, so nothing "
        f"here is measured against another currency.{resid}</div></div>")


# ---------------------------------------------------------------- host handlers

def host_price(explain, *values):
    """Price and friction first. The explanation follows in a chained event.

    a single handler would make the host wait for the breakdown before seeing the
    number they asked for. The chained pass re-scores f(x) as one of its coalitions,
    so the duplication is one call in 129.
    """
    ui = dict(zip(HOST_FIELDS, values))
    try:
        resp = price_listing(ui, explain=False)
    except (TypeError, ValueError) as e:
        return (f"<div class='card'>Cannot build the request: {esc(str(e))}</div>", "", "{}",
                _empty("Explanation", "Fix the request first."), None)
    if not resp.get("ok") and resp.get("problems"):
        return (_problems_card(resp["problems"]), "", json.dumps(resp["request"], indent=2),
                _empty("Explanation", "Fix the request first."), None)
    if resp.get("price_error"):
        return (f"<div class='card'>{resp['price_error']}</div>", "",
                json.dumps(resp["request"], indent=2),
                _empty("Explanation", "The price endpoint did not answer."), None)
    pending = (_empty("Working out the drivers", f"128 coalitions against {PRICE_ENDPOINT}.")
               if explain else
               _empty("Explanation", "Turn on 'Explain the price' to see the driver breakdown."))
    return (_price_card(resp), _friction_card(resp),
            json.dumps(resp["request"], indent=2), pending, resp)


def host_explain(explain, resp, *values):
    if not explain or not resp or not resp.get("ok"):
        # host_price already put the reason on screen -- the validation problems, the
        # endpoint error, or the note that explaining is switched off. Leave it there.
        return gr.update(), gr.update()
    ui = dict(zip(HOST_FIELDS, values))
    try:
        ex = explain_price(build_payload(ui))
    except Exception as e:
        return f"<div class='card'>{explain_error(e)}</div>", "{}"
    return _explain_card(ex), json.dumps(
        {k: v for k, v in ex.items() if k != "reference_listing"}, indent=2)


# gradio injects the authenticated user by looking for a gr.Request TYPE HINT. Under
# `from __future__ import annotations` every annotation is a string, and gradio
# resolves them with typing.get_type_hints while catching only NameError and
# TypeError -- so `request: gr.Request` against a lazily-imported gradio either
# raises AttributeError or resolves to nothing and silently stops injecting, which is
# how the previous app recorded every save as anonymous while deletes ran as the
# logged-in user. Binding the class object directly cannot be evaluated wrongly
# because it is never evaluated at all. `request` MUST also come first: gradio builds
# its positional list by walking the signature and stopping at the first
# non-positional parameter, so a request sitting after *values is never reached.
def host_save_event(request, listing_id, resp, *values):
    return host_save(current_user(request), listing_id, resp, *values)


def host_delete_event(request, listing_id):
    return host_delete(current_user(request), listing_id)


def host_my_listings_event(request):
    return host_my_listings(current_user(request))


def _bind_request_annotations() -> None:
    for fn in (host_save_event, host_delete_event, host_my_listings_event):
        fn.__annotations__["request"] = gr.Request


def host_reference(city):
    try:
        return json.dumps(baseline_listing(city), indent=2)
    except Exception as e:
        return f"// reference unavailable: {e}"


def host_save(username, listing_id, resp, *values):
    """Save the priced listing so guests can find it.

    Prices first if the host never pressed the price button, rather than refusing:
    the listing they typed is right there, and "price it, then save it" is a step the
    app can take on its own.
    """
    ui = dict(zip(HOST_FIELDS, values))
    lid = (listing_id or "").strip()
    if resp is None or not resp.get("ok"):
        try:
            resp = price_listing(ui, explain=False)
        except (TypeError, ValueError) as e:
            return f"Cannot build the request: {e}", host_my_listings(username)
        if not resp.get("ok"):
            problems = resp.get("problems") or [resp.get("price_error", "the price endpoint failed")]
            return "Fix these first:\n" + "\n".join(f"- {p}" for p in problems), \
                host_my_listings(username)
    try:
        rec = save_listing(lid, resp, username)
    except PermissionError as e:
        return str(e), host_my_listings(username)
    except ValueError as e:
        return str(e), host_my_listings(username)
    except Exception as e:
        return f"Save failed.\n\n{explain_error(e)}", host_my_listings(username)

    who = f"as `{username}`" if username else "unowned (this app has no auth)"
    if rec["instant_book_prob"] is None:
        tail = ("It is **not yet visible to guests**: the booking endpoint did not answer, "
                "and without an instant-booking score the listing has no position in the "
                "similarity space. Price it again once the endpoint is warm, then save.")
    else:
        tail = (f"Guests can now find it: it is searchable as a result and selectable as a "
                f"starting point, at {rec['asking_price']:,.0f} {rec['currency']} with a "
                f"modelled value of {rec['suggested_price']:,.0f} and "
                f"{rec['instant_book_prob']:.0%} instant booking.")
    return (f"Saved **{lid}** {who} at {rec['updated_at'][:16]}.\n\n{tail}\n\n"
            f"Store: `{store_backend_name()}`"), host_my_listings(username)


def host_delete(username, listing_id):
    out = delete_listing((listing_id or "").strip(), username)
    return out, host_my_listings(username)


def _money(v) -> str:
    return "-" if v is None else f"{float(v):,.0f}"


def host_my_listings(username=None):
    """Every saved listing with its owner, not just yours.

    Hiding records you cannot edit is how one becomes unreachable from the UI with no
    explanation of where it went.
    """
    store = store_read()
    if not store:
        return "No listings saved yet."
    rows = ["| listing | owner | asking | modelled | instant book | guests see it | updated |",
            "|---|---|---:|---:|---:|---|---|"]
    unscored = 0
    for lid, rec in sorted(store.items()):
        owner = rec.get("owner")
        who = ("_unowned_" if owner is None else
               (f"`{owner}` (you)" if owner == username else f"`{owner}` &mdash; read only"))
        prob, suggested = rec.get("instant_book_prob"), rec.get("suggested_price")
        # Every value here is read defensively. Records written by an earlier version
        # of this app are upgraded on read but cannot have fields invented for them,
        # so a missing number has to render as a dash and explain itself rather than
        # take the page down -- which is exactly what it did.
        if suggested is None:
            seen = "no &mdash; never priced"
            unscored += 1
        elif prob is None:
            seen = "no &mdash; no booking score yet"
            unscored += 1
        else:
            stale = record_is_stale(rec)
            seen = f"yes &mdash; {stale}" if stale else "yes"
        rows.append(f"| `{lid}` | {who} | {_money(rec.get('asking_price'))} | "
                    f"{_money(suggested)} | "
                    f"{'-' if prob is None else f'{prob:.0%}'} | {seen} | "
                    f"{(rec.get('updated_at') or '')[:16] or '-'} |")
    signed = (f"Signed in as `{username}`." if username
              else "Not signed in, so saves are unowned and anyone can edit them.")
    tail = ""
    if unscored:
        tail = (f"\n\n{unscored} listing(s) have no instant-booking score, so guests "
                f"cannot find them. Most were saved by an earlier version of this app, "
                f"which stored the price only. Press **Price this listing** and "
                f"**Save** again, or fill them all in at once with "
                f"`python host_guest_app.py --rescore-store`.")
    return signed + "\n\n" + "\n".join(rows) + tail


CORPUS_FILL = ["asking_price", "city", "neighbourhood", "property_type", "room_type",
               "accommodates", "bedrooms", "minimum_nights", "instant_bookable",
               "latitude", "longitude"]


def host_corpus_options(city, room_type):
    rows = catalogue_browse(city, min_guests=1, room_type=room_type, limit=25)
    if not rows:
        return gr.update(choices=[], value=None), "Nothing in the corpus matched. Try another city."
    labels = [f"{r['listing_id']} - {r['room_type']}, sleeps {int(r['accommodates'])}, "
              f"{float(r['price']):,.0f} {CURRENCY.get(city, '')}" for r in rows]
    return (gr.update(choices=labels, value=labels[0]),
            f"{len(labels)} listings sampled from {city}. Pick one and load it.")


def host_apply_corpus(label):
    """Fill the form from a corpus listing, and say what could not be filled.

    The corpus is read through the publication allowlist, so it has no coordinates,
    no host profile and no review scores -- the same withholding that protects hosts
    on the guest side applies to us here. Those fields keep their current values and
    the status line says so, because a price built on nine real fields and eighteen
    defaults should not look like a price built on twenty-seven real ones.
    """
    lid = (label or "").split(" - ", 1)[0].strip()
    row = catalogue_get(lid)
    if row is None:
        return [gr.update() for _ in CORPUS_FILL] + [f"No listing `{lid}` in the corpus."]
    city = row["city"]
    lat, lon = CITY_CENTERS.get(city, (0.0, 0.0))
    values = {
        "asking_price": round(float(row["price"]), 2),
        "city": city,
        "neighbourhood": row.get("neighbourhood") or None,
        "property_type": row.get("property_type") or "Other",
        "room_type": row.get("room_type"),
        "accommodates": int(row["accommodates"]),
        "bedrooms": int(row["bedrooms"]) if not (isinstance(row.get("bedrooms"), float)
                                                 and math.isnan(row["bedrooms"])) else 1,
        "minimum_nights": int(row["minimum_nights"]) if not (
            isinstance(row.get("minimum_nights"), float) and math.isnan(row["minimum_nights"])) else 1,
        "instant_bookable": str(row.get("instant_bookable")).lower() in ("1", "1.0", "t", "true"),
        "latitude": lat, "longitude": lon,
    }
    note = (f"Loaded `{lid}`: 9 attributes from the corpus. Coordinates fell back to the "
            f"{esc(city)} centre and the host profile and review scores kept their current "
            f"values, because the corpus publishes none of those. Set them for a price you "
            f"can act on.")
    return [gr.update(value=values[k]) for k in CORPUS_FILL] + [note]


def host_body(host_state=None) -> None:
    """The pricing assistant, rendered into whatever Blocks context is open."""
    rdef, preset = review_defaults(), host_preset()
    gr.Markdown(f"""
        # Price a listing
        `{PRICE_ENDPOINT}` &middot; `{BOOKING_ENDPOINT}`

        Describe the listing and get three answers: what it should cost, which of its
        attributes carry that price, and how easily a listing like it books at that rate.
    """)
    h: dict[str, Any] = {}
    with gr.Row():
        with gr.Column(scale=3):
            with gr.Accordion("Start from a real listing (optional)", open=False):
                with gr.Row():
                    c_city = gr.Dropdown(label="City", choices=catalogue_cities(), value="Paris",
                                         scale=2)
                    c_room = gr.Dropdown(label="Room type", choices=["Any"] + room_type_choices(),
                                         value="Any", scale=2)
                    c_find = gr.Button("Show listings", scale=1)
                c_pick = gr.Dropdown(label="Corpus listing", choices=[], value=None)
                c_load = gr.Button("Load into the form")
                c_note = gr.Markdown()
            listing_form(h, preset, rdef)

        with gr.Column(scale=2):
            explain_cb = gr.Checkbox(
                label="Explain the price", value=True,
                info="Adds ~128 endpoint calls for an exact driver breakdown.")
            price_btn = gr.Button("Price this listing", variant="primary", size="lg")
            price_out = gr.HTML(_empty(
                "No price yet",
                "The first call after an idle period is slow: the endpoints scale to zero."))
            friction_out = gr.HTML(_empty(
                "Booking friction",
                "Scored at the suggested price once there is one."))
            explain_out = gr.HTML(_empty(
                "Explanation", "Price the listing to see what drives the number."))
            resp_state = gr.State(None)
            save_btn = gr.Button("Save so guests can find it", variant="secondary")
            save_status = gr.Markdown()
            with gr.Accordion("Saved listings", open=False):
                mine = gr.Markdown()
                with gr.Row():
                    gr.Button("Refresh", size="sm").click(host_my_listings_event, None, mine)
                    del_id = gr.Textbox(label="", placeholder="listing id to delete",
                                        show_label=False)
                    del_btn = gr.Button("Delete", variant="stop", size="sm")
            with gr.Accordion("The reference listing this is compared against", open=False):
                ref_view = gr.Code(language="json", label="Typical listing in this city")
                gr.Markdown("<small>Every driver number above is measured against this. "
                            "Medians come from the fitted preprocessing artifacts where they "
                            "exist.</small>")
            with gr.Accordion(f"Request JSON ({len(SERVING_COLUMNS)} contract fields)", open=False):
                req_view = gr.Code(language="json", label="POST body")
            with gr.Accordion("Driver values (JSON)", open=False):
                shap_view = gr.Code(language="json", label="Shapley values")

    inputs = [h[k] for k in HOST_FIELDS]
    # One chain, not three listeners on one button. Registering the hand-off to the
    # guest tab as its own .click() looked equivalent and was not: gradio would have
    # read resp_state as it stood when the button was pressed, i.e. the PREVIOUS
    # response, so the guest tab would seed from the listing priced before last.
    # .then() runs after the state is written.
    event = price_btn.click(
        host_price, [explain_cb, *inputs],
        [price_out, friction_out, req_view, explain_out, resp_state])
    if host_state is not None:                 # shared with the guest tab in --role both
        event = event.then(lambda r: r, resp_state, host_state)
    event.then(host_explain, [explain_cb, resp_state, *inputs], [explain_out, shap_view])

    save_btn.click(host_save_event, [h["listing_id"], resp_state, *inputs],
                   [save_status, mine])
    del_btn.click(host_delete_event, del_id, [save_status, mine])
    _LOAD_HOOKS.append((host_my_listings_event, None, mine))

    def _recentre(c):
        return (gr.update(choices=known_neighbourhoods(c),
                          value=(known_neighbourhoods(c) or [None])[0]),
                gr.update(value=CITY_CENTERS.get(c, (0, 0))[0]),
                gr.update(value=CITY_CENTERS.get(c, (0, 0))[1]),
                host_reference(c))

    h["city"].change(_recentre, h["city"],
                     [h["neighbourhood"], h["latitude"], h["longitude"], ref_view])
    c_find.click(host_corpus_options, [c_city, c_room], [c_pick, c_note])
    c_load.click(host_apply_corpus, c_pick, [h[k] for k in CORPUS_FILL] + [c_note])
    _LOAD_HOOKS.append((lambda: host_reference(preset["city"]), None, ref_view))


# --------------------------------------------------------------- guest handlers

BROWSE_HEADERS = ["#", "listing", "sleeps", "price", "model value", "instant book",
                  "ml score"]
BROWSE_TYPES = ["number", "str", "number", "str", "str", "str", "str"]


def guest_recommend(city, min_guests, max_price, w_value, w_convenience, only_saved=False):
    """POST /recommend, rendered."""
    out = recsys_recommend(city, min_guests, max_price, w_value, w_convenience,
                           only_saved=only_saved)
    saved_here = [r for r in store_rows()
                  if str(r.get("city", "")).lower() == str(city).lower()]
    if not out.get("results"):
        msg = out.get("message") or f"Nothing in {city} matches those filters."
        if only_saved and not saved_here:
            msg = (f"No host has saved a listing in {city} yet. Price one on the host "
                   f"tab and save it, then search again.")
        return [], msg, json.dumps(out, indent=2)
    ccy = CURRENCY.get(city, "")
    saved_ids = {r["listing_id"] for r in store_rows()}
    table = []
    for i, r in enumerate(out["results"], 1):
        mark = " [saved here]" if str(r["listing_id"]) in saved_ids else ""
        table.append([i, f"{r['listing_id']}{mark}", int(r["accommodates"]),
                      f"{r['price']:,.0f}", f"{r['pred_price']:,.0f}",
                      f"{r['instant_book_prob'] * 100:.0f}%", f"{r['ml_score']:.4f}"])
    served = ("served by the recommender API" if out.get("backend") == "api"
              else "ranked in this process")
    n_saved = sum(1 for r in out["results"] if str(r["listing_id"]) in saved_ids)
    note = (f"**{out['count']} listings** in {city}, {served} through `POST /recommend`. "
            f"Prices are {esc(ccy)}. **model value** is the price model's estimate; "
            f"**ml score** is `w_value x value + w_convenience x instant book`.")
    if only_saved:
        note += (f" Restricted to the {len(saved_here)} listing(s) hosts have saved "
                 f"in {city}.")
    elif n_saved:
        note += (f" {n_saved} of these were priced and saved by a host in this app, "
                 f"so they are in the pool this endpoint ranks.")
    elif saved_here:
        # Saying "they are in the pool" while none are on screen is the kind of claim
        # nobody can check. Say where they went and how to see them.
        note += (f" {len(saved_here)} listing(s) saved by hosts in {city} were ranked "
                 f"too but did not reach the top 10 &mdash; tick **only listings saved "
                 f"here** to see them.")
    if out.get("backend_note"):
        note += f"\n\n_{esc(out['backend_note'])}_"
    if out.get("warning"):
        # Surfaced, not buried. Moving the value slider and watching nothing happen is
        # the most informative thing this screen can show about the endpoint.
        note += f"\n\n**The value weight is not doing anything here.** {esc(out['warning'])}"
    return table, note, json.dumps(out, indent=2)


GUEST_HEADERS = ["#", "listing", "from", "area", "sleeps", "price", "vs yours",
                 "fair price", "instant book", "match", "attr-only rank", "why"]
GUEST_TYPES = ["number", "str", "str", "str", "number", "str", "str", "str",
               "str", "str", "number", "str"]


def guest_body(host_state=None) -> None:
    """The discovery assistant: POST /recommend, with its five SearchQuery fields.

    city, min_guests, max_price, w_value, w_convenience -- the same request the client
    notebook posted, as controls. What comes back is ranked on both model outputs:
    w_value weights how far under the price model's estimate a listing sits, and
    w_convenience weights the classifier's instant-booking probability.
    """
    cities = recsys_cities()
    gr.Markdown(f"""
        # Find a place to stay
        `{recsys_backend_name()}`

        Two model predictions decide the order. **Value** is how far below the price
        model's estimate a listing is priced. **Convenience** is the instant-booking
        probability. The sliders are the `w_value` and `w_convenience` weights the
        endpoint takes.
    """)
    with gr.Row():
        with gr.Column(scale=1):
            g_city = gr.Dropdown(label="City", choices=cities,
                                 value="Paris" if "Paris" in cities else
                                 (cities[0] if cities else "Paris"))
            g_guests = gr.Number(label="Guests", value=2, precision=0, minimum=1,
                                 info="min_guests")
            g_budget = gr.Number(label="Max nightly price", value=150.0, minimum=1,
                                 info="max_price")
            g_wv = gr.Slider(label="Weight: value", minimum=0, maximum=1, step=0.05,
                             value=0.5, info="w_value")
            g_wc = gr.Slider(label="Weight: convenience", minimum=0, maximum=1, step=0.05,
                             value=0.5, info="w_convenience")
            g_only = gr.Checkbox(
                label="Only listings saved here", value=False,
                info="Hosts' own listings are always ranked alongside the corpus; this "
                     "narrows to them so a newly saved one is findable.")
            g_search = gr.Button("Search listings", variant="primary", size="lg")
        with gr.Column(scale=3):
            g_table = gr.Dataframe(headers=BROWSE_HEADERS, datatype=BROWSE_TYPES,
                                   interactive=False, wrap=True)
            g_note = gr.Markdown()
            with gr.Accordion("Response JSON", open=False):
                g_json = gr.Code(language="json",
                                 label="What a search service would receive")

    # The two weights are independent: SearchQuery does not constrain them to sum to
    # one, and pinning them here would hide what happens at w_value=1.
    g_search.click(guest_recommend, [g_city, g_guests, g_budget, g_wv, g_wc, g_only],
                   [g_table, g_note, g_json])


# ------------------------------------------------------------------- assembly

def _flush_load_hooks(demo) -> None:
    for fn, i, o in _LOAD_HOOKS:
        demo.load(fn, i, o)
    _LOAD_HOOKS.clear()


def build_host_ui():
    _gradio()
    with gr.Blocks(title=f"{COURSE} {TEAM_ID} - pricing assistant") as demo:
        inject_css()
        host_body()
        _flush_load_hooks(demo)
    return demo


def build_guest_ui():
    _gradio()
    with gr.Blocks(title=f"{COURSE} {TEAM_ID} - discovery assistant") as demo:
        inject_css()
        guest_body()
        _flush_load_hooks(demo)
    return demo


def build_both_ui():
    """Both journeys in one process, sharing one session state.

    That shared state is the demo's point: price a listing on the host tab, then hit
    "use the listing I just priced" on the guest tab and the same endpoint outputs
    become the seed of the similarity search. It is also NOT a trust boundary --
    gradio's auth is per application, not per tab, so anyone who can reach this can
    open the host tab. Run --role host and --role guest as separate processes for a
    split enforced by the process rather than by a tab label.
    """
    _gradio()
    with gr.Blocks(title=f"{COURSE} {TEAM_ID} - listing app (demo mode)") as demo:
        inject_css()
        gr.Markdown(f"# {COURSE} {TEAM_ID} &middot; pricing and discovery (demo mode)")
        shared = gr.State(None)
        with gr.Tabs():
            with gr.Tab("Host - price a listing"):
                host_body(host_state=shared)
            with gr.Tab("Guest - find something similar"):
                guest_body(host_state=shared)
        _flush_load_hooks(demo)
    return demo


# ============================================================================
# 11. SELF TEST
# ============================================================================
# No UI, no AWS, no gradio, no boto3. Both journeys are exercised with the endpoints
# replaced by known functions, which is the only way to check that the Shapley
# implementation is right: against a live model every number is plausible and none
# is verifiable.
# ============================================================================

def _fake_price_fn(rows: list[dict]) -> list[float]:
    """A known price function, so the attributions have a right answer.

    Deliberately shaped to test the axioms:
      Capacity and Booking rules contribute an identical +10 step   -> symmetry
      Reviews are not read at all                                   -> dummy
      Host profile multiplies, rather than adds                     -> interaction,
        which is the case a leave-one-out "explanation" gets wrong and Shapley splits.
    """
    out = []
    for r in rows:
        p = 50.0
        p += 10.0 if int(r["accommodates"]) > 3 else 0.0            # Capacity
        p += 10.0 if int(r["minimum_nights"]) > 2 else 0.0          # Booking rules
        p += 12.0 if r["room_type"] == "Entire place" else 0.0      # Property type
        p += 0.5 * len(json.loads(r["amenities"]))                  # Amenities
        p += 0.0 if r["neighbourhood"] == "REF" else 6.0            # Location
        p *= 1.20 if r["host_is_superhost"] == "t" else 1.0         # Host profile
        out.append(p)
    return out


def _fake_booking_fn(payload: dict) -> dict:
    """Cheaper listings book more easily. Monotone in price, so the lever is testable."""
    price = float(payload["price"])
    return {"probability_class_1": max(0.02, min(0.98, 1.2 - price / 200.0))}


def _notebook_rank(df, city, min_guests=2, max_price=150.0,
                   w_value=0.5, w_convenience=0.5, fit="load"):
    """Both notebooks' ranking, transcribed. They differ in one place: WHERE fit_transform
    is called relative to the filter.

    fit="load"        Test_API_Gradio.ipynb, and notebook 05 cell 14:

        df['deal_value'] = df['pred_price'] - df['price']
        df['value_score'] = scaler.fit_transform(df[['deal_value']])      # before filtering
        candidates = df[city & accommodates & price]

    fit="candidates"  notebook 05 cell 19 (compare_recsys) and cell 21 (evaluate_ndcg):

        candidates = df[city & accommodates & price]                      # filter first
        candidates['deal_value'] = candidates['pred_price'] - candidates['price']
        candidates['scaled_value_score'] = scaler.fit_transform(candidates[['deal_value']])

    Then both do the same thing:

        ml_score = w_value * scaled + w_convenience * instant_book_prob
        ranked = candidates.sort_values(by='ml_score', ascending=False).head(10)

    Kept as a literal second implementation so api_recommend is checked against the
    notebooks themselves rather than against my description of them. MinMaxScaler is
    (x - min) / (max - min), written out so this runs without sklearn; the self test
    cross-checks against the real one when it is installed.
    """
    d = df.copy()
    if fit == "load":
        d["deal_value"] = d["pred_price"] - d["price"]
        lo, hi = float(d["deal_value"].min()), float(d["deal_value"].max())
        d["scaled"] = (d["deal_value"] - lo) / (hi - lo)
    candidates = d[(d["city"].str.lower() == city.lower())
                   & (d["accommodates"] >= min_guests)
                   & (d["price"] <= max_price)].copy()
    if candidates.empty:
        return []
    if fit == "candidates":
        candidates["deal_value"] = candidates["pred_price"] - candidates["price"]
        lo = float(candidates["deal_value"].min())
        hi = float(candidates["deal_value"].max())
        candidates["scaled"] = (candidates["deal_value"] - lo) / (hi - lo)
    candidates["ml_score"] = (w_value * candidates["scaled"]
                              + w_convenience * candidates["instant_book_prob"])
    ranked = candidates.sort_values(by="ml_score", ascending=False).head(10)
    return [{"listing_id": str(r["listing_id"]), "ml_score": float(r["ml_score"])}
            for _, r in ranked.iterrows()]


def _fake_corpus():
    import pandas as pd
    #                      the seed, three alternatives that differ on ONE axis each,
    #                      and an out-of-city price outlier that sets the global min/max.
    return pd.DataFrame({
        "listing_id": ["S", "A", "B", "C", "D", "OUT"],
        "city": ["Paris"] * 5 + ["Rio de Janeiro"],
        "neighbourhood": ["Montmartre", "Montmartre", "Montmartre", "Marais", "Montmartre", "z"],
        "property_type": ["Entire apartment"] * 5 + ["Entire house"],
        "room_type": ["Entire place"] * 6,
        "accommodates": [4, 4, 4, 8, 4, 4],
        "bedrooms": [2, 2, 2, 4, 2, 2],
        "minimum_nights": [2, 2, 2, 7, 2, 2],
        "price": [100.0, 95.0, 40.0, 105.0, 98.0, 40000.0],
        "instant_bookable": [1, 1, 1, 1, 0, 1],
        #            A: near twin.  B: same attributes, far cheaper tier.
        #            C: different attributes, same tier.  D: same tier, hard to book.
        "pred_price": [110.0, 108.0, 45.0, 110.0, 109.0, 42000.0],
        "instant_book_prob": [0.90, 0.88, 0.90, 0.90, 0.10, 0.5],
    })


def self_test() -> int:
    load_fitted_state()
    ok = "  ok  "

    # ---- the two contracts ------------------------------------------------
    ui = host_preset()
    ui["asking_price"] = 90.0
    payload = build_payload(ui)
    assert set(payload) == set(SERVING_COLUMNS), set(payload) ^ set(SERVING_COLUMNS)
    assert not validate_host(ui, payload), validate_host(ui, payload)
    assert "price" not in payload, "the regression contract must not carry price"
    print(f"{ok}contract        {len(payload)} fields from {CONTRACT_SOURCE}, validation clean")

    bp = build_booking_payload(ui, 90.0)
    assert set(bp) == set(BOOKING_COLUMNS), set(bp) ^ set(BOOKING_COLUMNS)
    assert bp["price"] == 90.0 and build_booking_payload(ui, 42.0)["price"] == 42.0
    assert "neighbourhood" not in bp, "not in the classifier's feature schema"
    assert bp["host_is_superhost"] == 1 and payload["host_is_superhost"] == "t", \
        "booleans are ints for the classifier and 't'/'f' for the regression"
    print(f"{ok}booking payload {len(bp)} fields, price carried, flags coerced to int")

    # ---- the explanation groups partition the contract ---------------------
    assert set(_GROUPED) | EXPLAIN_FIXED == set(SERVING_COLUMNS)
    base = baseline_listing("Paris")
    assert set(base) == set(SERVING_COLUMNS), set(base) ^ set(SERVING_COLUMNS)
    assert base["city"] == "Paris"
    print(f"{ok}groups          {len(EXPLAIN_GROUPS)} groups + city cover all "
          f"{len(SERVING_COLUMNS)} fields; reference listing is contract-shaped")

    # ---- Shapley: the three axioms, against a function with a known answer --
    # Both the listing and its reference are pinned here rather than taken from the
    # artifacts. Symmetry is a statement about two groups that differ from the
    # reference in the same way, so a reference that happens to match the listing on
    # minimum_nights would make Booking rules a correct zero and the test a false alarm.
    x = dict(payload)
    x["neighbourhood"] = "SOMEWHERE"          # Location differs   -> +6
    x["accommodates"] = 4                     # Capacity differs   -> +10
    x["minimum_nights"] = 5                   # Booking rules      -> +10, identically
    ref = baseline_listing("Paris")
    ref.update({"neighbourhood": "REF", "accommodates": 3, "minimum_nights": 2,
                "host_is_superhost": "f"})
    calls = {"n": 0}

    def scored(rows):
        calls["n"] += len(rows)
        return _fake_price_fn(rows)

    import unittest.mock as _mock
    with _mock.patch(f"{__name__}.baseline_listing", return_value=ref):
        ex = explain_price(x, score_batch=scored)
    phi = {c["group"]: c["value"] for c in ex["contributions"]}

    assert ex["endpoint_calls"] == 128, ex["endpoint_calls"]
    assert abs(ex["residual"]) < 1e-9, f"efficiency violated: {ex['residual']}"
    print(f"{ok}shap efficiency {ex['reference_price']:.2f} + parts = "
          f"{ex['suggested_nightly_price']:.2f}, residual {ex['residual']:+.1e}")

    assert abs(phi["Reviews"]) < 1e-9, f"dummy violated: Reviews got {phi['Reviews']}"
    print(f"{ok}shap dummy      a group the model never reads is attributed "
          f"{phi['Reviews']:+.1e}")

    assert abs(phi["Capacity"] - phi["Booking rules"]) < 1e-9, \
        f"symmetry violated: {phi['Capacity']} vs {phi['Booking rules']}"
    print(f"{ok}shap symmetry   two groups with identical effect both get "
          f"{phi['Capacity']:+.2f}")

    # The superhost multiplier is worth 20% of a base that the other groups build.
    # Leave-one-out would hand that entire interaction to Host profile; Shapley
    # splits it, so every additive group ends up above its own standalone effect.
    assert phi["Host profile"] > 0, phi
    assert phi["Capacity"] > 10.0, f"interaction not shared with Capacity: {phi['Capacity']}"
    loo = _fake_price_fn([x])[0] - _fake_price_fn(
        [_coalition(x, ref, list(EXPLAIN_GROUPS), EXPLAIN_GROUPS,
                    [i for i, g in enumerate(EXPLAIN_GROUPS) if g != "Capacity"])])[0]
    assert abs(loo - phi["Capacity"]) > 0.5, "interaction test is not exercising anything"
    print(f"{ok}shap interaction Capacity gets {phi['Capacity']:+.2f}; leave-one-out would "
          f"have said {loo:+.2f}")

    ranked = [c["group"] for c in ex["contributions"]]
    assert ranked == sorted(ranked, key=lambda g: -abs(phi[g])), "not sorted by magnitude"
    assert all(c["detail"] for c in ex["contributions"]), "a driver has no plain-language detail"
    print(f"{ok}shap output     sorted by magnitude, top driver is "
          f"{ranked[0]!r} at {phi[ranked[0]]:+.2f}")

    # ---- sampling agrees with the exact answer -----------------------------
    phi_s, _ = shapley_sampled(x, ref, list(EXPLAIN_GROUPS), EXPLAIN_GROUPS, scored, 200, seed=7)
    worst = max(abs(a - phi[g]) for g, a in zip(EXPLAIN_GROUPS, phi_s))
    assert worst < 0.75, f"sampled estimate is {worst:.3f} from exact"
    assert abs(sum(phi_s) - sum(phi.values())) < 1e-9, "sampling broke efficiency"
    print(f"{ok}shap sampling   200 permutations land within {worst:.3f} of exact, "
          f"efficiency still exact")

    # ---- the whole host journey, endpoints stubbed --------------------------
    resp = price_listing(ui, explain=True, score_batch=_fake_price_fn,
                         booking_fn=_fake_booking_fn)
    assert resp["ok"] and resp["suggested_nightly_price"] > 0
    assert resp["friction"]["at_price"] == resp["suggested_nightly_price"]
    assert resp["friction_at_asking"]["at_price"] == 90.0
    moved = (resp["friction_at_asking"]["instant_book_prob"]
             - resp["friction"]["instant_book_prob"])
    assert abs(moved) > 0.01, "price is a classifier feature; the reading must move with it"
    print(f"{ok}host journey    {resp['suggested_nightly_price']:,.2f} "
          f"{resp['currency']}, friction {resp['friction']['instant_book_prob']:.0%} at the "
          f"suggestion and {resp['friction_at_asking']['instant_book_prob']:.0%} at 90")

    for prob, level in [(0.91, "low"), (0.60, "moderate"), (0.31, "raised"), (0.04, "high")]:
        assert friction_reading(prob)["level"] == level, (prob, friction_reading(prob))
    assert friction_reading(None)["level"] == "unknown"
    print(f"{ok}friction        four readings, monotone, no error band on a classifier")

    bad = price_listing({**ui, "minimum_nights": 30, "maximum_nights": 2},
                        score_batch=_fake_price_fn, booking_fn=_fake_booking_fn)
    assert not bad["ok"] and any("Minimum nights" in p for p in bad["problems"])
    print(f"{ok}validation      a contradictory listing is refused before any endpoint call")

    # ---- privacy ------------------------------------------------------------
    view = public_view(payload, asking_price=90.0)
    assert not (HOST_PRIVATE & set(view)) and "dist_center_km" in view and "latitude" not in view
    assert "host_secret_note" not in public_view({**payload, "host_secret_note": "no"})
    assert not (set(CORPUS_COLUMNS) & HOST_PRIVATE)
    print(f"{ok}privacy         {len(view)} published fields, {len(HOST_PRIVATE)} withheld, "
          f"unclassified fields withheld by default")

    # ---- lineage stamps are per endpoint ------------------------------------
    _LINEAGE_CACHE.clear()
    _LINEAGE_CACHE[PRICE_ENDPOINT] = {"model_package": "arn:.../8"}
    _LINEAGE_CACHE[BOOKING_ENDPOINT] = {"model_package": "arn:.../clf-3"}
    assert model_lineage(PRICE_ENDPOINT)["model_package"] == "arn:.../8"
    assert model_lineage(BOOKING_ENDPOINT)["model_package"] == "arn:.../clf-3", \
        "cache is not keyed by endpoint: model 2 received model 1's stamp"
    print(f"{ok}lineage         stamps are per endpoint, not shared across models")

    # ---- the store, and the save -> discover path ---------------------------
    global USE_S3, STORE_LOCAL
    USE_S3, STORE_LOCAL = False, "._selftest_store.json"
    if os.path.exists(STORE_LOCAL):
        os.remove(STORE_LOCAL)
    _LINEAGE_CACHE.clear()
    _LINEAGE_CACHE[PRICE_ENDPOINT] = {"model_package": "arn:.../models/8"}
    _LINEAGE_CACHE[BOOKING_ENDPOINT] = {"model_package": "arn:.../clf-3"}

    priced = price_listing({**ui, "asking_price": 120.0}, explain=True,
                           score_batch=_fake_price_fn, booking_fn=_fake_booking_fn)
    rec = save_listing("PAR-001", priced, "alice")
    assert rec["owner"] == "alice" and rec["asking_price"] == 120.0
    assert rec["suggested_price"] == priced["suggested_nightly_price"]
    assert rec["drivers"], "the top drivers should be stored alongside the price"
    # The classifier takes price as a feature and the corpus scores each listing at
    # its own asking price, so a saved listing must be scored at 120, not at the 97
    # the model suggested -- otherwise the friction dimension compares a shelf price
    # against a hypothetical one.
    assert rec["instant_book_prob_at"] == rec["asking_price"] == 120.0, rec["instant_book_prob_at"]
    assert abs(rec["instant_book_prob"]
               - _fake_booking_fn({"price": 120.0})["probability_class_1"]) < 1e-9, \
        "friction was stored at the suggested price, not the asking price"
    print(f"{ok}save            PAR-001 -> {store_backend_name().rsplit('/', 1)[-1]}, "
          f"asking {rec['asking_price']:,.0f}, modelled {rec['suggested_price']:,.0f}, "
          f"book {rec['instant_book_prob']:.0%}")

    # No asking price means publishing at the model's number, not publishing nothing.
    at_model = save_listing("PAR-002", price_listing(
        {**ui, "asking_price": None}, explain=False,
        score_batch=_fake_price_fn, booking_fn=_fake_booking_fn), None)
    assert at_model["asking_price"] == at_model["suggested_price"]
    assert at_model["owner"] is None, "an unauthenticated save must be unowned, not 'anonymous'"
    print(f"{ok}save defaults   a blank asking price publishes at the suggestion; "
          f"an unauthenticated save is unowned")

    published = store_rows()
    assert {r["listing_id"] for r in published} == {"PAR-001", "PAR-002"}
    for row in published:
        assert not (HOST_PRIVATE & set(row)), sorted(HOST_PRIVATE & set(row))
        assert set(row) == set(CORPUS_COLUMNS) | {"source"}, set(row) ^ set(CORPUS_COLUMNS)
    print(f"{ok}publication     saved listings reach guests as {len(CORPUS_COLUMNS)} corpus "
          f"columns; all {len(HOST_PRIVATE)} private fields dropped on the way out")

    # A listing the booking endpoint never scored has no position in the similarity
    # space, so it must be stored but withheld rather than published unranked.
    half = price_listing({**ui, "asking_price": 99.0}, explain=False,
                         score_batch=_fake_price_fn,
                         booking_fn=lambda p: {"probability_class_1": None})
    save_listing("PAR-003", half, "alice")
    assert "PAR-003" in store_read() and "PAR-003" not in {r["listing_id"] for r in store_rows()}
    print(f"{ok}half scored     a listing with no booking score is kept but withheld from "
          f"guests, not published without a friction reading")

    try:
        save_listing("PAR-001", priced, "mallory")
        raise SystemExit("ownership check failed")
    except PermissionError as e:
        assert "belongs to `alice`" in str(e) and "you are `mallory`" in str(e), str(e)
    assert "--set-owner" in delete_listing("PAR-001", "mallory")
    print(f"{ok}ownership       refusal names the owner AND the caller, and says how to "
          f"repair it from the CLI")

    _LINEAGE_CACHE[PRICE_ENDPOINT] = {"model_package": "arn:.../models/9"}   # a redeploy
    assert rec["model"]["price"]["model_package"] == "arn:.../models/8", "stamp mutated under us"
    assert "re-price" in (record_is_stale(rec) or "")
    _LINEAGE_CACHE[PRICE_ENDPOINT] = {"model_package": "arn:.../models/8"}
    print(f"{ok}staleness       the stamp is immutable and a redeploy is detected")

    # ---- a record written by the PREVIOUS app must not break this one --------
    # Same S3 key, different schema. This is the exact shape the old app wrote:
    # the price nested under `prediction`, no booking probability, and `model` as a
    # single flat lineage stamp.
    v1 = {"payload": priced["request"], "asking_price": 88.0,
          "prediction": {"suggested_nightly_price": 70.9, "currency": "EUR"},
          "model": {"endpoint": PRICE_ENDPOINT, "model_package": "arn:.../models/8"},
          "owner": "bob", "updated_at": "2026-01-02T03:04:05", "updated_by": "bob"}
    store_write({**store_read(), "OLD-001": v1})
    back = store_read()["OLD-001"]
    assert back["suggested_price"] == 70.9 and back["currency"] == "EUR"
    assert back["owner"] == "bob" and back["asking_price"] == 88.0, "migration lost a field"
    assert back["model"]["price"]["model_package"] == "arn:.../models/8"
    assert back["instant_book_prob"] is None, "a probability was invented for a v1 record"
    table = host_my_listings("alice")
    assert "OLD-001" in table and "--rescore-store" in table, \
        "a v1 record must render with a way out, not raise KeyError"
    assert "OLD-001" not in {r["listing_id"] for r in store_rows()}, \
        "a record with no friction score must stay out of the guest catalogue"
    print(f"{ok}v1 migration    a record from the previous app upgrades on read, renders, "
          f"and is withheld from guests until it has both scores")

    # Recoverable without the form it was typed into, which is what --rescore-store
    # relies on: the stored payload alone is enough to rebuild the classifier request.
    revived = booking_payload_from_serving(back["payload"], back["asking_price"])
    assert set(revived) == set(BOOKING_COLUMNS) and revived["price"] == 88.0
    assert revived["host_is_superhost"] in (0, 1)
    print(f"{ok}rescore path    the classifier request rebuilds from the stored payload "
          f"alone, so --rescore-store needs no host input")

    for lid in ("OLD-001", "PAR-003"):
        st = store_read(); st.pop(lid, None); store_write(st)

    # ---- guest journey: /recommend, both ways of scaling the value term -----
    global RECOMMEND_SCALING
    df = _fake_corpus()
    globals()["_CORPUS"] = df.assign(source="corpus")
    _CATALOGUE_CACHE.clear()
    _GLOBAL_SPAN.clear()

    # The corpus carries an out-of-city outlier priced at 40,000, which is what a
    # ten-currency dataset looks like and what sets the global min and max.
    RECOMMEND_SCALING = "global"
    g = api_recommend("Paris", 2, 150.0, 0.5, 0.5)
    order_g = [r["listing_id"] for r in g["results"]]
    assert "warning" in g, "the flattened value term must be reported"
    assert order_g == [r["listing_id"] for r in
                       api_recommend("Paris", 2, 150.0, 0.0, 1.0)["results"]], \
        "under global scaling w_value=0.5 must order exactly as w_value=0"
    print(f"{ok}recommend global {order_g} -- same order as ranking on instant_book_prob "
          f"alone, and the endpoint says so")

    RECOMMEND_SCALING = "candidates"
    _GLOBAL_SPAN.clear()
    c = api_recommend("Paris", 2, 150.0, 0.5, 0.5)
    order_c = [r["listing_id"] for r in c["results"]]
    assert "warning" not in c, "with candidate scaling the value term should do something"
    assert order_c != order_g, "candidate scaling changed nothing, so it is not fitted"
    # B is the cheapest listing but priced almost exactly at its model value; A is
    # underpriced against a higher model value. Value-aware ranking must prefer A.
    assert order_c.index("A") < order_c.index("B"), order_c
    print(f"{ok}recommend cands  {order_c} -- the value term now moves the ranking, and "
          f"the underpriced listing outranks the merely cheap one")

    spread_g = max(r["ml_score"] for r in g["results"]) - min(r["ml_score"] for r in g["results"])
    spread_c = max(r["ml_score"] for r in c["results"]) - min(r["ml_score"] for r in c["results"])
    print(f"{ok}scaling         ml_score spread {spread_g:.4f} global vs {spread_c:.4f} "
          f"candidates: the same weights, one of them inert")
    RECOMMEND_SCALING = "global"
    _GLOBAL_SPAN.clear()

    assert not (set(CORPUS_COLUMNS) & HOST_PRIVATE)
    saved_in_pool = {r["listing_id"] for r in store_rows()} & set(
        catalogue()["listing_id"])
    assert saved_in_pool, "saved listings never reached the pool /recommend ranks"
    print(f"{ok}save -> discover host-saved listings are in the pool the endpoint ranks: "
          f"{sorted(saved_in_pool)}")

    empty = api_recommend("Paris", 2, 1.0, 0.5, 0.5)
    assert empty["count"] == 0 and "No listings match" in empty["message"]
    print(f"{ok}empty result    an impossible budget returns a message, not an exception")

    # ---- fidelity: does this reproduce the notebooks, or merely resemble them? --
    assert _SCALING_DEFAULT == "candidates", (
        f"the default is {_SCALING_DEFAULT!r}. notebook 05 evaluated candidate-set "
        f"scaling in cells 19 and 21, so shipping global means the reported NDCG and "
        f"Utility Lift do not describe what is served.")
    print(f"{ok}scaling default {_SCALING_DEFAULT!r} -- the one notebook 05 evaluated")

    pool = catalogue()
    key = lambda r: (-round(r["ml_score"], 12), str(r["listing_id"]))
    for label, mode, fit in [("Test_API_Gradio / cell 14", "global", "load"),
                             ("notebook 05 cells 19+21", "candidates", "candidates")]:
        RECOMMEND_SCALING = mode
        _GLOBAL_SPAN.clear()
        theirs = _notebook_rank(pool, "Paris", 2, 150.0, 0.5, 0.5, fit=fit)
        mine = api_recommend("Paris", 2, 150.0, 0.5, 0.5)["results"]
        assert len(theirs) == len(mine), (label, len(theirs), len(mine))
        # Compared as (id, score) sorted by both, because the notebooks leave pandas on
        # its default quicksort while this uses a stable sort -- the one deliberate
        # deviation, and it can only reorder listings whose scores are already equal.
        for a, b in zip(sorted(theirs, key=key), sorted(mine, key=key)):
            assert a["listing_id"] == b["listing_id"], (label, a, b)
            assert abs(a["ml_score"] - b["ml_score"]) < 1e-12, (label, a, b)
        print(f"{ok}parity {mode:<10} api_recommend reproduces {label} exactly: "
              f"{len(mine)} rows, ml_scores identical to 1e-12")

    try:
        from sklearn.preprocessing import MinMaxScaler
    except ImportError:
        print(f"{ok}sklearn parity  skipped, sklearn is not installed here")
    else:
        deal = (pool["pred_price"] - pool["price"]).to_numpy(dtype=float).reshape(-1, 1)
        sk = MinMaxScaler().fit_transform(deal).ravel()
        ours = (deal.ravel() - deal.min()) / (deal.max() - deal.min())
        assert abs(sk - ours).max() < 1e-12, abs(sk - ours).max()
        print(f"{ok}sklearn parity  the hand-written min-max matches MinMaxScaler to "
              f"{abs(sk - ours).max():.1e}, so the transcriptions are faithful")

    RECOMMEND_SCALING = _SCALING_DEFAULT
    _GLOBAL_SPAN.clear()

    # ---- a guest must be able to find what a host just saved ----------------
    saved_only = api_recommend("Paris", 2, 150.0, 0.5, 0.5, only_saved=True)
    saved_ids = {r["listing_id"] for r in store_rows()}
    got = {r["listing_id"] for r in saved_only["results"]}
    assert got and got <= saved_ids, (got, saved_ids)
    print(f"{ok}only saved      {sorted(got)} -- a host's own listings are reachable "
          f"without having to out-rank a whole city")

    # The filter must not move the value term's reference point, or the same listing
    # would score differently depending on a filter that is about visibility, not value.
    full = {r["listing_id"]: r["ml_score"] for r in
            api_recommend("Paris", 2, 150.0, 0.5, 0.5)["results"]}
    shared = got & set(full)
    assert shared and all(
        abs(next(r["ml_score"] for r in saved_only["results"] if r["listing_id"] == lid)
            - full[lid]) < 1e-12 for lid in shared), "the filter changed the scores"
    print(f"{ok}filter is inert only_saved narrows the result set without changing any "
          f"listing's score, under {RECOMMEND_SCALING} scaling")

    # ---- the API, over a real socket ----------------------------------------
    # The server under test is the one that ships, not a stand-in written for the
    # test: a mocked _post_json would prove the call site and nothing about numpy
    # scalars json.dumps refuses or NaN that serialises as invalid JSON.
    import threading
    global RECSYS_BACKEND, RECSYS_API_URL

    srv = _stdlib_server("127.0.0.1", 0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    RECSYS_API_URL, RECSYS_BACKEND = f"http://127.0.0.1:{srv.server_address[1]}", "api"
    assert _wait_for_api(srv.server_address[1], timeout=10), "the API did not come up"

    health = _get_json(f"{RECSYS_API_URL}/health")
    assert health["ok"] and health["endpoints"] == ["/recommend", "/health"]
    print(f"{ok}api health      {health['listings']} listings, routes "
          f"{', '.join(health['endpoints'])}")

    over_http = recsys_recommend("Paris", 2, 150.0, 0.5, 0.5)
    assert over_http["backend"] == "api" and over_http["count"]
    assert set(over_http["results"][0]) == {
        "listing_id", "city", "accommodates", "price", "pred_price",
        "instant_book_prob", "ml_score"}, sorted(over_http["results"][0])
    assert [r["listing_id"] for r in over_http["results"]] == \
           [r["listing_id"] for r in api_recommend("Paris", 2, 150.0, 0.5, 0.5)["results"]], \
        "the API and the in-process path disagree, so they are not the same ranker"
    print(f"{ok}api transport   /recommend over HTTP matches the in-process call, and "
          f"keeps the original seven response fields")

    try:
        __import__("fastapi")
    except ImportError:
        print(f"{ok}api fastapi     skipped, fastapi is not installed here")
    else:
        route = next(r for r in _fastapi_app().routes
                     if getattr(r, "path", None) == "/recommend")
        assert getattr(route, "body_field", None) is not None, (
            "FastAPI did not resolve SearchQuery, so it is reading the request body as "
            "a query parameter and every POST will 422")
        print(f"{ok}api fastapi     /recommend resolved SearchQuery as a body field, not "
              f"a query parameter")

    srv.shutdown()
    dead = recsys_recommend("Paris", 2, 150.0, 0.5, 0.5)
    assert not dead.get("results") and "--recsys-backend inline" in dead["message"]
    print(f"{ok}api down        --recsys-backend api refuses with a route out")

    RECSYS_BACKEND = "auto"
    fell_back = recsys_recommend("Paris", 2, 150.0, 0.5, 0.5)
    assert fell_back["count"] and fell_back["backend"] == "inline" and fell_back["backend_note"]
    print(f"{ok}api fallback    auto ranks in process when nothing is serving, and says "
          f"so rather than failing the demo")
    RECSYS_BACKEND = "inline"

    os.path.exists(STORE_LOCAL) and os.remove(STORE_LOCAL)
    print("\nSELF TEST PASSED")
    return 0


# ============================================================================
# 12. COMMAND LINE
# ============================================================================
# The journeys are functions, so they are reachable without the UI. This is the same
# code path a platform's search service or host dashboard would call over HTTP.
# ============================================================================

def cmd_recommend(city: str, min_guests: int, max_price: float, w_value: float,
                  w_convenience: float, as_json: bool, only_saved: bool = False) -> int:
    """The guest journey from the command line: the same five fields the API takes."""
    out = recsys_recommend(city, min_guests, max_price, w_value, w_convenience,
                           only_saved=only_saved)
    if as_json:
        print(json.dumps(out, indent=2))
        return 0 if out.get("results") else 1
    if not out.get("results"):
        print(out.get("message") or f"Nothing in {city} matches those filters.")
        return 1
    ccy = CURRENCY.get(city, "")
    print(f"\n{city}: sleeps {min_guests}+, at most {max_price:,.0f} {ccy}, "
          f"w_value={w_value} w_convenience={w_convenience}")
    print(f"ranked by {out.get('backend')} | value scaling "
          f"{out.get('value_scaling', RECOMMEND_SCALING)}\n")
    print(f"  {'#':<3}{'listing':<14}{'sleeps':>7}{'price':>9}{'model':>9}"
          f"{'value':>9}{'book':>7}{'score':>9}")
    for i, r in enumerate(out["results"], 1):
        print(f"  {i:<3}{str(r['listing_id']):<14}{int(r['accommodates']):>7}"
              f"{r['price']:>9,.0f}{r['pred_price']:>9,.0f}"
              f"{r['pred_price'] - r['price']:>+9,.0f}"
              f"{r['instant_book_prob'] * 100:>6.0f}%{r['ml_score']:>9.4f}")
    if out.get("warning"):
        print(f"\nNOTE: {out['warning']}")
    return 0


def cmd_price_json(path: str, explain: bool = True) -> int:
    ui = json.loads(Path(path).read_text())
    out = price_listing(ui, explain=explain)
    print(json.dumps(out, indent=2))
    return 0 if out.get("ok") else 1


def cmd_list_store() -> int:
    store = store_read()
    print(f"store: {store_backend_name()}")
    if not store:
        print("(empty)")
        return 0
    print(f"{len(store)} listing(s)\n")
    print(f"  {'listing':<16s}{'owner':<14s}{'asking':>9s}{'model':>9s}{'book':>7s}  "
          f"{'guests see it':<20s}  updated")
    legacy = 0
    for lid, rec in sorted(store.items()):
        prob = rec.get("instant_book_prob")
        if rec.get("suggested_price") is None or prob is None:
            seen, legacy = "no (needs scoring)", legacy + 1
        else:
            seen = "stale" if record_is_stale(rec) else "yes"
        print(f"  {lid:<16s}{(rec.get('owner') or '(unowned)'):<14s}"
              f"{_money(rec.get('asking_price')):>9s}{_money(rec.get('suggested_price')):>9s}"
              f"{'-' if prob is None else f'{prob * 100:.0f}%':>7s}  {seen:<20s}  "
              f"{(rec.get('updated_at') or '-')[:16]}")
    if legacy:
        print(f"\n{legacy} listing(s) cannot be found by guests yet. "
              f"Run --rescore-store to score them from their stored payloads.")
    return 0


def cmd_migrate_store() -> int:
    """Make the on-read upgrade permanent. No endpoint calls, no scores invented."""
    store = store_read()                      # upgraded in memory by store_read
    if not store:
        print("Store is empty, nothing to migrate.")
        return 0
    store_write(store)
    needs = [lid for lid, r in store.items()
             if r.get("instant_book_prob") is None or r.get("suggested_price") is None]
    print(f"Rewrote {len(store)} record(s) at {store_backend_name()} in schema "
          f"v{STORE_SCHEMA}.")
    if needs:
        print(f"{len(needs)} still have no scores and stay invisible to guests: "
              f"{', '.join(needs[:8])}{' ...' if len(needs) > 8 else ''}")
        print("Run --rescore-store to fill those in from their stored payloads.")
    return 0


def cmd_rescore_store(all_records: bool = False) -> int:
    """Re-run both endpoints over saved listings and fill in the missing scores.

    Possible only because a saved record keeps the full 27-field payload, and because
    the classifier contract is derived from it rather than from the form -- so a
    listing saved by a version of this app that had no classifier can be brought into
    the guest catalogue without the host touching it.
    """
    store = store_read()
    targets = [lid for lid, rec in sorted(store.items())
               if all_records or rec.get("instant_book_prob") is None
               or rec.get("suggested_price") is None]
    if not targets:
        print("Every saved listing already has both scores. Use --all to re-score anyway.")
        return 0
    print(f"Scoring {len(targets)} listing(s) against {PRICE_ENDPOINT} "
          f"and {BOOKING_ENDPOINT}.\nThe first call may take ~60 s: the endpoints "
          f"scale to zero.\n")
    done = failed = 0
    for lid in targets:
        rec = store[lid]
        payload = rec.get("payload")
        if not payload:
            print(f"  {lid:<16s} skipped: the record has no stored listing payload")
            failed += 1
            continue
        # Checked here rather than left to the endpoint or to the dict projection: a
        # truncated payload otherwise surfaces as a bare KeyError on one field, which
        # says nothing about how many are missing or whether the record is salvageable.
        gaps = [c for c in SERVING_COLUMNS if c not in payload]
        if gaps:
            print(f"  {lid:<16s} skipped: stored payload is missing {len(gaps)} of "
                  f"{len(SERVING_COLUMNS)} contract fields ({', '.join(gaps[:3])}"
                  f"{', ...' if len(gaps) > 3 else ''}). Re-save it from the host form.")
            failed += 1
            continue
        try:
            pred, _ = invoke_price(payload)
            suggested = float(pred["suggested_nightly_price"])
            # Scored at what a guest would pay, matching how the corpus was scored.
            asking = rec.get("asking_price") or suggested
            book, _ = invoke_booking(booking_payload_from_serving(payload, asking))
            prob = book.get("probability_class_1")
            rec.update({
                "suggested_price": suggested,
                "asking_price": float(asking),
                "instant_book_prob": float(prob) if prob is not None else None,
                "instant_book_prob_at": float(asking),
                "currency": CURRENCY.get(payload.get("city"), rec.get("currency", "")),
                "model": {"price": model_lineage(PRICE_ENDPOINT),
                          "booking": model_lineage(BOOKING_ENDPOINT)},
                "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "updated_by": "rescore",
                "schema": STORE_SCHEMA,
            })
            print(f"  {lid:<16s} asking {asking:>8,.0f}  modelled {suggested:>8,.0f}  "
                  f"book {'-' if prob is None else f'{prob * 100:.0f}%'}")
            done += 1
        except Exception as e:
            print(f"  {lid:<16s} {e.__class__.__name__}: {e}")
            failed += 1
    # Written once at the end: a store_write per listing would multiply the chance of
    # a partial rewrite across a run that is already several endpoint calls long.
    store_write(store)
    print(f"\n{done} scored, {failed} failed. Owners are unchanged.")
    return 0 if failed == 0 else 1


def cmd_set_owner(spec: str) -> int:
    if ":" not in spec:
        print("Expected --set-owner LISTING_ID:USERNAME (a trailing ':' unowns it).")
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
    if not assume_yes and input("Type DELETE to confirm: ").strip() != "DELETE":
        print("Nothing deleted.")
        return 1
    store_write({})
    print("Store cleared.")
    return 0


def cmd_api_test(top_n: int = 5) -> int:
    """The checks that used to live in API_Test_Client.ipynb.

    Separate from --self-test on purpose: that one proves the maths with the network
    stubbed out, this one proves a server is up and answering over a real socket. One
    can pass while the other fails, and conflating them would hide which.
    """
    base = RECSYS_API_URL.rstrip("/")
    print(f"Recommender API at {base}\n")
    try:
        h = _get_json(f"{base}/health", timeout=10)
    except Exception as e:
        print(_api_unreachable(e))
        return 1
    print(f"  /health      {h['listings']:,} listings across {len(h['cities'])} cities, "
          f"{h['saved_listings']} saved by hosts")
    print(f"               ranker {h['ranker']}, value scaling {h['value_scaling']}")
    city = "Paris" if "Paris" in h["cities"] else (h["cities"][0] if h["cities"] else "Paris")

    # The original client's query, unchanged.
    q = {"city": city, "min_guests": 2, "max_price": 150.0,
         "w_value": 0.5, "w_convenience": 0.5}
    rec = _post_json(f"{base}/recommend", q, timeout=60)
    if not rec.get("results"):
        print(f"\n  /recommend   no results for {city}: {rec.get('message', '')}")
        return 1
    ccy = CURRENCY.get(city, "")
    print(f"\n  /recommend   {rec['count']} listings for "
          f"city={city} min_guests=2 max_price=150 w_value=0.5 w_convenience=0.5")
    for i, r in enumerate(rec["results"][:3], 1):
        deal = r["pred_price"] - r["price"]
        print(f"    Rank #{i}: listing {r['listing_id']}")
        print(f"       Price: {r['price']:,.0f} {ccy}  (model value "
              f"{r['pred_price']:,.2f}, so {deal:+,.2f})")
        print(f"       Instant book: {r['instant_book_prob'] * 100:.1f}%   "
              f"ML score: {r['ml_score']:.4f}")

    # What each weight is actually contributing. This is the check worth reading:
    # if the value column does not move between w_value=0 and w_value=0.5, the
    # endpoint is ranking on one model output while claiming to use two.
    print("\n  weightings   same query, the two weights swept")
    orders = {}
    for label, (wv, wc) in [("w_value=0.0", (0.0, 1.0)), ("w_value=0.5", (0.5, 0.5)),
                            ("w_value=1.0", (1.0, 0.0))]:
        r = _post_json(f"{base}/recommend", {**q, "w_value": wv, "w_convenience": wc},
                       timeout=60)
        orders[label] = [x["listing_id"] for x in r.get("results", [])][:top_n]
        print(f"    {label:<14}{orders[label]}")
    if orders["w_value=0.0"] == orders["w_value=0.5"]:
        print(f"\n    NOTE: w_value=0.5 gives the same order as w_value=0, so the value "
              f"term\n          is contributing nothing at the blended weighting. "
              f"{rec.get('warning', '')}")
    else:
        print("\n    The value term changes the order at w_value=0.5, so both model "
              "outputs\n          are reaching the ranking.")

    print("\nAPI TEST PASSED")
    return 0


def parse_users(spec: str):
    pairs = [p.split(":", 1) for p in spec.split(",") if ":" in p]
    return [(u.strip(), p.strip()) for u, p in pairs]


def main(argv=None) -> int:
    global CORPUS_CSV, USE_S3, RECSYS_BACKEND, RECSYS_API_URL, API_PORT, RECOMMEND_SCALING

    # `--similar X | head` is a normal thing to type, and python's default SIGPIPE
    # handling turns it into a traceback on stderr after correct output on stdout.
    try:
        import signal
        signal.signal(signal.SIGPIPE, signal.SIG_DFL)
    except (ImportError, AttributeError, ValueError):
        pass                                          # not POSIX, or not the main thread

    parser = argparse.ArgumentParser(description="ITI113 Airbnb pricing and discovery assistants")
    parser.add_argument("--role", choices=["host", "guest", "customer", "both"], default="both",
                        help="'customer' is accepted as an alias for 'guest'")
    parser.add_argument("--port", type=int, default=None)
    # The previous app took a --share flag and then passed share=True regardless, so
    # it always opened a public tunnel. Studio demos depend on that link, so the
    # behaviour is kept -- but it is now visible and --no-share turns it off.
    parser.add_argument("--share", action=argparse.BooleanOptionalAction, default=True,
                        help="open a public gradio.live tunnel (default: on)")
    parser.add_argument("--corpus", help="path or s3:// URI of the pre-scored corpus CSV")
    parser.add_argument("--recsys-backend", choices=["auto", "api", "inline"], default=None,
                        help="where the guest query is ranked (default auto)")
    parser.add_argument("--serve-api", action="store_true",
                        help="serve the recommender API and nothing else, then block")
    parser.add_argument("--with-api", action="store_true",
                        help="serve the API in this process alongside the UI, and rank "
                             "the guest journey through it over HTTP")
    parser.add_argument("--api-test", action="store_true",
                        help="run the API client checks, then exit")
    parser.add_argument("--api-port", type=int, default=API_PORT)
    parser.add_argument("--recsys-url", default=None,
                        help=f"base URL of the notebook's recommender API "
                             f"(default {RECSYS_API_URL})")
    parser.add_argument("--self-test", action="store_true",
                        help="exercise both journeys with the endpoints stubbed, then exit")
    parser.add_argument("--recommend", metavar="CITY",
                        help="run the guest journey from the command line, then exit")
    parser.add_argument("--guests", type=int, default=2, help="min_guests for --recommend")
    parser.add_argument("--max-price", type=float, default=150.0,
                        help="max_price for --recommend")
    parser.add_argument("--w-value", type=float, default=0.5)
    parser.add_argument("--w-convenience", type=float, default=0.5)
    parser.add_argument("--only-saved", action="store_true",
                        help="with --recommend, rank only listings hosts saved here")
    parser.add_argument("--scaling", choices=["global", "candidates"], default=None,
                        help="where /recommend's value term takes its min and max")
    parser.add_argument("--price-json", metavar="PATH",
                        help="run the host journey over a listing JSON file, then exit")
    parser.add_argument("--dump-preset", action="store_true",
                        help="print a listing JSON template for --price-json, then exit")
    parser.add_argument("--json", action="store_true",
                        help="raw JSON output for --recommend")
    parser.add_argument("--no-explain", action="store_true",
                        help="skip the driver breakdown in --price-json")
    parser.add_argument("--local-store", action="store_true",
                        help="keep saved listings in a local JSON file instead of S3")
    parser.add_argument("--list-store", action="store_true",
                        help="print every saved listing with its owner, then exit")
    parser.add_argument("--set-owner", metavar="ID:USER",
                        help="reassign ownership, e.g. PAR-001:host ('PAR-001:' to unown)")
    parser.add_argument("--reset-store", action="store_true",
                        help="delete every saved listing, then exit")
    parser.add_argument("--migrate-store", action="store_true",
                        help="rewrite saved listings in the current schema, then exit")
    parser.add_argument("--rescore-store", action="store_true",
                        help="score saved listings that have no model outputs, then exit")
    parser.add_argument("--all", action="store_true",
                        help="with --rescore-store, re-score every listing, not just gaps")
    parser.add_argument("--yes", action="store_true", help="skip the reset confirmation")
    args = parser.parse_args(argv)

    if args.corpus:
        CORPUS_CSV = args.corpus
    if args.recsys_backend:
        RECSYS_BACKEND = args.recsys_backend
    if args.recsys_url:
        RECSYS_API_URL = args.recsys_url
    API_PORT = args.api_port
    if args.scaling:
        RECOMMEND_SCALING = args.scaling
    if args.local_store:
        USE_S3 = False
    if args.role == "customer":
        args.role = "guest"

    if args.self_test:
        return self_test()
    if args.serve_api and not args.api_test:
        load_fitted_state()
        print(f"{COURSE} {TEAM_ID} / {STUDENT_ID} | recommender API | region={REGION}")
        print(f"corpus   : {corpus_status()}")
        serve_api(port=API_PORT)
        return 0
    if args.api_test:
        load_fitted_state()
        if args.serve_api or args.with_api:
            # One command that stands the server up and then exercises it, so the
            # demo does not depend on remembering to start something in another tab.
            RECSYS_API_URL = f"http://127.0.0.1:{API_PORT}"
            print(f"Started {serve_api(port=API_PORT, background=True)}\n")
        return cmd_api_test()
    if args.list_store:
        return cmd_list_store()
    if args.set_owner:
        return cmd_set_owner(args.set_owner)
    if args.reset_store:
        return cmd_reset_store(args.yes)
    if args.migrate_store:
        return cmd_migrate_store()
    if args.rescore_store:
        return cmd_rescore_store(args.all)
    load_fitted_state()
    if args.dump_preset:
        print(json.dumps(host_preset(), indent=2))
        return 0
    if args.recommend:
        return cmd_recommend(args.recommend, args.guests, args.max_price,
                             args.w_value, args.w_convenience, args.json,
                             only_saved=args.only_saved)
    if args.price_json:
        return cmd_price_json(args.price_json, explain=not args.no_explain)

    price_lin = model_lineage(PRICE_ENDPOINT)
    book_lin = model_lineage(BOOKING_ENDPOINT)
    print(f"{COURSE} {TEAM_ID} / {STUDENT_ID} | role={args.role} | region={REGION}")
    print(f"price    : {PRICE_ENDPOINT} ({price_lin.get('status')}) "
          f"{len(SERVING_COLUMNS)} fields from {CONTRACT_SOURCE}")
    print(f"           {price_lin.get('model_package') or 'model package unknown'}")
    print(f"booking  : {BOOKING_ENDPOINT} ({book_lin.get('status')}) "
          f"{len(BOOKING_COLUMNS)} fields")
    print(f"           {book_lin.get('model_package') or 'model package unknown'}")
    print(f"explain  : {len(EXPLAIN_GROUPS)} groups, {2 ** len(EXPLAIN_GROUPS)} coalitions, "
          f"{'batched' if SHAP_BATCH else 'sequential'}, city held fixed")
    print(f"recommend: value term scaled over the {RECOMMEND_SCALING}")
    print(f"recsys   : {recsys_backend_name()}")
    if RECSYS_BACKEND != "api":
        print(f"corpus   : {corpus_status()}")
    print(f"store    : {store_backend_name()} ({len(store_read())} saved)")

    # The corpus was scored by whatever endpoint notebook 05 pointed at and carries no
    # stamp of its own, so pred_price can silently belong to a model the price tab no
    # longer serves. The host journey scores live and cannot drift; the guest journey
    # reads pre-computed columns and can.
    print("           note: corpus pred_price and instant_book_prob are unstamped. "
          "Re-run notebook 05 after any redeploy.")

    if args.with_api:
        # The guest journey then genuinely goes over HTTP -- the same call a platform's
        # search service would make -- rather than short-cutting to a function call in
        # the same process. One command, one process, nothing to keep running in
        # another tab.
        RECSYS_API_URL = f"http://127.0.0.1:{API_PORT}"
        RECSYS_BACKEND = "api"
        print(f"api      : {serve_api(port=API_PORT, background=True)}")

    users = parse_users(HOST_USERS)
    if args.role == "guest":
        demo, port, auth = build_guest_ui(), args.port or 7862, None
    elif args.role == "host":
        demo, port, auth = build_host_ui(), args.port or 7861, users
    else:
        print("\nDEMO MODE: both journeys in one process behind one login. This is NOT a")
        print("trust boundary -- anyone who can reach the app can open the host tab. Run")
        print("--role host and --role guest separately for a split the process enforces.\n")
        demo, port, auth = build_both_ui(), args.port or 7859, users

    print(f"auth     : {'enabled for ' + str([u for u, _ in auth]) if auth else 'none (public read-only view)'}")
    demo.launch(server_name="0.0.0.0", server_port=port, share=args.share,
                auth=auth, show_error=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
