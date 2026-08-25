#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Host / customer prototype on top of the ITI113 Airbnb price endpoint
====================================================================

Two audiences, one model, one shared listing store.

    HOST      fills a listing, gets the model's suggested nightly rate, sets an
              asking price, and saves. Writes are authenticated.
    CUSTOMER  browses saved listings and sees a *fairness verdict* -- how the
              host's asking price compares with what the model expects for a
              listing like this. Read only, no login.

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
    python host_customer_app.py --role both       # single-process demo    :7860
    python host_customer_app.py --self-test       # no UI, no AWS needed

    HOST_USERS='alice:pw1,bob:pw2' python host_customer_app.py --role host

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

BUCKET = os.environ.get("BUCKET", "nyp-26s1-iti113")
PREFIX = f"iti113/{TEAM_ID}/data/airbnb-listings"
PROCESSED_PREFIX = f"{PREFIX}/processed"
ARTIFACTS_PREFIX = f"{PREFIX}/artifacts"

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

_LINEAGE_CACHE: dict[str, Any] = {}


def model_lineage(endpoint_name: str = PRICE_ENDPOINT, refresh: bool = False) -> dict:
    # Always hand back a COPY. Callers persist this into the store, and a cached
    # dict returned by reference would let a later refresh silently rewrite the
    # stamp on every record already saved -- defeating the whole point of it.
    if _LINEAGE_CACHE and not refresh:
        return dict(_LINEAGE_CACHE)
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
    _LINEAGE_CACHE.clear()
    _LINEAGE_CACHE.update(out)
    return dict(out)


def invoke_price(payload: dict) -> tuple[dict, float]:
    t0 = time.perf_counter()
    resp = runtime.invoke_endpoint(EndpointName=PRICE_ENDPOINT, ContentType="application/json",
                                   Body=json.dumps(payload))
    records = json.loads(resp["Body"].read())
    return records[0], (time.perf_counter() - t0) * 1000.0


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


def save_listing(listing_id: str, payload: dict, asking_price: float,
                 prediction: dict | None, username: str) -> dict:
    store = store_read()
    existing = store.get(listing_id)
    if existing and existing.get("owner") not in (None, username):
        raise PermissionError(f"{listing_id} belongs to another host.")

    store[listing_id] = {
        "payload": payload,
        "asking_price": float(asking_price),
        "prediction": prediction,
        "model": model_lineage(),
        "owner": existing.get("owner") if existing else username,
        "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "updated_by": username,
    }
    store_write(store)
    return store[listing_id]


def delete_listing(listing_id: str, username: str) -> str:
    store = store_read()
    rec = store.get(listing_id)
    if rec is None:
        return f"No listing `{listing_id}`."
    if rec.get("owner") not in (None, username):
        return f"`{listing_id}` belongs to another host; not deleted."
    del store[listing_id]
    store_write(store)
    return f"Deleted `{listing_id}`."


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


def host_suggest(*values):
    ui = dict(zip(HOST_FIELDS, values))
    try:
        payload = build_payload(ui)
    except (TypeError, ValueError) as e:
        return f"<div class='card'>Cannot build the request: {e}</div>", "", None
    problems = validate_host(ui, payload)
    if problems:
        bullets = "".join(f"<li>{p}</li>" for p in problems)
        return f"<div class='card'><div class='eyebrow'>Fix these</div><ul>{bullets}</ul></div>", \
               json.dumps(payload, indent=2), None
    try:
        pred, ms = invoke_price(payload)
    except Exception as e:
        return f"<div class='card'>{explain_error(e)}</div>", json.dumps(payload, indent=2), None
    return _suggestion_card(pred, ms), json.dumps(payload, indent=2), pred


def host_save(pred_state, *values, request: gr.Request = None):
    # Returns (status markdown, refreshed "your listings" table).
    ui = dict(zip(HOST_FIELDS, values))
    username = getattr(request, "username", None) or "anonymous"
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
    username = getattr(request, "username", None) or "anonymous"
    return delete_listing((listing_id or "").strip(), username)


def host_my_listings(request: gr.Request = None):
    username = getattr(request, "username", None) or "anonymous"
    store = store_read()
    mine = {k: v for k, v in store.items() if v.get("owner") in (None, username)}
    if not mine:
        return "No listings saved yet."
    rows = ["| listing | asking | model | gap | updated |", "|---|---:|---:|---:|---|"]
    for lid, rec in sorted(mine.items()):
        pred = rec.get("prediction") or {}
        sug = pred.get("suggested_nightly_price")
        gap = (f"{(rec['asking_price'] - sug) / sug * 100:+.0f}%" if sug else "-")
        rows.append(f"| `{lid}` | {rec['asking_price']:,.0f} | "
                    f"{sug:,.0f} | {gap} | {rec['updated_at'][:16]} |" if sug else
                    f"| `{lid}` | {rec['asking_price']:,.0f} | - | - | {rec['updated_at'][:16]} |")
    return "\n".join(rows)


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

        Fill in the listing, get the model's suggested nightly rate, set your asking price,
        and save. Five fields never reach customers: your exact coordinates, your join date,
        your acceptance rate and your total listing count.
    """)
    h: dict[str, Any] = {}
    with gr.Row():
        with gr.Column(scale=3):
            with gr.Row():
                h["listing_id"] = gr.Textbox(label="Listing ID", value=preset["listing_id"],
                                             info="3-32 chars. Used as the store key.")
                h["asking_price"] = gr.Number(label="Your asking price", value=preset["asking_price"],
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

        with gr.Column(scale=2):
            suggest_btn = gr.Button("Get model suggestion", variant="secondary", size="lg")
            save_btn = gr.Button("Save listing", variant="primary", size="lg")
            suggestion = gr.HTML("<div class='card'><div class='eyebrow'>No suggestion yet"
                                 "</div><div class='meta'>The first call after an idle period "
                                 "is slow: the endpoint scales to zero.</div></div>")
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
    suggest_btn.click(host_suggest, inputs, [suggestion, request_view, pred_state])
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
        host_body()
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


def build_customer_ui() -> gr.Blocks:
    with gr.Blocks(title=f"{COURSE} {TEAM_ID} - browse") as demo:
        inject_css()
        customer_body()
    return demo


def build_both_ui() -> gr.Blocks:
    """Host and customer in one Blocks. Not a trust boundary -- see the module docstring."""
    with gr.Blocks(title=f"{COURSE} {TEAM_ID} - listing app (demo mode)") as demo:
        inject_css()
        gr.Markdown(f"# {COURSE} {TEAM_ID} - listing app (demo mode)")
        with gr.Tabs():
            with gr.Tab("Host"):
                host_body()
            with gr.Tab("Customer"):
                customer_body()
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
    _LINEAGE_CACHE.update({"endpoint": PRICE_ENDPOINT, "model_package": "arn:.../models/8",
                           "model": "m-2026", "status": "InService"})
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
    assert rec["model"] is not _LINEAGE_CACHE, "stamp aliases the live cache"
    _LINEAGE_CACHE["model_package"] = "arn:.../models/9"          # simulate a redeploy
    assert rec["model"]["model_package"] == "arn:.../models/8", "stamp mutated under us"
    assert "re-save" in (verdict_is_stale(rec) or "")
    print("staleness       : stamp is immutable; a redeploy is detected and surfaced")

    try:
        save_listing("PAR-001", payload, 99.0, pred, "mallory")
        raise SystemExit("ownership check failed")
    except PermissionError:
        print("ownership       : a second host cannot overwrite alice's listing")

    assert delete_listing("PAR-001", "mallory").endswith("not deleted.")
    assert "Deleted" in delete_listing("PAR-001", "alice")
    assert store_read() == {}
    print("delete          : owner-only, store empty again")

    os.path.exists(STORE_LOCAL) and os.remove(STORE_LOCAL)
    print("\nSELF TEST PASSED")
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
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)

    if args.self_test:
        return self_test()

    global USE_S3
    if args.local_store:
        USE_S3 = False

    load_fitted_state()
    lineage = model_lineage()
    print(f"{COURSE} {TEAM_ID} / {STUDENT_ID} | role={args.role} | region={REGION}")
    print(f"endpoint : {PRICE_ENDPOINT} ({lineage.get('status')})")
    print(f"model    : {lineage.get('model_package') or 'unknown'}")
    print(f"store    : {store_backend_name()}")
    print(f"contract : {len(SERVING_COLUMNS)} fields from {CONTRACT_SOURCE}")
    print(f"listings : {len(store_read())}")

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
        port, auth = args.port or 7860, users

    if auth:
        print(f"auth     : enabled for {[u for u, _ in auth]}")
    else:
        print("auth     : none (read-only customer view)")

    demo.launch(server_name="0.0.0.0", server_port=port, share=True,
                auth=auth, show_error=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
