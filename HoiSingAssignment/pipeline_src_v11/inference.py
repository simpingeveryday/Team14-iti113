"""SageMaker inference handler — Serverless endpoint for the Airbnb price regressor.

Deserializes the SAME fitted pipeline that training serialized (train-serve
consistency by construction). Accepts raw listing JSON on the SERVING_COLUMNS
contract; returns the suggested fair advertised nightly rate in local currency.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.append(HERE)

import joblib
import numpy as np
import pandas as pd
from pipeline_lib import CURRENCY, SERVING_COLUMNS


def model_fn(model_dir):
    return joblib.load(os.path.join(model_dir, "model_pipeline.joblib"))


def input_fn(body, content_type="application/json"):
    if content_type != "application/json":
        raise ValueError(f"Unsupported content type: {content_type}")
    payload = json.loads(body)
    if isinstance(payload, dict):
        payload = [payload]
    df = pd.DataFrame(payload)
    missing = [c for c in SERVING_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required raw listing fields: {missing}")
    return df[SERVING_COLUMNS]


def predict_fn(data, model):
    prices = model.predict(data)          # pipeline returns PRICE units (expm1 inside)
    return data["city"].tolist(), prices


def output_fn(prediction, accept="application/json"):
    cities, prices = prediction
    response = [{
        "suggested_nightly_price": round(float(p), 2),
        "currency": CURRENCY.get(c, "unknown"),
        "city": c,
        "log_price": round(float(np.log1p(p)), 4),
    } for c, p in zip(cities, prices)]
    return json.dumps(response), accept
