"""SageMaker Training Job — fits the full production pipeline for the price regressor.

Parameterized: champion hyperparameters (Notebook 02 tuning winner) are the defaults,
overridable per execution. Reads cleaned splits from /processed, fits the serializable
raw-rows -> price Pipeline from pipeline_lib, evaluates with the Notebook 02 metric
suite, prints regex-capturable metrics for the SageMaker quality gate, and saves
model_pipeline.joblib + evaluation_report.json to the model dir (-> /artifacts).

MLflow logging is intentionally performed OUTSIDE this container, by the notebook,
after a successful pipeline execution.
"""
import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
for p in (HERE, "/opt/ml/processing/input/code"):
    if p not in sys.path:
        sys.path.append(p)

import joblib
import numpy as np
import pandas as pd
from pipeline_lib import (SERVING_COLUMNS, apply_price_caps, build_model_pipeline,
                          evaluate_predictions, fit_price_caps, __version__ as LIB_VERSION)

parser = argparse.ArgumentParser()
# --- champion hyperparameters (Notebook 02 tuning winner) — all overridable
parser.add_argument("--learning-rate", type=float, default=0.10)
parser.add_argument("--max-leaf-nodes", type=int, default=127)
parser.add_argument("--max-iter", type=int, default=300)
parser.add_argument("--min-samples-leaf", type=int, default=50)
parser.add_argument("--l2-regularization", type=float, default=1.0)
parser.add_argument("--random-state", type=int, default=42)
parser.add_argument("--gate-r2", type=float, default=0.65)
# --- run metadata (kept in SageMaker training-job hyperparameters)
parser.add_argument("--team-id", type=str, default=os.environ.get("TEAM_ID", "unknown-team"))
parser.add_argument("--student-id", type=str, default=os.environ.get("STUDENT_ID", "s000"))
# --- SageMaker channel paths (local runs override these)
parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN", "/opt/ml/input/data/train"))
parser.add_argument("--test", type=str, default=os.environ.get("SM_CHANNEL_TEST", "/opt/ml/input/data/test"))
parser.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
args = parser.parse_args()
os.makedirs(args.model_dir, exist_ok=True)

HPARAMS = {"learning_rate": args.learning_rate, "max_leaf_nodes": args.max_leaf_nodes,
           "max_iter": args.max_iter, "min_samples_leaf": args.min_samples_leaf,
           "l2_regularization": args.l2_regularization}
print(f"=== Training environment (pipeline_lib v{LIB_VERSION}) ===")
print(f"train: {args.train} | test: {args.test} | model_dir: {args.model_dir}")
print(f"hyperparameters: {HPARAMS} | random_state: {args.random_state}")

train_df = pd.read_csv(os.path.join(args.train, "train.csv"), low_memory=False)
test_df = pd.read_csv(os.path.join(args.test, "test.csv"), low_memory=False)
print(f"loaded train {train_df.shape} | test {test_df.shape}")

# Label hygiene (training labels only): per-city p99.5 winsorisation, caps fitted on TRAIN
price_caps = fit_price_caps(train_df["price"], train_df["city"])
y_train = apply_price_caps(train_df["price"], train_df["city"], price_caps)

pipe = build_model_pipeline(HPARAMS, random_state=args.random_state)
t0 = time.time()
pipe.fit(train_df[SERVING_COLUMNS], y_train)
fit_seconds = round(time.time() - t0, 1)
print(f"pipeline fitted on {len(train_df):,} rows in {fit_seconds}s")

pred_test = pipe.predict(test_df[SERVING_COLUMNS])
pooled, per_city = evaluate_predictions(pred_test, test_df["price"], test_df["city"], price_caps)
pred_train = pipe.predict(train_df[SERVING_COLUMNS])
pooled_train, _ = evaluate_predictions(pred_train, train_df["price"], train_df["city"], price_caps)
train_metrics = {k.replace("test_", "train_"): v for k, v in pooled_train.items()}

# --- metric lines captured by SageMaker metric_definitions regexes (quality gate)
for k, v in pooled.items():
    print(f"{k}: {v}")
for k, v in train_metrics.items():
    print(f"{k}: {v}")
gate_passed = pooled["test_r2_log"] >= args.gate_r2
print(f"quality_gate_r2log: threshold={args.gate_r2} -> {'PASS' if gate_passed else 'FAIL'}")
print("\nPer-city evaluation (local currency):")
print(per_city.to_string())

joblib.dump(pipe, os.path.join(args.model_dir, "model_pipeline.joblib"))
report = {
    "hyperparameters": HPARAMS, "random_state": args.random_state,
    "fit_seconds": fit_seconds,
    "metrics": {**pooled, **train_metrics},
    "per_city": per_city.reset_index().to_dict(orient="records"),
    "price_caps": {k: float(v) for k, v in price_caps.items()},
    "quality_gate": {"metric": "test_r2_log", "threshold": args.gate_r2, "passed": bool(gate_passed)},
    "n_train": int(len(train_df)), "n_test": int(len(test_df)),
    "pipeline_lib_version": LIB_VERSION,
    "team_id": args.team_id, "student_id": args.student_id,
    "created_utc": datetime.now(timezone.utc).isoformat(),
}
with open(os.path.join(args.model_dir, "evaluation_report.json"), "w") as f:
    json.dump(report, f, indent=2)
print(f"\nsaved model_pipeline.joblib + evaluation_report.json -> {args.model_dir}")
