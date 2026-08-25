"""
SageMaker Processing Job — champion-vs-baseline evaluation and gate report.

Writes ``evaluation.json``, which is consumed twice:

  1. by the pipeline's ConditionStep, through a ``PropertyFile`` (JsonGet reads
     scalars out of it to decide whether to register at all), and
  2. by the ModelStep, as ``ModelMetrics`` attached to the registered package, so
     the numbers that justified the promotion are permanently bound to the
     version rather than living in a notebook someone will re-run.

Evaluating both models in one job — rather than trusting the metrics each training
job printed about itself — means the comparison is guaranteed apples-to-apples:
same test rows, same metric code, same process.
"""

import json
import os
import sys
import tarfile

import numpy as np
import pandas as pd

sys.path.insert(0, "/opt/ml/processing/code")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import joblib  # noqa: E402
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score  # noqa: E402

from preprocessing import CURRENCY, PriceTargetTransformer  # noqa: E402

TEST_DIR = "/opt/ml/processing/test"
OUT_DIR = "/opt/ml/processing/evaluation"
os.makedirs(OUT_DIR, exist_ok=True)


def load_model(model_dir):
    """SageMaker delivers training output as model.tar.gz; unpack then load."""
    tar_path = os.path.join(model_dir, "model.tar.gz")
    if os.path.exists(tar_path):
        with tarfile.open(tar_path) as tar:
            tar.extractall(path=model_dir)
    return joblib.load(os.path.join(model_dir, "model.joblib"))


def score(artifact, frame, price_raw):
    pipeline = artifact["pipeline"]
    target_tf = artifact["target_transformer"]
    y_log = target_tf.transform(price_raw, frame["city"].values)
    pred_log = pipeline.predict(frame)
    pred_raw = PriceTargetTransformer.inverse_transform(pred_log)
    return {
        "mae_log": round(float(mean_absolute_error(y_log, pred_log)), 4),
        "rmse_log": round(float(np.sqrt(mean_squared_error(y_log, pred_log))), 4),
        "r2_log": round(float(r2_score(y_log, pred_log)), 4),
        "mape": round(float(np.mean(np.abs(pred_raw - price_raw) / price_raw)) * 100, 2),
    }, pred_log, pred_raw, y_log


test_df = pd.read_csv(os.path.join(TEST_DIR, "test_raw.csv"), low_memory=False)
price_test = test_df["price"].values.astype("float64")
print(f"Evaluating on {len(test_df):,} held-out listings")

champion = load_model("/opt/ml/processing/champion")
baseline = load_model("/opt/ml/processing/baseline")

champ_metrics, pred_log, pred_raw, y_log = score(champion, test_df, price_test)
base_metrics, _, _, _ = score(baseline, test_df, price_test)

print("champion:", champ_metrics)
print("baseline:", base_metrics)

# ------------------------------------------------------------ per city
rows = []
for c in sorted(test_df["city"].dropna().unique()):
    m = (test_df["city"] == c).values
    rows.append({
        "city": c, "currency": CURRENCY.get(c, "?"), "n": int(m.sum()),
        "mae_local": round(float(mean_absolute_error(price_test[m], pred_raw[m])), 1),
        "mape": round(float(np.mean(np.abs(pred_raw[m] - price_test[m]) / price_test[m])) * 100, 1),
        "r2_log": round(float(r2_score(y_log[m], pred_log[m])), 3),
    })
per_city = pd.DataFrame(rows)
print(per_city.to_string(index=False))

# ------------------------------------------- calibration across the range
resid = y_log - pred_log
within = lambda f: round(float(np.mean(np.abs(resid) < np.log(1 + f)) * 100), 1)

uplift = round(base_metrics["mae_log"] - champ_metrics["mae_log"], 4)

# SageMaker's model-quality schema keeps the registry UI readable; the flat
# custom block is what the ConditionStep actually reads.
evaluation = {
    "regression_metrics": {
        "mae_log": {"value": champ_metrics["mae_log"]},
        "rmse_log": {"value": champ_metrics["rmse_log"]},
        "r2_log": {"value": champ_metrics["r2_log"]},
        "mape": {"value": champ_metrics["mape"]},
    },
    "gate": {
        "test_r2_log": champ_metrics["r2_log"],
        "test_mae_log": champ_metrics["mae_log"],
        "baseline_mae_log": base_metrics["mae_log"],
        "mae_log_uplift_vs_baseline": uplift,
        "worst_city_r2_log": round(float(per_city["r2_log"].min()), 4),
        "worst_city_mape": round(float(per_city["mape"].max()), 2),
    },
    "champion": champ_metrics,
    "baseline": base_metrics,
    "calibration": {"within_10_pct": within(0.10), "within_25_pct": within(0.25),
                    "within_50_pct": within(0.50)},
    "per_city": per_city.to_dict(orient="records"),
    "n_test_rows": int(len(test_df)),
    "champion_metadata": champion["metadata"],
    "baseline_metadata": baseline["metadata"],
}

with open(os.path.join(OUT_DIR, "evaluation.json"), "w") as f:
    json.dump(evaluation, f, indent=2)
per_city.to_csv(os.path.join(OUT_DIR, "per_city_metrics.csv"), index=False)

print("\n=== GATE INPUTS ===")
print(json.dumps(evaluation["gate"], indent=2))
print(f"Predictions within +/-25% of the advertised rate: {within(0.25)}%")
