"""SageMaker Processing Job — versioned cleaning + split for the Airbnb price regressor.

Reads the immutable raw extract from /raw, applies the Notebook 01 auditable cleaning
sequence (via pipeline_lib), performs the city-stratified split, and writes the
versioned cleaned datasets + a data manifest to /processed.
"""
import argparse
import json
import os
import sys
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
for p in (HERE, "/opt/ml/processing/input/code"):
    if p not in sys.path:
        sys.path.append(p)

import pandas as pd
from sklearn.model_selection import train_test_split
from pipeline_lib import clean_listings, get_data_version, __version__ as LIB_VERSION

parser = argparse.ArgumentParser()
parser.add_argument("--input", type=str, default="/opt/ml/processing/input/Listings.csv")
parser.add_argument("--output-dir", type=str, default="/opt/ml/processing/output")
parser.add_argument("--test-size", type=float, default=0.20)
parser.add_argument("--random-state", type=int, default=42)
args = parser.parse_args()
os.makedirs(args.output_dir, exist_ok=True)

print(f"pipeline_lib v{LIB_VERSION} | reading {args.input}")
try:
    df_raw = pd.read_csv(args.input, encoding="utf-8", encoding_errors="replace", low_memory=False)
except TypeError:
    with open(args.input, "r", encoding="utf-8", errors="replace", newline="") as _fh:
        df_raw = pd.read_csv(_fh, low_memory=False)
data_version = get_data_version(args.input)
print(f"raw rows: {len(df_raw):,} | data_version: {data_version}")

df, audit = clean_listings(df_raw)
for a in audit:
    print(f"  clean: {a['step']:28s} -{a['rows_removed']:>4} rows  ({a['note']})")
print(f"clean rows: {len(df):,} ({len(df_raw) - len(df)} removed)")

train_df, test_df = train_test_split(df, test_size=args.test_size,
                                     random_state=args.random_state, stratify=df["city"])
train_df.to_csv(os.path.join(args.output_dir, "train.csv"), index=False)
test_df.to_csv(os.path.join(args.output_dir, "test.csv"), index=False)

manifest = {
    "data_version": data_version,
    "raw_rows": int(len(df_raw)), "clean_rows": int(len(df)),
    "train_rows": int(len(train_df)), "test_rows": int(len(test_df)),
    "test_size": args.test_size, "random_state": args.random_state,
    "stratify": "city", "audit": audit, "pipeline_lib_version": LIB_VERSION,
    "columns": df.columns.tolist(),
    "created_utc": datetime.now(timezone.utc).isoformat(),
}
with open(os.path.join(args.output_dir, "data_manifest.json"), "w") as f:
    json.dump(manifest, f, indent=2)

print(f"train: {len(train_df):,} rows | test: {len(test_df):,} rows")
print("wrote train.csv, test.csv, data_manifest.json")
