"""S3 -> EventBridge -> SageMaker Pipeline trigger for the Airbnb price regressor.

Env vars: PIPELINE_NAME, KEY_PREFIX (e.g. iti113/team14/trigger/input/), GATE_R2 (optional).
IAM: sagemaker:StartPipelineExecution on the pipeline ARN, s3:GetObject on the watched
prefix (for the schema pre-check), CloudWatch Logs basics.
"""
import hashlib
import json
import os

import boto3

sm = boto3.client("sagemaker")
s3 = boto3.client("s3")

PIPELINE_NAME = os.environ["PIPELINE_NAME"]
KEY_PREFIX = os.environ.get("KEY_PREFIX", "")
GATE_R2 = os.environ.get("GATE_R2", "0.65")

# The serving contract's raw header — a 4-KB range read rejects schema drift
# before a single container starts (see the May incident in the runbook).
REQUIRED_FIELDS = {
    "city", "latitude", "longitude", "neighbourhood", "property_type", "room_type",
    "accommodates", "bedrooms", "amenities", "minimum_nights", "maximum_nights",
    "instant_bookable", "host_since", "price",
}


def handler(event, context):
    detail = event.get("detail", {})
    bucket = detail.get("bucket", {}).get("name")
    key = detail.get("object", {}).get("key", "")
    etag = detail.get("object", {}).get("etag", "")
    print(json.dumps({"received": {"bucket": bucket, "key": key, "etag": etag}}))

    # --- guards: watched prefix, CSV suffix, non-empty
    if not key.startswith(KEY_PREFIX) or not key.endswith(".csv"):
        print(json.dumps({"skipped": key, "reason": "outside contract (prefix/suffix)"}))
        return {"status": "skipped"}
    if int(detail.get("object", {}).get("size", 0)) == 0:
        print(json.dumps({"skipped": key, "reason": "empty object"}))
        return {"status": "skipped"}

    # --- schema pre-check: header sniff via a ranged GET (cheap, container-free)
    head = s3.get_object(Bucket=bucket, Key=key, Range="bytes=0-4095")["Body"].read()
    header = set(head.decode("utf-8", errors="replace").splitlines()[0].split(","))
    missing = REQUIRED_FIELDS - header
    if missing:
        print(json.dumps({"rejected": key, "reason": "schema drift",
                          "missing_fields": sorted(missing)}))
        return {"status": "rejected", "missing": sorted(missing)}

    # --- idempotency: one execution per object VERSION (key + etag), at-least-once safe
    token = "s3" + hashlib.sha1(f"{key}|{etag}".encode()).hexdigest()[:30]
    input_uri = f"s3://{bucket}/{key}"
    resp = sm.start_pipeline_execution(
        PipelineName=PIPELINE_NAME,
        PipelineExecutionDisplayName=token,
        ClientRequestToken=token,
        PipelineParameters=[
            {"Name": "InputDataUrl", "Value": input_uri},
            {"Name": "QualityGateR2Log", "Value": GATE_R2},
        ],
    )
    print(json.dumps({"started": resp["PipelineExecutionArn"], "input": input_uri}))
    return {"status": "started", "execution_arn": resp["PipelineExecutionArn"]}
