# Conversation Log — Baseline Models in the Airbnb Gradio Demo

**Date:** 24 August 2026
**Participants:** User · Claude
**Project context:** ITI113 / team14 — Airbnb price regression + instant-booking classification on SageMaker (serverless endpoints, SageMaker MLflow App `app-PIGMOQJH46PS`, region `ap-southeast-1`)

**Files discussed**

| File | Role |
|---|---|
| `04_gradio_four_model_serverless_demo (1).ipynb` | Gradio UI: price champion, instant-booking champion, recommender, Ridge baseline |
| `02_baseline_experiments_sagemaker_mlflow_app_(with_team_tag_check).ipynb` | Classification baseline experiments (LogisticRegression, RF, RF tuning) |
| `04_gradio_five_model_serverless_demo.ipynb` | **Deliverable** — notebook 04 updated with a Baseline Logistic (classification) tab, Sections 6.8a–6.8c |

---

## Quick reference — key facts established

- **Ridge regression baseline** (already in notebook 04, Section 6.7): run `team14_s1402_ridge_baseline`, run ID `60fdc0210519493aa5e23d1c77703819`, loaded from `runs:/60fdc021…/model` on the MLflow App. Bare `RidgeCV` (alpha 0.01) fitted on the 88-column matrix; scored **in-process** using the price champion's fitted preprocessor pulled from the deployed pipeline (`Pipeline[:-1]`). Reported MAE(log) 0.4255 on *validation*.
- **Logistic classification baseline** (new tab): run `team14_s1401_logistic_regression_baseline`, run ID `5449338e75254622a4d50c70beec773b`, saved **only** via `mlflow.sklearn.log_model(artifact_path="model")` — never to S3 or disk. Test metrics: Accuracy 0.7049 · ROC-AUC 0.7612 · PR-AUC 0.6895 · F1 0.6069. Fitted on the 100-column classification matrix (222,513 × 100 train).
- **Classification champion**: tuned RandomForest, parent run `6f91c874f6954ebc9b042a468429c350`, `runs:/6f91c874…/best_model` (this is what `best_model.json` hands to Notebook 03). Test: Accuracy 0.7657 · ROC-AUC 0.8383 · PR-AUC 0.8111 · F1 0.6841 — **same test split** as the logistic baseline (55,629 rows), so the comparison is like-for-like.
- **Feature bridge for the new tab**: the booking endpoint's own `create_features`, imported from its `model.tar.gz` (resolved live via endpoint → config → model → model package → `ModelDataUrl`). In this container it takes **3 required arguments**, so 6.8b self-binds them by reading `model_fn`, the tarball's JSON files, and the container's own call sites — gated by 6.8b-verify before any prediction is trusted.

---

## Transcript

### 1 · User

> *(uploaded `04_gradio_four_model_serverless_demo (1).ipynb`)*
>
> how is the baseline model being ran? does it actually do the prediction?

### Claude

*\[Read the notebook; traced Sections 6.7a–6.7c and the Gradio wiring; inspected cell execution state.\]*

Yes — it's a real `.predict()`, not a stub or a canned number. But it runs in a fundamentally different way from the other three tabs.

**The chain**

- **6.7a — load the estimator.** Three sources tried cheapest-first, first success wins: a local `ridge_baseline.joblib`, then an S3 key, then `runs:/60fdc021…/model` from the SageMaker MLflow app. In practice only the third exists, since Notebook 02's publish step mirrored only the champion to S3. Nothing raises — every attempt is appended to `BASELINE_ATTEMPTS`, and failure just leaves `BASELINE_MODEL = None`.
- **6.7b — build the 88-column matrix.** The Ridge is a bare estimator with no preprocessing attached, so the notebook pulls the champion's `model.tar.gz` off the live endpoint (endpoint → config → model → model package → `ModelDataUrl`), unpacks it, loads `model_pipeline.joblib`, and keeps `Pipeline[:-1]` — dropping the HistGradientBoosting estimator, keeping the fitted transformers. A tolerant unpickler stubs classes the newer scikit-learn can't resolve (the `__pyx_unpickle_CyHalfSquaredError` loss object), which is safe precisely because that symbol belongs to the step being discarded.
- **6.7b-verify — the gate.** Un-scales `accommodates` and `minimum_nights` from a transformed probe row using the published `mean`/`scale` in `preprocessing_artifacts.json` and checks the originals come back. Any `FAIL` sets `PREPROCESSOR = None` and disables the tab.
- **6.7c — the prediction itself.** Two real lines:

```python
X = raw_to_features(payload)                                    # 27 raw → 88 float32
log_price = float(np.asarray(BASELINE_MODEL.predict(X)).ravel()[0])
```

then `np.expm1(log_price)` for the rate. Genuine sklearn inference against the fitted coefficients, in-kernel, sub-millisecond, `served_by: "in-process (no endpoint)"`. The only hard-coded numbers anywhere near it are `BASELINE_SCORES` — display metadata quoted from Notebook 02, not used to compute the price.

**What to watch**

- **These cells have never been run in this file.** Code cells 15–22 — all four baseline cells plus the Gradio launch — have `execution_count: None` and zero outputs. Only the first 14 ran, so nothing in the file confirms the estimator actually loaded or the transform passed verification.
- **Three failure modes all look calm.** If either object is `None`, the handler returns an "unavailable" card — the tab renders, explains itself, and never predicts. If verification only *skips* (e.g. no `scaler` block in `preprocessing_artifacts.json`), the tab predicts anyway with an "unconfirmed" note.
- **The pyfunc fallback quietly degrades the safety checks.** If `mlflow.sklearn.load_model` fails and `mlflow.pyfunc.load_model` succeeds, `.predict()` still works but `feature_names_in_` comes back empty, weakening the column-order reindex; diagnostics also degrade (`PyFuncModel` instead of `RidgeCV`).
- **"No network" is conditional.** The compare checkbox defaults to `True`, so a default click does hit the champion serverless endpoint — cold start and all. The baseline half is still local.

---

### 2 · User

> so does it predict using the baseline model? where is the prediction made? in the notebook?

### Claude

Yes. It's the actual RidgeCV object doing the prediction, and it happens inside the notebook kernel — the same Python process running the Gradio app.

Concretely: **6.7a** loads the fitted estimator into memory as `BASELINE_MODEL` (in practice via `mlflow.sklearn.load_model("runs:/60fdc021…/model")` — downloaded once, then it lives in RAM). **6.7b** loads the champion's fitted preprocessor into memory as `PREPROCESSOR`. **6.7c** calls `BASELINE_MODEL.predict(X)` directly on that in-memory object.

Both the feature engineering and the inference run locally. There's no `model.tar.gz` for the Ridge, no container, no `invoke_endpoint` call — the record it returns even labels itself `"served_by": "in-process (no endpoint)"`, and the timing is a `time.perf_counter()` measurement of local compute, not a round trip.

That's the whole contrast with the other three tabs, which all end in a `sagemaker-runtime` network call to a serverless endpoint. The only network traffic the baseline tab generates is at load time (fetching the estimator and the champion tarball), plus the optional champion comparison if that checkbox is ticked. The number on that tab is real — just computed on your machine rather than AWS's.

---

### 3 · User

> why is there a ml run id RidgeCV from runs:/60fdc0210519493aa5e23d1c77703819/model?

### Claude

That hex string is an MLflow **run ID** — MLflow assigns every run a 32-character identifier, and `60fdc0210519493aa5e23d1c77703819` is the one belonging to the run named `team14_s1402_ridge_baseline` in Notebook 02 section 4. The `runs:/<run-id>/model` syntax is MLflow's own URI scheme, meaning "the artifact logged at path `model/` inside that run" — `model` because `log_sklearn_model(ridge, X_val)` used MLflow's default artifact path.

**It's there because that's the only place the Ridge exists:**

| | Champion | Ridge baseline |
|---|---|---|
| MLflow | registered model `iti113-team14-airbnb-price-regressor` v7, alias `champion` | logged to a run, never registered |
| S3 | `model.tar.gz` (+ the 8A mirror) | nothing |
| SageMaker | approved model package, serverless endpoint | nothing |

The champion went through the full promotion path; the baseline got one line of logging — which makes sense for something whose job was to be a number to beat, not something to serve. It's the *third* fallback because it's the slowest and least portable source (network, IAM to that specific MLflow App ARN, MLflow client).

**What's brittle about it:** a run URI is an opaque, non-semantic pointer — no alias, no version, no "latest". Re-run Notebook 02 section 4 and a new run ID gets minted while the hardcoded constant keeps pointing at the old artifact, quietly stale. It's also single-copy: delete the MLflow App and the estimator is gone. Section 11 of the notebook says this outright and ships `nb02_publish_ridge_baseline.py` to mirror the joblib to S3, after which `_from_s3()` wins and the run ID becomes a fallback of a fallback. The lineage-parity alternative: register the Ridge as a model version and load `models:/iti113-team14-airbnb-price-regressor@baseline` — a stable name instead of a hash.

---

### 4 · User

> *(uploaded both notebooks)*
>
> **Context:** I am working with two Jupyter notebooks: `02_baseline_experiments_sagemaker_mlflow_app_(with_team_tag_check)` (contains my baseline logistic Regression model). `04_gradio_four_model_serverless_demo (1).ipynb` (contains my current Gradio UI running a champion classification model).
> **Objective:** I need to update the Gradio notebook to include a new tab that runs live inference using the baseline Ridge Regression model. This tab should mirror the exact functionality of the champion model tab so users can compare the two and see the improvement.
> **Constraints & Implementation Details:** The baseline logistic regression model is not deployed as an endpoint. Please analyze notebook 02 to determine how the baseline model was saved. If it was saved to S3 or locally, write the necessary code in notebook 04 to download/load the model artifact (e.g., using boto3 for S3 or joblib/pickle for local storage). If the storage path is unclear, provide clear placeholder code where I can input my S3 bucket/file path.
> **Deliverable:** Generate the updated Python code for `04_gradio_four_model_serverless_demo (1).ipynb` with the fully integrated baseline model tab.

### Claude

*\[Analyzed notebook 02 (classification): 20 executed cells, 100-feature matrix, LR baseline run + metrics, RF baseline, RF tuning, best-model hand-off. Cross-checked notebook 04's booking-tab contract, then built and validated the updated notebook programmatically.\]*

One correction first, because it changes what needed building: **notebook 04 already has the Ridge regression baseline tab** (Section 6.7, "Baseline Ridge (regression)"), and the uploaded notebook 02 contains no Ridge at all — it's the *classification* baseline notebook (target `instant_bookable`), whose baseline is the **LogisticRegression** from Run 1. The genuinely missing piece is a baseline tab for the *booking champion* — so that's what was built: a **Baseline Logistic (classification)** tab mirroring the Instant-booking tab, sitting right after it.

**How the baseline was saved:** MLflow only. Run 1 persists it with `mlflow.sklearn.log_model(sk_model=model_lr, artifact_path="model")` inside run `team14_s1401_logistic_regression_baseline`, run ID `5449338e75254622a4d50c70beec773b`. It was never written to S3 or disk — the only local save in that notebook is the data CSVs, and `best_model.json` points Notebook 03 at the tuned RF (`6f91c874…`) instead. The loader therefore tries local file → S3 → MLflow, cheapest first, with the first two as clearly marked placeholders (`LRB_LOCAL_CANDIDATES`, `LRB_S3_KEY`) and the run URI as the source that exists today.

**What's in the updated notebook — Sections 6.8a–6.8c, mirroring the 6.7 pattern:**

- **6.8a** loads the estimator (three sources, recorded attempts, nothing raises; pyfunc fallback if the sklearn flavour fails).
- **6.8b** solves the real problem: the LR was fitted on the **100-column** matrix (not the price model's 88), and the raw→100 transform is *code*, not a fitted pickle — the `create_features` inside the booking endpoint's own `model.tar.gz`. The cell resolves that tarball from the live endpoint via the same lineage walk as 6.7b, extracts it safely, and imports `create_features` directly, so both classifiers eat byte-identical features and the tabs differ in exactly one component. `BOOKING_CREATE_FEATURES_OVERRIDE` is the escape hatch.
- **6.8b-verify** gates it. No published scaler stats exist for this matrix, so the vetoing checks are structural: the 100 columns must be *earned* rather than manufactured by `reindex(fill_value=0)` (the zero-fill count is tracked precisely because Section 5.3's silent-zero-fill hazard could otherwise mask a wrong import), no NaNs, and two different hosts must produce different matrices. Raw-scale spot checks (`host_tenure_days`, `accommodates_per_bedroom`) warn without vetoing. Any hard FAIL disables the tab rather than letting it quote an indefensible probability.
- **6.8c + the tab** reuse `build_booking_payload`, `validate_booking`, and the derived-features preview unchanged, render the same probability bar, and the compare checkbox posts the identical payload to the champion endpoint for a side-by-side card. Better than the Ridge card: LR (ROC-AUC 0.7612, F1 0.6069) and the tuned RF (0.8383, 0.6841) are quoted from the **same test split**, so the delta is a fair subtraction — and the card says so.

The header now reads five tabs, with a `logistic baseline:` status line. Caveats: the new cells ship unexecuted (as 6.7's did) — run top-to-bottom in Studio; and if the Ridge tab was what was wanted, it already existed.

**Deliverable:** `04_gradio_five_model_serverless_demo.ipynb`

---

### 5 · User

> ```
> ### Baseline unavailable
> - **Transform**: model.tar.gz reached, but no importable create_features — see the attempts below
>
> Load attempts:
>   - `_lrb_from_local_file` -> not found
>   - `_lrb_from_s3` -> not found
>   - `_lrb_from_mlflow` -> loaded from runs:/5449338e75254622a4d50c70beec773b/model
>   - `inference.py` -> create_features wants 3 required args, not just a DataFrame —
>     set BOOKING_CREATE_FEATURES_OVERRIDE
>
> Fix it in Section 6.8a/6.8b: point `LRB_S3_KEY` at your mirror, add a path to
> `LRB_LOCAL_CANDIDATES`, check that `LRB_RUN_ID` and the MLflow URI match the run that
> logged the baseline, or supply `BOOKING_CREATE_FEATURES_OVERRIDE`. The other four
> tabs are unaffected.
> ```

### Claude

*\[Rewrote cell 6.8b as a self-binding loader; unit-tested it against three mock container layouts; fixed a numpy-array truthiness bug the tests exposed; rebuilt and re-validated the notebook.\]*

Good news in that error report: two of the three pieces worked — the estimator loaded from MLflow, and the tarball walk found `inference.py`. The refusal was deliberate: the container's `create_features` wants **three** arguments, and calling a 3-arg function with guessed values is exactly the "predicts happily and is wrong" failure this notebook keeps defending against. The old cell just gave up too early.

The updated notebook replaces **cell 6.8b only** (plus its markdown intro) with a self-binding version that reads the container's own wiring to discover what those three arguments should be:

1. **Records the signature** — the output now shows the actual parameter names, e.g. `create_features(df, artifacts, final_feature_columns)`.
2. **Builds a pantry of candidate values** — whatever `model_fn(model_dir)` returns (the tarball *is* the model dir, so this reproduces the endpoint's own startup), every JSON file in the tarball, and the module's globals.
3. **Scans the container's own call sites** — the line in `inference.py` where *it* calls `create_features(...)` is the most reliable witness to what belongs in each slot, so those expressions are tried too.
4. **Binds each required parameter by name** — exact pantry match first, then the call-site expression, then name heuristics (`*column*/*col*/*feature*` → the column list, `*artifact*/*stats*` → the JSON blob, `*snapshot*/*date*` → `BOOKING_SNAPSHOT`, `*dir*/*path*` → the unpacked directory).

Every binding is printed, so the cell's output shows exactly what each parameter received and where it came from — and the binding is *provisional*: 6.8b-verify still gates it (earned-columns, NaN, input-sensitivity), so a wrong binding disables the tab instead of quoting a bad probability.

Tested against three mock containers shaped like the realistic possibilities — bundle-dict + exact names; a broken `model_fn` with oddly named params and the column list buried inside `preprocessing_artifacts.json`; and a snapshot-date parameter. All three bind and pass verification. (Testing also caught a real bug in the earlier code: a numpy-array truthiness error that was silently discarding the fitted column order.)

**Next step:** re-run cells 6.8b → 6.8b-verify → 6.8c, then relaunch the UI cell. If binding still can't resolve a parameter, the diagnostic prints the signature and the container's own call site verbatim, and the override becomes a one-liner of the shape `BOOKING_CREATE_FEATURES_OVERRIDE = lambda df: create_features(df, <artifacts>, <columns>)` with the two blanks read straight off that printout.

**Deliverable (superseded in place):** `04_gradio_five_model_serverless_demo.ipynb`

---

### 6 · User

> create the md file of the conversation

### Claude

Produced this document: `conversation_log_baseline_model_tabs.md`.

---

*End of log.*
