# hotel-cancellation-mlops

Predict whether a hotel booking will be canceled, and build the MLOps system around the model:
automated training pipeline, experiment tracking and model registry, containerized prediction API,
and drift monitoring. Course project for UChicago MS-ADS, ADSP 32021 (Machine Learning Operations
and Inference Engineering), Autumn 2026.

Status: work in progress. See `docs/` as it is filled in.

## Quickstart (stages implemented so far)

Requires Python 3.12 or newer.

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt

python -m src.ingestion.download      # data/raw/hotels.csv, verified against a pinned SHA-256
python -m src.data.make_splits        # train / val / holdout test by arrival date
python -m src.data.make_splits --check  # recompute and compare with docs/split_manifest.json
pytest -q && ruff check .
```

Split (by arrival date): train up to 2016-12-31, validation 2017-01-01 to 2017-03-31,
test 2017-04-01 onward. The test file is written to `data/holdout/` and is not read by any
code before production validation. `data/` is not committed.
