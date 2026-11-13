# hotel-cancellation-mlops

Predict whether a hotel booking will be canceled, and build the MLOps system around the model:
automated training pipeline, experiment tracking and model registry, containerized prediction API,
and drift monitoring. Course project for UChicago MS-ADS, ADSP 32021 (Machine Learning Operations
and Inference Engineering), Autumn 2026.

**Data:** Hotel Booking Demand datasets (Antonio, de Almeida, Nunes 2019, *Data in Brief* 22, 41-49,
https://doi.org/10.1016/j.dib.2018.11.126), in the combined TidyTuesday 2020-02-11 version: 119,390 bookings
from two Portuguese hotels, arrivals 2015-07-01 to 2017-08-31. Target: `is_canceled`.

## Status

| Stage | State |
|---|---|
| Data ingestion (verified download) and time-based split with an isolated test set | done |
| EDA (train and validation only), column timing audit, clean-data drift check | done, see [`docs/eda_findings.md`](docs/eda_findings.md) |
| Orchestration, experiment tracking, model registry | not started |
| Containerized API | not started |
| Monitoring and drift simulation | not started |

Requirement-by-requirement status against the syllabus: [`docs/syllabus_checklist.md`](docs/syllabus_checklist.md).

Main EDA results (details and caveats in the findings document):

- Cancellation rate 35.8% on train + validation; non-refundable bookings are canceled 99.2% of the time.
- Some columns may only be known at or after check-in (`country`, `required_car_parking_spaces`,
  `assigned_room_type`, `booking_changes`). Removing country, parking and year costs about 0.04 PR-AUC
  with an untuned gradient-boosting model, so reported scores depend on this choice.
- Evidently's default drift settings flag 39% to 50% of columns on clean validation data, so a "30% of
  columns drifted" alert would fire on clean traffic. Alert levels must be calibrated on clean runs.

## Quickstart (stages implemented so far)

Requires Python 3.12 or newer.

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt

python -m src.ingestion.download        # data/raw/hotels.csv, verified against a pinned SHA-256
python -m src.data.make_splits          # train / val / holdout test by arrival date
python -m src.data.make_splits --check  # recompute and compare with docs/split_manifest.json
python -m src.eda.run_eda               # docs/eda_results.json and docs/figures/ (about 45 s)
pytest -q && ruff check .
```

Split (by arrival date): train up to 2016-12-31, validation 2017-01-01 to 2017-03-31, test 2017-04-01
onward. The test file is written to `data/holdout/` and no code reads it before production validation.
`data/` is not committed.

## Repository layout

```
src/ingestion/   download.py          verified download of the raw CSV
src/data/        columns.py           column groups and feature sets
                 splits.py            time-based split, design-data loader (train + val only)
                 make_splits.py       writes the split files and docs/split_manifest.json
src/eda/         run_eda.py           EDA, drift check, exploratory baselines, figures
tests/                                pytest suite
docs/                                 findings, checklist, split manifest, figures
.github/workflows/ci.yml              ruff and pytest on Python 3.12
```

Planned (added as the stages land): `flows/` (orchestration), `src/training`, `src/serving`,
`src/monitoring`, `docker/`, `scripts/` (drift simulation).

## Team and contributions

| Member | GitHub | Contribution |
|---|---|---|
| Hoon | jhoonyum | to be filled in |
| to be added | | to be filled in |

The syllabus asks every member to state their own engineering contribution in the presentation.

## AI assistance

Code, documentation and analysis in this repository were produced with the assistance of Claude
(Anthropic), model `claude-sonnet-5-5`, through Claude Code. Commits carry a `Co-Authored-By` trailer.

## References

- Antonio, N., de Almeida, A., Nunes, L. (2019). Hotel booking demand datasets. *Data in Brief*, 22, 41-49.
  https://doi.org/10.1016/j.dib.2018.11.126
- TidyTuesday (2020-02-11). Hotel bookings data, cleaned by Thomas Mock and Antoine Bichat.
  https://github.com/rfordatascience/tidytuesday
