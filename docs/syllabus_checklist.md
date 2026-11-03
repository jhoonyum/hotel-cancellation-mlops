# Syllabus checklist

Source: ADSP 32021 Machine Learning Operations and Inference Engineering, Autumn 2026 (Arnab Bose, Ph.D.),
sections "Final Project", "Technical Project Outline", "Deliverables" and the grading list.
Status key: **done** = in the repository and checked; **partial** = started, see note; **todo** = not started.

## 1. Rules and logistics

| Requirement (syllabus) | Status | Evidence / note |
|---|---|---|
| Python 3.12 or higher | done | `.python-version` 3.12; CI runs 3.12; all code was run on Python 3.12.3 |
| Team project; group size announced in class | open | Team roster not recorded yet (README, "Team and contributions") |
| Topic selection and proposal (up to 300 words) due end of week 4 | open | Past the date. The syllabus says topics are approved first-come, first-served to prevent project replication: confirm with the instructor that this dataset and topic are approved for this group |
| Final presentation in week 10, 10 minutes, every member states their own engineering contribution | todo | Contributions table in the README must be filled in |
| Clearly outline which member did which part | todo | Same table |
| Disclose AI tools used (product and version) | done | README, "AI assistance" |
| Repository "public/accessible" for the slides link | open | Repository is currently private. Make it public or give the instructor and TA access before submission |

## 2. Technical project outline (five stages)

| # | Requirement | Status | Evidence / note |
|---|---|---|---|
| 1a | Dataset with a clear target solving a specific problem | done | Hotel bookings, target `is_canceled`; `docs/eda_findings.md` |
| 1b | Evaluation metric aligned with the problem's constraints | partial | PR-AUC proposed (no-skill 0.33), threshold from a cost ratio. The cost ratio is not agreed yet (`docs/eda_findings.md`, section 8) |
| 1c | Train/test split; test set isolated until production validation | done | `src/data/splits.py`, `make_splits.py`; test file in `data/holdout/`; `load_design_data` returns train and val only; split hashes in `docs/split_manifest.json`; tests in `tests/test_splits.py`. Disclosure about an earlier look at the test split: `docs/eda_findings.md`, section 0 |
| 1 | Baselines | partial | Majority class, rule and logistic regression are scored in the EDA on validation. A formal baseline module with tracked runs comes with stage 2 |
| 2a | Automated orchestrator for preprocessing and training | todo | Planned: Prefect |
| 2b | Experiment tracking and model logging (MLflow or similar), AutoML optional | todo | Planned: MLflow |
| 2c | Finalized model in a registry with semantic versioning | todo | Planned: MLflow Model Registry |
| 3a | Package the registered model and deploy (Docker with FastAPI, Flask or BentoML) | todo | Planned: FastAPI and Docker |
| 3b | Accept test inputs and return real-time predictions | todo | |
| 4a | Monitoring framework or dashboard for data and model performance | todo | Evidently; clean-data drift study done (`docs/eda_findings.md`, section 6) and shows that default alert levels need calibration |
| 4b | Baseline validation: pass the clean test dataset through the deployed API and compare with the monitoring baseline | todo | Needs the holdout test file and the deployed API |
| 5 | Stress test: corrupt the test data (out-of-bounds values, swapped columns, schema changes), send it to the deployed model, document how monitoring alerts | todo | |

## 3. Deliverables

| Deliverable | Status | Evidence / note |
|---|---|---|
| GitHub: fully commented source code with check-in and PR logs | partial | Work goes through feature branches and pull requests; CI runs ruff and pytest |
| GitHub: README explaining how to run and reproduce the whole pipeline locally | partial | README covers the stages that exist (ingestion, splits, EDA, tests) |
| GitHub: dependency file (requirements.txt, environment.yaml or Dockerfile) | done | `requirements.txt`, `requirements-dev.txt`; Dockerfile arrives with stage 3 |
| Slides (one per team): problem statement and EDA | partial | Material ready: `docs/eda_findings.md` and `docs/figures/` |
| Slides: evaluation metric justification | partial | Section 7 and 8 of `docs/eda_findings.md` |
| Slides: system architecture diagram (training and deployment) | todo | |
| Slides: experiment tracking dashboard showing algorithm selection | todo | |
| Slides: deployment and monitoring demo or screenshots | todo | |
| Slides: drift analysis, visual evidence of monitoring reacting to corrupted data | todo | `fig5` shows the clean-data baseline only |
| Slides: visible link to the repository | open | See visibility note above |

## 4. Grading list (each item 5 points)

| Item | Where it stands |
|---|---|
| a. Problem statement and system architecture | problem and EDA done; architecture diagram todo |
| b. Data ingestion and baselines | ingestion, isolated split and exploratory baselines done; formal tracked baselines todo |
| c. Pipeline automation and experiment tracking | todo |
| d. Containerization and deployment | todo |
| e. Monitoring and drift engineering | clean-data drift calibration done; monitoring and simulation todo |
| f. GitHub usage and repository structure | in progress: pull requests, CI, tests, modular `src/` |
| g. Challenges and future work | first list in `docs/eda_findings.md`, sections 8 and 9 (column timing, duplicates, alert calibration, label delay) |
| h. Presentation and team collaboration | todo |

## 5. Open decisions

1. Confirm topic approval with the instructor (section 1).
2. Agree the prediction time and the feature set (`docs/eda_findings.md`, section 8, points 1 to 3).
3. Agree the cost ratio behind the decision threshold.
4. Record the team roster and each member's part.
5. Decide repository visibility for submission.
