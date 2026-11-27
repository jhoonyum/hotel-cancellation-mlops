"""Train the baseline candidates and track every run in MLflow.

Each candidate is fit on train and scored on validation; the test split is never loaded.
One MLflow run per candidate records:
    params   candidate, algorithm, feature set, cost ratio, row counts, classifier settings
    metrics  validation PR-AUC, ROC-AUC, Brier, cost-minimizing threshold, and at that
             threshold: cost per 1,000 bookings, precision, recall, flagged share
    tags     lineage: git commit (and whether the tree had uncommitted changes), SHA-256 of
             the train and val files
    model    the fitted sklearn Pipeline (preprocessing + classifier) with an input example,
             plus the confusion counts at the threshold

Usage:
    python -m src.training.train                         # tracks to ./mlflow.db, artifacts in ./mlruns/
    mlflow ui --backend-store-uri sqlite:///mlflow.db    # browse the runs at http://127.0.0.1:5000
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import mlflow
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline

from src.data.columns import DEFAULT_FEATURES, TARGET
from src.data.splits import PROCESSED_DIR, REPO_ROOT, load_design_data
from src.ingestion.download import sha256_of
from src.training.evaluate import confusion, evaluate
from src.training.models import make_model

EXPERIMENT = "hotel-cancellation"
COST_RATIO = 2.0  # a false positive costs 2 empty rooms (docs/stage1_decisions.md, section 3)
DEFAULT_TRACKING_URI = f"sqlite:///{REPO_ROOT / 'mlflow.db'}"
DEFAULT_ARTIFACT_LOCATION = (REPO_ROOT / "mlruns").as_uri()

# MLflow saves sklearn models with skops, which loads only allow-listed types (a pickle file
# can run arbitrary code when loaded). These are the extra types our pipelines contain. We
# trust them because we only load models this pipeline trained and logged itself, never
# model files from outside.
SKOPS_TRUSTED_TYPES = [
    "numpy.dtype",  # encoder dtype settings
    "functools.partial",  # HistGradientBoosting internals
    "sklearn.utils.validation.check_array",
    "sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor",
]


def candidates() -> dict[str, BaseEstimator]:
    """The baselines every later model must beat, simplest first. All take DEFAULT_FEATURES."""
    return {
        # Predicts the training cancellation rate for everyone: the "flag nothing" floor.
        "majority": DummyClassifier(strategy="prior"),
        # The proposal's simple rule without deposit_type (now excluded): longer lead time,
        # higher risk. One feature, so its threshold is a lead-time cutoff.
        "rule_lead_time": make_model(["lead_time"], "logreg"),
        "logreg": make_model(DEFAULT_FEATURES, "logreg"),
        # Same recipe as the exploratory run in docs/stage1_decisions.md (cost 255.5 at r = 2).
        "hgb_default": make_model(DEFAULT_FEATURES, "hgb"),
    }


def lineage(processed_dir: Path = PROCESSED_DIR) -> dict[str, str]:
    """Tags that tie a run to the exact code and data that produced it."""
    try:
        commit = _git("rev-parse", "HEAD")
        dirty = str(bool(_git("status", "--porcelain", "--untracked-files=no")))
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = "unknown", "unknown"
    return {
        "git_commit": commit,
        "git_dirty": dirty,
        "data_train_sha256": sha256_of(Path(processed_dir) / "train.csv"),
        "data_val_sha256": sha256_of(Path(processed_dir) / "val.csv"),
    }


def train_all(
    design: pd.DataFrame,
    tracking_uri: str,
    artifact_location: str | None = None,
    tags: dict[str, str] | None = None,
    r: float = COST_RATIO,
) -> list[dict]:
    """Fit every candidate on train, score it on val, log one MLflow run each.

    `design` is the output of `load_design_data()`. `artifact_location` is used only when
    the experiment does not exist yet. Returns one dict per candidate with its run id,
    model URI and validation metrics.
    """
    found = set(design["split"])
    if found != {"train", "val"}:  # training must never see the holdout
        raise ValueError(f"expected train and val rows only, got {sorted(found)}")
    train, val = design[design["split"] == "train"], design[design["split"] == "val"]
    x_tr, y_tr = train[DEFAULT_FEATURES], train[TARGET].to_numpy()
    x_va, y_va = val[DEFAULT_FEATURES], val[TARGET].to_numpy()

    mlflow.set_tracking_uri(tracking_uri)
    experiment_id = _experiment_id(EXPERIMENT, artifact_location)
    results = []
    for name, model in candidates().items():
        model.fit(x_tr, y_tr)
        p = model.predict_proba(x_va)[:, 1]
        metrics = evaluate(y_va, p, r)
        with mlflow.start_run(experiment_id=experiment_id, run_name=name) as run:
            mlflow.set_tags({**(tags or {}), "candidate": name, "stage": "baseline"})
            mlflow.log_params(
                {
                    "candidate": name,
                    "algorithm": type(_classifier(model)).__name__,
                    "feature_set": "DEFAULT_FEATURES",
                    "n_features": len(DEFAULT_FEATURES),
                    "cost_ratio": r,
                    "train_rows": len(train),
                    "val_rows": len(val),
                    **{f"clf_{k}": v for k, v in _classifier(model).get_params().items()},
                }
            )
            mlflow.log_metrics({f"val_{k}": v for k, v in metrics.items() if v is not None})
            mlflow.log_dict(
                confusion(y_va, p >= metrics["threshold"]), "confusion_at_threshold.json"
            )
            info = mlflow.sklearn.log_model(
                model,
                name="model",
                input_example=x_va.head(5),
                skops_trusted_types=SKOPS_TRUSTED_TYPES,
            )
        results.append(
            {"candidate": name, "run_id": run.info.run_id, "model_uri": info.model_uri, **metrics}
        )
    return results


def _classifier(model: BaseEstimator) -> BaseEstimator:
    return model.named_steps["clf"] if isinstance(model, Pipeline) else model


def _experiment_id(name: str, artifact_location: str | None) -> str:
    found = mlflow.get_experiment_by_name(name)
    if found is not None:
        return found.experiment_id
    return mlflow.create_experiment(name, artifact_location=artifact_location)


def _git(*args: str) -> str:
    out = subprocess.run(
        ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=True
    )
    return out.stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tracking-uri", default=DEFAULT_TRACKING_URI)
    args = parser.parse_args()
    local = args.tracking_uri == DEFAULT_TRACKING_URI
    results = train_all(
        load_design_data(),
        args.tracking_uri,
        artifact_location=DEFAULT_ARTIFACT_LOCATION if local else None,
        tags=lineage(),
    )
    print(f"{'candidate':16s} {'PR-AUC':>7s} {'thresh':>6s} {'cost/1000':>9s} "
          f"{'prec':>6s} {'recall':>6s}")
    for m in results:
        prec = f"{m['precision']:.3f}" if m["precision"] is not None else "-"
        print(
            f"{m['candidate']:16s} {m['pr_auc']:7.4f} {m['threshold']:6.2f} "
            f"{m['cost_per_1000']:9.1f} {prec:>6s} {m['recall']:6.3f}"
        )
    print(f"flag nothing: {results[0]['cost_per_1000_flag_nothing']:.1f} per 1,000 bookings")


if __name__ == "__main__":
    main()
