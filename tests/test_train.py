"""Training pipeline tests on a small synthetic dataset: no network, no data/ directory."""

import mlflow
import numpy as np
import pandas as pd
import pytest

from src.data.columns import CATEGORICAL_ALL, DEFAULT_FEATURES, TARGET
from src.training.train import EXPERIMENT, candidates, train_all


def synthetic_design(n: int = 400, seed: int = 0) -> pd.DataFrame:
    """Rows shaped like load_design_data(): the 18 default features, the target, `split`."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(index=range(n))
    for c in DEFAULT_FEATURES:
        df[c] = rng.choice(["a", "b", "c"], n) if c in CATEGORICAL_ALL else rng.integers(0, 5, n)
    df["lead_time"] = rng.integers(0, 300, n)
    df["adr"] = rng.uniform(40, 200, n)
    df[TARGET] = (rng.uniform(size=n) < 0.2 + df["lead_time"] / 500).astype(int)
    df["split"] = np.where(np.arange(n) < 300, "train", "val")
    return df


@pytest.fixture(scope="module")
def tracked(tmp_path_factory):
    """Train once into a throwaway MLflow store shared by the tests below."""
    root = tmp_path_factory.mktemp("mlflow")
    uri = f"sqlite:///{root / 'mlflow.db'}"
    results = train_all(
        synthetic_design(), uri, (root / "artifacts").as_uri(), tags={"git_commit": "test"}
    )
    return uri, {r["candidate"]: r for r in results}


def test_one_tracked_run_per_candidate(tracked):
    uri, results = tracked
    client = mlflow.MlflowClient(uri)
    runs = client.search_runs([client.get_experiment_by_name(EXPERIMENT).experiment_id])
    assert {r.info.run_name for r in runs} == set(candidates()) == set(results)
    for run in runs:
        assert {"val_pr_auc", "val_cost_per_1000", "val_threshold", "val_brier"} <= set(
            run.data.metrics
        )
        assert run.data.params["feature_set"] == "DEFAULT_FEATURES"
        assert run.data.tags["git_commit"] == "test"  # lineage tags reach every run


def test_majority_baseline_equals_flagging_nothing(tracked):
    _, results = tracked
    val = synthetic_design().query("split == 'val'")
    m = results["majority"]
    assert m["pr_auc"] == pytest.approx(val[TARGET].mean())  # no-skill PR-AUC = base rate
    assert m["roc_auc"] == 0.5
    assert m["cost_per_1000"] == m["cost_per_1000_flag_nothing"]


@pytest.mark.parametrize("name", ["logreg", "hgb_default"])
def test_logged_model_scores_raw_rows_with_unseen_categories(tracked, name):
    """The logged artifact holds preprocessing + model: raw columns in, probabilities out."""
    _, results = tracked
    model = mlflow.sklearn.load_model(results[name]["model_uri"])
    rows = synthetic_design().query("split == 'val'")[DEFAULT_FEATURES].head(10).copy()
    rows["hotel"] = "never seen in training"
    p = model.predict_proba(rows)[:, 1]
    assert p.shape == (10,) and ((p >= 0) & (p <= 1)).all()


def test_refuses_rows_from_the_holdout(tmp_path):
    design = synthetic_design()
    design.loc[:9, "split"] = "test"
    with pytest.raises(ValueError, match="train and val rows only"):
        train_all(design, f"sqlite:///{tmp_path / 'mlflow.db'}")
