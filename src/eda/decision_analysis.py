"""Evidence for the stage-1 decisions: the cost ratio behind the decision threshold, and how
the threshold behaves for the default feature set (STRICT without deposit_type).

Design data only (train + validation). The test split is never loaded.

Cost model (unit = the cost of one empty room):
    false negative  (missed cancellation, room stays empty)            cost 1
    false positive  (flagged, the guest shows up; under overbooking
                     this can mean walking the guest)                   cost r
    true positive, true negative                                        cost 0
With calibrated probabilities the expected-cost threshold is r / (1 + r). We also find the
threshold that minimizes the observed cost on validation, check how stable it is, and test
whether isotonic calibration on the last train months brings the model to r / (1 + r).

Writes docs/decision_results.json and docs/figures/fig7_cost_threshold.png.

Usage:
    python -m src.eda.decision_analysis
"""

from __future__ import annotations

import json
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, brier_score_loss

from src.data.columns import DEFAULT_FEATURES, STRICT, TARGET
from src.data.splits import REPO_ROOT, load_design_data
from src.eda.run_eda import (
    BLUE,
    INK2,
    MUTED,
    N_BOOT,
    ORANGE,
    SEED,
    make_model,
    save,
    style,
)

RESULTS_PATH = REPO_ROOT / "docs" / "decision_results.json"
RATIOS = (1.0, 2.0, 3.0)  # cost of a false positive / cost of a false negative
PRIMARY_RATIO = 2.0
GRID = np.round(np.arange(0.05, 0.96, 0.01), 2)
GREEN = "#1f9e74"
CALIBRATION_START = pd.Timestamp("2016-10-01")  # last three train months hold out for calibration


def confusion(y: np.ndarray, flag: np.ndarray) -> dict:
    tp = int((flag & (y == 1)).sum())
    fp = int((flag & (y == 0)).sum())
    fn = int((~flag & (y == 1)).sum())
    return {"tp": tp, "fp": fp, "fn": fn, "tn": int(len(y) - tp - fp - fn)}


def cost(y: np.ndarray, flag: np.ndarray, r: float) -> float:
    c = confusion(y, flag)
    return c["fn"] + r * c["fp"]


def summary(y: np.ndarray, flag: np.ndarray, r: float) -> dict:
    c = confusion(y, flag)
    n_flag = c["tp"] + c["fp"]
    return {
        "flagged_share": round(n_flag / len(y), 4),
        "precision": round(c["tp"] / n_flag, 4) if n_flag else None,
        "recall": round(c["tp"] / (c["tp"] + c["fn"]), 4),
        "cost_per_1000": round(1000 * cost(y, flag, r) / len(y), 1),
        **c,
    }


def best_threshold(y: np.ndarray, p: np.ndarray, r: float) -> float:
    costs = [cost(y, p >= t, r) for t in GRID]
    return float(GRID[int(np.argmin(costs))])


def calibration(y: np.ndarray, p: np.ndarray) -> dict:
    edges = np.quantile(p, np.linspace(0, 1, 11))
    bins = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, 9)
    table = [
        {
            "decile": int(b + 1),
            "mean_predicted": round(float(p[bins == b].mean()), 4),
            "observed_rate": round(float(y[bins == b].mean()), 4),
            "n": int((bins == b).sum()),
        }
        for b in range(10)
    ]
    return {
        "brier": round(float(brier_score_loss(y, p)), 4),
        "mean_predicted": round(float(p.mean()), 4),
        "observed_rate": round(float(y.mean()), 4),
        "deciles": table,
    }


def saving_vs_nothing(y: np.ndarray, flag: np.ndarray, r: float, idx: np.ndarray) -> dict:
    """Cost saved per 1,000 bookings against flagging nothing, with a bootstrap interval."""
    none = np.zeros(len(y), dtype=bool)
    boot = [1000 * (cost(y[i], none[i], r) - cost(y[i], flag[i], r)) / len(i) for i in idx]
    return {
        "point": round(1000 * (cost(y, none, r) - cost(y, flag, r)) / len(y), 1),
        "ci95": [round(float(np.percentile(boot, q)), 1) for q in (2.5, 97.5)],
    }


def calibration_experiment(train, val, y: np.ndarray, r: float) -> dict:
    """Fit on train before CALIBRATION_START, fit isotonic calibration on the rest of train,
    and compare cost on validation at r / (1 + r) and at the cost-minimizing threshold."""
    early = train[train["arrival_date"] < CALIBRATION_START]
    late = train[train["arrival_date"] >= CALIBRATION_START]
    model = make_model(DEFAULT_FEATURES, "hgb").fit(early[DEFAULT_FEATURES], early[TARGET])
    iso = IsotonicRegression(out_of_bounds="clip").fit(
        model.predict_proba(late[DEFAULT_FEATURES])[:, 1], late[TARGET]
    )
    raw = model.predict_proba(val[DEFAULT_FEATURES])[:, 1]
    t = r / (1 + r)
    out = {"fit_rows": len(early), "calibration_rows": len(late), "calibration_start": "2016-10-01"}
    for name, q in (("uncalibrated", raw), ("isotonic", iso.predict(raw))):
        t_best = best_threshold(y, q, r)
        out[name] = {
            "pr_auc": round(float(average_precision_score(y, q)), 4),
            "brier": round(float(brier_score_loss(y, q)), 4),
            "cost_at_theoretical_threshold": summary(y, q >= t, r)["cost_per_1000"],
            "cost_minimizing_threshold": t_best,
            "cost_at_cost_minimizing_threshold": summary(y, q >= t_best, r)["cost_per_1000"],
        }
    return out


def non_refund_profile(df: pd.DataFrame) -> dict:
    """Who the Non Refund bookings are (train + validation), to test the published explanation."""
    nr = df[df["deposit_type"] == "Non Refund"]
    rest = df[df["deposit_type"] != "Non Refund"]
    raw_cols = [c for c in df.columns if c not in ("split", "arrival_date")]
    canceled = df[df[TARGET] == 1]
    days_before = (
        canceled["arrival_date"] - pd.to_datetime(canceled["reservation_status_date"])
    ).dt.days
    is_nr = canceled["deposit_type"] == "Non Refund"
    seg = nr["market_segment"].value_counts(normalize=True)
    return {
        "rows": len(nr),
        "cancel_rate": round(float(nr[TARGET].mean()), 4),
        "share_country_PRT": round(float((nr["country"] == "PRT").mean()), 4),
        "share_market_segment": {k: round(float(v), 4) for k, v in seg.head(4).items()},
        "share_exact_duplicate_rows": round(float(nr.duplicated(subset=raw_cols).mean()), 4),
        "median_lead_time": float(nr["lead_time"].median()),
        "median_lead_time_other_deposit_types": float(rest["lead_time"].median()),
        "canceled_median_days_before_arrival": float(days_before[is_nr].median()),
        "canceled_median_days_before_arrival_other": float(days_before[~is_nr].median()),
    }


def main() -> None:
    warnings.filterwarnings("ignore")
    style()
    df = load_design_data()
    assert set(df["split"]) == {"train", "val"}
    train, val = df[df["split"] == "train"], df[df["split"] == "val"]
    y_tr, y = train[TARGET].to_numpy(), val[TARGET].to_numpy()

    p = make_model(DEFAULT_FEATURES, "hgb").fit(train[DEFAULT_FEATURES], y_tr)
    p = p.predict_proba(val[DEFAULT_FEATURES])[:, 1]
    # Ablation only: the same model with deposit_type, whose timing cannot be checked.
    p_dep = make_model(STRICT, "hgb").fit(train[STRICT], y_tr).predict_proba(val[STRICT])[:, 1]
    non_refund = (val["deposit_type"] == "Non Refund").to_numpy()
    flag_none = np.zeros(len(y), dtype=bool)

    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(y), size=(N_BOOT, len(y)))

    by_ratio = {}
    for r in RATIOS:
        t_theory = round(r / (1 + r), 4)
        t_best = best_threshold(y, p, r)
        boot_t = np.array([best_threshold(y[i], p[i], r) for i in idx])
        by_ratio[str(r)] = {
            "theoretical_threshold": t_theory,
            "cost_minimizing_threshold": t_best,
            "bootstrap_threshold_p5_p50_p95": [
                float(np.percentile(boot_t, q)) for q in (5, 50, 95)
            ],
            "at_theoretical_threshold": summary(y, p >= t_theory, r),
            "at_cost_minimizing_threshold": summary(y, p >= t_best, r),
            "flag_nothing": summary(y, flag_none, r),
            "saving_vs_flag_nothing_per_1000": {
                "at_theoretical_threshold": saving_vs_nothing(y, p >= t_theory, r, idx),
                "at_cost_minimizing_threshold": saving_vs_nothing(y, p >= t_best, r, idx),
            },
            "ablation_with_deposit_type": summary(y, p_dep >= t_theory, r),
            "reference_rule_non_refund": summary(y, non_refund, r),
        }

    # Proposed rule: the threshold that minimizes validation cost at the primary ratio.
    t_primary = by_ratio[str(PRIMARY_RATIO)]["cost_minimizing_threshold"]
    flag_primary = p >= t_primary

    months = {}
    for m in val["arrival_date_month"].unique():
        mask = (val["arrival_date_month"] == m).to_numpy()
        months[str(m)] = {
            "n": int(mask.sum()),
            "cancel_rate": round(float(y[mask].mean()), 4),
            **summary(y[mask], flag_primary[mask], PRIMARY_RATIO),
        }

    results = {
        "scope": "train + validation only; the test split is never loaded",
        "model": "HistGradientBoostingClassifier, defaults, DEFAULT_FEATURES (18 columns)",
        "features": DEFAULT_FEATURES,
        "validation_rows": len(y),
        "validation_pr_auc": round(float(average_precision_score(y, p)), 4),
        "validation_pr_auc_ablation_with_deposit_type": round(
            float(average_precision_score(y, p_dep)), 4
        ),
        "cost_unit": "one empty room (false negative = 1, false positive = r)",
        "primary_ratio": PRIMARY_RATIO,
        "by_ratio": by_ratio,
        "calibration": calibration(y, p),
        "calibration_experiment": calibration_experiment(train, val, y, PRIMARY_RATIO),
        "non_refund_profile_train_and_validation": non_refund_profile(df),
        "non_refund_rows_flagged_without_deposit_type": {
            "threshold": t_primary,
            "non_refund_rows": int(non_refund.sum()),
            "flagged": int((flag_primary & non_refund).sum()),
        },
        "by_month_at_primary_threshold": months,
        "bootstrap": {"resamples": N_BOOT, "seed": SEED},
    }
    RESULTS_PATH.write_text(json.dumps(results, indent=2) + "\n")

    # Figure: cost per 1000 bookings against the threshold, one line per ratio.
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    colors = {1.0: BLUE, 2.0: ORANGE, 3.0: GREEN}
    for r in RATIOS:
        curve = [1000 * cost(y, p >= t, r) / len(y) for t in GRID]
        t_b = by_ratio[str(r)]["cost_minimizing_threshold"]
        ax.plot(GRID, curve, color=colors[r], lw=2, label=f"r = {r:g}  (lowest cost at {t_b:.2f})")
        ax.plot([t_b], [1000 * cost(y, p >= t_b, r) / len(y)], "o", color=colors[r])
    ax.axhline(1000 * y.mean(), color=MUTED, ls=":", lw=1.4)
    ax.text(0.06, 1000 * y.mean() + 5, "flag nothing", color=INK2, fontsize=8.5)
    ax.set_ylim(180, 420)
    ax.set_xlabel("decision threshold on predicted cancellation probability")
    ax.set_ylabel("cost per 1,000 bookings\n(unit: one empty room)")
    ax.set_title("Validation cost by threshold, default features (r = false-positive cost)")
    ax.legend(loc="upper right", fontsize=8.5)
    save(fig, "fig7_cost_threshold.png")
    print(f"wrote {RESULTS_PATH.relative_to(REPO_ROOT)} and fig7_cost_threshold.png")


if __name__ == "__main__":
    main()
