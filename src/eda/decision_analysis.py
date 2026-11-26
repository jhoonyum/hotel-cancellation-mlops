"""Evidence for the two stage-1 decisions: the cost ratio behind the decision threshold,
and how the threshold behaves for the strict feature set.

Design data only (train + validation). The test split is never loaded.

Cost model (unit = the cost of one empty room):
    false negative  (missed cancellation, room stays empty)            cost 1
    false positive  (flagged, the guest shows up; under overbooking
                     this can mean walking the guest)                   cost r
    true positive, true negative                                        cost 0
With calibrated probabilities the expected-cost threshold is r / (1 + r). We also find the
threshold that minimizes the observed cost on validation and check how stable it is.

Writes docs/decision_results.json and docs/figures/fig7_cost_threshold.png.

Usage:
    python -m src.eda.decision_analysis
"""

from __future__ import annotations

import json
import warnings

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss

from src.data.columns import STRICT, TARGET
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


def main() -> None:
    warnings.filterwarnings("ignore")
    style()
    df = load_design_data()
    assert set(df["split"]) == {"train", "val"}
    train, val = df[df["split"] == "train"], df[df["split"] == "val"]
    y_tr, y = train[TARGET].to_numpy(), val[TARGET].to_numpy()

    model = make_model(STRICT, "hgb").fit(train[STRICT], y_tr)
    p = model.predict_proba(val[STRICT])[:, 1]
    non_refund = (val["deposit_type"] == "Non Refund").to_numpy()
    flag_none = np.zeros(len(y), dtype=bool)

    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(y), size=(N_BOOT, len(y)))

    by_ratio = {}
    for r in RATIOS:
        t_theory = round(r / (1 + r), 4)
        t_best = best_threshold(y, p, r)
        flag = p >= t_best
        # Threshold stability and the saving against "flag nothing", over the same resamples.
        boot_t = np.array([best_threshold(y[i], p[i], r) for i in idx])
        saving = np.array(
            [
                1000 * (cost(y[i], flag_none, r) - cost(y[i], flag[i], r)) / len(i)
                for i in idx
            ]
        )
        flag_theory = p >= t_theory

        def vs_rule(f: np.ndarray, r: float = r) -> dict:
            # Positive = the model costs less than the Non Refund rule.
            d = [1000 * (cost(y[i], non_refund[i], r) - cost(y[i], f[i], r)) / len(i) for i in idx]
            point = 1000 * (cost(y, non_refund, r) - cost(y, f, r)) / len(y)
            return {
                "point": round(point, 1),
                "ci95": [round(float(np.percentile(d, q)), 1) for q in (2.5, 97.5)],
            }

        by_ratio[str(r)] = {
            "theoretical_threshold": t_theory,
            "cost_minimizing_threshold": t_best,
            "bootstrap_threshold_p5_p50_p95": [
                float(np.percentile(boot_t, q)) for q in (5, 50, 95)
            ],
            "at_cost_minimizing_threshold": summary(y, flag, r),
            "at_theoretical_threshold": summary(y, p >= t_theory, r),
            "flag_nothing": summary(y, flag_none, r),
            "rule_non_refund": summary(y, non_refund, r),
            "saving_vs_flag_nothing_per_1000": {
                "point": round(1000 * (cost(y, flag_none, r) - cost(y, flag, r)) / len(y), 1),
                "ci95": [round(float(np.percentile(saving, q)), 1) for q in (2.5, 97.5)],
            },
            "saving_vs_rule_non_refund_per_1000": {
                "at_cost_minimizing_threshold": vs_rule(flag),
                "at_theoretical_threshold": vs_rule(flag_theory),
            },
        }

    # The proposed rule: the theoretical threshold at the primary ratio.
    t_primary = by_ratio[str(PRIMARY_RATIO)]["theoretical_threshold"]
    flag_primary = p >= t_primary
    extra = flag_primary & ~non_refund
    decomposition = {
        "threshold": t_primary,
        "non_refund_bookings": int(non_refund.sum()),
        "non_refund_flagged_by_model": int((flag_primary & non_refund).sum()),
        "other_flags": int(extra.sum()),
        "other_flags_precision": round(float(y[extra].mean()), 4),
        "break_even_precision": round(PRIMARY_RATIO / (1 + PRIMARY_RATIO), 4),
        "pr_auc_without_non_refund_rows": round(
            float(average_precision_score(y[~non_refund], p[~non_refund])), 4
        ),
        "prevalence_without_non_refund_rows": round(float(y[~non_refund].mean()), 4),
    }

    # Per month at the proposed threshold: is its behavior stable inside validation?
    months = {}
    for m in val["arrival_date_month"].unique():
        mask = (val["arrival_date_month"] == m).to_numpy()
        months[str(m)] = {
            "n": int(mask.sum()),
            "cancel_rate": round(float(y[mask].mean()), 4),
            **summary(y[mask], p[mask] >= t_primary, PRIMARY_RATIO),
        }

    results = {
        "scope": "train + validation only; the test split is never loaded",
        "model": "HistGradientBoostingClassifier, defaults, strict feature set (19 columns)",
        "features": STRICT,
        "validation_pr_auc": round(float(average_precision_score(y, p)), 4),
        "validation_rows": len(y),
        "cost_unit": "one empty room (false negative = 1, false positive = r)",
        "primary_ratio": PRIMARY_RATIO,
        "by_ratio": by_ratio,
        "calibration": calibration(y, p),
        "where_the_model_adds_over_the_rule": decomposition,
        "by_month_at_primary_threshold": months,
        "bootstrap": {"resamples": N_BOOT, "seed": SEED},
    }
    RESULTS_PATH.write_text(json.dumps(results, indent=2) + "\n")

    # Figure: expected cost per 1000 bookings against the threshold, one line per ratio.
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    colors = {1.0: BLUE, 2.0: ORANGE, 3.0: GREEN}
    for r in RATIOS:
        curve = [1000 * cost(y, p >= t, r) / len(y) for t in GRID]
        t_best = by_ratio[str(r)]["cost_minimizing_threshold"]
        ax.plot(GRID, curve, color=colors[r], lw=2, label=f"r = {r:g}  (best t = {t_best:.2f})")
        ax.plot([t_best], [1000 * cost(y, p >= t_best, r) / len(y)], "o", color=colors[r])
    ax.axhline(1000 * y.mean(), color=MUTED, ls=":", lw=1.4)
    ax.text(0.06, 1000 * y.mean() + 5, "flag nothing", color=INK2, fontsize=8.5)
    # The rule's cost barely moves with r (it has 7 false positives), so one line at r = 2.
    rule_cost = by_ratio[str(PRIMARY_RATIO)]["rule_non_refund"]["cost_per_1000"]
    ax.axhline(rule_cost, color=INK2, ls="--", lw=1.2)
    ax.text(0.06, rule_cost - 14, "rule: flag Non Refund", color=INK2, fontsize=8.5)
    ax.set_ylim(180, 420)
    ax.set_xlabel("decision threshold on predicted cancellation probability")
    ax.set_ylabel("cost per 1,000 bookings\n(unit: one empty room)")
    ax.set_title("Validation cost by threshold, strict feature set (r = false-positive cost)")
    ax.legend(loc="upper right", fontsize=8.5)
    save(fig, "fig7_cost_threshold.png")
    print(f"wrote {RESULTS_PATH.relative_to(REPO_ROOT)} and fig7_cost_threshold.png")


if __name__ == "__main__":
    main()
