"""Cost-based evaluation shared by the EDA and the training pipeline.

Cost model (unit = the cost of one empty room), from docs/stage1_decisions.md:
    false negative  (missed cancellation, room stays empty)            cost 1
    false positive  (flagged, the guest shows up and may be walked)     cost r
    true positive, true negative                                        cost 0

The decision threshold is the grid value that minimizes this cost on validation.
"""

from __future__ import annotations

import numpy as np

GRID = np.round(np.arange(0.05, 0.96, 0.01), 2)  # candidate decision thresholds


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
