import numpy as np

from src.eda.decision_analysis import best_threshold, confusion, cost, summary

Y = np.array([1, 1, 1, 0, 0, 0, 0, 0])
P = np.array([0.95, 0.80, 0.30, 0.70, 0.20, 0.10, 0.05, 0.02])


def test_confusion_counts_add_up():
    c = confusion(Y, P >= 0.5)
    assert c == {"tp": 2, "fp": 1, "fn": 1, "tn": 4}


def test_cost_weights_false_positives_by_ratio():
    flag = P >= 0.5  # one false negative, one false positive
    assert cost(Y, flag, 1.0) == 2
    assert cost(Y, flag, 3.0) == 4


def test_flag_nothing_costs_one_per_cancellation():
    assert cost(Y, np.zeros(len(Y), dtype=bool), 2.0) == Y.sum()


def test_higher_false_positive_cost_never_lowers_the_threshold():
    t1, t3 = best_threshold(Y, P, 1.0), best_threshold(Y, P, 3.0)
    assert t3 >= t1
    # At r = 3 flagging the 0.70 negative is not worth it, so the threshold moves above it.
    assert t3 > 0.70


def test_summary_handles_no_flags():
    s = summary(Y, np.zeros(len(Y), dtype=bool), 2.0)
    assert s["precision"] is None
    assert s["recall"] == 0.0
    assert s["cost_per_1000"] == 375.0
