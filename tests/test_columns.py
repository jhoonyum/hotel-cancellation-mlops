import pytest

from src.data.columns import (
    CALENDAR,
    CATEGORICAL,
    CATEGORICAL_ALL,
    FEATURE_SETS,
    NUMERIC,
    NUMERIC_ALL,
    OUTCOME_COLUMNS,
    STRICT,
    TARGET,
    TIMING_RISK,
)


def test_numeric_and_categorical_do_not_overlap():
    assert not set(NUMERIC_ALL) & set(CATEGORICAL_ALL)


@pytest.mark.parametrize("name", list(FEATURE_SETS))
def test_every_feature_in_every_set_is_classified(name):
    """A column that is in neither list would be dropped silently by the model builders."""
    unclassified = [c for c in FEATURE_SETS[name] if c not in NUMERIC_ALL + CATEGORICAL_ALL]
    assert unclassified == []


@pytest.mark.parametrize("name", list(FEATURE_SETS))
def test_no_feature_set_contains_target_or_outcome_columns(name):
    assert TARGET not in FEATURE_SETS[name]
    assert not set(OUTCOME_COLUMNS) & set(FEATURE_SETS[name])


def test_strict_set_excludes_timing_risk_columns_and_year():
    assert not set(TIMING_RISK) & set(STRICT)
    assert "arrival_date_year" not in STRICT


def test_calendar_columns_exist_in_the_proposal_lists():
    assert set(CALENDAR) <= set(NUMERIC + CATEGORICAL)


def test_ablation_sets_differ_from_strict():
    """Guards against an ablation that is silently identical to the baseline set."""
    for name, cols in FEATURE_SETS.items():
        if name != "strict":
            assert set(cols) != set(STRICT), name
