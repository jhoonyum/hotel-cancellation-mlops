import pandas as pd
import pytest

from src.data.splits import (
    TEST_START,
    VAL_START,
    add_arrival_date,
    assign_split,
    load_design_data,
)


def _frame(dates: list[str]) -> pd.DataFrame:
    ts = pd.to_datetime(dates)
    return pd.DataFrame(
        {
            "arrival_date_year": ts.year,
            "arrival_date_month": ts.strftime("%B"),
            "arrival_date_day_of_month": ts.day,
        }
    )


def test_arrival_date_is_built_from_year_month_name_day():
    df = add_arrival_date(_frame(["2015-07-01", "2016-02-29", "2017-08-31"]))
    assert df["arrival_date"].dt.strftime("%Y-%m-%d").tolist() == [
        "2015-07-01",
        "2016-02-29",
        "2017-08-31",
    ]


@pytest.mark.parametrize(
    "date, expected",
    [
        ("2015-07-01", "train"),
        ("2016-12-31", "train"),
        ("2017-01-01", "val"),
        ("2017-03-31", "val"),
        ("2017-04-01", "test"),
        ("2017-08-31", "test"),
    ],
)
def test_split_boundaries(date, expected):
    df = add_arrival_date(_frame([date]))
    assert assign_split(df).iloc[0] == expected


def test_boundaries_match_documented_dates():
    assert VAL_START == pd.Timestamp("2017-01-01")
    assert TEST_START == pd.Timestamp("2017-04-01")


def test_splits_partition_the_rows_without_overlap():
    dates = pd.date_range("2015-07-01", "2017-08-31", freq="7D").strftime("%Y-%m-%d").tolist()
    df = add_arrival_date(_frame(dates))
    labels = assign_split(df)
    assert set(labels) == {"train", "val", "test"}
    assert len(labels) == len(df)
    # time order: every train date precedes every val date precedes every test date
    by = df.assign(split=labels).groupby("split")["arrival_date"]
    assert by.max()["train"] < by.min()["val"]
    assert by.max()["val"] < by.min()["test"]


def test_design_loader_reads_only_train_and_val(tmp_path):
    """The loader must work from a folder that has no holdout files at all."""
    for name, day in (("train", "2016-05-01"), ("val", "2017-02-01")):
        pd.DataFrame({"arrival_date": [day], "is_canceled": [0]}).to_csv(
            tmp_path / f"{name}.csv", index=False
        )
    out = load_design_data(tmp_path)
    assert sorted(out["split"].unique()) == ["train", "val"]
    assert not (tmp_path / "test.csv").exists()
