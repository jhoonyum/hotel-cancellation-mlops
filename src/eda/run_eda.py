"""Exploratory data analysis on the design data (train + validation) only.

The test split is never loaded: this module reads data through `load_design_data`,
which returns train and val rows only.

Sections (each writes numbers to docs/eda_results.json and figures to docs/figures/):
    1. overview and data quality
    2. cancellation over time
    3. what separates canceled from kept bookings
    4. feature-timing audit (columns that may only be known at or after check-in)
    5. drift check: Evidently defaults on the clean validation window
    6. exploratory baselines (fit on train, scored on validation) and a feature-set ablation

Usage:
    python -m src.data.make_splits   # once, after downloading the raw file
    python -m src.eda.run_eda
"""

from __future__ import annotations

import json
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from evidently import DataDefinition, Dataset, Report
from evidently.presets import DataDriftPreset
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from src.data.columns import (
    CALENDAR,
    CATEGORICAL,
    CATEGORICAL_ALL,
    FEATURE_SETS,
    NUMERIC,
    NUMERIC_ALL,
    TARGET,
)
from src.data.splits import REPO_ROOT, VAL_START, load_design_data

FIG_DIR = REPO_ROOT / "docs" / "figures"
RESULTS_PATH = REPO_ROOT / "docs" / "eda_results.json"

# Two colors from a color-vision-deficiency-checked categorical palette, plus chart ink.
BLUE, ORANGE = "#2a78d6", "#eb6834"
SURFACE, INK, INK2, MUTED, GRID, AXIS = (
    "#fcfcfb",
    "#0b0b0b",
    "#52514e",
    "#898781",
    "#e1e0d9",
    "#c3c2b7",
)
SEED = 0
N_BOOT = 300


def style() -> None:
    plt.switch_backend("Agg")
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": AXIS,
            "axes.labelcolor": INK2,
            "xtick.color": INK2,
            "ytick.color": INK2,
            "text.color": INK,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "axes.axisbelow": True,
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "legend.frameon": False,
        }
    )


def save(fig: plt.Figure, name: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / name, dpi=130, bbox_inches="tight")
    plt.close(fig)


def rate_by(df: pd.DataFrame, key, order: list | None = None) -> pd.DataFrame:
    out = df.groupby(key, observed=True)[TARGET].agg(n="size", rate="mean")
    return out.reindex(order) if order is not None else out


def rows(table: pd.DataFrame) -> dict:
    return {
        str(k): {"n": int(r.n), "rate": round(float(r.rate), 4)} for k, r in table.iterrows()
    }


def hbar(ax, labels, rates, ns, base_rate=None, color=BLUE) -> None:
    y = np.arange(len(labels))[::-1]
    ax.barh(y, np.asarray(rates) * 100, color=color, height=0.55)
    ax.set_yticks(y, labels)
    ax.set_xlim(0, 135)
    ax.set_xticks([0, 25, 50, 75, 100])
    for yi, r, n in zip(y, rates, ns, strict=True):
        ax.text(r * 100 + 1.5, yi, f"{r:.0%}  (n={n:,})", va="center", fontsize=8.5, color=INK2)
    if base_rate is not None:
        ax.axvline(base_rate * 100, color=MUTED, ls=":", lw=1.4)
    ax.set_xlabel("cancellation rate (%)")
    ax.grid(axis="y", visible=False)


# --------------------------------------------------------------------------- 1. overview
def overview(df: pd.DataFrame) -> dict:
    raw_cols = [c for c in df.columns if c not in ("split", "arrival_date")]
    dup = df.duplicated(subset=raw_cols)
    guests = df["adults"] + df["children"].fillna(0) + df["babies"]
    nights = df["stays_in_weekend_nights"] + df["stays_in_week_nights"]
    by_split = df.groupby("split").agg(
        rows=(TARGET, "size"),
        cancel_rate=(TARGET, "mean"),
        first=("arrival_date", "min"),
        last=("arrival_date", "max"),
    )
    return {
        "design_rows": len(df),
        "columns": len(raw_cols),
        "by_split": {
            s: {
                "rows": int(r.rows),
                "cancel_rate": round(float(r.cancel_rate), 4),
                "first": r["first"].strftime("%Y-%m-%d"),
                "last": r["last"].strftime("%Y-%m-%d"),
            }
            for s, r in by_split.iterrows()
        },
        "cancel_rate_by_hotel": rows(rate_by(df, "hotel")),
        "missing": {
            "children": int(df["children"].isna().sum()),
            "country": int(df["country"].isna().sum()),
            "agent (not applicable)": int(df["agent"].isna().sum()),
            "company (not applicable)": int(df["company"].isna().sum()),
        },
        "duplicate_rows_after_first_copy": int(dup.sum()),
        "duplicate_share": round(float(dup.mean()), 4),
        "cancel_rate_of_duplicates": round(float(df.loc[dup, TARGET].mean()), 4),
        "data_quality": {
            "adr_negative": int((df["adr"] < 0).sum()),
            "adr_zero": int((df["adr"] == 0).sum()),
            "adr_max": float(df["adr"].max()),
            "adr_at_least_1000": int((df["adr"] >= 1000).sum()),
            "no_guests": int((guests == 0).sum()),
            "zero_nights": int((nights == 0).sum()),
            "lead_time_max": int(df["lead_time"].max()),
            "meal_undefined": int((df["meal"] == "Undefined").sum()),
            "market_segment_undefined": int((df["market_segment"] == "Undefined").sum()),
            "distribution_channel_undefined": int(
                (df["distribution_channel"] == "Undefined").sum()
            ),
        },
    }


# --------------------------------------------------------------------------- 2. time
def time_profile(df: pd.DataFrame) -> dict:
    m = df.groupby(df["arrival_date"].dt.to_period("M")).agg(
        n=(TARGET, "size"), rate=(TARGET, "mean")
    )
    x = m.index.to_timestamp()
    fig, (a, b) = plt.subplots(
        2, 1, figsize=(9, 5.6), sharex=True, gridspec_kw={"height_ratios": [3, 1.4]}
    )
    a.plot(x, m["rate"] * 100, color=BLUE, lw=2, marker="o", ms=4.5)
    a.set_ylabel("cancellation rate (%)")
    a.set_title("Cancellation rate and bookings by arrival month (train + validation)")
    a.axvline(VAL_START, color=MUTED, ls="--", lw=1.2)
    a.text(VAL_START, a.get_ylim()[1], "  validation", va="top", color=INK2, fontsize=9)
    b.bar(x, m["n"], width=22, color=BLUE)
    b.axvline(VAL_START, color=MUTED, ls="--", lw=1.2)
    b.set_ylabel("bookings")
    b.grid(axis="x", visible=False)
    fig.tight_layout()
    save(fig, "fig1_cancellation_over_time.png")
    top, bottom = m["rate"].idxmax(), m["rate"].idxmin()
    return {
        "months": {str(p): {"n": int(r.n), "rate": round(float(r.rate), 4)} for p, r in m.iterrows()},
        "highest_month": {"month": str(top), "rate": round(float(m.loc[top, "rate"]), 4)},
        "lowest_month": {"month": str(bottom), "rate": round(float(m.loc[bottom, "rate"]), 4)},
    }


# --------------------------------------------------------------------------- 3. drivers
def drivers(df: pd.DataFrame) -> dict:
    base = float(df[TARGET].mean())
    tables = {
        c: rate_by(df, c).sort_values("rate", ascending=False)
        for c in ["deposit_type", "market_segment", "customer_type", "hotel"]
    }
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.2))
    for ax, (col, t) in zip(axes.ravel(), tables.items(), strict=True):
        hbar(ax, list(t.index), list(t["rate"]), list(t["n"]), base_rate=base)
        ax.set_title(col)
    fig.suptitle(
        f"Cancellation rate by group (dotted line: overall {base:.0%})",
        x=0.01,
        ha="left",
        fontweight="bold",
        fontsize=12,
    )
    fig.tight_layout()
    save(fig, "fig2_cancellation_by_category.png")

    lead = pd.cut(
        df["lead_time"],
        [-1, 7, 30, 90, 180, 365, 10_000],
        labels=["0-7", "8-30", "31-90", "91-180", "181-365", "366+"],
    )
    lead_t = rate_by(df, lead)
    req = df["total_of_special_requests"].clip(upper=3).map({0: "0", 1: "1", 2: "2", 3: "3+"})
    req_t = rate_by(df, req, order=["0", "1", "2", "3+"])
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 3.8))
    for ax, t, title, xlabel in (
        (a, lead_t, "Lead time", "days between booking and arrival"),
        (b, req_t, "Special requests", "number of special requests on the booking"),
    ):
        pos = np.arange(len(t))
        ax.bar(pos, t["rate"] * 100, color=BLUE, width=0.6)
        ax.set_xticks(pos, list(t.index))
        ax.set_ylim(0, 100)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("cancellation rate (%)")
        ax.set_title(title)
        ax.grid(axis="x", visible=False)
        for p, (r, n) in enumerate(zip(t["rate"], t["n"], strict=True)):
            ax.text(p, r * 100 + 1.5, f"{r:.0%}\nn={n:,}", ha="center", fontsize=8.5, color=INK2)
    fig.tight_layout()
    save(fig, "fig3_lead_time_and_requests.png")

    other = {
        "market_segment": rows(tables["market_segment"]),
        "distribution_channel": rows(rate_by(df, "distribution_channel")),
        "meal": rows(rate_by(df, "meal")),
        "previous_cancellations_gt0": rows(rate_by(df, df["previous_cancellations"] > 0)),
        "is_repeated_guest": rows(rate_by(df, "is_repeated_guest")),
    }
    return {
        "overall_rate": round(base, 4),
        "deposit_type": rows(tables["deposit_type"]),
        "customer_type": rows(tables["customer_type"]),
        "hotel": rows(tables["hotel"]),
        "lead_time_buckets": rows(lead_t),
        "special_requests": rows(req_t),
        "other": other,
    }


# --------------------------------------------------------------------------- 4. timing audit
def timing_audit(df: pd.DataFrame) -> dict:
    with_country = df[df["country"].notna()]
    items = [
        ("parking: any requested / none", df, df["required_car_parking_spaces"] > 0),
        ("country: PRT / other", with_country, with_country["country"] == "PRT"),
        ("room: assigned differs from reserved / same", df,
         df["assigned_room_type"] != df["reserved_room_type"]),
        ("booking changes: 1+ / none", df, df["booking_changes"] > 0),
        ("special requests: 1+ / none", df, df["total_of_special_requests"] > 0),
        ("deposit type: Non Refund / other", df, df["deposit_type"] == "Non Refund"),
    ]
    out, pairs = {}, []
    for label, sub, mask in items:
        flag, rest = sub[mask], sub[~mask]
        out[label] = {
            "n_flag": len(flag),
            "rate_flag": round(float(flag[TARGET].mean()), 4),
            "n_other": len(rest),
            "rate_other": round(float(rest[TARGET].mean()), 4),
        }
        pairs.append((label, out[label]))

    fig, ax = plt.subplots(figsize=(10, 5.4))
    y = 0
    ticks, labels = [], []
    for label, r in pairs:
        name, groups = label.split(": ")
        g1, g2 = groups.split(" / ")
        for yy, grp, rate, n, color in (
            (y, g1, r["rate_flag"], r["n_flag"], BLUE),
            (y - 1, g2, r["rate_other"], r["n_other"], ORANGE),
        ):
            ax.barh(yy, rate * 100, color=color, height=0.8)
            ax.text(rate * 100 + 1.2, yy, f"{rate:.1%}  (n={n:,})", va="center", fontsize=8.5,
                    color=INK2)
            ticks.append(yy)
            labels.append(f"{name}: {grp}")
        y -= 3
    ax.set_yticks(ticks, labels)
    ax.set_xlim(0, 125)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel("cancellation rate (%)")
    ax.set_title("Cancellation rate inside vs outside groups defined by columns with doubtful timing")
    ax.grid(axis="y", visible=False)
    ax.legend(
        handles=[
            plt.Rectangle((0, 0), 1, 1, color=BLUE, label="group flagged by the column"),
            plt.Rectangle((0, 0), 1, 1, color=ORANGE, label="comparison group"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.14),
        ncol=2,
    )
    fig.tight_layout()
    save(fig, "fig4_timing_audit.png")

    status = pd.crosstab(df["reservation_status"], df[TARGET])
    out["reservation_status_vs_target"] = {
        s: {"kept": int(r.get(0, 0)), "canceled": int(r.get(1, 0))} for s, r in status.iterrows()
    }
    return out


# --------------------------------------------------------------------------- 5. drift
def drift_run(ref: pd.DataFrame, cur: pd.DataFrame, columns: list[str]) -> dict:
    num = [c for c in NUMERIC if c in columns]
    cat = [c for c in CATEGORICAL if c in columns]
    cols = num + cat
    definition = DataDefinition(numerical_columns=num, categorical_columns=cat)
    snapshot = Report([DataDriftPreset(columns=cols)]).run(
        current_data=Dataset.from_pandas(cur[cols].reset_index(drop=True), data_definition=definition),
        reference_data=Dataset.from_pandas(ref[cols].reset_index(drop=True), data_definition=definition),
    )
    count = share = None
    scores: dict[str, tuple[float, float | None]] = {}
    for metric in snapshot.dict()["metrics"]:
        kind = metric["config"]["type"].split(":")[-1]
        if kind == "DriftedColumnsCount":
            count, share = int(metric["value"]["count"]), float(metric["value"]["share"])
        elif kind == "ValueDrift":
            scores[metric["config"]["column"]] = (
                float(metric["value"]),
                metric["config"].get("threshold"),
            )
    flagged = [c for c, (s, t) in scores.items() if t is not None and s >= t]
    if count is None or len(flagged) != count or len(scores) != len(cols):
        raise RuntimeError(
            f"cannot reconcile Evidently output: count={count} flagged={len(flagged)} "
            f"scores={len(scores)} columns={len(cols)}"
        )
    # All columns use distance-based tests with the default threshold 0.1, so the share at
    # other thresholds follows directly from the scores (checked against Evidently's count).
    sweep = {
        str(t): round(sum(s >= t for s, _ in scores.values()) / len(cols), 4)
        for t in (0.1, 0.15, 0.2, 0.3)
    }
    if {th for _, th in scores.values()} != {0.1} or sweep["0.1"] != round(share, 4):
        raise RuntimeError("threshold sweep does not reproduce Evidently's own count")
    return {
        "columns": len(cols),
        "drifted": count,
        "share": round(share, 4),
        "share_if_threshold": sweep,
        "flagged_columns": sorted(flagged),
        "scores": {c: round(s, 3) for c, (s, _) in scores.items()},
        "thresholds": sorted({t for _, t in scores.values() if t is not None}),
    }


def drift_check(df: pd.DataFrame) -> dict:
    train, val = df[df["split"] == "train"], df[df["split"] == "val"]
    months = train["arrival_date"].dt.month
    same_season = train[(train["arrival_date"].dt.year == 2016) & (months <= 3)]
    every = NUMERIC + CATEGORICAL
    no_year = [c for c in every if c != "arrival_date_year"]
    no_calendar = [c for c in every if c not in CALENDAR]
    scenarios = {
        "S1 validation vs all train, all columns": (train, every),
        "S2 validation vs all train, calendar columns removed": (train, no_calendar),
        "S3 validation vs Jan-Mar 2016, year removed": (same_season, no_year),
        "S4 validation vs Jan-Mar 2016, calendar columns removed": (same_season, no_calendar),
    }
    out = {}
    for name, (ref, cols) in scenarios.items():
        out[name] = {"reference_rows": len(ref)} | drift_run(ref, val, cols)

    s1 = out["S1 validation vs all train, all columns"]
    flagged = set(s1["flagged_columns"])
    ordered = sorted(s1["scores"].items(), key=lambda kv: kv[1])
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 5.6), gridspec_kw={"width_ratios": [1.5, 1]})
    names = [k for k, _ in ordered]
    vals = [v for _, v in ordered]
    a.barh(names, vals, color=[ORANGE if k in flagged else BLUE for k in names], height=0.65)
    thr = s1["thresholds"]
    if len(thr) == 1:
        a.axvline(thr[0], color=MUTED, ls="--", lw=1.2)
        a.text(thr[0], len(names) - 0.4, f" threshold {thr[0]}", color=INK2, fontsize=8.5, va="bottom")
    a.set_xlabel("Evidently drift score (distance; larger = more different)")
    a.set_title("Clean validation window vs training data: score per column")
    a.grid(axis="y", visible=False)
    a.legend(
        handles=[
            plt.Rectangle((0, 0), 1, 1, color=ORANGE, label="flagged as drifted"),
            plt.Rectangle((0, 0), 1, 1, color=BLUE, label="not flagged"),
        ],
        loc="lower right",
    )
    short = [
        "all train,\nall columns",
        "all train,\nno calendar",
        "Jan-Mar 2016,\nno year",
        "Jan-Mar 2016,\nno calendar",
    ]
    shares = [v["share"] for v in out.values()]
    pos = np.arange(4)
    b.bar(pos, np.asarray(shares) * 100, color=BLUE, width=0.6)
    b.axhline(30, color=ORANGE, ls="--", lw=1.4)
    b.legend(
        handles=[plt.Line2D([0], [0], color=ORANGE, ls="--", lw=1.4, label="30% candidate alert level")],
        loc="upper right",
    )
    b.tick_params(axis="x", labelsize=8)
    for p, (s, v) in enumerate(zip(shares, out.values(), strict=True)):
        b.text(p, s * 100 + 1.5, f"{v['drifted']}/{v['columns']}", ha="center", fontsize=9,
               color=INK2)
    b.set_xticks(pos, short)
    b.set_ylim(0, 75)
    b.set_ylabel("columns flagged (%)")
    b.set_title("Share flagged by scenario")
    b.grid(axis="x", visible=False)
    fig.tight_layout()
    save(fig, "fig5_drift_on_clean_data.png")
    return out


# --------------------------------------------------------------------------- 6. baselines
def make_model(features: list[str], kind: str) -> Pipeline:
    unknown = [c for c in features if c not in NUMERIC_ALL + CATEGORICAL_ALL]
    if unknown:  # never drop a requested column silently
        raise ValueError(f"columns not classified as numeric or categorical: {unknown}")
    cat = [c for c in features if c in CATEGORICAL_ALL]
    num = [c for c in features if c in NUMERIC_ALL]
    if kind == "hgb":
        pre = ColumnTransformer(
            [
                (
                    "cat",
                    OrdinalEncoder(
                        handle_unknown="use_encoded_value",
                        unknown_value=-1,
                        encoded_missing_value=-1,
                    ),
                    cat,
                ),
                ("num", "passthrough", num),
            ]
        )
        clf = HistGradientBoostingClassifier(
            random_state=SEED, categorical_features=list(range(len(cat))) or None
        )
    else:
        pre = ColumnTransformer(
            [
                ("cat", OneHotEncoder(handle_unknown="ignore"), cat),
                ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), num),
            ]
        )
        clf = LogisticRegression(max_iter=2000)
    return Pipeline([("pre", pre), ("clf", clf)])


def baselines(df: pd.DataFrame) -> dict:
    train, val = df[df["split"] == "train"], df[df["split"] == "val"]
    y_tr, y_va = train[TARGET].to_numpy(), val[TARGET].to_numpy()
    prevalence = float(y_va.mean())

    non_refund = (val["deposit_type"] == "Non Refund").to_numpy()
    tp = int((non_refund & (y_va == 1)).sum())
    out = {
        "validation_prevalence": round(prevalence, 4),
        "majority_class": {"pr_auc": round(prevalence, 4), "roc_auc": 0.5, "recall": 0.0},
        "rule_non_refund": {
            "flagged_share": round(float(non_refund.mean()), 4),
            "precision": round(tp / int(non_refund.sum()), 4),
            "recall": round(tp / int(y_va.sum()), 4),
        },
    }

    runs = [("hgb", name, cols) for name, cols in FEATURE_SETS.items()]
    runs += [("logreg", n, FEATURE_SETS[n]) for n in ("strict", "proposal")]
    feature_cols = {(kind, name): cols for kind, name, cols in runs}
    probs = {}
    for kind, name, cols in runs:
        model = make_model(cols, kind).fit(train[cols], y_tr)
        probs[(kind, name)] = model.predict_proba(val[cols])[:, 1]

    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(y_va), size=(N_BOOT, len(y_va)))
    boot = {
        key: np.array([average_precision_score(y_va[i], p[i]) for i in idx])
        for key, p in probs.items()
    }
    point = {key: float(average_precision_score(y_va, p)) for key, p in probs.items()}
    for kind in ("hgb", "logreg"):
        out[kind] = {}
        ref_key = (kind, "strict")
        for (k, name), p in probs.items():
            if k != kind:
                continue
            diff = boot[(k, name)] - boot[ref_key]
            lo, hi = np.percentile(boot[(k, name)], [2.5, 97.5])
            dlo, dhi = np.percentile(diff, [2.5, 97.5])
            out[kind][name] = {
                "pr_auc": round(point[(k, name)], 4),
                "pr_auc_ci95": [round(float(lo), 4), round(float(hi), 4)],
                "roc_auc": round(float(roc_auc_score(y_va, p)), 4),
                "pr_auc_minus_strict": round(point[(k, name)] - point[ref_key], 4),
                "pr_auc_minus_strict_ci95": [round(float(dlo), 4), round(float(dhi), 4)],
                "n_features": len(feature_cols[(k, name)]),
            }

    fig, ax = plt.subplots(figsize=(10, 5.6))
    names = list(FEATURE_SETS)
    ypos = np.arange(len(names))[::-1]
    for offset, kind, color, label in ((0.12, "hgb", BLUE, "gradient boosting"),
                                       (-0.12, "logreg", ORANGE, "logistic regression")):
        for yy, name in zip(ypos, names, strict=True):
            r = out[kind].get(name)
            if r is None:
                continue
            ax.plot(r["pr_auc_ci95"], [yy + offset] * 2, color=color, lw=2)
            ax.plot(r["pr_auc"], yy + offset, "o", color=color, ms=7, label=label)
            ax.text(r["pr_auc_ci95"][1] + 0.002, yy + offset, f"{r['pr_auc']:.3f}", va="center",
                    fontsize=8.5, color=INK2)
    handles, labels_ = ax.get_legend_handles_labels()
    uniq = dict(zip(labels_, handles, strict=True))
    ax.legend(uniq.values(), uniq.keys(), loc="lower right")
    ax.set_yticks(ypos, names)
    ax.set_xlabel("validation PR-AUC with 95% bootstrap interval (higher is better)")
    ax.set_title(
        f"Same model, different feature sets (no-skill PR-AUC = {prevalence:.2f}; untuned, one split)"
    )
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    save(fig, "fig6_feature_set_ablation.png")
    return out


def main() -> None:
    warnings.filterwarnings("ignore")
    style()
    df = load_design_data()
    assert set(df["split"]) == {"train", "val"}
    results = {
        "scope": "train + validation only; the test split is never loaded",
        "overview": overview(df),
        "time": time_profile(df),
        "drivers": drivers(df),
        "timing_audit": timing_audit(df),
        "drift_check": drift_check(df),
        "baselines": baselines(df),
    }
    RESULTS_PATH.write_text(json.dumps(results, indent=2) + "\n")
    print(f"wrote {RESULTS_PATH.relative_to(REPO_ROOT)} and {len(list(FIG_DIR.glob('*.png')))} figures")


if __name__ == "__main__":
    main()
