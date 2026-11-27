"""Model recipes shared by the EDA and the training pipeline.

Every model is one sklearn Pipeline: preprocessing + classifier. The pipeline takes the raw
booking columns, so serving can pass requests straight in and runs exactly the preprocessing
used in training (no training-serving skew).
"""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from src.data.columns import CATEGORICAL_ALL, NUMERIC_ALL

RANDOM_STATE = 0


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
            random_state=RANDOM_STATE, categorical_features=list(range(len(cat))) or None
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
