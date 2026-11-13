"""Shared feature columns and preprocessing builder."""

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

CATEGORICAL_FEATURES = ["plan", "contract"]
NUMERIC_FEATURES = [
    "tenure_months",
    "monthly_charge",
    "support_calls_90d",
    "late_payments_90d",
    "usage_hours_monthly",
]
FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES
TARGET = "churned"


def make_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        [
            ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
            ("numeric", StandardScaler(), NUMERIC_FEATURES),
        ]
    )
