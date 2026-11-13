"""Generate a small, reproducible synthetic customer churn dataset."""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
DATA_PATH = DATA_DIR / "customers.csv"


def make_dataset(rows: int = 2500, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    plan = rng.choice(["Basic", "Plus", "Premium"], rows, p=[0.48, 0.34, 0.18])
    contract = rng.choice(["Monthly", "Annual", "Two-year"], rows, p=[0.52, 0.31, 0.17])
    tenure = rng.integers(1, 73, rows)
    monthly_charge = np.round(rng.normal(78, 25, rows).clip(20, 180), 2)
    support_calls = rng.poisson(1.2, rows).clip(0, 8)
    late_payments = rng.poisson(0.45, rows).clip(0, 5)
    usage_hours = np.round(rng.gamma(4.0, 8.0, rows).clip(1, 100), 1)

    # A transparent simulation rule creates a learnable target, not real-world labels.
    logit = (
        -2.1
        + 0.65 * (contract == "Monthly")
        + 0.55 * (plan == "Basic")
        + 0.52 * support_calls
        + 0.42 * late_payments
        + 0.65 * (tenure < 8)
        + 0.012 * (monthly_charge - 75)
        - 0.018 * usage_hours
    )
    probability = 1 / (1 + np.exp(-logit))
    churned = rng.binomial(1, probability)

    return pd.DataFrame(
        {
            "plan": plan,
            "contract": contract,
            "tenure_months": tenure,
            "monthly_charge": monthly_charge,
            "support_calls_90d": support_calls,
            "late_payments_90d": late_payments,
            "usage_hours_monthly": usage_hours,
            "churned": churned,
        }
    )


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df = make_dataset()
    df.to_csv(DATA_PATH, index=False)
    print(f"Created {len(df):,} synthetic rows: {DATA_PATH}")
    print(f"Simulated churn rate: {df['churned'].mean():.1%}")


if __name__ == "__main__":
    main()
