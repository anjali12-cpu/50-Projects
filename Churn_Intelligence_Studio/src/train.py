"""Train classical ML models and a small PyTorch MLP."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.features import FEATURES, TARGET, make_preprocessor

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "customers.csv"
ARTIFACTS = ROOT / "artifacts"
SEED = 42


class ChurnMLP(nn.Module):
    def __init__(self, input_size: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x).squeeze(1)


def metrics(y_true: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    pred = (probability >= 0.5).astype(int)
    return {
        "accuracy": round(float(accuracy_score(y_true, pred)), 4),
        "f1": round(float(f1_score(y_true, pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, probability)), 4),
    }


def train() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError("Dataset not found. Run: python -m src.data")
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_PATH)
    X = df[FEATURES]
    y = df[TARGET].astype(int).to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )

    models = {
        "logistic_regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "random_forest": RandomForestClassifier(
            n_estimators=250, min_samples_leaf=3, class_weight="balanced", random_state=SEED
        ),
    }
    report: dict[str, dict[str, float]] = {}
    for name, estimator in models.items():
        pipeline = Pipeline([("preprocessor", make_preprocessor()), ("model", estimator)])
        pipeline.fit(X_train, y_train)
        probability = pipeline.predict_proba(X_test)[:, 1]
        report[name] = metrics(y_test, probability)
        joblib.dump(pipeline, ARTIFACTS / f"{name}.joblib")

    preprocessor = make_preprocessor()
    train_array = preprocessor.fit_transform(X_train).astype(np.float32)
    test_array = preprocessor.transform(X_test).astype(np.float32)
    if hasattr(train_array, "toarray"):
        train_array, test_array = train_array.toarray(), test_array.toarray()

    train_ds = TensorDataset(
        torch.from_numpy(train_array), torch.from_numpy(y_train.astype(np.float32))
    )
    loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    net = ChurnMLP(train_array.shape[1])
    optimizer = torch.optim.Adam(net.parameters(), lr=0.003)
    loss_fn = nn.BCEWithLogitsLoss()
    net.train()
    for _ in range(35):
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = loss_fn(net(xb), yb)
            loss.backward()
            optimizer.step()

    net.eval()
    with torch.no_grad():
        torch_probability = torch.sigmoid(net(torch.from_numpy(test_array))).numpy()
    report["pytorch_mlp"] = metrics(y_test, torch_probability)

    torch.save(
        {
            "state_dict": net.state_dict(),
            "input_size": train_array.shape[1],
            "feature_names": preprocessor.get_feature_names_out().tolist(),
        },
        ARTIFACTS / "pytorch_mlp.pt",
    )
    joblib.dump(preprocessor, ARTIFACTS / "pytorch_preprocessor.joblib")
    (ARTIFACTS / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Saved models and report in {ARTIFACTS}")


if __name__ == "__main__":
    train()
