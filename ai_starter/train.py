from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def train_and_save(model_path: Path, test_size: float, random_state: int) -> dict[str, float]:
    iris = load_iris()
    x_train, x_test, y_train, y_test = train_test_split(
        iris.data,
        iris.target,
        test_size=test_size,
        random_state=random_state,
        stratify=iris.target,
    )

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=300, random_state=random_state)),
        ]
    )
    model.fit(x_train, y_train)

    predictions = model.predict(x_test)
    metrics = {
        "accuracy": round(float(accuracy_score(y_test, predictions)), 4),
        "f1_macro": round(float(f1_score(y_test, predictions, average="macro")), 4),
        "test_size": test_size,
        "random_state": float(random_state),
    }

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)

    metrics_path = model_path.with_suffix(".metrics.json")
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and export a starter ML model artifact.")
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("model/model.joblib"),
        help="Path where the trained model artifact will be saved.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Test split ratio.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducible training and splitting.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = train_and_save(args.model_path, args.test_size, args.random_state)
    print(f"Saved model to: {args.model_path}")
    print("Evaluation metrics:")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
