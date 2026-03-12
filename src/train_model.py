from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

from data_preprocessing import build_features, load_dataset, split_dataset


RANDOM_STATE = 42


def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float]:
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_prob)),
    }


def train_models(data_path: str | Path, model_output: str | Path) -> dict:
    df = load_dataset(data_path)
    X, y, prep_artifact = build_features(df, training=True)

    if y is None:
        raise ValueError("Target column 'Churn' is missing from dataset.")

    X_train, X_test, y_train, y_test = split_dataset(X, y, test_size=0.2, random_state=RANDOM_STATE)

    logistic_model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    random_forest_model = RandomForestClassifier(
        n_estimators=400,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    logistic_model.fit(X_train, y_train)
    random_forest_model.fit(X_train, y_train)

    logistic_metrics = evaluate_model(logistic_model, X_test, y_test)
    random_forest_metrics = evaluate_model(random_forest_model, X_test, y_test)

    model_scores = {
        "LogisticRegression": logistic_metrics,
        "RandomForest": random_forest_metrics,
    }

    best_model_name = max(model_scores.keys(), key=lambda name: model_scores[name]["roc_auc"])
    best_model = logistic_model if best_model_name == "LogisticRegression" else random_forest_model

    artifact = {
        "model": best_model,
        "model_name": best_model_name,
        "metrics": model_scores,
        "feature_columns": prep_artifact.feature_columns,
        "numeric_columns": prep_artifact.numeric_columns,
        "scaler": prep_artifact.scaler,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
    }

    model_output = Path(model_output)
    model_output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, model_output)

    return artifact


def main() -> None:
    parser = argparse.ArgumentParser(description="Train customer churn model and save best model artifact.")
    parser.add_argument(
        "--data-path",
        default="data/Telco-Customer-Churn.csv",
        help="Path to CSV dataset.",
    )
    parser.add_argument(
        "--model-output",
        default="models/churn_model.pkl",
        help="Output path for saved model artifact.",
    )
    args = parser.parse_args()

    artifact = train_models(args.data_path, args.model_output)

    print("Training complete.")
    print(f"Best model: {artifact['model_name']}")
    print("Model comparison:")
    for model_name, metrics in artifact["metrics"].items():
        metric_summary = ", ".join([f"{k}={v:.4f}" for k, v in metrics.items()])
        print(f"  - {model_name}: {metric_summary}")
    print(f"Saved model to: {args.model_output}")


if __name__ == "__main__":
    main()
