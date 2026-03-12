from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

DEFAULT_CUSTOMER_TEMPLATE = {
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "No",
    "Dependents": "No",
    "tenure": 12,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 75.0,
    "TotalCharges": 900.0,
}


class ChurnPredictor:
    def __init__(self, model_path: str | Path) -> None:
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model artifact not found: {model_path}")

        artifact = joblib.load(model_path)
        self.model = artifact["model"]
        self.feature_columns = artifact["feature_columns"]
        self.numeric_columns = artifact["numeric_columns"]
        self.scaler = artifact["scaler"]

    def _build_single_row(self, payload: dict[str, Any]) -> pd.DataFrame:
        row = DEFAULT_CUSTOMER_TEMPLATE.copy()
        row.update(payload)

        row["SeniorCitizen"] = int(row["SeniorCitizen"])
        row["tenure"] = float(row["tenure"])
        row["MonthlyCharges"] = float(row["MonthlyCharges"])

        if "TotalCharges" not in payload or payload["TotalCharges"] in (None, ""):
            row["TotalCharges"] = row["tenure"] * row["MonthlyCharges"]
        else:
            row["TotalCharges"] = float(payload["TotalCharges"])

        return pd.DataFrame([row])

    def _transform(self, payload: dict[str, Any]) -> pd.DataFrame:
        input_df = self._build_single_row(payload)
        encoded = pd.get_dummies(input_df, drop_first=False)
        encoded = encoded.reindex(columns=self.feature_columns, fill_value=0)

        numeric_cols = [c for c in self.numeric_columns if c in encoded.columns]
        if numeric_cols:
            for col in numeric_cols:
                encoded[col] = encoded[col].astype(float)
            transformed = self.scaler.transform(encoded[numeric_cols])
            for idx, col in enumerate(numeric_cols):
                encoded[col] = transformed[:, idx]

        return encoded

    def predict_proba(self, payload: dict[str, Any]) -> float:
        transformed = self._transform(payload)
        probability = float(self.model.predict_proba(transformed)[0][1])
        return probability

    def predict_label(self, payload: dict[str, Any], threshold: float = 0.5) -> str:
        prob = self.predict_proba(payload)
        return "Yes" if prob >= threshold else "No"


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict churn probability for a single customer.")
    parser.add_argument("--model-path", default="models/churn_model.pkl", help="Path to trained model artifact.")
    parser.add_argument("--gender", default="Female")
    parser.add_argument("--senior-citizen", type=int, default=0)
    parser.add_argument("--tenure", type=float, default=12)
    parser.add_argument("--contract", default="Month-to-month")
    parser.add_argument("--internet-service", default="Fiber optic")
    parser.add_argument("--monthly-charges", type=float, default=75.0)
    parser.add_argument("--payment-method", default="Electronic check")
    args = parser.parse_args()

    payload = {
        "gender": args.gender,
        "SeniorCitizen": args.senior_citizen,
        "tenure": args.tenure,
        "Contract": args.contract,
        "InternetService": args.internet_service,
        "MonthlyCharges": args.monthly_charges,
        "PaymentMethod": args.payment_method,
    }

    predictor = ChurnPredictor(args.model_path)
    prob = predictor.predict_proba(payload)
    label = "Yes" if prob >= 0.5 else "No"

    print(f"Churn probability: {prob:.2%}")
    print(f"Predicted churn: {label}")


if __name__ == "__main__":
    main()
