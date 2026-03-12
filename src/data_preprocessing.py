from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


TARGET_COL = "Churn"
ID_COL = "customerID"
NUMERIC_COLS_BASE = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]


@dataclass
class PreprocessingArtifact:
    feature_columns: list[str]
    numeric_columns: list[str]
    scaler: StandardScaler


def load_dataset(data_path: str | Path) -> pd.DataFrame:
    """Load Telco churn dataset."""
    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")
    return pd.read_csv(data_path)


def clean_telco_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and standardize dataset columns."""
    clean_df = df.copy()

    if "TotalCharges" in clean_df.columns:
        clean_df["TotalCharges"] = pd.to_numeric(clean_df["TotalCharges"], errors="coerce")

    clean_df = clean_df.dropna().reset_index(drop=True)

    if ID_COL in clean_df.columns:
        clean_df = clean_df.drop(columns=[ID_COL])

    return clean_df


def build_features(
    df: pd.DataFrame,
    training: bool = True,
    feature_columns: Optional[list[str]] = None,
    scaler: Optional[StandardScaler] = None,
) -> tuple[pd.DataFrame, Optional[pd.Series], Optional[PreprocessingArtifact]]:
    """
    Build model-ready features with one-hot encoding + scaling.

    Returns:
        X_transformed, y, preprocessing_artifact (artifact is only returned during training)
    """
    working_df = clean_telco_dataframe(df)

    y: Optional[pd.Series] = None
    if TARGET_COL in working_df.columns:
        y = working_df[TARGET_COL].map({"Yes": 1, "No": 0}).astype(int)
        working_df = working_df.drop(columns=[TARGET_COL])

    X = pd.get_dummies(working_df, drop_first=False)

    numeric_columns = [c for c in NUMERIC_COLS_BASE if c in X.columns]

    if training:
        scaler = StandardScaler()
        if numeric_columns:
            for col in numeric_columns:
                X[col] = X[col].astype(float)
            transformed = scaler.fit_transform(X[numeric_columns])
            for idx, col in enumerate(numeric_columns):
                X[col] = transformed[:, idx]

        artifact = PreprocessingArtifact(
            feature_columns=X.columns.tolist(),
            numeric_columns=numeric_columns,
            scaler=scaler,
        )
        return X, y, artifact

    if feature_columns is None or scaler is None:
        raise ValueError("For inference, feature_columns and scaler must be provided.")

    X = X.reindex(columns=feature_columns, fill_value=0)

    numeric_columns = [c for c in NUMERIC_COLS_BASE if c in X.columns]
    if numeric_columns:
        for col in numeric_columns:
            X[col] = X[col].astype(float)
        transformed = scaler.transform(X[numeric_columns])
        for idx, col in enumerate(numeric_columns):
            X[col] = transformed[:, idx]

    return X, y, None


def split_dataset(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Create train-test split with stratification."""
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
