"""
preprocess.py

Handles train-test splitting and scaling for the Flood Risk system.

DESIGN PRINCIPLES:
- Train-test split ALWAYS before scaling
- Scaler fitted ONLY on training data
- Test data NEVER influences training statistics
- Artifacts saved for backend inference
"""

import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
from typing import Tuple


def split_data(
    df: pd.DataFrame,
    target_col: str,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Splits dataframe into train and test sets.

    Parameters:
    - df: full dataframe (features + target)
    - target_col: name of target column
    - test_size: proportion for test split
    - random_state: reproducibility

    Returns:
    - X_train, X_test, y_train, y_test
    """

    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    return X_train, X_test, y_train, y_test


def scale_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    scaler_path: str = "ml/artifacts/scaler.pkl"
):
    """
    Scales numerical - features using StandardScaler.

    IMPORTANT:
    - Fit ONLY on X_train
    - Transform X_train and X_test
    """

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Save scaler for backend inference
    joblib.dump(scaler, scaler_path)

    return X_train_scaled, X_test_scaled
