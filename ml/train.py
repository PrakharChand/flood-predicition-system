"""
train.py

Trains the Random Forest model for Flood Risk Assessment.

PIPELINE ORDER (DO NOT CHANGE):
1. Load processed dataset
2. Apply feature engineering
3. Train-test split
4. Scaling (fit on train only)
5. Model training
6. Save artifacts

This file is executed ONCE for training.
Backend will NEVER touch this.
"""

import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier

from features import add_engineered_features
from preprocess import split_data, scale_features


# =========================
# CONFIGURATION
# =========================

DATA_PATH = "ml/data/flood_data_processed.csv"
TARGET_COLUMN = "Flood_Occurred"

MODEL_PATH = "ml/artifacts/model.pkl"
FEATURE_COLUMNS_PATH = "ml/artifacts/feature_columns.pkl"


# =========================
# LOAD DATA
# =========================

df = pd.read_csv(DATA_PATH)

# Safety check
if TARGET_COLUMN not in df.columns:
    raise ValueError(f"Target column '{TARGET_COLUMN}' not found in dataset")


# =========================
# FEATURE ENGINEERING
# =========================

df = add_engineered_features(df)


# =========================
# TRAIN-TEST SPLIT
# =========================

X_train, X_test, y_train, y_test = split_data(
    df=df,
    target_col=TARGET_COLUMN,
    test_size=0.2,
    random_state=42
)

# Save feature column order (CRITICAL for backend inference)
joblib.dump(
    X_train.columns.tolist(),
    FEATURE_COLUMNS_PATH
)


# =========================
# SCALING
# =========================

X_train_scaled, X_test_scaled = scale_features(
    X_train=X_train,
    X_test=X_test
)


# =========================
# MODEL TRAINING
# =========================

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=12,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1
)

model.fit(X_train_scaled, y_train)


# =========================
# SAVE MODEL
# =========================

joblib.dump(model, MODEL_PATH)

print("✅ Model training completed successfully.")
print("📦 Artifacts saved:")
print(" - Model:", MODEL_PATH)
print(" - Feature Columns:", FEATURE_COLUMNS_PATH)
print(" - Scaler: ml/artifacts/scaler.pkl")
