"""
evaluate.py

Evaluates the trained Flood Risk model using probability-based metrics.

WHY THIS MATTERS:
- Flood systems care about RISK, not just correctness
- Probability calibration and ROC-AUC matter more than accuracy
"""

import pandas as pd
import joblib
import numpy as np

from sklearn.metrics import (
    roc_auc_score,
    classification_report,
    confusion_matrix
)

from features import add_engineered_features
from preprocess import split_data, scale_features


# =========================
# CONFIGURATION
# =========================

DATA_PATH = "ml/data/flood_data_processed.csv"
TARGET_COLUMN = "Flood_Occurred"

MODEL_PATH = "ml/artifacts/model.pkl"
SCALER_PATH = "ml/artifacts/scaler.pkl"
FEATURE_COLUMNS_PATH = "ml/artifacts/feature_columns.pkl"


# =========================
# LOAD ARTIFACTS
# =========================

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
feature_columns = joblib.load(FEATURE_COLUMNS_PATH)


# =========================
# LOAD DATA
# =========================

df = pd.read_csv(DATA_PATH)

df = add_engineered_features(df)


# =========================
# SPLIT DATA (SAME SEED)
# =========================

X_train, X_test, y_train, y_test = split_data(
    df=df,
    target_col=TARGET_COLUMN,
    test_size=0.2,
    random_state=42
)

# Ensure column order consistency
X_test = X_test[feature_columns]


# =========================
# SCALE TEST DATA
# =========================

X_test_scaled = scaler.transform(X_test)


# =========================
# PREDICTIONS
# =========================

y_proba = model.predict_proba(X_test_scaled)[:, 1]
y_pred = (y_proba >= 0.5).astype(int)


# =========================
# METRICS
# =========================

roc_auc = roc_auc_score(y_test, y_proba)

print("\n📊 FLOOD RISK MODEL EVALUATION")
print("-" * 40)
print(f"ROC-AUC Score: {roc_auc:.4f}\n")

print("Classification Report (threshold = 0.5):")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# =========================
# PROBABILITY INSPECTION
# =========================

print("\n🔍 Sample predicted probabilities:")
print(np.round(y_proba[:10], 3))
