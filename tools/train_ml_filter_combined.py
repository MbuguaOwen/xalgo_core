import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import matplotlib.pyplot as plt
import os

# ─────────────────────────────────────────────
# 📥 Load and Filter Data
# ─────────────────────────────────────────────
print("📂 Loading features_triangular_labeled.csv...")
df = pd.read_csv("features_triangular_labeled.csv", parse_dates=["timestamp"])
df = df[df["label"].isin([-1, 1])]  # Drop label == 0

X = df[[
    "spread", "spread_zscore", "vol_spread", "spread_ewma", "spread_kalman"
]]
y = df["label"]

# ─────────────────────────────────────────────
# 🔀 Time-Series Aware Train/Calibrate/Test Split
# ─────────────────────────────────────────────
print("⏱️ Performing time series split...")
tscv = TimeSeriesSplit(n_splits=5)
splits = list(tscv.split(X))

train_idx, test_idx = splits[-1]
val_idx = splits[-2][1]  # Previous test becomes calibration set

X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]

# ─────────────────────────────────────────────
# 🧠 Train + Calibrate Model
# ─────────────────────────────────────────────
print("🧠 Training RandomForest model...")
clf = RandomForestClassifier(n_estimators=100, max_depth=6, class_weight="balanced", random_state=42)
clf.fit(X_train, y_train)

print("🎯 Calibrating with isotonic regression...")
calibrated = CalibratedClassifierCV(clf, method="isotonic", cv="prefit")
calibrated.fit(X_val, y_val)

# ─────────────────────────────────────────────
# 📈 Evaluate on Test Set
# ─────────────────────────────────────────────
print("📊 Evaluating on test set...")
y_pred = calibrated.predict(X_test)
