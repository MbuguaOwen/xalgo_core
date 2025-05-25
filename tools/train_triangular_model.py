import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report
import joblib
import os

# ─────────────────────────────────────────────
# 📥 Load Labeled Data
# ─────────────────────────────────────────────
print("📥 Loading labeled data...")
df = pd.read_csv("features_triangular_labeled.csv", parse_dates=["timestamp"])

# Drop only if essential features are missing
required_cols = [
    "btc_usd", "eth_usd", "eth_btc",
    "implied_ethbtc", "spread", "spread_zscore",
    "vol_spread", "spread_ewma", "spread_kalman", "label"
]
df = df.dropna(subset=required_cols)

# Filter out HOLDs (label == 0)
df = df[df["label"] != 0]

if df.empty:
    raise ValueError("❌ No rows remaining after filtering. Check your label distribution or data quality.")

# ─────────────────────────────────────────────
# 🧠 Feature & Label Extraction
# ─────────────────────────────────────────────
FEATURES = [
    "btc_usd", "eth_usd", "eth_btc",
    "implied_ethbtc", "spread", "spread_zscore",
    "vol_spread", "spread_ewma", "spread_kalman"
]
X = df[FEATURES]

# ✅ Remap labels: -1 → 0, +1 → 1 for XGBoost compatibility
label_map = {-1: 0, 1: 1}
y = df["label"].map(label_map)

# Inverse map (note for main.py): 0 → -1 (SELL), 1 → +1 (BUY)
inverse_label_map = {0: -1, 1: 1}

# ─────────────────────────────────────────────
# 🔀 Time-Series Aware Train/Val Split
# ─────────────────────────────────────────────
print("🔀 Performing time-series split...")
split = TimeSeriesSplit(n_splits=5)
for train_idx, val_idx in split.split(X):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    break  # Use the first fold only

# ─────────────────────────────────────────────
# 🚀 Train XGBoost Classifier
# ─────────────────────────────────────────────
print("🚀 Training model...")
model = XGBClassifier(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.1,
    use_label_encoder=False,
    eval_metric="logloss"
)
model.fit(X_train, y_train)

# ─────────────────────────────────────────────
# 🎯 Calibrate Probabilities
# ─────────────────────────────────────────────
print("🎯 Calibrating confidence scores...")
calibrated = CalibratedClassifierCV(model, method="isotonic", cv="prefit")
calibrated.fit(X_val, y_val)

# ─────────────────────────────────────────────
# 📈 Evaluation
# ─────────────────────────────────────────────
print("📈 Classification report:\n")
y_pred = calibrated.predict(X_val)
print(classification_report(y_val, y_pred, digits=4))

# ─────────────────────────────────────────────
# 💾 Save Model
# ─────────────────────────────────────────────
os.makedirs("ml_model", exist_ok=True)
joblib.dump(calibrated, "ml_model/triangular_rf_model.pkl")
print("✅ Model saved to ml_model/triangular_rf_model.pkl")
