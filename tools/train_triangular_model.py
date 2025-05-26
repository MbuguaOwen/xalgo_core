# train_triangular_leader_model.py – Optimized XGBoost Model Training with Pair Leader

import pandas as pd
import numpy as np
import time
from xgboost import XGBClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report
import joblib
import os

# ─────────────────────────────────────────────
# ⏱️ Timed Label Logger
# ─────────────────────────────────────────────
LABELS = []
start_time = time.time()

def log(step):
    now = time.time()
    LABELS.append((step, now))
    print(f"✅ Step: {step} @ {time.strftime('%H:%M:%S', time.localtime(now))}")

log("Begin Model Training Script")

# ─────────────────────────────────────────────
# Load Data
# ─────────────────────────────────────────────
log("Loading features_triangular_base.csv...")
df = pd.read_csv("features_triangular_base.csv", parse_dates=["timestamp"])

df = df.dropna(subset=[
    "btc_price", "eth_price", "ethbtc_price",
    "implied_ethbtc", "spread", "spread_zscore",
    "vol_spread", "spread_ewma", "spread_kalman",
    "pair_leader_encoded"
])

if "label" not in df.columns:
    raise ValueError("❌ 'label' column is missing. Please run label generation first.")

df = df[df["label"].isin([-1, 1])]
y = df["label"].map({-1: 0, 1: 1})

log("Cleaned and filtered labeled data")

# ─────────────────────────────────────────────
# Features
# ─────────────────────────────────────────────
FEATURES = [
    "btc_price", "eth_price", "ethbtc_price",
    "implied_ethbtc", "spread", "spread_zscore",
    "vol_spread", "spread_ewma", "spread_kalman",
    "pair_leader_encoded"
]
X = df[FEATURES]
log("Selected Features")

# ─────────────────────────────────────────────
# Time Series Split
# ─────────────────────────────────────────────
log("Performing time-series split...")
split = TimeSeriesSplit(n_splits=5)
for train_idx, val_idx in split.split(X):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    break

log("Prepared Train/Validation sets")

# ─────────────────────────────────────────────
# Model Training
# ─────────────────────────────────────────────
log("Training XGBoost model...")
model = XGBClassifier(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.1,
    use_label_encoder=False,
    eval_metric="logloss"
)
model.fit(X_train, y_train)
log("Trained XGBoost base model")

# ─────────────────────────────────────────────
# Calibration
# ─────────────────────────────────────────────
log("Calibrating probabilities...")
calibrated = CalibratedClassifierCV(model, method="isotonic", cv="prefit")
calibrated.fit(X_val, y_val)
log("Calibrated probabilities using isotonic regression")

# ─────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────
print("\n📊 Evaluation Metrics:")
y_pred = calibrated.predict(X_val)
print(classification_report(y_val, y_pred, digits=4))
log("Printed classification report")

# ─────────────────────────────────────────────
# Save Model
# ─────────────────────────────────────────────
os.makedirs("ml_model", exist_ok=True)
joblib.dump(calibrated, "ml_model/triangular_rf_leader.pkl")
log("Saved calibrated model to disk")

# ─────────────────────────────────────────────
# Summary Timing
# ─────────────────────────────────────────────
print("\n🕓 Step Duration Breakdown:")
for i in range(1, len(LABELS)):
    prev, t0 = LABELS[i - 1]
    curr, t1 = LABELS[i]
    print(f"{prev} ➝ {curr}: {t1 - t0:.2f}s")

print(f"\n✅ Done. Total Time: {time.time() - start_time:.2f}s")
