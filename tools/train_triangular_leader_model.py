# train_triangular_leader_model.py – Leader-Aware XGBoost Trainer (Upgraded for btc_usd / eth_usd / eth_btc)

import pandas as pd
import numpy as np
import time
import os
from xgboost import XGBClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report
import joblib

# ─── Step Logger ───
LABELS = []
start_time = time.time()

def log(step):
    now = time.time()
    LABELS.append((step, now))
    print(f"✅ Step: {step} @ {time.strftime('%H:%M:%S', time.localtime(now))}")

log("Start Training Script")

# ─── Load Labeled Feature Set ───
df = pd.read_csv("features_triangular_labeled.csv", parse_dates=["timestamp"])
log("Loaded features_triangular_base.csv")

if "label" not in df.columns:
    raise ValueError("❌ Missing 'label' column. Please label the data before training.")

df = df[df["label"].isin([-1, 1])]
log(f"Filtered to directional labels only. Counts:\n{df['label'].value_counts()}")

# ─── Feature Matrix + Labels ───
FEATURES = [
    "btc_usd", "eth_usd", "eth_btc",
    "implied_ethbtc", "spread", "spread_zscore",
    "vol_spread", "spread_ewma", "spread_kalman"
]

missing = [f for f in FEATURES if f not in df.columns]
if missing:
    raise ValueError(f"❌ Missing expected features: {missing}")

X = df[FEATURES]
y = df["label"].map({-1: 0, 1: 1})
log("Prepared features and labels")

# ─── Time-Series Split ───
split = TimeSeriesSplit(n_splits=5)
for train_idx, val_idx in split.split(X):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    break

log("Performed time-series aware train/test split")

# ─── Model Training ───
model = XGBClassifier(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.1,
    use_label_encoder=False,
    eval_metric="logloss"
)
model.fit(X_train, y_train)
log("Trained XGBoost classifier")

# ─── Calibration ───
calibrated = CalibratedClassifierCV(model, method="isotonic", cv="prefit")
calibrated.fit(X_val, y_val)
log("Calibrated probabilities")

# ─── Evaluation ───
y_pred = calibrated.predict(X_val)
print("\n📊 Classification Report:\n")
print(classification_report(y_val, y_pred, digits=4))
log("Evaluated and printed report")

# ─── Save Model ───
os.makedirs("ml_model", exist_ok=True)
joblib.dump(calibrated, "ml_model/triangular_rf_leader.pkl")
log("Saved model to ml_model/triangular_rf_leader.pkl")

# ─── Duration Summary ───
print("\n🕓 Process Duration Breakdown:")
for i in range(1, len(LABELS)):
    label, t1 = LABELS[i]
    prev_label, t0 = LABELS[i - 1]
    print(f"{prev_label} ➔ {label}: {t1 - t0:.2f}s")

print(f"\n✅ Done. Total Training Time: {time.time() - start_time:.2f}s")
