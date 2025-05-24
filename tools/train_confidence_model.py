#!/usr/bin/env python3
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report, confusion_matrix, log_loss
import joblib
from xgboost import XGBClassifier

# ─────────────────────────────────────────────
# ⚙️ Config
# ─────────────────────────────────────────────
DATA_PATH = "features_triangular_labeled.csv"
MODEL_DIR = "ml_model"
os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_OUTPUT = os.path.join(MODEL_DIR, "triangular_rf_model.pkl")
AUDIT_CSV = os.path.join(MODEL_DIR, "audit_confidence_model.csv")
CONF_PLOT = os.path.join(MODEL_DIR, "confidence_histogram_conf_model.png")

FEATURE_COLUMNS = [
    "btc_usd", "eth_usd", "eth_btc",
    "implied_ethbtc", "spread", "spread_zscore",
    "vol_spread", "spread_ewma", "spread_kalman",
    "direction"
]

# Remapping logic for binary model
label_map = {-1: 0, 1: 1}
inverse_map = {0: -1, 1: 1}

# ─────────────────────────────────────────────
# 📥 Load Data
# ─────────────────────────────────────────────
print("📂 Loading features_triangular_labeled.csv...")
df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])
df = df[df["label"].isin([-1, 1])].dropna(subset=FEATURE_COLUMNS)

df["y"] = df["label"].map(label_map)

print(f"🔍 Filtered label counts:\n{df['label'].value_counts().sort_index()}")

X = df[FEATURE_COLUMNS]
y = df["y"]

# ─────────────────────────────────────────────
# ⏳ Time-Aware Split
# ─────────────────────────────────────────────
tscv = TimeSeriesSplit(n_splits=5)
splits = list(tscv.split(X))
train_idx, test_idx = splits[-1]
val_idx = splits[-2][1]

X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
X_val, y_val     = X.iloc[val_idx],   y.iloc[val_idx]
X_test, y_test   = X.iloc[test_idx],  y.iloc[test_idx]

# ─────────────────────────────────────────────
# 🧠 Train + Calibrate
# ─────────────────────────────────────────────
print("⚡ Training XGBoost (binary)...")
xgb = XGBClassifier(
    n_estimators=150, max_depth=5, learning_rate=0.05,
    subsample=0.9, colsample_bytree=0.8,
    use_label_encoder=False, eval_metric="logloss",
    random_state=42, n_jobs=-1
)
xgb.fit(X_train, y_train)

cal_model = CalibratedClassifierCV(xgb, method="isotonic", cv="prefit")
cal_model.fit(X_val, y_val)

# ─────────────────────────────────────────────
# 📊 Evaluate on Test Set
# ─────────────────────────────────────────────
print("\n📈 Evaluation on Test Set:")
y_pred_mapped = cal_model.predict(X_test)
y_pred = pd.Series(y_pred_mapped).map(inverse_map).values
y_true = pd.Series(y_test).map(inverse_map).values
probas = cal_model.predict_proba(X_test)
conf = np.max(probas, axis=1)

print(classification_report(y_true, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_true, y_pred))
print(f"Log Loss: {log_loss(y_test, probas):.6f}")
print(f"Confidence stats: min={conf.min():.2f}, mean={conf.mean():.2f}, max={conf.max():.2f}")

# ─────────────────────────────────────────────
# 💾 Save Model
# ─────────────────────────────────────────────
joblib.dump(cal_model, MODEL_OUTPUT)
print(f"✅ Saved model to: {MODEL_OUTPUT}")

# ─────────────────────────────────────────────
# 📋 Save Audit Log
# ─────────────────────────────────────────────
audit_df = pd.DataFrame({
    "timestamp": df.iloc[test_idx]["timestamp"].values,
    "true_label": y_true,
    "predicted": y_pred,
    "confidence": conf
})
audit_df.to_csv(AUDIT_CSV, index=False)
print(f"📋 Audit saved to: {AUDIT_CSV}")

# ─────────────────────────────────────────────
# 📈 Confidence Histogram
# ─────────────────────────────────────────────
plt.hist(conf[y_pred == 1], bins=50, alpha=0.6, label="Long (+1)")
plt.hist(conf[y_pred == -1], bins=50, alpha=0.6, label="Short (–1)")
plt.title("Confidence Histogram – Binary Filter")
plt.xlabel("Confidence")
plt.ylabel("Frequency")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(CONF_PLOT)
plt.close()
print(f"📊 Saved confidence plot to: {CONF_PLOT}")
