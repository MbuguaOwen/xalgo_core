
import pandas as pd
import joblib
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

INPUT_CSV = "features_cointegration_labeled.csv"
MODEL_PATH = "ml_model/cointegration_score_model.pkl"

# 📥 Load and filter
df = pd.read_csv(INPUT_CSV)
df = df[df["label_cointegration"].isin([0, 1])]

# ✅ Features (may be extended)
features = [
    "spread", "spread_zscore", "vol_spread", "spread_ewma", "spread_kalman",
    "btc_usd", "eth_usd", "eth_btc", "implied_ethbtc",
    "btc_vol", "eth_vol", "ethbtc_vol"
]

X = df[features]
y = df["label_cointegration"]

# 🎓 Split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# 🤖 Train
model = XGBClassifier(use_label_encoder=False, eval_metric="logloss")
model.fit(X_train, y_train)

# 🧪 Evaluate
print("📊 Classification Report:")
print(classification_report(y_test, model.predict(X_test)))

# 💾 Save
joblib.dump(model, MODEL_PATH)
print(f"✅ Model saved to: {MODEL_PATH}")
