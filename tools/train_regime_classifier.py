
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np

INPUT_CSV = "features_cointegration_labeled.csv"
OUTPUT_MODEL = "ml_model/regime_classifier.pkl"

# 📥 Load
df = pd.read_csv(INPUT_CSV)

# 🧠 Label regimes using spread std quantiles
spread_std = df["spread"].rolling(window=50).std()

q1 = spread_std.quantile(0.33)
q2 = spread_std.quantile(0.66)

def classify_regime(std):
    if std < q1:
        return "flat"
    elif std < q2:
        return "trending"
    else:
        return "volatile"

df["regime"] = spread_std.apply(lambda s: classify_regime(s) if not np.isnan(s) else "flat")

# 🔎 Debug distribution
print("📊 Regime Distribution:")
print(df["regime"].value_counts())

# 🔢 Encode target
regime_map = {"flat": 0, "trending": 1, "volatile": 2}
df["regime_encoded"] = df["regime"].map(regime_map)

# ✅ Features
features = [
    "spread", "spread_zscore", "vol_spread", "spread_kalman", "spread_ewma",
    "btc_usd", "eth_usd", "eth_btc", "implied_ethbtc"
]
X = df[features].dropna()
y = df.loc[X.index, "regime_encoded"]

# 🎓 Train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42)
model.fit(X_train, y_train)

print("\n📊 Classification Report:")
print(classification_report(y_test, model.predict(X_test), target_names=["flat", "trending", "volatile"]))

# 💾 Save model
joblib.dump(model, OUTPUT_MODEL)
print(f"✅ Model saved to: {OUTPUT_MODEL}")
