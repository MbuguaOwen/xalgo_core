
import pandas as pd
import joblib
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

# 📂 Load cleaned dataset
df = pd.read_csv("features_pair_selection.csv")

# 🧹 Keep only labeled rows
df = df[df["label"].isin(["BTC", "ETH"])]

# ⚖️ Balance BTC and ETH
min_count = min(df["label"].value_counts().values)
df_balanced = df.groupby("label").sample(n=min_count, random_state=42)

# ✅ Define features
features = [
    "btc_usd", "eth_usd", "eth_btc", "implied_ethbtc", "spread", "spread_zscore",
    "btc_vol", "eth_vol", "ethbtc_vol", "momentum_btc", "momentum_eth",
    "rolling_corr", "vol_ratio"
]
X = df_balanced[features]
y = df_balanced["label"].map({"BTC": 0, "ETH": 1})

# 🎓 Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# 🤖 Train model
model = XGBClassifier(use_label_encoder=False, eval_metric="mlogloss")
model.fit(X_train, y_train)

# 💾 Save model
joblib.dump(model, "ml_model/pair_selector_model.pkl")

# 📊 Diagnostics
print("✅ Model saved to: ml_model/pair_selector_model.pkl")
print("Label counts (balanced):")
print(df_balanced['label'].value_counts())






