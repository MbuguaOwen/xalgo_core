import pandas as pd
import numpy as np
from pykalman import KalmanFilter
from scipy.stats import zscore
from tqdm import tqdm
import os

# ─────────────────────────────────────────────
# 🔧 Parameters
# ─────────────────────────────────────────────
TP_MULTIPLIER = 1.5
SL_MULTIPLIER = 1.0
HORIZON = 30
Z_THRESH = 2.0

# ─────────────────────────────────────────────
# 📥 Load Raw Data
# ─────────────────────────────────────────────
def load_price_data(path, col_name):
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["open_time"], errors="coerce")
    df = df[["timestamp", "close"]].rename(columns={"close": col_name})
    return df.dropna().sort_values("timestamp")

btc = load_price_data("data/BTCUSDT.csv", "btc_usd")
eth = load_price_data("data/ETHUSDT.csv", "eth_usd")
ethbtc = load_price_data("data/ETHBTC.csv", "eth_btc")

# ─────────────────────────────────────────────
# 🧩 Align by Timestamp
# ─────────────────────────────────────────────
df = pd.merge_asof(btc, eth, on="timestamp", direction="nearest")
df = pd.merge_asof(df, ethbtc, on="timestamp", direction="nearest")
df = df.dropna().reset_index(drop=True)

# ─────────────────────────────────────────────
# 🧠 Feature Engineering
# ─────────────────────────────────────────────
df["implied_ethbtc"] = df["eth_usd"] / df["btc_usd"]
df["spread"] = df["implied_ethbtc"] - df["eth_btc"]
df["spread_zscore"] = zscore(df["spread"].ffill())
df["vol_spread"] = df["spread"].rolling(window=20, min_periods=1).std()
df["spread_ewma"] = df["spread"].ewm(span=20, adjust=False).mean()

# Kalman filter smoothing
kf = KalmanFilter(initial_state_mean=0, n_dim_obs=1)
kf = kf.em(df["spread"].values, n_iter=5)
state_means, _ = kf.filter(df["spread"].values)
df["spread_kalman"] = state_means.flatten()

# ─────────────────────────────────────────────
# 🏷️ Directional Triple-Barrier Labeling
# ─────────────────────────────────────────────
df["label"] = 0
total = len(df)

print("🧠 Applying directional triple-barrier labeling...")
for i in tqdm(range(total - HORIZON)):
    spread = df.loc[i, "spread"]
    vol = df.loc[i, "vol_spread"]
    z = df.loc[i, "spread_zscore"]

    if pd.isna(vol) or vol < 1e-6 or abs(z) < Z_THRESH:
        continue

    tp = spread + TP_MULTIPLIER * vol
    sl = spread - SL_MULTIPLIER * vol
    forward = df.loc[i+1:i+HORIZON, "spread"]

    for future_spread in forward:
        if future_spread >= tp:
            df.loc[i, "label"] = 1
            break
        elif future_spread <= sl:
            df.loc[i, "label"] = -1
            break

# ─────────────────────────────────────────────
# 💾 Save Final Output
# ─────────────────────────────────────────────
final_cols = [
    "timestamp", "btc_usd", "eth_usd", "eth_btc",
    "implied_ethbtc", "spread", "spread_zscore",
    "vol_spread", "spread_ewma", "spread_kalman", "label"
]

os.makedirs("labeled", exist_ok=True)
df[final_cols].to_csv("features_triangular_labeled.csv", index=False)
print("✅ Saved to features_triangular_labeled.csv")
