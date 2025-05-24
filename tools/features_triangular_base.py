import pandas as pd
import numpy as np
from scipy.stats import zscore
from pykalman import KalmanFilter

# ─────────────────────────────────────────────
# 📥 Load & Normalize Timestamps
# ─────────────────────────────────────────────
def load_and_prepare(filepath, price_col, rename_col):
    print(f"🔍 Loading {filepath}...")
    df = pd.read_csv(filepath)
    df["timestamp"] = pd.to_datetime(df["open_time"], errors='coerce')
    df = df[["timestamp", price_col]].dropna()
    df = df.rename(columns={price_col: rename_col})
    return df.sort_values("timestamp").reset_index(drop=True)

btc_df = load_and_prepare("data/BTCUSDT.csv", "close", "btc_usd")
eth_df = load_and_prepare("data/ETHUSDT.csv", "close", "eth_usd")
ethbtc_df = load_and_prepare("data/ETHBTC.csv", "close", "eth_btc")

# ─────────────────────────────────────────────
# 🧩 Timestamp Alignment (Forward-Fill Logic)
# ─────────────────────────────────────────────
print("🔗 Aligning timestamps with merge_asof...")
df = pd.merge_asof(btc_df, eth_df, on="timestamp", direction="nearest")
df = pd.merge_asof(df, ethbtc_df, on="timestamp", direction="nearest")
df.dropna(inplace=True)

# ─────────────────────────────────────────────
# 🧠 Feature Engineering
# ─────────────────────────────────────────────
print("🧠 Computing features...")
df["implied_ethbtc"] = df["eth_usd"] / df["btc_usd"]
df["spread"] = df["implied_ethbtc"] - df["eth_btc"]
df["spread_zscore"] = zscore(df["spread"].ffill())
df["vol_spread"] = df["spread"].rolling(window=20, min_periods=1).std()
df["spread_ewma"] = df["spread"].ewm(span=20, adjust=False).mean()

# ─────────────────────────────────────────────
# 📈 Kalman Filter Smoothing
# ─────────────────────────────────────────────
print("📈 Applying Kalman filter...")
spread_series = df["spread"].ffill().values
kf = KalmanFilter(initial_state_mean=0, n_dim_obs=1)
kf = kf.em(spread_series, n_iter=5)
state_means, _ = kf.filter(spread_series)
df["spread_kalman"] = state_means.flatten()

# ─────────────────────────────────────────────
# 🏷️ Placeholders for Labeling and Regime
# ─────────────────────────────────────────────
df["regime"] = np.nan
df["label"] = np.nan

# ─────────────────────────────────────────────
# 💾 Save Output
# ─────────────────────────────────────────────
final_cols = [
    "timestamp", "btc_usd", "eth_usd", "eth_btc", "implied_ethbtc",
    "spread", "spread_zscore", "vol_spread", "spread_ewma", "spread_kalman",
    "regime", "label"
]
print("💾 Saving to features_triangular_base.csv...")
df[final_cols].to_csv("features_triangular_base.csv", index=False)
print("✅ Done: features_triangular_base.csv created successfully.")
