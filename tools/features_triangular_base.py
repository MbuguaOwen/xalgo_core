# features_triangular_base.py – Optimized, Fast, and Powerful Feature Generator

import pandas as pd
import numpy as np
import time
from utils.pair_leader import pair_leader_classifier

# ─── CONFIG ───
Z_WINDOW = 100
ROLLING_WINDOW = 20
LABELS = []

def log(label):
    t = time.time()
    LABELS.append((label, t))
    print(f"✅ Step: {label} @ {time.strftime('%H:%M:%S', time.localtime(t))}")

start_time = time.time()
log("Start Script")

# ─── FAST KALMAN APPROX (Lightweight, Causal) ───
def fast_kalman_approx(series, alpha=0.15):
    estimate = series.iloc[0]
    filtered = []
    for obs in series:
        estimate = alpha * obs + (1 - alpha) * estimate
        filtered.append(estimate)
    return filtered

# ─── FAST CSV LOADER ───
def fast_load_csv(path, rename_to):
    df = pd.read_csv(path, usecols=["open_time", "close"])
    df["timestamp"] = pd.to_datetime(df["open_time"], errors="coerce", utc=True)
    df = df.dropna(subset=["timestamp"])
    df = df.rename(columns={"close": rename_to})
    return df[["timestamp", rename_to]]

log("Begin Loading Data")

btc_df = fast_load_csv("data/BTCUSDT.csv", "btc_usd")
eth_df = fast_load_csv("data/ETHUSDT.csv", "eth_usd")
ethbtc_df = fast_load_csv("data/ETHBTC.csv", "eth_btc")

log("Finished Loading Data")

# ─── TIMESTAMP ALIGNMENT ───
df = pd.merge_asof(btc_df, eth_df, on="timestamp", direction="nearest")
df = pd.merge_asof(df, ethbtc_df, on="timestamp", direction="nearest")
df = df.dropna().reset_index(drop=True)

log("Aligned Timestamps")

# ─── CORE TRIANGLE FEATURES ───
df["implied_ethbtc"] = df["eth_usd"] / df["btc_usd"]
df["spread"] = df["implied_ethbtc"] - df["eth_btc"]

df["spread_zscore"] = (
    df["spread"] - df["spread"].rolling(window=Z_WINDOW, min_periods=10).mean()
) / df["spread"].rolling(window=Z_WINDOW, min_periods=10).std()

df["vol_spread"] = df["spread"].rolling(window=ROLLING_WINDOW).std()
df["spread_ewma"] = df["spread"].ewm(span=20, adjust=False).mean()

log("Calculated Spread/Zscore/EWMA")

# ─── FAST KALMAN FILTER (APPROX) ───
df["spread_kalman"] = fast_kalman_approx(df["spread"])
log("Applied Fast Kalman Approximation")

# ─── VOLATILITY + Z-SCORES for LEADER ───
df["z_btc"] = df["btc_usd"].pct_change().rolling(Z_WINDOW).mean().fillna(0)
df["z_eth"] = df["eth_usd"].pct_change().rolling(Z_WINDOW).mean().fillna(0)
df["z_btc_prev"] = df["z_btc"].shift(1)
df["z_eth_prev"] = df["z_eth"].shift(1)

df["vol_btc"] = df["btc_usd"].rolling(ROLLING_WINDOW).std()
df["vol_eth"] = df["eth_usd"].rolling(ROLLING_WINDOW).std()

log("Computed BTC/ETH Z-Scores & Volatility")

# ─── PAIR LEADER CLASSIFICATION ───
def classify_row(row):
    return pair_leader_classifier({
        "z_btc": row["z_btc"],
        "z_eth": row["z_eth"],
        "z_btc_prev": row["z_btc_prev"],
        "z_eth_prev": row["z_eth_prev"],
        "vol_btc": row["vol_btc"],
        "vol_eth": row["vol_eth"]
    })

df["pair_leader"] = df.apply(classify_row, axis=1)
df["pair_leader_encoded"] = df["pair_leader"].map({"BTC": 0, "ETH": 1, "NEUTRAL": 2})

log("Classified Pair Leader")

# ─── EXPORT FINAL FEATURE SET ───
final_cols = [
    "timestamp", "btc_usd", "eth_usd", "eth_btc",
    "implied_ethbtc", "spread", "spread_zscore", "vol_spread",
    "spread_ewma", "spread_kalman", "z_btc", "z_eth",
    "vol_btc", "vol_eth", "pair_leader", "pair_leader_encoded"
]

print("\n🤎 pair_leader distribution:")
print(df["pair_leader"].value_counts())

print("\n📊 Z-Score Volatility Summary:")
print(df[["z_btc", "z_eth", "vol_btc", "vol_eth"]].describe())

df[final_cols].dropna().to_csv("features_triangular_base.csv", index=False)
log("Saved features_triangular_base.csv")

# ─── TIMING SUMMARY ───
print("\n🕓 Process Duration Breakdown:")
for i in range(1, len(LABELS)):
    label, t = LABELS[i]
    prev_t = LABELS[i - 1][1]
    print(f"{LABELS[i-1][0]} ➔ {label}: {t - prev_t:.2f}s")

print(f"\n✅ Done. Total Time: {time.time() - start_time:.2f}s")
