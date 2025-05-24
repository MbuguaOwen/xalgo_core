# Merged chunked preprocessing & feature generation
# core/feature_pipeline.py – Feature Generator (Live + Batch)

import pandas as pd
import numpy as np
from datetime import datetime
from collections import deque

# ─────────────────────────────────────────────────────────
# 🔧 Parameters for rolling z-score and volatility
# ─────────────────────────────────────────────────────────
Z_SCORE_WINDOW = 100


def compute_triangle_features(btc_price, eth_price, ethbtc_price, spread_window):
    implied_ethbtc = eth_price / btc_price
    spread = implied_ethbtc - ethbtc_price

    spread_window.append(spread)
    values = list(spread_window)

    if len(values) >= Z_SCORE_WINDOW:
        mean = np.mean(values)
        std = np.std(values)
        z_score = (spread - mean) / std if std > 1e-6 else 0.0
        vol_spread = std
    else:
        z_score = 0.0
        vol_spread = 0.0

    return {
        "btc_usd": btc_price,
        "eth_usd": eth_price,
        "eth_btc": ethbtc_price,
        "implied_ethbtc": implied_ethbtc,
        "spread": spread,
        "spread_zscore": z_score,
        "vol_spread": vol_spread
    }


def generate_features_from_csvs(csv_paths):
    # Batch mode for historical CSVs
    btc_df = pd.read_csv(csv_paths["BTCUSDT"], parse_dates=["open_time"])
    eth_df = pd.read_csv(csv_paths["ETHUSDT"], parse_dates=["open_time"])
    ethbtc_df = pd.read_csv(csv_paths["ETHBTC"], parse_dates=["open_time"])

    btc_df = btc_df.rename(columns={"open_time": "timestamp", "close": "btc_price"})
    eth_df = eth_df.rename(columns={"open_time": "timestamp", "close": "eth_price"})
    ethbtc_df = ethbtc_df.rename(columns={"open_time": "timestamp", "close": "ethbtc_price"})

    df = btc_df[["timestamp", "btc_price"]] \
        .merge(eth_df[["timestamp", "eth_price"]], on="timestamp") \
        .merge(ethbtc_df[["timestamp", "ethbtc_price"]], on="timestamp")

    spread_window = deque(maxlen=Z_SCORE_WINDOW)
    features = []

    for _, row in df.iterrows():
        f = compute_triangle_features(row["btc_price"], row["eth_price"], row["ethbtc_price"], spread_window)
        f["timestamp"] = row["timestamp"]
        features.append(f)

    return pd.DataFrame(features)


def generate_live_features(btc_price, eth_price, ethbtc_price, spread_window):
    # Live mode: single-tick input
    return compute_triangle_features(btc_price, eth_price, ethbtc_price, spread_window)
