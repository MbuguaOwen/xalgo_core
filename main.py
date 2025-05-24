#!/usr/bin/env python
# main.py – ML Signal Engine with Cluster Guard & Binary Directional Filter

import asyncio
import logging
import pandas as pd
from collections import deque
from datetime import datetime
import sys
import os

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from utils.filters import MLFilter
from core.feature_pipeline import generate_live_features
from core.execution_engine import execute_trade
from core.trade_logger import log_signal_event
from data.binance_ingestor import BinanceIngestor

# ─────────────────────────────────────────────
# 🔧 Config
# ─────────────────────────────────────────────
MODEL_PATH = "ml_model/triangular_rf_model.pkl"
CONFIDENCE_THRESHOLD = 0.90
executed_signals = deque(maxlen=10)

WINDOW_MAP = {
    "volatile": 30,
    "flat": 100,
    "trending": 200
}

spread_windows = {
    "volatile": deque(maxlen=WINDOW_MAP["volatile"]),
    "flat": deque(maxlen=WINDOW_MAP["flat"]),
    "trending": deque(maxlen=WINDOW_MAP["trending"])
}

# ─────────────────────────────────────────────
# 🧠 Spread & Direction Inference
# ─────────────────────────────────────────────
def evaluate_triangle_paths(btc_price, eth_price, ethbtc_price):
    implied_ethbtc_A = eth_price / btc_price
    spread_A = implied_ethbtc_A - ethbtc_price
    profit_A = (btc_price * ethbtc_price) / eth_price

    implied_ethbtc_B = btc_price / eth_price
    spread_B = implied_ethbtc_B - (1 / ethbtc_price)
    profit_B = (eth_price / ethbtc_price) / btc_price

    if profit_A > profit_B:
        direction = -1 if spread_A > 0 else 1
        return "ETH/USDT", spread_A, eth_price, direction
    else:
        direction = -1 if spread_B > 0 else 1
        return "BTC/USDT", spread_B, btc_price, direction

def detect_regime(spread_zscore, vol_spread):
    if vol_spread > 0.000004:
        return "volatile"
    elif abs(spread_zscore) < 0.5:
        return "flat"
    else:
        return "trending"

# ─────────────────────────────────────────────
# 🔁 Tick Handler
# ─────────────────────────────────────────────
try:
    ml_filter = MLFilter(model_path=MODEL_PATH)
except Exception as e:
    logging.error(f"[MLFilter] ❌ Error loading model: {e}")
    sys.exit(1)

def process_tick(timestamp: datetime, btc_price: float, eth_price: float, ethbtc_price: float):
    chosen_pair, spread, entry_price, expected_direction = evaluate_triangle_paths(
        btc_price, eth_price, ethbtc_price
    )

    if abs(spread) < 1e-6:
        reason = "no_opportunity"
        log_signal_event(timestamp, spread, 0.0, 0, 0, reason)
        return

    temp_window = deque(maxlen=200)
    _ = generate_live_features(btc_price, eth_price, ethbtc_price, temp_window)
    if not temp_window:
        return

    spread_z = temp_window[-1]
    vol = pd.Series(temp_window).std()
    regime = detect_regime(spread_z, vol)
    window = spread_windows[regime]
    features = generate_live_features(btc_price, eth_price, ethbtc_price, window)

    if not features or any(k not in features for k in [
        "implied_ethbtc", "spread", "spread_zscore", "vol_spread", "spread_ewma", "spread_kalman"
    ]):
        return

    model_features = [
        btc_price,
        eth_price,
        ethbtc_price,
        features["implied_ethbtc"],
        features["spread"],
        features["spread_zscore"],
        features["vol_spread"],
        features["spread_ewma"],
        features["spread_kalman"],
        expected_direction
    ]

    x_input = pd.DataFrame([model_features], columns=[
        "btc_usd", "eth_usd", "eth_btc",
        "implied_ethbtc", "spread", "spread_zscore",
        "vol_spread", "spread_ewma", "spread_kalman", "direction"
    ])

    confidence, signal = ml_filter.predict_with_confidence(x_input)

    same_signal_count = sum(
        1 for t, s, p in executed_signals if s == signal and p == chosen_pair
    )
    max_cluster = 5 if regime == "volatile" else 3

    logging.info(
        f"📊 {timestamp} | Pair={chosen_pair} | Spread={spread:.5e} | "
        f"Signal={signal} | Conf={confidence:.3f} | Match={signal == expected_direction} | Regime={regime}"
    )

    if confidence >= CONFIDENCE_THRESHOLD and signal == expected_direction and same_signal_count < max_cluster:
        reason = "model_approved"
        execute_trade(signal_type=signal, pair=chosen_pair, timestamp=timestamp, spread=spread, price=entry_price)
        executed_signals.append((timestamp, signal, chosen_pair))
    elif same_signal_count >= max_cluster:
        reason = "cluster_guard_blocked"
    elif confidence >= CONFIDENCE_THRESHOLD:
        reason = "direction_mismatch"
    else:
        reason = "low_confidence"

    log_signal_event(
        timestamp=timestamp,
        entry_price=spread,
        confidence=confidence,
        model_signal=signal,
        final_decision=int(reason == "model_approved"),
        reason=reason
    )

# ─────────────────────────────────────────────
# 🚀 Launch
# ─────────────────────────────────────────────
async def main():
    logging.basicConfig(level=logging.INFO)
    logging.info("🚀 XAlgo Signal Engine starting (Binance WebSocket mode)...")
    ingestor = BinanceIngestor()
    await ingestor.stream(process_tick)

if __name__ == "__main__":
    asyncio.run(main())
