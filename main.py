#!/usr/bin/env python
# main.py – ML Signal Engine with Smart Cluster Guard, Optimized Entry, and Trade Lifecycle Monitoring

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

# ─── Config ───
MODEL_PATH = "ml_model/triangular_rf_model.pkl"
CONFIDENCE_THRESHOLD = 0.93
executed_signals = deque(maxlen=50)
active_trades = {}

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

# ─── Signal Direction and Regime ───
def evaluate_triangle_paths(btc_price, eth_price, ethbtc_price):
    implied_ethbtc = eth_price / btc_price
    spread = implied_ethbtc - ethbtc_price
    direction = 1 if spread < 0 else -1
    return spread, direction

def detect_regime(spread_zscore, vol_spread):
    if vol_spread > 0.000004:
        return "volatile"
    elif abs(spread_zscore) < 0.5:
        return "flat"
    else:
        return "trending"

# ─── Trade Execution Logic ───
def execute_triangular_trade(signal_type, timestamp, spread, entry_price):
    if signal_type == 1:
        long_leg = "ETHUSDT"
        short_leg = "BTCUSDT"
    else:
        long_leg = "BTCUSDT"
        short_leg = "ETHUSDT"

    execute_trade(signal_type=1, pair=long_leg, timestamp=timestamp, spread=spread, price=entry_price)
    execute_trade(signal_type=-1, pair=short_leg, timestamp=timestamp, spread=spread, price=entry_price)
    trade_id = f"{timestamp.isoformat()}_{signal_type}"
    active_trades[trade_id] = {
        "timestamp": timestamp,
        "signal": signal_type,
        "spread": spread,
        "price": entry_price,
        "long": long_leg,
        "short": short_leg
    }
    return trade_id

# ─── Smart Cluster Guard ───
def should_block_by_cluster(signal, regime):
    same_signals = [1 for _, s, _ in executed_signals if s == signal]
    max_allowed = 5 if regime == "volatile" else 3
    return len(same_signals) >= max_allowed

# ─── Tick Handler ───
ml_filter = MLFilter(model_path=MODEL_PATH)

def process_tick(timestamp: datetime, btc_price: float, eth_price: float, ethbtc_price: float):
    spread, expected_direction = evaluate_triangle_paths(btc_price, eth_price, ethbtc_price)

    if abs(spread) < 1e-6:
        return  # no opportunity

    temp_window = deque(maxlen=200)
    _ = generate_live_features(btc_price, eth_price, ethbtc_price, temp_window)
    spread_z = temp_window[-1]
    vol = pd.Series(temp_window).std()
    regime = detect_regime(spread_z, vol)
    window = spread_windows[regime]
    features = generate_live_features(btc_price, eth_price, ethbtc_price, window)

    if not features:
        return

    x_input = pd.DataFrame([[
        btc_price,
        eth_price,
        ethbtc_price,
        features["implied_ethbtc"],
        features["spread"],
        features["spread_zscore"],
        features["vol_spread"],
        features["spread_ewma"],
        features["spread_kalman"]
    ]], columns=[
        "btc_usd", "eth_usd", "eth_btc",
        "implied_ethbtc", "spread", "spread_zscore",
        "vol_spread", "spread_ewma", "spread_kalman"
    ])

    confidence, model_signal = ml_filter.predict_with_confidence(x_input)

    if confidence >= CONFIDENCE_THRESHOLD and model_signal == expected_direction and not should_block_by_cluster(model_signal, regime):
        trade_id = execute_triangular_trade(model_signal, timestamp, spread, eth_price if model_signal == 1 else btc_price)
        executed_signals.append((timestamp, model_signal, trade_id))

        log_signal_event(
            timestamp=timestamp,
            entry_price=spread,
            confidence=confidence,
            model_signal=model_signal,
            final_decision=1,
            reason="model_approved"
        )
        logging.info(f"✅ APPROVED TRADE | {timestamp} | ID: {trade_id} | Signal: {model_signal} | Confidence: {confidence:.4f}")

    elif confidence >= CONFIDENCE_THRESHOLD and model_signal != expected_direction:
        logging.info(f"⚠️ DIRECTION MISMATCH | Signal: {model_signal}, Expected: {expected_direction}")
    elif should_block_by_cluster(model_signal, regime):
        logging.info(f"🚫 BLOCKED BY CLUSTER GUARD | Signal: {model_signal}, Regime: {regime}")
    else:
        logging.info(f"❌ LOW CONFIDENCE | {confidence:.4f}")

# ─── Launch ───
async def main():
    logging.basicConfig(level=logging.INFO)
    logging.info("🚀 XAlgo Smart Signal Engine Launching...")
    ingestor = BinanceIngestor()
    await ingestor.stream(process_tick)

if __name__ == "__main__":
    asyncio.run(main())
