#!/usr/bin/env python
# main.py – ML Signal Engine with Grand Signal Selection, Auto-Exit, and Lifecycle Precision

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
active_trade = None
active_trades = {}
last_signal = {"type": None, "timestamp": None, "confidence": 0.0}

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

# ─── Utility Logic ───
def detect_regime(spread_zscore, vol_spread):
    if vol_spread > 0.000004:
        return "volatile"
    elif abs(spread_zscore) < 0.5:
        return "flat"
    else:
        return "trending"

def has_spread_reverted(current_spread: float, entry_spread: float, threshold: float = 0.5) -> bool:
    return abs(current_spread) < abs(entry_spread) * threshold

# ─── Smart Trade Execution ───
def execute_triangular_trade(signal_type, timestamp, spread, btc_price, eth_price):
    if signal_type == 1:
        long_leg = "ETHUSDT"
        short_leg = "BTCUSDT"
        long_price = eth_price
        short_price = btc_price
    else:
        long_leg = "BTCUSDT"
        short_leg = "ETHUSDT"
        long_price = btc_price
        short_price = eth_price

    execute_trade(signal_type=1, pair=long_leg, timestamp=timestamp, spread=spread, price=long_price)
    execute_trade(signal_type=-1, pair=short_leg, timestamp=timestamp, spread=spread, price=short_price)
    trade_id = f"{timestamp.isoformat()}_{signal_type}"
    active_trades[trade_id] = {
        "timestamp": timestamp,
        "signal": signal_type,
        "spread": spread,
        "price": long_price,
        "long": long_leg,
        "short": short_leg
    }
    return trade_id

# ─── Guards ───
def should_block_by_cluster(signal, regime):
    same_signals = [1 for _, s, _ in executed_signals if s == signal]
    max_allowed = 5 if regime == "volatile" else 3
    return len(same_signals) >= max_allowed

def should_block_flip(new_signal, timestamp, min_seconds=60):
    if last_signal["type"] is None:
        return False
    time_diff = (timestamp - last_signal["timestamp"]).total_seconds()
    return new_signal != last_signal["type"] and time_diff < min_seconds

# ─── Tick Handler ───
ml_filter = MLFilter(model_path=MODEL_PATH)
stability_buffer = deque(maxlen=3)

def process_tick(timestamp: datetime, btc_price: float, eth_price: float, ethbtc_price: float):
    global active_trade

    implied_ethbtc = eth_price / btc_price
    spread = implied_ethbtc - ethbtc_price
    expected_direction = 1 if spread < 0 else -1

    # ✅ Check for auto-clear
    if active_trade:
        _, _, trade_id = executed_signals[-1]
        trade_info = active_trades.get(trade_id, {})
        entry_spread = trade_info.get("spread")
        if entry_spread and has_spread_reverted(spread, entry_spread):
            logging.info(f"🔁 Spread reverted. Closing trade: {trade_id}")
            active_trade = None
            active_trades.pop(trade_id, None)
        else:
            return

    if abs(spread) < 1e-6:
        return

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
    stability_buffer.append(model_signal)
    if not all(s == model_signal for s in stability_buffer):
        return

    if should_block_flip(model_signal, timestamp):
        logging.info(f"⚠️ Flip-blocked: Prev={last_signal['type']} → New={model_signal} too soon")
        return

    if confidence >= CONFIDENCE_THRESHOLD and model_signal == expected_direction and not should_block_by_cluster(model_signal, regime):
        trade_id = execute_triangular_trade(model_signal, timestamp, spread, btc_price, eth_price)
        executed_signals.append((timestamp, model_signal, trade_id))
        last_signal["type"] = model_signal
        last_signal["timestamp"] = timestamp
        last_signal["confidence"] = confidence
        active_trade = trade_id

        log_signal_event(
            timestamp=timestamp,
            entry_price=spread,
            confidence=confidence,
            model_signal=model_signal,
            final_decision=1,
            reason="grand_signal"
        )
        logging.info(f"✅ GRAND SIGNAL | {timestamp} | ID: {trade_id} | Signal: {model_signal} | Conf: {confidence:.4f}")

    elif confidence >= CONFIDENCE_THRESHOLD and model_signal != expected_direction:
        logging.info(f"⚠️ DIRECTION MISMATCH | Signal: {model_signal}, Expected: {expected_direction}")
    elif should_block_by_cluster(model_signal, regime):
        logging.info(f"🚫 CLUSTER BLOCKED | Signal: {model_signal}, Regime: {regime}")
    else:
        logging.info(f"❌ REJECTED | Confidence: {confidence:.4f}")

# ─── Launch ───
async def main():
    logging.basicConfig(level=logging.INFO)
    logging.info("🚀 XAlgo Nexus – Grand Signal Engine Launching...")
    ingestor = BinanceIngestor()
    await ingestor.stream(process_tick)

if __name__ == "__main__":
    asyncio.run(main())
