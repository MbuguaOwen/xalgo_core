#!/usr/bin/env python
# main.py – ML Signal Engine with Dual-Model Decision (Aligned Feature Schema)

import asyncio
import logging
import pandas as pd
from collections import deque
from datetime import datetime
import sys
import os

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from utils.filters import MLFilter
from utils.pair_leader import pair_leader_classifier
from core.feature_pipeline import generate_live_features
from core.execution_engine import execute_trade
from core.trade_logger import log_signal_event
from data.binance_ingestor import BinanceIngestor

# ─── Config ───
PRIMARY_MODEL_PATH = "ml_model/triangular_rf_model.pkl"
LEADER_MODEL_PATH = "ml_model/triangular_rf_leader.pkl"
CONFIDENCE_THRESHOLD = 0.93
MAX_HOLD_SECONDS = 180

executed_signals = deque(maxlen=50)
active_trades = {}
active_trade = None
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

def has_spread_reverted(current_spread, entry_spread, threshold=0.5):
    return abs(current_spread) < abs(entry_spread) * threshold

# ─── Trade Execution ───
def execute_triangular_trade(signal_type, timestamp, spread, btc_price, eth_price):
    long_leg = "ETHUSDT" if signal_type == 1 else "BTCUSDT"
    short_leg = "BTCUSDT" if signal_type == 1 else "ETHUSDT"
    long_price = eth_price if signal_type == 1 else btc_price
    short_price = btc_price if signal_type == 1 else eth_price

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

# ─── Signal Guards ───
def should_block_by_cluster(signal, regime):
    same_signals = [1 for _, s, _ in executed_signals if s == signal]
    max_allowed = 5 if regime == "volatile" else 3
    return len(same_signals) >= max_allowed

def should_block_flip(new_signal, timestamp, min_seconds=60):
    if last_signal["type"] is None:
        return False
    time_diff = (timestamp - last_signal["timestamp"]).total_seconds()
    return new_signal != last_signal["type"] and time_diff < min_seconds

# ─── Inference Engine Setup ───
primary_model = MLFilter(PRIMARY_MODEL_PATH)
leader_model = MLFilter(LEADER_MODEL_PATH)
stability_buffer = deque(maxlen=3)

# ─── Tick Handler ───
def process_tick(timestamp, btc_price, eth_price, ethbtc_price):
    global active_trade

    implied_ethbtc = eth_price / btc_price
    spread = implied_ethbtc - ethbtc_price
    expected_direction = 1 if spread < 0 else 0

    if active_trade:
        _, _, trade_id = executed_signals[-1]
        trade_info = active_trades.get(trade_id, {})
        entry_spread = trade_info.get("spread")
        entry_time = trade_info.get("timestamp")
        age = (timestamp - entry_time).total_seconds() if entry_time else 0

        if entry_spread and has_spread_reverted(spread, entry_spread):
            profit_pct = (abs(entry_spread) - abs(spread)) / abs(entry_spread)
            log_signal_event(timestamp, entry_spread, last_signal["confidence"], trade_info["signal"], 1, "spread_reverted_exit", profit_pct)
            logging.info(f"💰 PROFIT | {trade_id} | Entry: {entry_spread:.5f} → Exit: {spread:.5f} | PnL: {profit_pct:.4%}")
            active_trade = None
            active_trades.pop(trade_id, None)
            return

        if age > MAX_HOLD_SECONDS:
            profit_pct = (abs(entry_spread) - abs(spread)) / abs(entry_spread)
            log_signal_event(timestamp, entry_spread, last_signal["confidence"], trade_info["signal"], 0 if profit_pct < 0 else 1, "forced_exit_loss" if profit_pct < 0 else "forced_exit", profit_pct)
            logging.warning(f"⏱️ TIMEOUT | {trade_id} | Held: {age:.1f}s | PnL: {profit_pct:.4%}")
            active_trade = None
            active_trades.pop(trade_id, None)
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
        btc_price, eth_price, ethbtc_price,
        features["implied_ethbtc"], features["spread"], features["spread_zscore"],
        features["vol_spread"], features["spread_ewma"], features["spread_kalman"]
    ]], columns=[
        "btc_usd", "eth_usd", "eth_btc",
        "implied_ethbtc", "spread", "spread_zscore",
        "vol_spread", "spread_ewma", "spread_kalman"
    ])

    conf_primary, signal_primary = primary_model.predict_with_confidence(x_input)
    conf_leader, signal_leader = leader_model.predict_with_confidence(x_input)

    agreed_signal = signal_primary == signal_leader == expected_direction
    final_confidence = min(conf_primary, conf_leader)

    stability_buffer.append(signal_leader)
    if not all(s == signal_leader for s in stability_buffer):
        return

    if should_block_flip(signal_leader, timestamp):
        return

    if final_confidence >= CONFIDENCE_THRESHOLD and agreed_signal and not should_block_by_cluster(signal_leader, regime):
        trade_id = execute_triangular_trade(signal_leader, timestamp, spread, btc_price, eth_price)
        executed_signals.append((timestamp, signal_leader, trade_id))
        last_signal.update({"type": signal_leader, "timestamp": timestamp, "confidence": final_confidence})
        active_trade = trade_id

        log_signal_event(timestamp, spread, final_confidence, signal_leader, 1, "grand_dual_signal")
        logging.info(f"✅ GRAND SIGNAL | {timestamp} | ID: {trade_id} | Conf: {final_confidence:.4f}")

# ─── Launch ───
async def main():
    logging.basicConfig(level=logging.INFO)
    logging.info("🚀 XAlgo Nexus – Dual-Model Signal Engine Starting...")
    ingestor = BinanceIngestor()
    await ingestor.stream(process_tick)

if __name__ == "__main__":
    asyncio.run(main())
