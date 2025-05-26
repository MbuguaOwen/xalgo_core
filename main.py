
#!/usr/bin/env python
# main.py – Hybrid Engine: Enhanced Veto Logs + Z-Score + Cointegration + Confidence + Leg Selector

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
from core.kalman_cointegration_monitor import KalmanCointegrationMonitor

# ─── Config ───
GATE_MODEL_PATH = "ml_model/triangular_rf_model.pkl"
PAIR_MODEL_PATH = "ml_model/pair_selector_model.pkl"
CONFIDENCE_THRESHOLD = 0.85
COINTEGRATION_THRESHOLD = 0.7
ZSCORE_THRESHOLD = 0.7
MAX_HOLD_SECONDS = 300

executed_signals = deque(maxlen=50)
active_trades = {}
active_trade = None
last_signal = {"type": None, "timestamp": None, "confidence": 0.0}
WINDOW = deque(maxlen=200)

# ─── Filters ───
confidence_filter = MLFilter(GATE_MODEL_PATH)
pair_selector = MLFilter(PAIR_MODEL_PATH)
cointegration_monitor = KalmanCointegrationMonitor()
reverse_pair_map = {0: "BTC", 1: "ETH"}

# ─── Trade Execution ───
def execute_single_leg_trade(timestamp, spread, btc_price, eth_price, selected_leg):
    if selected_leg == "ETH":
        execute_trade(signal_type=1, pair="ETHUSDT", timestamp=timestamp, spread=spread, price=eth_price)
        execute_trade(signal_type=-1, pair="BTCUSDT", timestamp=timestamp, spread=spread, price=btc_price)
    elif selected_leg == "BTC":
        execute_trade(signal_type=1, pair="BTCUSDT", timestamp=timestamp, spread=spread, price=btc_price)
        execute_trade(signal_type=-1, pair="ETHUSDT", timestamp=timestamp, spread=spread, price=eth_price)

    trade_id = f"{timestamp.isoformat()}_{selected_leg}"
    active_trades[trade_id] = {
        "timestamp": timestamp,
        "leg": selected_leg,
        "spread": spread,
        "btc_price": btc_price,
        "eth_price": eth_price
    }
    return trade_id

# ─── Tick Handler ───
def process_tick(timestamp, btc_price, eth_price, ethbtc_price):
    global active_trade

    implied_ethbtc = eth_price / btc_price
    spread = implied_ethbtc - ethbtc_price

    if abs(spread) < 1e-6:
        logging.info("🔇 SKIPPED: Spread too close to 0")
        return

    features = generate_live_features(btc_price, eth_price, ethbtc_price, WINDOW)
    if not features:
        logging.warning("❌ SKIPPED: Feature generation failed")
        return

    spread_z = features["spread_zscore"]
    if abs(spread_z) < ZSCORE_THRESHOLD:
        logging.info(f"🛑 VETOED: Z-score too weak (z={spread_z:.2f} < {ZSCORE_THRESHOLD})")
        return

    if active_trade:
        _, _, trade_id = executed_signals[-1]
        trade_info = active_trades.get(trade_id, {})
        entry_spread = trade_info.get("spread")
        entry_time = trade_info.get("timestamp")
        age = (timestamp - entry_time).total_seconds() if entry_time else 0

        if entry_spread and abs(spread) < abs(entry_spread) * 0.5:
            profit_pct = (abs(entry_spread) - abs(spread)) / abs(entry_spread)
            log_signal_event(timestamp, entry_spread, last_signal["confidence"], None, 1, "reverted_exit", profit_pct)
            logging.info(f"💰 EXIT | {trade_id} | Entry: {entry_spread:.5f} → Exit: {spread:.5f} | PnL: {profit_pct:.4%}")
            active_trades.pop(trade_id, None)
            active_trade = None
            return

        if age > MAX_HOLD_SECONDS:
            profit_pct = (abs(entry_spread) - abs(spread)) / abs(entry_spread)
            log_signal_event(timestamp, entry_spread, last_signal["confidence"], None, 0 if profit_pct < 0 else 1, "forced_exit", profit_pct)
            logging.warning(f"⏱️ TIMEOUT | {trade_id} | Held: {age:.1f}s | PnL: {profit_pct:.4%}")
            active_trades.pop(trade_id, None)
            active_trade = None
            return

        logging.info("🧱 VETOED: Trade already active")
        return

    gate_input = pd.DataFrame([[features[k] for k in [
        "btc_usd", "eth_usd", "eth_btc", "implied_ethbtc", "spread", "spread_zscore",
        "vol_spread", "spread_ewma", "spread_kalman"
    ]]], columns=[
        "btc_usd", "eth_usd", "eth_btc", "implied_ethbtc", "spread", "spread_zscore",
        "vol_spread", "spread_ewma", "spread_kalman"
    ])
    confidence, _ = confidence_filter.predict_with_confidence(gate_input)
    if confidence < CONFIDENCE_THRESHOLD:
        logging.info(f"🛑 VETOED: Confidence too low ({confidence:.4f} < {CONFIDENCE_THRESHOLD})")
        return

    stability_score = cointegration_monitor.update(spread)
    if stability_score < COINTEGRATION_THRESHOLD:
        logging.info(f"🛑 VETOED: Cointegration stability too low ({stability_score:.2f} < {COINTEGRATION_THRESHOLD})")
        return

    required_pair_features = [
        "btc_usd", "eth_usd", "eth_btc", "implied_ethbtc", "spread", "spread_zscore",
        "btc_vol", "eth_vol", "ethbtc_vol", "momentum_btc", "momentum_eth",
        "rolling_corr", "vol_ratio"
    ]
    missing = [k for k in required_pair_features if k not in features]
    if missing:
        logging.error(f"❌ Missing features for pair selector: {missing}")
        return

    pair_input = pd.DataFrame([[features[k] for k in required_pair_features]], columns=required_pair_features)
    pair_code = pair_selector.predict(pair_input)[0]
    selected_leg = reverse_pair_map.get(pair_code)

    logging.info(f"🧠 SIGNAL APPROVED: leg={selected_leg} | z={spread_z:.2f} | conf={confidence:.4f} | stab={stability_score:.2f}")

    trade_id = execute_single_leg_trade(timestamp, spread, btc_price, eth_price, selected_leg)
    executed_signals.append((timestamp, selected_leg, trade_id))
    last_signal.update({"type": selected_leg, "timestamp": timestamp, "confidence": confidence})
    active_trade = trade_id

    log_signal_event(timestamp, spread, confidence, None, 1, f"cointegrated_trade_{selected_leg}")
    logging.info(f"✅ EXECUTED | {selected_leg} | Conf: {confidence:.4f} | z={spread_z:.2f} | Stab: {stability_score:.2f} | ID: {trade_id}")

# ─── Launch ───
async def main():
    logging.basicConfig(level=logging.INFO)
    logging.info("🚀 XAlgo Nexus – Final Z-Score-Gated Cointegrated Pair Selector Running")
    ingestor = BinanceIngestor()
    await ingestor.stream(process_tick)

if __name__ == "__main__":
    asyncio.run(main())
