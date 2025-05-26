
#!/usr/bin/env python
# main.py – Hybrid Engine: ML-Gated + Cointegration-Model + Dynamic SL/TP + Leg Selector

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
from core.execution_engine import execute_trade, calculate_dynamic_sl_tp
from core.trade_logger import log_signal_event
from data.binance_ingestor import BinanceIngestor

# ─── Config ───
GATE_MODEL_PATH = "ml_model/triangular_rf_model.pkl"
PAIR_MODEL_PATH = "ml_model/pair_selector_model.pkl"
COINT_MODEL_PATH = "ml_model/cointegration_score_model.pkl"

CONFIDENCE_THRESHOLD = 0.97
ZSCORE_THRESHOLD = 1.5
MAX_HOLD_SECONDS = 600 

executed_signals = deque(maxlen=50)
active_trades = {}
active_trade = None
last_signal = {"type": None, "timestamp": None, "confidence": 0.0}
WINDOW = deque(maxlen=200)

# ─── Filters ───
confidence_filter = MLFilter(GATE_MODEL_PATH)
pair_selector = MLFilter(PAIR_MODEL_PATH)
cointegration_model = MLFilter(COINT_MODEL_PATH)
reverse_pair_map = {0: "BTC", 1: "ETH"}

# ─── Trade Execution ───
def execute_single_leg_trade(timestamp, spread, btc_price, eth_price, selected_leg, features, confidence):
    regime = features.get("regime", "flat")
    zscore = features.get("spread_zscore", 0.0)
    vol_spread = features.get("vol_spread", 0.001)

    stop_loss_pct, take_profit_pct = calculate_dynamic_sl_tp(
        spread_zscore=zscore,
        vol_spread=vol_spread,
        confidence=confidence,
        regime=regime
    )

    if selected_leg == "ETH":
        execute_trade(1, "ETHUSDT", timestamp, spread, eth_price, zscore, vol_spread, confidence, regime)
        execute_trade(-1, "BTCUSDT", timestamp, spread, btc_price, zscore, vol_spread, confidence, regime)
    elif selected_leg == "BTC":
        execute_trade(1, "BTCUSDT", timestamp, spread, btc_price, zscore, vol_spread, confidence, regime)
        execute_trade(-1, "ETHUSDT", timestamp, spread, eth_price, zscore, vol_spread, confidence, regime)

    trade_id = f"{timestamp.isoformat()}_{selected_leg}"
    active_trades[trade_id] = {
        "timestamp": timestamp,
        "leg": selected_leg,
        "spread": spread,
        "btc_price": btc_price,
        "eth_price": eth_price,
        "sl": stop_loss_pct,
        "tp": take_profit_pct
    }

    log_signal_event(
        timestamp=timestamp,
        entry_price=spread,
        confidence=confidence,
        spread_zscore=zscore,
        model_signal=None,
        final_decision=1,
        reason=f"ml_cointegrated_trade_{selected_leg}",
        stop_loss_pct=stop_loss_pct,
        take_profit_pct=take_profit_pct
    )

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
            log_signal_event(
                timestamp, entry_spread, last_signal["confidence"], spread_z,
                None, 1, "reverted_exit", profit_pct,
                trade_info.get("sl"), trade_info.get("tp")
            )
            logging.info(f"💰 EXIT | {trade_id} | Entry: {entry_spread:.5f} → Exit: {spread:.5f} | PnL: {profit_pct:.4%}")
            active_trades.pop(trade_id, None)
            active_trade = None
            return

        if age > MAX_HOLD_SECONDS:
            profit_pct = (abs(entry_spread) - abs(spread)) / abs(entry_spread)
            log_signal_event(
                timestamp, entry_spread, last_signal["confidence"], spread_z,
                None, 0 if profit_pct < 0 else 1, "forced_exit", profit_pct,
                trade_info.get("sl"), trade_info.get("tp")
            )
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

    # 🧠 ML Cointegration filter replaces hard-coded threshold
    cointegration_input = pd.DataFrame([[features[k] for k in [
        "spread", "spread_zscore", "vol_spread", "spread_ewma", "spread_kalman",
        "btc_usd", "eth_usd", "eth_btc", "implied_ethbtc",
        "btc_vol", "eth_vol", "ethbtc_vol"
    ]]], columns=[
        "spread", "spread_zscore", "vol_spread", "spread_ewma", "spread_kalman",
        "btc_usd", "eth_usd", "eth_btc", "implied_ethbtc",
        "btc_vol", "eth_vol", "ethbtc_vol"
    ])
    conf_cointegration, _ = cointegration_model.predict_with_confidence(cointegration_input)
    if conf_cointegration < 0.5:
        logging.info(f"🛑 VETOED: ML cointegration score too low ({conf_cointegration:.4f})")
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

    logging.info(f"🧠 SIGNAL APPROVED: leg={selected_leg} | z={spread_z:.2f} | conf={confidence:.4f} | ml_coint={conf_cointegration:.4f}")

    trade_id = execute_single_leg_trade(timestamp, spread, btc_price, eth_price, selected_leg, features, confidence)
    executed_signals.append((timestamp, selected_leg, trade_id))
    last_signal.update({"type": selected_leg, "timestamp": timestamp, "confidence": confidence})
    active_trade = trade_id

    logging.info(f"✅ EXECUTED | {selected_leg} | Conf: {confidence:.4f} | z={spread_z:.2f} | ML_Coint: {conf_cointegration:.4f} | ID: {trade_id}")

# ─── Launch ───
async def main():
    logging.basicConfig(level=logging.INFO)
    logging.info("🚀 XAlgo Nexus – ML-Gated Signal Engine with Trained Cointegration Filter")
    ingestor = BinanceIngestor()
    await ingestor.stream(process_tick)

if __name__ == "__main__":
    asyncio.run(main())
