#!/usr/bin/env python
# main.py – Quant-Grade Triangular Arbitrage ML Signal Engine (BUY & SELL Ready)

import asyncio
import logging
import pandas as pd
from collections import deque
from datetime import datetime, timezone
import sys
import os

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from utils.filters import MLFilter
from core.feature_pipeline import generate_live_features
from core.execution_engine import execute_trade, calculate_dynamic_sl_tp
from core.trade_logger import log_signal_event
from data.binance_ingestor import BinanceIngestor

# ==== CONFIGURABLE USER THRESHOLDS ====
MAX_HOLD_SECONDS = 1200
MIN_SPREAD_MAGNITUDE = 0.0000
MIN_SLOPE_MAGNITUDE = 0.0000
ADJUSTED_ZSCORE_THRESHOLD = 0.0

USER_CONFIDENCE_THRESHOLD = None  # None for adaptive, or float for override
USER_COINTEGRATION_THRESHOLD = None

GATE_MODEL_PATH = "ml_model/triangular_rf_model.pkl"
PAIR_MODEL_PATH = "ml_model/pair_selector_model.pkl"
COINT_MODEL_PATH = "ml_model/cointegration_score_model.pkl"
REGIME_MODEL_PATH = "ml_model/regime_classifier.pkl"

GATE_FEATURES = [
    "spread", "spread_zscore", "vol_spread", "spread_kalman", "spread_ewma",
    "btc_usd", "eth_usd", "eth_btc", "implied_ethbtc"
]
COINT_FEATURES = GATE_FEATURES + ["btc_vol", "eth_vol", "ethbtc_vol"]
PAIR_FEATURES = [
    "btc_usd", "eth_usd", "eth_btc", "implied_ethbtc", "spread", "spread_zscore",
    "btc_vol", "eth_vol", "ethbtc_vol", "momentum_btc", "momentum_eth",
    "rolling_corr", "vol_ratio", "spread_slope"
]

reverse_pair_map = {0: "BTC", 1: "ETH"}
regime_map = {0: "flat", 1: "volatile", 2: "trending"}

executed_signals = deque(maxlen=50)
active_trades = {}
active_trade = None
last_signal = {"type": None, "timestamp": None, "confidence": 0.0}
WINDOW = deque(maxlen=200)

def ensure_datetime(ts):
    if isinstance(ts, datetime):
        return ts
    elif isinstance(ts, (int, float)):
        return datetime.fromtimestamp(ts, tz=timezone.utc)
    return datetime.utcnow().replace(tzinfo=timezone.utc)

def get_adaptive_thresholds(regime: str, volatility: float, confidence_hint: float = 0.85):
    if regime == "trending":
        z = 1.2 + volatility * 0.3
        conf = max(0.88, confidence_hint + 0.02)
        coint = max(0.5, 0.7 - volatility * 0.15)
    elif regime == "volatile":
        z = 1.0 + volatility * 0.2
        conf = max(0.86, confidence_hint)
        coint = 0.65 - volatility * 0.1
    else:
        z = 0.8 + volatility * 0.1
        conf = 0.85
        coint = 0.7
    return round(z, 3), round(conf, 3), round(max(min(coint, 0.75), 0.5), 3)

def check_model_features(model, features_dict, model_name):
    expected = set(model.feature_names_in_)
    actual = set(features_dict.keys())
    missing = expected - actual
    if missing:
        logging.critical(f"🚨 [FATAL] {model_name} missing required features: {missing}")
        raise RuntimeError(f"{model_name}: Model expects features missing in current pipeline: {missing}")

def check_all_models_loaded():
    dummy = {k: 1.0 for k in set(GATE_FEATURES + COINT_FEATURES + PAIR_FEATURES)}
    for model, name in [
        (confidence_filter.model, "Confidence Model"),
        (cointegration_model.model, "Cointegration Model"),
        (pair_selector.model, "Pair Selector Model"),
        (regime_classifier.model, "Regime Classifier Model"),
    ]:
        check_model_features(model, dummy, name)

# --- Load Models and Check Features at Startup ---
confidence_filter = MLFilter(GATE_MODEL_PATH)
pair_selector = MLFilter(PAIR_MODEL_PATH)
cointegration_model = MLFilter(COINT_MODEL_PATH)
regime_classifier = MLFilter(REGIME_MODEL_PATH)
check_all_models_loaded()

def execute_single_leg_trade(timestamp, spread, btc_price, eth_price, selected_leg, features, confidence, direction):
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
        execute_trade(direction, "ETHUSDT", timestamp, spread, eth_price, zscore, vol_spread, confidence, regime)
    else:
        execute_trade(direction, "BTCUSDT", timestamp, spread, btc_price, zscore, vol_spread, confidence, regime)

    trade_id = f"{timestamp.isoformat()}_{selected_leg}_{'BUY' if direction==1 else 'SELL' if direction==-1 else 'HOLD'}"
    active_trades[trade_id] = {
        "timestamp": timestamp,
        "leg": selected_leg,
        "direction": direction,
        "spread": spread,
        "btc_price": btc_price,
        "eth_price": eth_price,
        "sl": stop_loss_pct,
        "tp": take_profit_pct
    }
    log_signal_event(
        timestamp, spread, confidence, zscore, direction,
        1 if direction != 0 else 0,
        f"ml_cointegrated_trade_{selected_leg}_{'BUY' if direction==1 else 'SELL' if direction==-1 else 'HOLD'}",
        None, stop_loss_pct, take_profit_pct,
        spread_slope=features.get("spread_slope"),
        regime=regime
    )
    return trade_id

def process_tick(timestamp, btc_price, eth_price, ethbtc_price):
    global active_trade

    timestamp = ensure_datetime(timestamp)
    implied_ethbtc = eth_price / btc_price
    spread = implied_ethbtc - ethbtc_price

    # Gate 1: Spread Magnitude
    if abs(spread) < MIN_SPREAD_MAGNITUDE:
        log_signal_event(
            timestamp, spread, 0.0, 0.0, None, 0, "veto_spread_too_small",
            spread_slope=None, regime=None
        )
        logging.info(f"🛑 VETO: Spread magnitude too small: {spread:.8f} < {MIN_SPREAD_MAGNITUDE}")
        return

    features = generate_live_features(btc_price, eth_price, ethbtc_price, WINDOW)
    if not features:
        log_signal_event(
            timestamp, spread, 0.0, 0.0, None, 0, "veto_feature_fail",
            spread_slope=None, regime=None
        )
        logging.warning("❌ SKIPPED: Feature generation failed")
        return

    spread_z = features.get("spread_zscore", 0.0)
    volatility = features.get("vol_spread", 0.001)
    spread_slope = features.get("spread_slope", 0.0)

    # --- Regime Classification ---
    regime_input = pd.DataFrame([features]).reindex(columns=regime_classifier.model.feature_names_in_)
    regime_code = regime_classifier.predict(regime_input)[0]
    regime = regime_map.get(regime_code, "flat")
    features["regime"] = regime

    # --- Thresholds ---
    zscore_threshold, default_confidence_threshold, default_cointegration_threshold = get_adaptive_thresholds(regime, volatility)
    confidence_threshold = USER_CONFIDENCE_THRESHOLD if USER_CONFIDENCE_THRESHOLD is not None else default_confidence_threshold
    cointegration_threshold = USER_COINTEGRATION_THRESHOLD if USER_COINTEGRATION_THRESHOLD is not None else default_cointegration_threshold

    # --- Veto Gates ---
    if abs(spread_z) < zscore_threshold:
        log_signal_event(
            timestamp, spread, 0.0, spread_z, None, 0, "veto_zscore_low",
            spread_slope=spread_slope, regime=regime
        )
        logging.info(f"🛑 VETO: Z-score too low: {spread_z:.2f} < {zscore_threshold}")
        return

    if abs(spread_slope) < MIN_SLOPE_MAGNITUDE:
        log_signal_event(
            timestamp, spread, 0.0, spread_z, None, 0, "veto_slope_low",
            spread_slope=spread_slope, regime=regime
        )
        logging.info(f"🛑 VETO: Spread slope too flat: {spread_slope:.6f} < {MIN_SLOPE_MAGNITUDE}")
        return

    adjusted_zscore = abs(spread_z) * (confidence_threshold or 1.0)
    if adjusted_zscore < ADJUSTED_ZSCORE_THRESHOLD:
        log_signal_event(
            timestamp, spread, 0.0, spread_z, None, 0, "veto_adjusted_zscore",
            spread_slope=spread_slope, regime=regime, adjusted_zscore=adjusted_zscore
        )
        logging.info(f"🛑 VETO: Adjusted Z-score too weak: {adjusted_zscore:.2f} < {ADJUSTED_ZSCORE_THRESHOLD}")
        return

    # --- Exit handling for open trade ---
    if active_trade:
        _, _, trade_id = executed_signals[-1]
        trade_info = active_trades.get(trade_id, {})
        entry_spread = trade_info.get("spread")
        entry_time = ensure_datetime(trade_info.get("timestamp"))
        age = (timestamp - entry_time).total_seconds() if entry_time else 0

        if entry_spread and abs(spread) < abs(entry_spread) * 0.5:
            profit_pct = (abs(entry_spread) - abs(spread)) / abs(entry_spread)
            log_signal_event(timestamp, entry_spread, last_signal["confidence"], spread_z,
                             trade_info.get("direction", 1), 1, "reverted_exit", profit_pct,
                             trade_info.get("sl"), trade_info.get("tp"),
                             spread_slope=spread_slope, regime=regime
            )
            logging.info(f"💰 EXIT: {trade_id} | Profit={profit_pct:.2%}")
            active_trades.pop(trade_id, None)
            active_trade = None
            return

        if age > MAX_HOLD_SECONDS:
            profit_pct = (abs(entry_spread) - abs(spread)) / abs(entry_spread)
            log_signal_event(timestamp, entry_spread, last_signal["confidence"], spread_z,
                             trade_info.get("direction", 1), 0 if profit_pct < 0 else 1, "forced_exit", profit_pct,
                             trade_info.get("sl"), trade_info.get("tp"),
                             spread_slope=spread_slope, regime=regime
            )
            logging.warning(f"⏱️ TIMEOUT: {trade_id} | Age={age}s")
            active_trades.pop(trade_id, None)
            active_trade = None
            return
        return

    # --- ML Confidence Gate ---
    gate_input = pd.DataFrame([features]).reindex(columns=confidence_filter.model.feature_names_in_)
    check_model_features(confidence_filter.model, features, "Confidence Model")
    confidence, direction = confidence_filter.predict_with_confidence(gate_input)
    # direction: -1 (SELL), 1 (BUY), 0 (HOLD)

    if direction == 0:
        log_signal_event(
            timestamp, spread, confidence, spread_z, direction, 0, "veto_direction_hold",
            spread_slope=spread_slope, regime=regime
        )
        logging.info("🛑 VETO: Model signaled HOLD (direction == 0)")
        return

    if confidence < confidence_threshold:
        log_signal_event(
            timestamp, spread, confidence, spread_z, direction, 0, "veto_confidence_low",
            spread_slope=spread_slope, regime=regime, confidence_threshold=confidence_threshold
        )
        logging.info(f"🛑 VETO: Confidence too low: {confidence:.4f} < {confidence_threshold}")
        return

    # --- ML Cointegration Gate ---
    coint_input = pd.DataFrame([features]).reindex(columns=cointegration_model.model.feature_names_in_)
    check_model_features(cointegration_model.model, features, "Cointegration Model")
    coint_score, _ = cointegration_model.predict_with_confidence(coint_input)
    if coint_score < cointegration_threshold:
        log_signal_event(
            timestamp, spread, confidence, spread_z, direction, 0, "veto_cointegration_low",
            spread_slope=spread_slope, regime=regime, coint_score=coint_score
        )
        logging.info(f"🛑 VETO: Cointegration too weak: {coint_score:.4f} < {cointegration_threshold}")
        return

    # --- ML Pair/Leader Gate ---
    pair_input = pd.DataFrame([features]).reindex(columns=pair_selector.model.feature_names_in_)
    check_model_features(pair_selector.model, features, "Pair Selector Model")
    pair_code = pair_selector.predict(pair_input)[0]
    selected_leg = reverse_pair_map.get(pair_code)

    trade_id = execute_single_leg_trade(timestamp, spread, btc_price, eth_price, selected_leg, features, confidence, direction)
    executed_signals.append((timestamp, selected_leg, trade_id))
    last_signal.update({"type": selected_leg, "timestamp": timestamp, "confidence": confidence})
    active_trade = trade_id

async def main():
    logging.basicConfig(level=logging.INFO)
    logging.info("🚀 Boss-Level Adaptive Signal Engine Starting...")
    ingestor = BinanceIngestor()
    await ingestor.stream(process_tick)

if __name__ == "__main__":
    asyncio.run(main())
