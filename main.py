#!/usr/bin/env python
# main.py – Live ML Signal Engine with Dynamic Pair Selection

import asyncio
import logging
import pandas as pd
from collections import deque
from datetime import datetime
from utils.filters import MLFilter
from core.feature_pipeline import generate_live_features
from core.execution_engine import execute_trade
from core.trade_logger import log_signal_event
from data.binance_ingestor import BinanceIngestor

# ─────────────────────────────────────────────
# 🔧 Config
# ─────────────────────────────────────────────
MODEL_PATH = "models/triangular_rf_model.pkl"
CONFIDENCE_THRESHOLD = 0.99
Z_SCORE_WINDOW = 100

# ─────────────────────────────────────────────
# 🧠 Evaluate Triangle Paths
# ─────────────────────────────────────────────
def evaluate_triangle_paths(btc_price, eth_price, ethbtc_price):
    # Path A: USDT → ETH → BTC → USDT
    implied_ethbtc_A = eth_price / btc_price
    spread_A = implied_ethbtc_A - ethbtc_price
    profit_A = (btc_price * ethbtc_price) / eth_price  # approximate final return

    # Path B: USDT → BTC → ETH → USDT
    implied_ethbtc_B = btc_price / eth_price
    spread_B = implied_ethbtc_B - (1 / ethbtc_price)
    profit_B = (eth_price / ethbtc_price) / btc_price

    if profit_A > profit_B:
        return "ETH/USDT", spread_A, eth_price
    else:
        return "BTC/USDT", spread_B, btc_price

# ─────────────────────────────────────────────
# 🔁 Signal Callback Logic
# ─────────────────────────────────────────────
ml_filter = MLFilter(model_path=MODEL_PATH)
spread_window = deque(maxlen=Z_SCORE_WINDOW)

def process_tick(timestamp: datetime, btc_price: float, eth_price: float, ethbtc_price: float):
    chosen_pair, spread, entry_price = evaluate_triangle_paths(btc_price, eth_price, ethbtc_price)

    features = generate_live_features(btc_price, eth_price, ethbtc_price, spread_window)
    x_input = pd.DataFrame([[
        features["btc_usd"],
        features["eth_usd"],
        features["eth_btc"],
        features["implied_ethbtc"],
        features["spread"],
        features["spread_zscore"],
        features["vol_spread"]
    ]], columns=[
        "btc_usd",
        "eth_usd",
        "eth_btc",
        "implied_ethbtc",
        "spread",
        "spread_zscore",
        "vol_spread"
    ])

    confidence, signal = ml_filter.predict_with_confidence(x_input)

    if confidence >= CONFIDENCE_THRESHOLD:
        reason = "model_approved"
        execute_trade(
            signal_type=signal,
            pair=chosen_pair,
            timestamp=timestamp,
            spread=spread,
            price=entry_price
        )
    else:
        reason = "low_confidence"

    log_signal_event(
        timestamp=timestamp,
        entry_price=spread,
        confidence=confidence,
        model_signal=signal,
        final_decision=int(confidence >= CONFIDENCE_THRESHOLD),
        reason=reason
    )

# ─────────────────────────────────────────────
# 🚀 Entry Point
# ─────────────────────────────────────────────
async def main():
    logging.basicConfig(level=logging.INFO)
    logging.info("🚀 XAlgo Live Signal Engine starting (Binance WebSocket mode)")
    ingestor = BinanceIngestor()
    await ingestor.stream(process_tick)

if __name__ == "__main__":
    asyncio.run(main())
