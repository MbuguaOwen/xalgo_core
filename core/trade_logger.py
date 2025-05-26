# core/trade_logger.py – Enhanced Signal Logging with SL/TP and Z-Score

import csv
import os
import logging
from datetime import datetime

LOG_FILE = "logs/signal_log.csv"
os.makedirs("logs", exist_ok=True)

# ─── Ensure File Has Correct Header ─────────────────────
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "timestamp", "entry_price", "confidence", "spread_zscore",
            "model_signal", "final_decision", "reason",
            "profit", "stop_loss_pct", "take_profit_pct"
        ])

# ─── Log Signal Event ───────────────────────────────────
def log_signal_event(
    timestamp,
    entry_price,
    confidence,
    spread_zscore,
    model_signal,
    final_decision,
    reason,
    profit=None,
    stop_loss_pct=None,
    take_profit_pct=None
):
    try:
        with open(LOG_FILE, mode="a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                timestamp,
                round(entry_price, 8),
                round(confidence, 4),
                round(spread_zscore, 4) if spread_zscore is not None else "",
                model_signal,
                final_decision,
                reason,
                round(profit, 6) if profit is not None else "",
                round(stop_loss_pct, 4) if stop_loss_pct is not None else "",
                round(take_profit_pct, 4) if take_profit_pct is not None else ""
            ])
    except Exception as e:
        logging.error(f"❌ Failed to log signal: {e}")
