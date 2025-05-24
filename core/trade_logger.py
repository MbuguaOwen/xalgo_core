 # Lightweight trade logging to CSV or TimescaleDB
 # core/trade_logger.py – Signal Logging (Production-grade, Extendable)

import csv
import os
from datetime import datetime

LOG_FILE = "logs/signal_log.csv"

os.makedirs("logs", exist_ok=True)

if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "entry_price", "confidence", "model_signal", "final_decision", "reason"])

def log_signal_event(timestamp, entry_price, confidence, model_signal, final_decision, reason):
    with open(LOG_FILE, mode="a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            timestamp,
            round(entry_price, 8),
            round(confidence, 4),
            model_signal,
            final_decision,
            reason
        ])
