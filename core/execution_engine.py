# core/execution_engine.py – Final Execution Display & Logging

import logging
from datetime import datetime
from termcolor import colored
import pytz
import os

# Logs every approved execution
# Auto-creates log file with header if missing

def execute_trade(signal_type: int, pair: str, timestamp: datetime, spread: float, price: float):
    action = {1: "BUY", -1: "SELL", 0: "HOLD"}.get(signal_type, "UNKNOWN")
    color = {"BUY": "green", "SELL": "red", "HOLD": "yellow"}.get(action, "white")

    # Convert timestamp to local time
    nairobi_tz = pytz.timezone("Africa/Nairobi")
    local_time = timestamp.astimezone(nairobi_tz).isoformat()

    # Print to console
    log_line = f"🟢 EXECUTE: [{action}] {pair} @ price={price:.2f} | spread={spread:.8f} | {local_time}"
    print(colored(log_line, color))

    # Ensure logs directory and file exist
    os.makedirs("logs", exist_ok=True)
    log_path = "logs/execution_log.csv"
    new_file = not os.path.exists(log_path)

    try:
        with open(log_path, mode="a", newline="") as f:
            if new_file:
                f.write("timestamp,action,pair,price,spread\n")
            f.write(f"{local_time},{action},{pair},{price:.2f},{spread:.8f}\n")
    except Exception as e:
        logging.error(f"❌ Failed to write to execution_log.csv: {e}")
