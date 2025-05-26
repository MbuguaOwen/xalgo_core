import csv
import os
import logging
from datetime import datetime, timezone

LOG_FILE = "logs/signal_log.csv"
os.makedirs("logs", exist_ok=True)

BASE_FIELDS = [
    "timestamp", "entry_price", "confidence", "spread_zscore",
    "model_signal", "final_decision", "reason",
    "profit", "stop_loss_pct", "take_profit_pct"
]

def ensure_datetime(ts):
    if isinstance(ts, datetime):
        return ts
    elif isinstance(ts, (int, float)):
        return datetime.fromtimestamp(ts, tz=timezone.utc)
    return datetime.utcnow().replace(tzinfo=timezone.utc)

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
    take_profit_pct=None,
    **kwargs
):
    try:
        timestamp = ensure_datetime(timestamp)
        iso_time = timestamp.isoformat()

        # Compose log entry
        log_data = {
            "timestamp": iso_time,
            "entry_price": round(entry_price, 8) if entry_price is not None else "",
            "confidence": round(confidence, 4) if confidence is not None else "",
            "spread_zscore": round(spread_zscore, 4) if spread_zscore is not None else "",
            "model_signal": model_signal if model_signal is not None else "",
            "final_decision": final_decision if final_decision is not None else "",
            "reason": reason if reason is not None else "",
            "profit": round(profit, 6) if profit is not None else "",
            "stop_loss_pct": round(stop_loss_pct, 4) if stop_loss_pct is not None else "",
            "take_profit_pct": round(take_profit_pct, 4) if take_profit_pct is not None else "",
        }
        # Add any extra fields (e.g., regime, spread_slope) if desired
        for k, v in kwargs.items():
            log_data[k] = v if v is not None else ""

        # Final field order: BASE_FIELDS + any extras sorted
        extras = sorted([k for k in log_data.keys() if k not in BASE_FIELDS])
        fieldnames = BASE_FIELDS + extras

        # ----> Never write the header <----
        with open(LOG_FILE, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writerow(log_data)

    except Exception as e:
        logging.error(f"❌ Failed to log signal: {e}")
