# XAlgo Core: Live ML-Based Triangular Arbitrage Bot

## 📈 Overview
XAlgo Core is a high-frequency trading system for live BTC/ETH/USDT triangular arbitrage using:
- Real-time Binance WebSocket data
- Machine learning–based execution logic
- Sub-second signal filtering
- Minimal architecture for low-latency deployment

---

## ⚙️ Features
- ✅ Binance WebSocket: real-time BTCUSDT, ETHUSDT, ETHBTC prices
- ✅ Feature generation: spread, z-score, implied ETHBTC, vol_spread
- ✅ ML signal scoring with calibrated confidence
- ✅ Execution simulation (real trade integration ready)
- ✅ Logs signals and executions in CSV format

---

## 🧱 Project Structure
```
.
├── core/
│   ├── execution_engine.py      # Logs or executes trades
│   ├── feature_pipeline.py      # Computes features from Binance ticks
│   └── trade_logger.py          # Signal audit logger
├── data/
│   └── binance_ingestor.py      # Streams real-time data
├── utils/
│   └── filters.py               # MLFilter using .pkl model
├── models/                      # Contains triangular_rf_model.pkl
├── logs/                        # Outputs: signal_log.csv, execution_log.csv
├── main.py                      # Live engine entrypoint
├── requirements.txt             # Python dependencies
└── .gitignore                   # Clean repo practices
```

---

## 🚀 Quickstart

### 1. Clone and Set Up Environment
```bash
git clone https://github.com/YOUR_USERNAME/xalgo_core.git
cd xalgo_core
python -m venv .venv
. .venv/Scripts/activate  # or source .venv/bin/activate on Unix
pip install -r requirements.txt
```

### 2. Run Live Trading Engine
```bash
python main.py
```
Logs will be saved to:
- `logs/signal_log.csv`
- `logs/execution_log.csv`

---

## 🧠 ML Model Details
The model used (`models/triangular_rf_model.pkl`) expects the following features:
```python
[
  "btc_usd", "eth_usd", "eth_btc",
  "implied_ethbtc", "spread", "spread_zscore", "vol_spread"
]
```
It was trained to identify high-confidence profitable mean-reversion signals.

---

## 📄 Notes
- Currently runs in simulation mode (executes mock trades)
- Ready for real order routing via Binance REST/FIX API
- Prometheus/Grafana hooks coming soon

---

© 2025 XAlgo Core — Precision. Speed. Profit.
