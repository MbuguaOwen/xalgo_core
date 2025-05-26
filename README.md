
# 📈 XAlgoNexus – Triangular Arbitrage Intelligence Engine

**XAlgoNexus** is an institutional-grade, real-time arbitrage engine designed for triangular trading opportunities in both **crypto** and **forex** markets.  
It fuses statistical rigor, machine learning intelligence, and adaptive execution logic to identify and exploit short-term inefficiencies across pairs like BTC/ETH/USDT and GBP/USD/EUR.

---

## 🚀 System Overview

### ✅ Live Signal Engine
- **Streaming Binance data**
- **Feature generation** from BTCUSDT, ETHUSDT, ETHBTC
- **Z-score + volatility-based filtering**
- **ML gating** via confidence model
- **ML-based cointegration classification**
- **Best-leg selector** to choose BTC or ETH for trade
- **Dynamic SL/TP** based on spread, volatility, regime

---

## 🤖 Models Used

| Model Path                           | Role                        |
|--------------------------------------|-----------------------------|
| `triangular_rf_model.pkl`            | Predicts confidence of reversion (–1 / +1) |
| `cointegration_score_model.pkl`      | Classifies stability of the spread |
| `pair_selector_model.pkl`            | Chooses best leg: BTC or ETH |
| `regime_classifier.pkl` *(optional)* | Classifies market regime: flat, trending, volatile |

---

## 📂 Project Structure

```bash
.
├── core/
│   ├── feature_pipeline.py
│   ├── kalman_cointegration_monitor.py
│   ├── execution_engine.py
│   └── adaptive_filters.py
├── utils/
│   └── filters.py
├── data/
│   ├── BTCUSDT.csv
│   ├── ETHUSDT.csv
│   └── ETHBTC.csv
├── ml_model/
│   ├── triangular_rf_model.pkl
│   ├── cointegration_score_model.pkl
│   ├── pair_selector_model.pkl
│   └── regime_classifier.pkl
├── logs/
│   └── signal_log.csv
├── main.py
└── README.md
```

---

## 🧪 Training Pipelines

Scripts to train each ML module:

- `train_cointegration_score_model.py`
- `train_pair_selector.py`
- `train_regime_classifier.py`
- `train_ml_filter_combined.py`

All models are trained on engineered features from `features_cointegration_labeled.csv`.

---

## ⚙️ Signal Flow Diagram

```text
[Live Binance Prices]
        ↓
[Feature Generator] ← historical .csv also supported
        ↓
[Z-Score Filter] → [Regime Classifier (optional)]
        ↓
[ML Confidence Model] (triangular_rf_model.pkl)
        ↓
[ML Cointegration Model] (cointegration_score_model.pkl)
        ↓
[Best Leg Selector] (pair_selector_model.pkl)
        ↓
[Execute BTC or ETH Legs] → [Log Signal + SL/TP Risk]
```

---

## 📊 Outputs

- Logged to: `logs/signal_log.csv`
- Format: `timestamp, entry_price, confidence, model_signal, final_decision, reason, profit, stop_loss_pct, take_profit_pct`

---

## 📡 Coming Soon

- ✅ Telegram/Discord alerts for high-confidence setups
- ✅ Prometheus + Grafana dashboards
- ✅ Automated backtesting + regime-aware evaluation
- ✅ Real capital integration via Binance API

---

## 🧠 Built By

> Owen Mbugua · Quant engineer @ XAlgo · 2025

---

## 📥 Get Started

Install requirements and run:
```bash
python main.py
```

Or run a specific model trainer:
```bash
python train_cointegration_score_model.py
```

---
