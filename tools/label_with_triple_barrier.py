# label_with_triple_barrier.py – Apply Adaptive Triple Barrier to Triangular Spread

import pandas as pd
import numpy as np
import time

# ─── CONFIG ───
TP_MULTIPLIER = 1.5     # take-profit = x * volatility
SL_MULTIPLIER = 1.0     # stop-loss = x * volatility
FORWARD_WINDOW = 20     # how many rows (minutes) ahead to look

start = time.time()
print("📅 Loading features_triangular_base.csv...")
df = pd.read_csv("features_triangular_base.csv", parse_dates=["timestamp"])

# ─── Ensure required columns ───
required_cols = ["timestamp", "spread", "vol_spread"]
for col in required_cols:
    if col not in df.columns:
        raise ValueError(f"❌ Missing column: {col}")

# ─── Compute Barriers and Label ───
labels = []

for i in range(len(df) - FORWARD_WINDOW):
    entry_spread = df.loc[i, "spread"]
    vol = df.loc[i, "vol_spread"]

    tp = entry_spread + TP_MULTIPLIER * vol
    sl = entry_spread - SL_MULTIPLIER * vol

    forward_spread = df.loc[i+1:i+FORWARD_WINDOW, "spread"]

    hit_tp = (forward_spread >= tp).any()
    hit_sl = (forward_spread <= sl).any()

    if hit_tp and not hit_sl:
        labels.append(1)    # BUY
    elif hit_sl and not hit_tp:
        labels.append(-1)   # SELL
    else:
        labels.append(0)    # HOLD

labels += [0] * (len(df) - len(labels))
df["label"] = labels

# Rename features to match btc_usd, eth_usd, eth_btc convention
rename_map = {
    "btc_price": "btc_usd",
    "eth_price": "eth_usd",
    "ethbtc_price": "eth_btc"
}
df.rename(columns=rename_map, inplace=True)

# ─── Save Labeled Output ───
df.to_csv("features_triangular_labeled.csv", index=False)
print("✅ Labeled dataset saved: features_triangular_labeled.csv")
print(f"🧠 Label distribution:\n{df['label'].value_counts()}")
print(f"🕒 Done in {time.time() - start:.2f} seconds.")
