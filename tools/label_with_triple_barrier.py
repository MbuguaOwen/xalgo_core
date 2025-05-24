import os

import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# ─────────────────────────────────────────────
# 📥 Load Features
# ─────────────────────────────────────────────
print("📂 Loading features_triangular_base.csv...")
df = pd.read_csv("features_triangular_base.csv", parse_dates=["timestamp"])
df = df.sort_values("timestamp").reset_index(drop=True)

spread = df["spread_kalman"].values
spread_z = df["spread_zscore"].values
timestamps = df["timestamp"].values

# 🧠 Dynamic volatility & movement context
past_vol = pd.Series(spread).rolling(window=30, min_periods=1).std().shift(1).values
past_move = pd.Series(np.abs(np.diff(spread, prepend=spread[0]))).rolling(window=30).mean().shift(1).values

labels, directions, regimes = [], [], []

# ─────────────────────────────────────────────
# 🧠 Adaptive Triple-Barrier Logic
# ─────────────────────────────────────────────
print("🧠 Labeling with dynamic triple-barrier logic...")
for i in tqdm(range(len(df))):
    entry = spread[i]
    vol_i = past_vol[i]
    move_i = past_move[i]
    zscore = spread_z[i]
    t0 = timestamps[i]

    if any(np.isnan([entry, vol_i, move_i, zscore])) or vol_i == 0 or move_i == 0:
        labels.append(np.nan)
        directions.append(np.nan)
        regimes.append(np.nan)
        continue

    # Skip weak reversion entries
    if abs(zscore) < 0.75:
        labels.append(0)
        directions.append(0)
        regimes.append("weak_reversion")
        continue

    # Dynamic TP/SL from percentiles
    lookback = past_move[max(0, i-30):i]
    tp = np.quantile(lookback, 0.90) if len(lookback) >= 10 else 2.0 * move_i
    sl = np.quantile(lookback, 0.10) if len(lookback) >= 10 else 1.0 * move_i

    # Adaptive forward time horizon
    forward_minutes = int(min(30, max(3, vol_i * 1e5)))

    label = 0
    direction = 0
    regime = "neutral"

    for j in range(i + 1, len(df)):
        t1 = timestamps[j]
        dt = (t1 - t0).astype('timedelta64[m]').astype(int)
        if dt > forward_minutes:
            break

        delta = spread[j] - entry

        if entry < 0:  # long
            if delta >= tp:
                label, direction, regime = +1, +1, "revert_long"
                break
            elif delta <= -sl:
                break
        elif entry > 0:  # short
            if delta <= -tp:
                label, direction, regime = -1, -1, "revert_short"
                break
            elif delta >= sl:
                break

    labels.append(label)
    directions.append(direction)
    regimes.append(regime)

df["label"] = labels
df["direction"] = directions
df["regime"] = regimes
df.dropna(subset=["label"], inplace=True)

# ─────────────────────────────────────────────
# 🔍 Label Distribution Audit
# ─────────────────────────────────────────────
print(f"🔍 Final label distribution:\n{df['label'].value_counts()}")
ambiguous = (df["label"] == 0).sum()
print(f"⚖️ Ambiguous labels: {ambiguous} / {len(df)} = {100 * ambiguous / len(df):.2f}%")

# 📊 Visualization
os.makedirs("ml_model", exist_ok=True)
sns.histplot(data=df, x="spread_kalman", hue="label", bins=100, stat="density", common_norm=False)
plt.title("Spread Kalman vs. Label")
plt.xlabel("spread_kalman")
plt.tight_layout()
plt.savefig("ml_model/spread_label_histogram_dynamic.png")
plt.close()

# ─────────────────────────────────────────────
# 💾 Save Output
# ─────────────────────────────────────────────
print("💾 Saving features_triangular_labeled.csv...")
df.to_csv("features_triangular_labeled.csv", index=False)
print("✅ Done: Adaptive triple-barrier labels saved.")
