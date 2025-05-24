import pandas as pd
import numpy as np
from tqdm import tqdm

# ─────────────────────────────────────────────
# 📥 Load Feature Set
# ─────────────────────────────────────────────
print("📂 Loading features_triangular_base.csv...")
df = pd.read_csv("features_triangular_base.csv", parse_dates=["timestamp"])
df = df.sort_values("timestamp").reset_index(drop=True)

# ─────────────────────────────────────────────
# ⚙️ Labeling Parameters
# ─────────────────────────────────────────────
TP_MULT = 2.0
SL_MULT = 1.0
FORWARD_MINUTES = 5

labels = []
directions = []
regimes = []

spread = df["spread_kalman"].values
vol = df["vol_spread"].ffill().values
timestamps = df["timestamp"].values

# ─────────────────────────────────────────────
# 🧠 Triple-Barrier Logic with Directional Entry
# ─────────────────────────────────────────────
print("🧠 Applying triple-barrier labeling...")
for i in tqdm(range(len(df))):
    entry = spread[i]
    vol_i = vol[i]
    t0 = timestamps[i]

    if np.isnan(vol_i) or vol_i == 0:
        labels.append(np.nan)
        directions.append(np.nan)
        regimes.append(np.nan)
        continue

    tp = TP_MULT * vol_i
    sl = SL_MULT * vol_i

    label = 0
    direction = 0
    regime = "neutral"

    for j in range(i + 1, len(df)):
        t1 = timestamps[j]
        dt = (t1 - t0).astype('timedelta64[m]').astype(int)
        if dt > FORWARD_MINUTES:
            break

        delta = spread[j] - entry

        if entry < 0:
            if delta >= tp:
                label = +1
                direction = +1
                regime = "mean_revert_long"
                break
            elif delta <= -sl:
                label = 0  # no label on failed long
                break
        elif entry > 0:
            if delta <= -tp:
                label = -1
                direction = -1
                regime = "mean_revert_short"
                break
            elif delta >= sl:
                label = 0  # no label on failed short
                break

    labels.append(label)
    directions.append(direction)
    regimes.append(regime)

# ─────────────────────────────────────────────
# 📝 Assign Labels to DataFrame
# ─────────────────────────────────────────────
df["label"] = labels
df["direction"] = directions
df["regime"] = regimes

# Drop rows where label is NaN (no volatility or invalid entry)
df.dropna(subset=["label"], inplace=True)

# ─────────────────────────────────────────────
# 💾 Save Output
# ─────────────────────────────────────────────
print("💾 Saving features_triangular_labeled.csv...")
df.to_csv("features_triangular_labeled.csv", index=False)
print("✅ Done: features_triangular_labeled.csv created.")
