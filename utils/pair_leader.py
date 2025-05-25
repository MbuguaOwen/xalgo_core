# pair_leader.py – Robust BTC/ETH Dominance Classifier (Final Fix)

import numpy as np

def pair_leader_classifier(features: dict) -> str:
    """
    Determine BTC or ETH leadership using direct slope and volatility comparison.
    No strict thresholds — just dominance logic.
    """
    z_btc = features.get("z_btc", 0.0)
    z_eth = features.get("z_eth", 0.0)
    z_btc_prev = features.get("z_btc_prev", 0.0)
    z_eth_prev = features.get("z_eth_prev", 0.0)
    vol_btc = features.get("vol_btc", 1e-8)
    vol_eth = features.get("vol_eth", 1e-8)

    slope_btc = z_btc - z_btc_prev
    slope_eth = z_eth - z_eth_prev

    slope_strength_btc = abs(slope_btc)
    slope_strength_eth = abs(slope_eth)

    if slope_strength_btc > slope_strength_eth and slope_strength_btc > 1e-5:
        return "BTC"
    elif slope_strength_eth > slope_strength_btc and slope_strength_eth > 1e-5:
        return "ETH"

    # Volatility backup dominance (if slope inconclusive)
    if vol_btc > vol_eth * 1.1:
        return "BTC"
    elif vol_eth > vol_btc * 1.1:
        return "ETH"

    return "NEUTRAL"

# Optional quick test
if __name__ == "__main__":
    test = {
        "z_btc": 0.0004,
        "z_eth": 0.0001,
        "z_btc_prev": 0.0001,
        "z_eth_prev": 0.0002,
        "vol_btc": 75,
        "vol_eth": 6.5
    }
    print("Leader:", pair_leader_classifier(test))
