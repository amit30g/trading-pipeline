"""
Layer 1: Market Regime Detector
Determines overall market posture: Aggressive / Normal / Cautious / Defensive / Cash
"""
import pandas as pd
import numpy as np
from config import REGIME_CONFIG, REGIME_POSTURE


def compute_regime(
    nifty_df: pd.DataFrame,
    all_stock_data: dict[str, pd.DataFrame],
) -> dict:
    """
    Analyze market regime based on:
    1. Nifty 50 vs 200 DMA
    2. 50 DMA vs 200 DMA (golden/death cross)
    3. Breadth: % of stocks above 50 DMA
    4. Breadth: % of stocks above 200 DMA
    5. Net new 52-week highs vs lows

    Returns dict with regime score, label, signals, and posture parameters.
    """
    close = nifty_df["Close"]
    cfg = REGIME_CONFIG

    signals = {}
    scores = []

    # ── Signal 1: Nifty 50 vs 200 DMA ──────────────────────────
    ma200 = close.rolling(200).mean()
    latest_close = close.iloc[-1]
    latest_ma200 = ma200.iloc[-1]
    pct_from_200 = ((latest_close - latest_ma200) / latest_ma200) * 100

    if pct_from_200 > cfg["index_near_200dma_pct"]:
        s1 = 1
        s1_label = f"Bullish (Nifty {pct_from_200:+.1f}% above 200 DMA)"
    elif pct_from_200 < -cfg["index_near_200dma_pct"]:
        s1 = -1
        s1_label = f"Bearish (Nifty {pct_from_200:+.1f}% below 200 DMA)"
    else:
        s1 = 0
        s1_label = f"Neutral (Nifty near 200 DMA, {pct_from_200:+.1f}%)"
    signals["index_vs_200dma"] = {"score": s1, "detail": s1_label}
    scores.append(s1)

    # ── Signal 2: 50 DMA vs 200 DMA crossover ──────────────────
    ma50 = close.rolling(50).mean()
    if ma50.iloc[-1] > ma200.iloc[-1]:
        if ma50.iloc[-5] <= ma200.iloc[-5]:  # recent golden cross
            s2 = 1
            s2_label = "Bullish (fresh golden cross)"
        else:
            s2 = 1
            s2_label = "Bullish (50 DMA > 200 DMA)"
    elif ma50.iloc[-1] < ma200.iloc[-1]:
        if ma50.iloc[-5] >= ma200.iloc[-5]:  # recent death cross
            s2 = -1
            s2_label = "Bearish (fresh death cross)"
        else:
            s2 = -1
            s2_label = "Bearish (50 DMA < 200 DMA)"
    else:
        s2 = 0
        s2_label = "Neutral (MAs converging)"
    signals["ma_crossover"] = {"score": s2, "detail": s2_label}
    scores.append(s2)

    # ── Signal 3: % of stocks above 50 DMA ─────────────────────
    above_50 = 0
    total = 0
    for ticker, df in all_stock_data.items():
        if len(df) < 50:
            continue
        total += 1
        stock_ma50 = df["Close"].rolling(50).mean()
        if df["Close"].iloc[-1] > stock_ma50.iloc[-1]:
            above_50 += 1

    breadth_50 = (above_50 / total * 100) if total > 0 else 50
    if breadth_50 >= cfg["breadth_50dma_bullish"]:
        s3 = 1
        s3_label = f"Bullish ({breadth_50:.0f}% above 50 DMA)"
    elif breadth_50 <= cfg["breadth_50dma_bearish"]:
        s3 = -1
        s3_label = f"Bearish ({breadth_50:.0f}% above 50 DMA)"
    else:
        s3 = 0
        s3_label = f"Neutral ({breadth_50:.0f}% above 50 DMA)"
    signals["breadth_50dma"] = {"score": s3, "detail": s3_label, "value": breadth_50}
    scores.append(s3)

    # ── Signal 4: % of stocks above 200 DMA ────────────────────
    above_200 = 0
    total_200 = 0
    for ticker, df in all_stock_data.items():
        if len(df) < 200:
            continue
        total_200 += 1
        stock_ma200 = df["Close"].rolling(200).mean()
        if df["Close"].iloc[-1] > stock_ma200.iloc[-1]:
            above_200 += 1

    breadth_200 = (above_200 / total_200 * 100) if total_200 > 0 else 50
    if breadth_200 >= cfg["breadth_200dma_bullish"]:
        s4 = 1
        s4_label = f"Bullish ({breadth_200:.0f}% above 200 DMA)"
    elif breadth_200 <= cfg["breadth_200dma_bearish"]:
        s4 = -1
        s4_label = f"Bearish ({breadth_200:.0f}% above 200 DMA)"
    else:
        s4 = 0
        s4_label = f"Neutral ({breadth_200:.0f}% above 200 DMA)"
    signals["breadth_200dma"] = {"score": s4, "detail": s4_label, "value": breadth_200}
    scores.append(s4)

    # ── Signal 5: Net new 52-week highs - lows ──────────────────
    new_highs = 0
    new_lows = 0
    for ticker, df in all_stock_data.items():
        if len(df) < 252:
            continue
        high_52w = df["High"].rolling(252).max().iloc[-1]
        low_52w = df["Low"].rolling(252).min().iloc[-1]
        current = df["Close"].iloc[-1]
        # Within 2% of 52-week high = new high
        if current >= high_52w * 0.98:
            new_highs += 1
        # Within 2% of 52-week low = new low
        if current <= low_52w * 1.02:
            new_lows += 1

    net_new_highs = new_highs - new_lows
    if net_new_highs >= cfg["net_new_highs_bullish"]:
        s5 = 1
        s5_label = f"Bullish (Net new highs: {net_new_highs}, H:{new_highs} L:{new_lows})"
    elif net_new_highs <= cfg["net_new_highs_bearish"]:
        s5 = -1
        s5_label = f"Bearish (Net new highs: {net_new_highs}, H:{new_highs} L:{new_lows})"
    else:
        s5 = 0
        s5_label = f"Neutral (Net new highs: {net_new_highs}, H:{new_highs} L:{new_lows})"
    signals["net_new_highs"] = {"score": s5, "detail": s5_label, "highs": new_highs, "lows": new_lows}
    scores.append(s5)

    # ── Aggregate Regime Score ──────────────────────────────────
    raw_score = sum(scores)  # range: -5 to +5
    # Map to -2..+2 scale
    if raw_score >= 4:
        regime_score = 2
    elif raw_score >= 2:
        regime_score = 1
    elif raw_score >= -1:
        regime_score = 0
    elif raw_score >= -3:
        regime_score = -1
    else:
        regime_score = -2

    posture = REGIME_POSTURE[regime_score]

    # ── Breadth trend (is breadth improving or deteriorating?) ──
    # Check if breadth_50 was higher or lower a week ago
    breadth_trend = "stable"
    above_50_prev = 0
    total_prev = 0
    for ticker, df in all_stock_data.items():
        if len(df) < 55:
            continue
        total_prev += 1
        stock_ma50 = df["Close"].rolling(50).mean()
        if len(stock_ma50) > 5 and df["Close"].iloc[-6] > stock_ma50.iloc[-6]:
            above_50_prev += 1
    if total_prev > 0:
        breadth_50_prev = above_50_prev / total_prev * 100
        if breadth_50 > breadth_50_prev + 3:
            breadth_trend = "improving"
        elif breadth_50 < breadth_50_prev - 3:
            breadth_trend = "deteriorating"

    return {
        "regime_score": regime_score,
        "raw_score": raw_score,
        "label": posture["label"],
        "posture": posture,
        "signals": signals,
        "breadth_trend": breadth_trend,
        "summary": (
            f"Market Regime: {posture['label']} (score {regime_score}, raw {raw_score}/5)\n"
            f"  Breadth trend: {breadth_trend}\n"
            f"  Max capital: {posture['max_capital_pct']}% | "
            f"Risk/trade: {posture['risk_per_trade_pct']}% | "
            f"Max new positions: {posture['max_new_positions']}"
        ),
    }


def print_regime(regime: dict) -> None:
    """Pretty-print regime analysis."""
    print("\n" + "=" * 65)
    print("  LAYER 1: MARKET REGIME ANALYSIS")
    print("=" * 65)
    print(f"\n  {regime['summary']}\n")
    print("  Individual Signals:")
    for name, sig in regime["signals"].items():
        icon = {1: "+", 0: "~", -1: "-"}.get(sig["score"], "?")
        print(f"    [{icon}] {name}: {sig['detail']}")
    print()
