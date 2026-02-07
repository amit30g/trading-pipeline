"""
Layer 3: Stock Screener — Rank stocks within hot sectors by leadership quality.
"""
import pandas as pd
import numpy as np
from config import SCREENER_CONFIG
from data_fetcher import get_sector_for_stock


def compute_stock_rs(
    stock_close: pd.Series,
    benchmark_close: pd.Series,
    period: int = None,
) -> float:
    """
    Compute relative strength of a stock vs a benchmark over `period` days.
    Returns percentage outperformance.
    """
    if period is None:
        period = SCREENER_CONFIG["rs_period"]

    # Align on common dates
    combined = pd.DataFrame({"stock": stock_close, "bench": benchmark_close}).dropna()
    if len(combined) < period:
        period = len(combined) - 1
    if period <= 0:
        return 0.0

    stock_ret = combined["stock"].iloc[-1] / combined["stock"].iloc[-period] - 1
    bench_ret = combined["bench"].iloc[-1] / combined["bench"].iloc[-period] - 1
    return round((stock_ret - bench_ret) * 100, 2)


def compute_accumulation_distribution(df: pd.DataFrame, lookback: int = None) -> float:
    """
    Measure accumulation by comparing volume on up-days vs down-days.
    Returns ratio: >1 means accumulation, <1 means distribution.
    """
    if lookback is None:
        lookback = SCREENER_CONFIG["accumulation_days_lookback"]

    recent = df.tail(lookback).copy()
    recent["change"] = recent["Close"].diff()

    up_vol = recent.loc[recent["change"] > 0, "Volume"].sum()
    down_vol = recent.loc[recent["change"] < 0, "Volume"].sum()

    if down_vol == 0:
        return 2.0  # strong accumulation
    return round(up_vol / down_vol, 2)


def distance_from_52w_high(df: pd.DataFrame) -> float:
    """Return % distance from 52-week high. 0 = at high, 20 = 20% below."""
    if len(df) < 252:
        high = df["High"].max()
    else:
        high = df["High"].tail(252).max()
    current = df["Close"].iloc[-1]
    return round((1 - current / high) * 100, 2)


def screen_stocks(
    stock_data: dict[str, pd.DataFrame],
    nifty_df: pd.DataFrame,
    sector_data: dict[str, pd.DataFrame],
    target_sectors: list[str],
) -> list[dict]:
    """
    Screen and rank stocks within target sectors.

    Filters applied:
    1. RS vs Nifty > threshold
    2. RS vs own sector > threshold
    3. Within X% of 52-week high
    4. Minimum average volume
    5. Accumulation > distribution

    Returns list of dicts sorted by composite leadership score.
    """
    cfg = SCREENER_CONFIG
    nifty_close = nifty_df["Close"]

    results = []

    for ticker, df in stock_data.items():
        if len(df) < 100:
            continue

        # Determine which sector this stock belongs to
        sector = get_sector_for_stock(ticker)
        if sector not in target_sectors:
            continue

        close = df["Close"]

        # ── RS vs Nifty ─────────────────────────────────────────
        rs_vs_nifty = compute_stock_rs(close, nifty_close)

        # ── RS vs Sector ────────────────────────────────────────
        rs_vs_sector = 0.0
        if sector in sector_data:
            sector_close = sector_data[sector]["Close"]
            rs_vs_sector = compute_stock_rs(close, sector_close)

        # ── Distance from 52-week high ──────────────────────────
        dist_high = distance_from_52w_high(df)

        # ── Average volume ──────────────────────────────────────
        avg_vol = df["Volume"].tail(50).mean()

        # ── Accumulation/Distribution ───────────────────────────
        acc_dist = compute_accumulation_distribution(df)

        # ── Apply filters ───────────────────────────────────────
        if rs_vs_nifty < cfg["min_rs_vs_nifty"]:
            continue
        if rs_vs_sector < cfg["min_rs_vs_sector"]:
            continue
        if dist_high > cfg["max_distance_from_52w_high_pct"]:
            continue
        if avg_vol < cfg["min_avg_volume"]:
            continue
        if acc_dist < cfg["min_accumulation_ratio"]:
            continue

        # ── Composite leadership score ──────────────────────────
        # Weight: RS vs Nifty (30%) + RS vs Sector (20%) +
        #         Proximity to high (25%) + Accumulation (25%)
        proximity_score = max(0, 25 - dist_high)  # 0-25 scale
        acc_score = min(acc_dist * 10, 25)  # cap at 25

        composite = (
            0.30 * rs_vs_nifty +
            0.20 * rs_vs_sector +
            0.25 * proximity_score +
            0.25 * acc_score
        )

        results.append({
            "ticker": ticker,
            "sector": sector,
            "rs_vs_nifty": rs_vs_nifty,
            "rs_vs_sector": rs_vs_sector,
            "dist_from_high_pct": dist_high,
            "avg_volume": int(avg_vol),
            "accumulation_ratio": acc_dist,
            "leadership_score": round(composite, 2),
            "close": round(float(close.iloc[-1]), 2),
        })

    results.sort(key=lambda x: x["leadership_score"], reverse=True)
    return results


def print_screener_results(results: list[dict], max_show: int = 30) -> None:
    """Pretty-print stock screening results."""
    print("\n" + "=" * 90)
    print("  LAYER 3: STOCK LEADERSHIP SCREENING")
    print("=" * 90)
    print(
        f"\n  {'#':<4} {'Ticker':<16} {'Sector':<18} {'RS/Nifty':>9} {'RS/Sect':>8} "
        f"{'%frHigh':>8} {'Acc/Dist':>9} {'Score':>7}"
    )
    print("  " + "-" * 86)

    for i, s in enumerate(results[:max_show]):
        print(
            f"  {i+1:<4} {s['ticker']:<16} {s['sector']:<18} "
            f"{s['rs_vs_nifty']:>9.2f} {s['rs_vs_sector']:>8.2f} "
            f"{s['dist_from_high_pct']:>7.1f}% "
            f"{s['accumulation_ratio']:>9.2f} {s['leadership_score']:>7.2f}"
        )

    if len(results) > max_show:
        print(f"\n  ... and {len(results) - max_show} more stocks")
    print(f"\n  Total stocks passing filters: {len(results)}")
    print()
