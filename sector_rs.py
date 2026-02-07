"""
Layer 2: Sector Relative Strength Scanner
Identifies sectors with emerging relative strength using Mansfield RS.
"""
import pandas as pd
import numpy as np
from config import SECTOR_CONFIG


def compute_mansfield_rs(
    sector_close: pd.Series,
    index_close: pd.Series,
    ma_period: int = None,
) -> pd.Series:
    """
    Compute Mansfield Relative Strength.
    RS = (sector / index) / SMA(sector / index, ma_period) - 1
    Positive = outperforming, Negative = underperforming.
    """
    if ma_period is None:
        ma_period = SECTOR_CONFIG["rs_ma_period"]

    # Align dates
    combined = pd.DataFrame({
        "sector": sector_close,
        "index": index_close,
    }).dropna()

    ratio = combined["sector"] / combined["index"]
    ratio_ma = ratio.rolling(min(ma_period, len(ratio) - 1)).mean()
    mansfield_rs = (ratio / ratio_ma - 1) * 100  # as percentage
    return mansfield_rs


def compute_rs_momentum(
    sector_close: pd.Series,
    index_close: pd.Series,
    periods: list[int] = None,
) -> dict[str, float]:
    """
    Compute rate of change of relative strength over multiple periods.
    Returns dict: {"1m": x, "3m": y, "6m": z}
    """
    if periods is None:
        periods = SECTOR_CONFIG["momentum_periods"]

    ratio = sector_close / index_close
    ratio = ratio.dropna()

    labels = {21: "1m", 63: "3m", 126: "6m"}
    result = {}
    for p in periods:
        label = labels.get(p, f"{p}d")
        if len(ratio) > p:
            roc = (ratio.iloc[-1] / ratio.iloc[-p] - 1) * 100
            result[label] = round(roc, 2)
        else:
            result[label] = None
    return result


def analyze_rs_trend(mansfield_rs: pd.Series, lookback: int = 21) -> str:
    """
    Determine RS trend direction over the last `lookback` days.
    Returns: "rising", "falling", or "flat"
    """
    if len(mansfield_rs) < lookback:
        return "flat"
    recent = mansfield_rs.iloc[-lookback:]
    # Simple linear regression slope
    x = np.arange(len(recent))
    slope = np.polyfit(x, recent.values, 1)[0]
    if slope > 0.02:
        return "rising"
    elif slope < -0.02:
        return "falling"
    return "flat"


def scan_sectors(
    sector_data: dict[str, pd.DataFrame],
    nifty_df: pd.DataFrame,
) -> list[dict]:
    """
    Analyze all sectors and return ranked list.

    Returns list of dicts sorted by composite RS score (best first):
    [{
        "sector": name,
        "mansfield_rs": latest_value,
        "rs_trend": "rising"/"falling"/"flat",
        "momentum": {"1m": x, "3m": y, "6m": z},
        "composite_score": float,
    }, ...]
    """
    nifty_close = nifty_df["Close"]
    results = []

    for sector_name, sector_df in sector_data.items():
        sector_close = sector_df["Close"]

        # Mansfield RS
        mrs = compute_mansfield_rs(sector_close, nifty_close)
        if len(mrs.dropna()) < 20:
            continue

        latest_rs = mrs.iloc[-1]
        rs_trend = analyze_rs_trend(mrs)
        momentum = compute_rs_momentum(sector_close, nifty_close)

        # Composite score: weighted combination
        # Mansfield RS level (40%) + RS trend (20%) + momentum blend (40%)
        trend_score = {"rising": 1, "flat": 0, "falling": -1}[rs_trend]

        mom_values = [v for v in momentum.values() if v is not None]
        avg_momentum = np.mean(mom_values) if mom_values else 0

        composite = (
            0.4 * latest_rs +
            0.2 * trend_score * 2 +  # scale trend to be comparable
            0.4 * avg_momentum
        )

        results.append({
            "sector": sector_name,
            "mansfield_rs": round(latest_rs, 2),
            "rs_trend": rs_trend,
            "momentum": momentum,
            "composite_score": round(composite, 2),
        })

    results.sort(key=lambda x: x["composite_score"], reverse=True)
    return results


def get_top_sectors(sector_rankings: list[dict], n: int = None) -> list[str]:
    """Return names of top N sectors by composite score."""
    if n is None:
        n = SECTOR_CONFIG["top_sectors_count"]
    # Only include sectors with positive RS trend or at least neutral
    top = [s for s in sector_rankings[:n] if s["rs_trend"] != "falling"]
    # If we filtered too many, take top N regardless
    if len(top) < 2:
        top = sector_rankings[:n]
    return [s["sector"] for s in top]


def print_sector_rankings(rankings: list[dict], top_n: int = None) -> None:
    """Pretty-print sector RS rankings."""
    if top_n is None:
        top_n = SECTOR_CONFIG["top_sectors_count"]

    print("\n" + "=" * 65)
    print("  LAYER 2: SECTOR RELATIVE STRENGTH")
    print("=" * 65)
    print(f"\n  {'Rank':<5} {'Sector':<22} {'RS':>7} {'Trend':<8} {'1m':>7} {'3m':>7} {'6m':>7} {'Score':>7}")
    print("  " + "-" * 73)

    for i, s in enumerate(rankings):
        mom = s["momentum"]
        marker = " <<" if i < top_n else ""
        trend_icon = {"rising": "^", "flat": "-", "falling": "v"}[s["rs_trend"]]
        print(
            f"  {i+1:<5} {s['sector']:<22} "
            f"{s['mansfield_rs']:>7.2f} {trend_icon:<8} "
            f"{mom.get('1m', 'N/A'):>7} {mom.get('3m', 'N/A'):>7} {mom.get('6m', 'N/A'):>7} "
            f"{s['composite_score']:>7.2f}{marker}"
        )
    print(f"\n  << = Top {top_n} sectors (hunting ground)")
    print()
