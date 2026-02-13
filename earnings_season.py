"""
Earnings Season Tracker — aggregate quarterly results across the stock universe.

Segments stocks by market cap (Nifty 100 / Midcap 150 / Small),
fetches quarterly financials, and computes aggregate revenue/PAT growth.
"""
import time
import pickle
import logging
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import numpy as np
import requests

from config import (
    NIFTY100_CSV_URL, NIFTY_MIDCAP150_CSV_URL,
    EARNINGS_SEASON_CONFIG, UNIVERSE_CACHE_TTL_HOURS,
)
from data_fetcher import _cache_age_hours, load_universe

logger = logging.getLogger(__name__)

CACHE_DIR = Path(__file__).parent / "scan_cache"
EARNINGS_CACHE = CACHE_DIR / "earnings_season.pkl"
NIFTY100_CACHE = CACHE_DIR / "nifty100_constituents.csv"
MIDCAP150_CACHE = CACHE_DIR / "midcap150_constituents.csv"

_NSE_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
    "Accept": "text/csv,text/html,application/xhtml+xml",
}


# ── Market Cap Segmentation ──────────────────────────────────────


def _download_index_csv(url: str, cache_path: Path) -> set[str]:
    """Download an NSE index CSV and return set of symbols."""
    try:
        resp = requests.get(url, headers=_NSE_HEADERS, timeout=30)
        resp.raise_for_status()
        CACHE_DIR.mkdir(exist_ok=True)
        cache_path.write_bytes(resp.content)
        df = pd.read_csv(cache_path)
        symbols = set(df["Symbol"].str.strip().tolist())
        logger.info("Downloaded %d constituents from %s", len(symbols), url)
        return symbols
    except Exception as e:
        logger.warning("Index CSV download failed (%s): %s", url, e)
        return set()


def _load_index_csv(url: str, cache_path: Path) -> set[str]:
    """Load index CSV from cache or download if stale."""
    age = _cache_age_hours(cache_path)
    if age < UNIVERSE_CACHE_TTL_HOURS and cache_path.exists():
        try:
            df = pd.read_csv(cache_path)
            return set(df["Symbol"].str.strip().tolist())
        except Exception:
            pass
    symbols = _download_index_csv(url, cache_path)
    if not symbols and cache_path.exists():
        try:
            df = pd.read_csv(cache_path)
            return set(df["Symbol"].str.strip().tolist())
        except Exception:
            pass
    return symbols


def load_mcap_segments(universe_symbols: list[str]) -> dict[str, str]:
    """Classify stocks as large/mid/small cap using index membership.

    Returns {symbol: "large"|"mid"|"small"}.
    """
    large_caps = _load_index_csv(NIFTY100_CSV_URL, NIFTY100_CACHE)
    mid_caps = _load_index_csv(NIFTY_MIDCAP150_CSV_URL, MIDCAP150_CACHE)

    segments = {}
    for sym in universe_symbols:
        clean = sym.replace(".NS", "").replace(".BO", "").strip().upper()
        if clean in large_caps:
            segments[sym] = "large"
        elif clean in mid_caps:
            segments[sym] = "mid"
        else:
            segments[sym] = "small"
    return segments


# ── Quarter Identification ───────────────────────────────────────


def determine_target_quarter() -> tuple[datetime, datetime, str]:
    """Determine the most recent completed quarter for earnings analysis.

    Indian FY quarters end: Jun 30 (Q1), Sep 30 (Q2), Dec 31 (Q3), Mar 31 (Q4).
    Target = most recent quarter-end that's at least filing_lag_days ago.

    Returns (target_qtr_end, yoy_qtr_end, quarter_label).
    """
    lag = EARNINGS_SEASON_CONFIG["quarter_filing_lag_days"]
    today = datetime.now()
    cutoff = today - timedelta(days=lag)

    # Quarter ends in calendar order
    year = cutoff.year
    quarter_ends = [
        datetime(year - 1, 6, 30),
        datetime(year - 1, 9, 30),
        datetime(year - 1, 12, 31),
        datetime(year, 3, 31),
        datetime(year, 6, 30),
        datetime(year, 9, 30),
        datetime(year, 12, 31),
        datetime(year + 1, 3, 31),
    ]

    # Find most recent quarter end before cutoff
    target = None
    for qe in reversed(quarter_ends):
        if qe <= cutoff:
            target = qe
            break

    if target is None:
        target = quarter_ends[0]

    # YoY comparison = same quarter last year
    yoy = datetime(target.year - 1, target.month, target.day)

    # Build label
    month = target.month
    if month <= 3:
        fy = target.year
        q = 4
    elif month <= 6:
        fy = target.year + 1
        q = 1
    elif month <= 9:
        fy = target.year + 1
        q = 2
    else:
        fy = target.year + 1
        q = 3

    label = f"Q{q} FY{fy % 100}"
    return target, yoy, label


# ── Fetch Quarterly Results ──────────────────────────────────────


def fetch_all_quarterly_results(
    symbols: list[str],
    progress_callback=None,
) -> dict[str, pd.DataFrame]:
    """Fetch quarterly results for all symbols using the NSE fetcher.

    Leverages per-stock 24h cache. progress_callback(current, total, symbol) optional.
    """
    from nse_data_fetcher import get_nse_fetcher
    nse = get_nse_fetcher()

    results = {}
    total = len(symbols)
    for i, sym in enumerate(symbols):
        if progress_callback:
            progress_callback(i, total, sym)
        try:
            df = nse.fetch_quarterly_results(sym)
            if df is not None and not df.empty:
                results[sym] = df
        except Exception as e:
            logger.debug("Failed to fetch quarterly for %s: %s", sym, e)
    return results


# ── Compute Aggregates ───────────────────────────────────────────


def _safe_pct(new_val, old_val):
    """Compute percentage change handling negative bases."""
    if old_val is None or new_val is None:
        return None
    if old_val == 0:
        return None
    return ((new_val - old_val) / abs(old_val)) * 100


def _match_quarter_row(df: pd.DataFrame, target_date: datetime, tolerance_days: int):
    """Find a row in quarterly results within tolerance of target date."""
    if df is None or df.empty or "date" not in df.columns:
        return None
    dates = pd.to_datetime(df["date"])
    delta = (dates - pd.Timestamp(target_date)).abs()
    min_delta = delta.min()
    if min_delta <= pd.Timedelta(days=tolerance_days):
        idx = delta.idxmin()
        return df.loc[idx]
    return None


def compute_earnings_season(
    quarterly_data: dict[str, pd.DataFrame],
    mcap_segments: dict[str, str],
    target_qtr: datetime,
    yoy_qtr: datetime,
    universe_df: pd.DataFrame,
) -> dict:
    """Compute aggregate earnings season metrics.

    Returns dict with headline numbers, segment breakdown, growth distribution,
    stock details, and sector breakdown.
    """
    tolerance = EARNINGS_SEASON_CONFIG["date_tolerance_days"]
    high_growth = EARNINGS_SEASON_CONFIG["high_growth_threshold_pct"]

    # Build industry map from universe
    industry_map = {}
    if not universe_df.empty:
        for _, row in universe_df.iterrows():
            sym = str(row.get("Symbol", "")).strip()
            if sym:
                ticker = f"{sym}.NS"
                industry_map[ticker] = str(row.get("Industry", "Unknown")).strip()

    all_symbols = set(mcap_segments.keys())
    stock_details = []
    sector_agg = {}  # industry -> {current_rev, yoy_rev, current_pat, yoy_pat, count, reported}

    # Aggregate sums per segment
    seg_data = {}
    for seg in ("large", "mid", "small"):
        seg_data[seg] = {
            "count": 0, "reported": 0,
            "cur_rev": 0, "yoy_rev": 0,
            "cur_pat": 0, "yoy_pat": 0,
            "has_yoy_count": 0,
        }

    reported_count = 0
    yoy_comparison_count = 0
    above_high_growth = 0
    above_25pct = 0
    negative_growth = 0

    for sym in all_symbols:
        segment = mcap_segments.get(sym, "small")
        seg_data[segment]["count"] += 1
        industry = industry_map.get(sym, "Unknown")

        if industry not in sector_agg:
            sector_agg[industry] = {
                "cur_rev": 0, "yoy_rev": 0, "cur_pat": 0, "yoy_pat": 0,
                "count": 0, "reported": 0,
            }
        sector_agg[industry]["count"] += 1

        qdf = quarterly_data.get(sym)
        if qdf is None:
            continue

        cur_row = _match_quarter_row(qdf, target_qtr, tolerance)
        if cur_row is None:
            continue

        # Stock reported
        reported_count += 1
        seg_data[segment]["reported"] += 1
        sector_agg[industry]["reported"] += 1

        cur_rev = cur_row.get("revenue")
        cur_pat = cur_row.get("net_income")
        cur_eps = cur_row.get("diluted_eps")
        cur_opm = cur_row.get("opm_pct")
        cur_npm = cur_row.get("npm_pct")

        yoy_row = _match_quarter_row(qdf, yoy_qtr, tolerance)
        yoy_rev = yoy_row.get("revenue") if yoy_row is not None else None
        yoy_pat = yoy_row.get("net_income") if yoy_row is not None else None

        rev_yoy_pct = _safe_pct(cur_rev, yoy_rev)
        pat_yoy_pct = _safe_pct(cur_pat, yoy_pat)

        # Aggregate sums (only where both current and YoY exist)
        if cur_rev is not None and yoy_rev is not None and cur_rev > 0 and yoy_rev > 0:
            seg_data[segment]["cur_rev"] += cur_rev
            seg_data[segment]["yoy_rev"] += yoy_rev
            sector_agg[industry]["cur_rev"] += cur_rev
            sector_agg[industry]["yoy_rev"] += yoy_rev

        if cur_pat is not None and yoy_pat is not None:
            seg_data[segment]["cur_pat"] += cur_pat
            seg_data[segment]["yoy_pat"] += yoy_pat
            seg_data[segment]["has_yoy_count"] += 1
            sector_agg[industry]["cur_pat"] += cur_pat
            sector_agg[industry]["yoy_pat"] += yoy_pat
            yoy_comparison_count += 1

            if pat_yoy_pct is not None:
                if pat_yoy_pct >= high_growth:
                    above_high_growth += 1
                if pat_yoy_pct >= 25:
                    above_25pct += 1
                if pat_yoy_pct < 0:
                    negative_growth += 1

        detail = {
            "symbol": sym.replace(".NS", ""),
            "segment": segment,
            "industry": industry,
            "revenue_cr": round(cur_rev / 1e7, 1) if cur_rev else None,
            "revenue_yoy_pct": round(rev_yoy_pct, 1) if rev_yoy_pct is not None else None,
            "pat_cr": round(cur_pat / 1e7, 1) if cur_pat else None,
            "pat_yoy_pct": round(pat_yoy_pct, 1) if pat_yoy_pct is not None else None,
            "eps": round(cur_eps, 2) if cur_eps is not None else None,
            "opm_pct": round(cur_opm, 1) if cur_opm is not None else None,
            "npm_pct": round(cur_npm, 1) if cur_npm is not None else None,
        }
        stock_details.append(detail)

    # Compute aggregate growth on summed values
    total_cur_rev = sum(s["cur_rev"] for s in seg_data.values())
    total_yoy_rev = sum(s["yoy_rev"] for s in seg_data.values())
    total_cur_pat = sum(s["cur_pat"] for s in seg_data.values())
    total_yoy_pat = sum(s["yoy_pat"] for s in seg_data.values())

    agg_rev_yoy = _safe_pct(total_cur_rev, total_yoy_rev)
    agg_pat_yoy = _safe_pct(total_cur_pat, total_yoy_pat)

    # Segment breakdown
    by_segment = {}
    for seg in ("large", "mid", "small"):
        sd = seg_data[seg]
        by_segment[seg] = {
            "count": sd["count"],
            "reported": sd["reported"],
            "revenue_yoy": round(_safe_pct(sd["cur_rev"], sd["yoy_rev"]), 1) if _safe_pct(sd["cur_rev"], sd["yoy_rev"]) is not None else None,
            "pat_yoy": round(_safe_pct(sd["cur_pat"], sd["yoy_pat"]), 1) if _safe_pct(sd["cur_pat"], sd["yoy_pat"]) is not None else None,
        }

    # Sector breakdown
    sector_breakdown = {}
    for industry, sa in sorted(sector_agg.items(), key=lambda x: -x[1]["reported"]):
        if sa["reported"] == 0:
            continue
        sector_breakdown[industry] = {
            "count": sa["count"],
            "reported": sa["reported"],
            "revenue_yoy": round(_safe_pct(sa["cur_rev"], sa["yoy_rev"]), 1) if _safe_pct(sa["cur_rev"], sa["yoy_rev"]) is not None else None,
            "pat_yoy": round(_safe_pct(sa["cur_pat"], sa["yoy_pat"]), 1) if _safe_pct(sa["cur_pat"], sa["yoy_pat"]) is not None else None,
        }

    total_universe = len(all_symbols)
    _, _, quarter_label = determine_target_quarter()

    return {
        "quarter_label": quarter_label,
        "target_date": target_qtr,
        "total_universe": total_universe,
        "reported_count": reported_count,
        "reported_pct": round(reported_count / total_universe * 100, 1) if total_universe else 0,
        "aggregate": {
            "revenue_yoy_pct": round(agg_rev_yoy, 1) if agg_rev_yoy is not None else None,
            "pat_yoy_pct": round(agg_pat_yoy, 1) if agg_pat_yoy is not None else None,
        },
        "by_segment": by_segment,
        "growth_distribution": {
            "above_15pct": above_high_growth,
            "above_15pct_pct": round(above_high_growth / yoy_comparison_count * 100, 1) if yoy_comparison_count else 0,
            "above_25pct": above_25pct,
            "negative_growth": negative_growth,
        },
        "stock_details": sorted(stock_details, key=lambda x: x.get("pat_yoy_pct") or -999, reverse=True),
        "sector_breakdown": sector_breakdown,
        "scan_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }


# ── Cache Helpers ────────────────────────────────────────────────


def load_earnings_cache() -> dict | None:
    """Load cached earnings season data if fresh enough."""
    if not EARNINGS_CACHE.exists():
        return None
    age = _cache_age_hours(EARNINGS_CACHE)
    if age > EARNINGS_SEASON_CONFIG["cache_ttl_hours"]:
        return None
    try:
        with open(EARNINGS_CACHE, "rb") as f:
            return pickle.load(f)
    except Exception:
        return None


def save_earnings_cache(data: dict):
    """Save earnings season data to disk cache."""
    CACHE_DIR.mkdir(exist_ok=True)
    with open(EARNINGS_CACHE, "wb") as f:
        pickle.dump(data, f)


# ── Top-Level Orchestrator ───────────────────────────────────────


def run_earnings_scan(universe_df: pd.DataFrame, progress_callback=None) -> dict:
    """Run the full earnings season scan.

    Args:
        universe_df: DataFrame from load_universe() with Symbol, Industry columns.
        progress_callback: Optional (current, total, symbol) callback for UI.

    Returns:
        Earnings season dict (see compute_earnings_season).
    """
    # Build symbol list
    symbols = []
    for _, row in universe_df.iterrows():
        sym = str(row.get("Symbol", "")).strip()
        if sym:
            symbols.append(f"{sym}.NS")

    if not symbols:
        return {}

    # Step 1: Market cap segments
    if progress_callback:
        progress_callback(0, len(symbols), "Loading market cap segments...")
    mcap_segments = load_mcap_segments(symbols)

    # Step 2: Determine target quarter
    target_qtr, yoy_qtr, label = determine_target_quarter()
    logger.info("Earnings scan: %s (target=%s, yoy=%s)", label, target_qtr.date(), yoy_qtr.date())

    # Step 3: Fetch all quarterly results
    quarterly_data = fetch_all_quarterly_results(symbols, progress_callback)

    # Step 4: Compute aggregates
    result = compute_earnings_season(quarterly_data, mcap_segments, target_qtr, yoy_qtr, universe_df)

    # Step 5: Cache
    save_earnings_cache(result)

    return result
