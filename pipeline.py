#!/usr/bin/env python3
"""
Trading Pipeline — Main Orchestrator
Chains all 5 layers: Regime → Sector RS → Stock Screen → Stage Filter → Fundamental Veto

Usage:
    python pipeline.py                          # full scan
    python pipeline.py --capital 2000000        # set capital to ₹20L
    python pipeline.py --sectors-only           # just show sector rankings
    python pipeline.py --regime-only            # just show market regime
    python pipeline.py --stock RELIANCE.NS      # analyze a single stock
"""
import argparse
import datetime as dt
import sys
import time

from config import POSITION_CONFIG, LOOKBACK_DAYS
from data_fetcher import (
    fetch_index_data,
    fetch_sector_data,
    fetch_stock_data_for_sectors,
    fetch_all_stock_data,
    NIFTY500_SECTOR_MAP,
)
from market_regime import compute_regime, print_regime
from sector_rs import scan_sectors, get_top_sectors, print_sector_rankings
from stock_screener import screen_stocks, print_screener_results
from stage_filter import (
    filter_stage2_candidates,
    analyze_stock_stage,
    print_stage_results,
)
from fundamental_veto import (
    generate_final_watchlist,
    print_final_watchlist,
    fetch_fundamentals,
    apply_fundamental_veto,
)


def run_full_pipeline(capital: float = None, days: int = None):
    """Run the complete 5-layer pipeline."""
    if capital is None:
        capital = POSITION_CONFIG["total_capital"]
    if days is None:
        days = LOOKBACK_DAYS

    print("\n" + "#" * 65)
    print("  TRADING PIPELINE — FULL SCAN")
    print(f"  Date: {dt.date.today()}")
    print(f"  Capital: Rs {capital:,.0f}")
    print("#" * 65)

    start_time = time.time()

    # ── STEP 1: Fetch Nifty 50 Index Data ───────────────────────
    print("\n[1/6] Fetching Nifty 50 index data...")
    nifty_df = fetch_index_data(days=days)
    print(f"  Nifty 50: {len(nifty_df)} trading days loaded.")

    # ── STEP 2: Fetch All Stock Data (for breadth) ──────────────
    print("\n[2/6] Fetching stock universe data (for breadth calculations)...")
    all_stock_data = fetch_all_stock_data(days=days)

    # ── STEP 3: Market Regime ───────────────────────────────────
    print("\n[3/6] Computing market regime...")
    regime = compute_regime(nifty_df, all_stock_data)
    print_regime(regime)

    if regime["regime_score"] == -2:
        print("\n  REGIME IS CASH — No new positions recommended.")
        print("  Pipeline complete. Run again when conditions improve.\n")
        return

    # ── STEP 4: Sector Relative Strength ────────────────────────
    print("[4/6] Scanning sector relative strength...")
    sector_data = fetch_sector_data(days=days)
    sector_rankings = scan_sectors(sector_data, nifty_df)
    print_sector_rankings(sector_rankings)
    top_sectors = get_top_sectors(sector_rankings)
    print(f"  Target sectors: {', '.join(top_sectors)}")

    # ── STEP 5: Stock Screening ─────────────────────────────────
    print(f"\n[5/6] Screening stocks in target sectors...")
    # Re-use already fetched data where possible; fetch remaining
    stock_data = {}
    needed_tickers = set()
    for sector in top_sectors:
        for t in NIFTY500_SECTOR_MAP.get(sector, []):
            if t in all_stock_data:
                stock_data[t] = all_stock_data[t]
            else:
                needed_tickers.add(t)

    if needed_tickers:
        print(f"  Fetching {len(needed_tickers)} additional tickers...")
        from data_fetcher import fetch_price_data
        extra = fetch_price_data(list(needed_tickers), days=days)
        stock_data.update(extra)

    screened = screen_stocks(stock_data, nifty_df, sector_data, top_sectors)
    print_screener_results(screened)

    if not screened:
        print("  No stocks pass screening filters. Pipeline complete.\n")
        return

    # ── STEP 6: Stage 2 Filter ──────────────────────────────────
    print("[6/6] Applying Stage 2 + breakout filter...")
    stage2_candidates = filter_stage2_candidates(stock_data, screened)
    print_stage_results(stage2_candidates)

    if not stage2_candidates:
        print("  No Stage 2 candidates found. Check watchlist for developing setups.\n")
        return

    # ── STEP 7: Fundamental Veto + Final Output ─────────────────
    print("Applying fundamental veto and computing position sizes...")
    watchlist = generate_final_watchlist(stage2_candidates, regime, capital)
    print_final_watchlist(watchlist, regime)

    elapsed = time.time() - start_time
    print(f"\n  Pipeline completed in {elapsed:.1f}s")
    print(f"  {len([w for w in watchlist if w['action'] == 'BUY'])} buy signals | "
          f"{len([w for w in watchlist if w['action'] in ('WATCH', 'WATCHLIST')])} on watchlist | "
          f"{len([w for w in watchlist if w['action'] == 'VETOED'])} vetoed")
    print()

    return watchlist


def run_regime_only(days: int = None):
    """Just show market regime."""
    if days is None:
        days = LOOKBACK_DAYS
    print("\n  Fetching data for regime analysis...")
    nifty_df = fetch_index_data(days=days)
    all_stock_data = fetch_all_stock_data(days=days)
    regime = compute_regime(nifty_df, all_stock_data)
    print_regime(regime)
    return regime


def run_sectors_only(days: int = None):
    """Just show sector rankings."""
    if days is None:
        days = LOOKBACK_DAYS
    print("\n  Fetching data for sector analysis...")
    nifty_df = fetch_index_data(days=days)
    sector_data = fetch_sector_data(days=days)
    rankings = scan_sectors(sector_data, nifty_df)
    print_sector_rankings(rankings)
    return rankings


def analyze_single_stock(ticker: str, days: int = None):
    """Deep-dive analysis on a single stock."""
    if days is None:
        days = LOOKBACK_DAYS

    print(f"\n  Analyzing {ticker}...")
    from data_fetcher import fetch_price_data
    data = fetch_price_data([ticker], days=days)
    if ticker not in data:
        print(f"  Error: Could not fetch data for {ticker}")
        return

    df = data[ticker]
    nifty_df = fetch_index_data(days=days)

    # Stage analysis
    stage = analyze_stock_stage(df, ticker)
    print(f"\n  Stage: {stage['stage']['detail']}")
    print(f"  S2 checks: {stage['stage'].get('s2_checks', {})}")
    print(f"  Bases found: {stage['bases_found']}, in Stage 2: {stage['base_count_in_stage2']}")

    if stage["breakout"]:
        bo = stage["breakout"]
        print(f"\n  BREAKOUT DETECTED!")
        print(f"    Date: {bo['breakout_date']}")
        print(f"    Price: {bo['breakout_price']}")
        print(f"    Volume: {bo['volume_ratio']}x average")

        if stage["entry_setup"]:
            es = stage["entry_setup"]
            print(f"    Entry: {es['entry_price']} | Stop: {es['effective_stop']} | Risk: {es['risk_pct']}%")

    if stage["vcp"]:
        vcp = stage["vcp"]
        print(f"\n  VCP: {'Yes' if vcp['is_vcp'] else 'No'} ({vcp['contractions']} contractions)")

    # Fundamentals
    print(f"\n  Fundamentals:")
    fund = fetch_fundamentals(ticker)
    veto = apply_fundamental_veto(fund)
    print(f"    Passes veto: {veto['passes']} ({veto['confidence']})")
    for r in veto["reasons"]:
        print(f"    - {r}")

    # Key numbers
    for key in ["pe_ratio", "peg_ratio", "roe", "debt_equity", "revenue_growth", "earnings_growth"]:
        val = fund.get(key)
        if val is not None:
            if "growth" in key or key == "roe":
                print(f"    {key}: {val*100:.1f}%")
            else:
                print(f"    {key}: {val:.2f}")

    # RS vs Nifty
    from stock_screener import compute_stock_rs
    rs = compute_stock_rs(df["Close"], nifty_df["Close"])
    print(f"\n  RS vs Nifty (6m): {rs:.2f}%")
    print()

    return stage


def main():
    parser = argparse.ArgumentParser(description="Trading Pipeline Scanner")
    parser.add_argument("--capital", type=float, default=None, help="Total capital in Rs")
    parser.add_argument("--days", type=int, default=None, help="Lookback days for historical data")
    parser.add_argument("--regime-only", action="store_true", help="Only show market regime")
    parser.add_argument("--sectors-only", action="store_true", help="Only show sector rankings")
    parser.add_argument("--stock", type=str, help="Analyze a single stock (e.g., RELIANCE.NS)")

    args = parser.parse_args()

    if args.stock:
        analyze_single_stock(args.stock, days=args.days)
    elif args.regime_only:
        run_regime_only(days=args.days)
    elif args.sectors_only:
        run_sectors_only(days=args.days)
    else:
        run_full_pipeline(capital=args.capital, days=args.days)


if __name__ == "__main__":
    main()
