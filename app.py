"""
Trading Pipeline Dashboard — Main Entry Point
Run with: streamlit run app.py
"""
import logging
import sys
import io
import os
import pickle
import datetime as dt
from pathlib import Path

import streamlit as st
import plotly.graph_objects as go

# Suppress yfinance noise
logging.getLogger("yfinance").setLevel(logging.CRITICAL)
logging.getLogger("peewee").setLevel(logging.CRITICAL)

# ── Page config (must be first st call) ─────────────────────────
st.set_page_config(
    page_title="Trading Pipeline",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

from config import POSITION_CONFIG, SECTOR_CONFIG
from data_fetcher import fetch_index_data, fetch_all_stock_data, fetch_sector_data, fetch_price_data
from market_regime import compute_regime
from sector_rs import scan_sectors, get_top_sectors
from stock_screener import screen_stocks
from stage_filter import filter_stage2_candidates
from fundamental_veto import generate_final_watchlist
from dashboard_helpers import regime_color, build_nifty_sparkline

# ── Scan Cache (disk persistence) ──────────────────────────────
CACHE_DIR = Path(__file__).parent / "scan_cache"
CACHE_FILE = CACHE_DIR / "last_scan.pkl"

CACHE_KEYS = [
    "scan_date", "capital", "nifty_df", "all_stock_data", "sector_data",
    "regime", "sector_rankings", "top_sectors", "stock_data",
    "screened_stocks", "stage2_candidates", "final_watchlist",
]


def save_scan_to_disk():
    """Persist current scan results to disk."""
    CACHE_DIR.mkdir(exist_ok=True)
    data = {k: st.session_state[k] for k in CACHE_KEYS if k in st.session_state}
    with open(CACHE_FILE, "wb") as f:
        pickle.dump(data, f)


def load_scan_from_disk():
    """Load previous scan results from disk into session state."""
    if not CACHE_FILE.exists():
        return False
    try:
        with open(CACHE_FILE, "rb") as f:
            data = pickle.load(f)
        for k, v in data.items():
            st.session_state[k] = v
        return True
    except Exception:
        return False


def is_cache_stale(max_age_hours: int = 24) -> bool:
    """Check if the cached scan data is older than max_age_hours."""
    scan_date_str = st.session_state.get("scan_date")
    if not scan_date_str:
        return True
    try:
        scan_dt = dt.datetime.strptime(scan_date_str, "%Y-%m-%d %H:%M")
        age = dt.datetime.now() - scan_dt
        return age.total_seconds() > max_age_hours * 3600
    except Exception:
        return True


# Auto-load cached scan on first visit
if "regime" not in st.session_state:
    if load_scan_from_disk():
        if is_cache_stale():
            st.toast(f"Cached scan from {st.session_state.get('scan_date', '?')} is stale — click Run Scan to refresh")
        else:
            st.toast(f"Loaded scan from {st.session_state.get('scan_date', 'disk')}")


# ── Sidebar ─────────────────────────────────────────────────────
with st.sidebar:
    st.title("Trading Pipeline")
    st.caption("Weinstein + O'Neil / Minervini")

    st.divider()

    capital = st.number_input(
        "Capital (INR)",
        min_value=100_000,
        value=POSITION_CONFIG["total_capital"],
        step=100_000,
        format="%d",
    )

    col_a, col_b = st.columns(2)
    with col_a:
        risk_pct = st.number_input("Risk %", min_value=0.25, max_value=3.0, value=1.0, step=0.25)
    with col_b:
        top_n = st.number_input("Top Sectors", min_value=2, max_value=8, value=SECTOR_CONFIG["top_sectors_count"], step=1)

    max_positions = st.number_input("Max Positions", min_value=1, max_value=20, value=6, step=1)

    st.divider()

    run_scan = st.button("Run Scan", type="primary", use_container_width=True)

    if "scan_date" in st.session_state:
        st.caption(f"Last scan: {st.session_state.scan_date}")
        if is_cache_stale():
            st.warning("Data is stale (>24h old)", icon="⚠️")

    st.divider()
    st.caption("Built with Streamlit + Plotly")


# ── Scan Orchestration ──────────────────────────────────────────
def run_pipeline_scan():
    """Run the full 5-layer pipeline and store results in session state."""
    # Capture stdout to suppress print output from pipeline modules
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()

    try:
        with st.status("Running pipeline scan...", expanded=True) as status:
            progress = st.progress(0)

            # Step 1: Fetch Nifty data
            st.write("Fetching Nifty 50 index data...")
            progress.progress(5)
            nifty_df = fetch_index_data()
            st.session_state.nifty_df = nifty_df

            # Step 2: Fetch all stock data
            st.write("Fetching stock universe data (this may take a minute)...")
            progress.progress(10)
            all_stock_data = fetch_all_stock_data()
            st.session_state.all_stock_data = all_stock_data
            st.write(f"  Loaded {len(all_stock_data)} stocks")
            progress.progress(30)

            # Step 3: Market regime
            st.write("Computing market regime...")
            regime = compute_regime(nifty_df, all_stock_data)
            st.session_state.regime = regime
            st.write(f"  Regime: {regime['label']} (score {regime['regime_score']:+d})")
            progress.progress(40)

            # Step 4: Sector RS
            st.write("Fetching sector data & computing RS...")
            sector_data = fetch_sector_data()
            st.session_state.sector_data = sector_data
            progress.progress(55)

            sector_rankings = scan_sectors(sector_data, nifty_df)
            st.session_state.sector_rankings = sector_rankings

            top_sectors = get_top_sectors(sector_rankings, n=top_n)
            st.session_state.top_sectors = top_sectors
            st.write(f"  Top sectors: {', '.join(top_sectors)}")
            progress.progress(60)

            # Step 5: Stock screening
            st.write("Screening stocks in top sectors...")
            # Reuse already-fetched data
            stock_data = dict(all_stock_data)
            # Fetch any missing tickers from target sectors
            from data_fetcher import get_all_stock_tickers, get_sector_for_stock, NIFTY500_SECTOR_MAP
            needed = []
            for sector in top_sectors:
                for t in NIFTY500_SECTOR_MAP.get(sector, []):
                    if t not in stock_data:
                        needed.append(t)
            if needed:
                extra = fetch_price_data(needed)
                stock_data.update(extra)
            st.session_state.stock_data = stock_data

            screened = screen_stocks(stock_data, nifty_df, sector_data, top_sectors)
            st.session_state.screened_stocks = screened
            st.write(f"  {len(screened)} stocks passed screening")
            progress.progress(75)

            # Step 6: Stage filter
            st.write("Running stage analysis...")
            stage2 = filter_stage2_candidates(stock_data, screened) if screened else []
            st.session_state.stage2_candidates = stage2
            st.write(f"  {len(stage2)} Stage 2 candidates")
            progress.progress(85)

            # Step 7: Fundamental veto + watchlist
            st.write("Applying fundamental veto & sizing positions...")
            watchlist = generate_final_watchlist(stage2, regime, capital) if stage2 else []
            st.session_state.final_watchlist = watchlist
            progress.progress(95)

            buy_count = sum(1 for w in watchlist if w.get("action") == "BUY")
            watch_count = sum(1 for w in watchlist if w.get("action") in ("WATCH", "WATCHLIST"))
            st.write(f"  {buy_count} BUY signals, {watch_count} watchlist")

            st.session_state.scan_date = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
            st.session_state.capital = capital

            # Save to disk for persistence across restarts
            st.write("Saving scan results to disk...")
            save_scan_to_disk()
            progress.progress(100)

            status.update(label="Scan complete!", state="complete")

    finally:
        sys.stdout = old_stdout


if run_scan:
    run_pipeline_scan()


# ── Home Page ───────────────────────────────────────────────────
st.title("Trading Pipeline Dashboard")

if "regime" not in st.session_state:
    st.info("Click **Run Scan** in the sidebar to start the pipeline.")
    st.stop()

regime = st.session_state.regime
watchlist = st.session_state.get("final_watchlist", [])
top_sectors = st.session_state.get("top_sectors", [])

# Regime banner
label = regime["label"]
color = regime_color(label)
score = regime["regime_score"]
st.markdown(
    f"""
    <div style="background: {color}22; border-left: 5px solid {color};
                padding: 15px 20px; border-radius: 0 8px 8px 0; margin-bottom: 20px;">
        <span style="font-size: 1.6em; font-weight: 700; color: {color};">
            {label.upper()}
        </span>
        <span style="font-size: 1.1em; margin-left: 15px; color: #ccc;">
            Score: {score:+d} &nbsp;|&nbsp; Max Capital: {regime['posture']['max_capital_pct']}%
            &nbsp;|&nbsp; Risk/Trade: {regime['posture']['risk_per_trade_pct']}%
        </span>
    </div>
    """,
    unsafe_allow_html=True,
)

# Summary metrics row
col1, col2, col3, col4 = st.columns(4)
buy_count = sum(1 for w in watchlist if w.get("action") == "BUY")
watch_count = sum(1 for w in watchlist if w.get("action") in ("WATCH", "WATCHLIST"))
screened_count = len(st.session_state.get("screened_stocks", []))
stage2_count = len(st.session_state.get("stage2_candidates", []))

col1.metric("Top Sectors", ", ".join(top_sectors[:4]))
col2.metric("Stocks Screened", screened_count)
col3.metric("Stage 2 Candidates", stage2_count)
col4.metric("BUY Signals", buy_count)

# Nifty sparkline
st.subheader("Nifty 50 — Last 90 Days")
nifty_df = st.session_state.nifty_df
fig = build_nifty_sparkline(nifty_df, days=90)
st.plotly_chart(fig, use_container_width=True)

# Quick links
st.divider()
st.markdown("""
**Navigate the pages in the sidebar:**
- **Market Regime** — Full regime analysis with charts
- **Sector Rotation** — RS rankings, heatmap, RS line chart
- **Stock Scanner** — Screened stocks with scatter plot
- **Stage Analysis** — Stage 2 candidates with candlestick charts
- **Watchlist** — Final BUY/WATCH list with position sizing
- **Stock Deep Dive** — Drill into any single stock
""")
