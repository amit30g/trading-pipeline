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

from config import POSITION_CONFIG, SECTOR_CONFIG, REGIME_CONFIG, SMART_MONEY_CONFIG
from data_fetcher import (
    fetch_index_data, fetch_all_stock_data, fetch_sector_data,
    fetch_price_data, fetch_macro_data, get_sector_map,
)
from market_regime import compute_regime
from sector_rs import scan_sectors, get_top_sectors
from stock_screener import screen_stocks
from stage_filter import filter_stage2_candidates
from fundamental_veto import generate_final_watchlist
from conviction_scorer import rank_candidates_by_conviction, get_top_conviction_ideas
from position_manager import get_positions_summary, load_positions
from nse_data_fetcher import get_nse_fetcher
from dashboard_helpers import (
    regime_color, build_nifty_sparkline, build_macro_pulse_html,
    build_mini_heatmap, compute_quality_radar,
)

# ── Scan Cache (disk persistence) ──────────────────────────────
CACHE_DIR = Path(__file__).parent / "scan_cache"
CACHE_FILE = CACHE_DIR / "last_scan.pkl"

CACHE_KEYS = [
    "scan_date", "capital", "nifty_df", "all_stock_data", "sector_data",
    "regime", "sector_rankings", "top_sectors", "stock_data",
    "screened_stocks", "stage2_candidates", "final_watchlist",
    "macro_data", "quality_radar", "universe_count",
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

    chart_tf = st.selectbox(
        "Chart Timeframe",
        ["Weekly", "Daily", "Monthly"],
        index=0,
        key="chart_timeframe",
    )

    st.divider()

    run_scan = st.button("Run Scan", type="primary", use_container_width=True)

    if "scan_date" in st.session_state:
        st.caption(f"Last scan: {st.session_state.scan_date}")
        if "universe_count" in st.session_state:
            st.caption(f"Universe: {st.session_state.universe_count} stocks")
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

            # Step 0: Fetch macro data
            st.write("Fetching macro data...")
            progress.progress(2)
            macro_data = fetch_macro_data()
            st.session_state.macro_data = macro_data

            # Step 1: Fetch Nifty data
            st.write("Fetching Nifty 50 index data...")
            progress.progress(5)
            nifty_df = fetch_index_data()
            st.session_state.nifty_df = nifty_df

            # Step 2: Fetch all stock data
            from data_fetcher import get_all_stock_tickers
            all_tickers = get_all_stock_tickers()
            st.session_state.universe_count = len(all_tickers)
            st.write(f"Fetching stock universe data ({len(all_tickers)} stocks — this may take a few minutes)...")
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
            stock_data = dict(all_stock_data)
            sector_map = get_sector_map()
            needed = []
            for sector in top_sectors:
                for t in sector_map.get(sector, []):
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
            progress.progress(92)

            # Step 8: Quality Radar
            st.write("Computing Quality Radar...")
            quality_radar = compute_quality_radar(watchlist)
            st.session_state.quality_radar = quality_radar
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


# ── Home Page — Command Center ────────────────────────────────
st.markdown(
    '<h1 style="margin-bottom:0;">Command Center</h1>'
    '<p style="color:#888; margin-top:0; font-size:0.95em;">'
    'Regime + Conviction Ideas + Positions + Smart Money</p>',
    unsafe_allow_html=True,
)

if "regime" not in st.session_state:
    st.info("Click **Run Scan** in the sidebar to start the pipeline.")
    st.stop()

regime = st.session_state.regime
watchlist = st.session_state.get("final_watchlist", [])
top_sectors = st.session_state.get("top_sectors", [])

# ══════════════════════════════════════════════════════════════════
# SECTION 1: Regime + FII/DII
# ══════════════════════════════════════════════════════════════════
label = regime["label"]
color = regime_color(label)
score = regime["regime_score"]

# Fetch FII/DII data
nse_fetcher = get_nse_fetcher()
fii_dii = None
try:
    fii_dii = nse_fetcher.fetch_fii_dii_data()
except Exception:
    pass

# Regime badge + FII/DII inline
fii_net_str = ""
dii_net_str = ""
if fii_dii:
    fii_net = fii_dii.get("fii_net", 0)
    dii_net = fii_dii.get("dii_net", 0)
    fii_color = "#26a69a" if fii_net >= 0 else "#ef5350"
    dii_color = "#26a69a" if dii_net >= 0 else "#ef5350"
    fii_net_str = (
        f'<span style="margin-left:15px; font-size:0.9em;">'
        f'FII: <span style="color:{fii_color}; font-weight:600;">{fii_net:+,.0f} Cr</span>'
        f' &nbsp; DII: <span style="color:{dii_color}; font-weight:600;">{dii_net:+,.0f} Cr</span>'
        f'</span>'
    )

st.markdown(
    f'''
    <div style="background:{color}22; border-left:5px solid {color};
                padding:12px 20px; border-radius:0 8px 8px 0; margin-bottom:12px;">
        <span style="font-size:1.8em; font-weight:800; color:{color};">{label.upper()}</span>
        <span style="font-size:1em; margin-left:15px; color:#ccc;">
            Score {score:+d} &nbsp;|&nbsp;
            Capital: {regime["posture"]["max_capital_pct"]}% &nbsp;|&nbsp;
            Risk/Trade: {regime["posture"]["risk_per_trade_pct"]}%
        </span>
        {fii_net_str}
    </div>
    ''',
    unsafe_allow_html=True,
)

# Breadth metrics
signals = regime.get("signals", {})
breadth_50 = signals.get("breadth_50dma", {}).get("value", 0)
breadth_200 = signals.get("breadth_200dma", {}).get("value", 0)
nh_signals = signals.get("net_new_highs", {})
net_highs = nh_signals.get("highs", 0)
net_lows = nh_signals.get("lows", 0)
net_nh = net_highs - net_lows
idx_detail = signals.get("index_vs_200dma", {}).get("detail", "N/A")

col_b1, col_b2, col_b3, col_b4 = st.columns(4)
col_b1.metric("% > 50 DMA", f"{breadth_50:.0f}%")
col_b2.metric("% > 200 DMA", f"{breadth_200:.0f}%")
col_b3.metric("Net New Highs", f"{net_nh:+d}", delta=f"H:{net_highs} L:{net_lows}")
col_b4.metric("Nifty vs 200 DMA", idx_detail)


# ══════════════════════════════════════════════════════════════════
# SECTION 2: Top 3 Conviction Ideas
# ══════════════════════════════════════════════════════════════════
st.markdown("#### Top Conviction Ideas")

with st.expander("How Conviction Scores Work"):
    st.markdown("""
**Conviction Score** (0-100) ranks stocks by combining multiple factors:

| Factor | Max Points | How It's Scored |
|--------|-----------|-----------------|
| Sector Rank | 40 pts | #1 ranked sector = 40, #2 = 35, #3 = 30, #4 = 25. Being in a top sector is the single biggest factor. |
| Stage 2 Score | 20 pts | Perfect S2 (7/7 criteria met) = 20 pts. Each missing criterion reduces the score. |
| Base Count | 10 pts | 1st base = 10 pts, 2nd = 7, 3rd = 4. First-base breakouts have the highest success rate. |
| RS Percentile | 15 pts | Top RS vs Nifty among all screened stocks = 15 pts, scaled by percentile rank. |
| Accumulation | 15 pts | Highest accumulation ratio = 15 pts, scaled by percentile rank. |

**Bonus points** (up to +10): VCP pattern (+5), tight risk <5% (+3), volume surge on breakout (+2).

**Scores above 60** = high conviction (green border). **40-60** = moderate (orange). **Below 40** = lower conviction (red).
""")

sector_rankings = st.session_state.get("sector_rankings", [])
stage2_candidates = st.session_state.get("stage2_candidates", [])

if watchlist and sector_rankings:
    # Rank watchlist by conviction
    ranked = rank_candidates_by_conviction(
        candidates=list(watchlist),
        sector_rankings=sector_rankings,
    )
    top_ideas = get_top_conviction_ideas(ranked, top_n=3)

    if top_ideas:
        idea_cols = st.columns(len(top_ideas))
        for idx, (col, idea) in enumerate(zip(idea_cols, top_ideas)):
            with col:
                conv_score = idea.get("conviction_score", 0)
                ticker_name = idea.get("ticker", "").replace(".NS", "")
                sector = idea.get("sector", "")
                es = idea.get("entry_setup", {}) or {}
                pos = idea.get("position", {})
                targets = idea.get("targets", {})
                vcp = idea.get("vcp")

                # Build rationale chips
                rationale = []
                if sector in [r.get("sector") or r.get("name", "") for r in sector_rankings[:2]]:
                    rationale.append("Top Sector")
                s2_score = idea.get("stage", {}).get("s2_score", 0)
                if s2_score == 7:
                    rationale.append("Perfect S2")
                if vcp and vcp.get("is_vcp"):
                    rationale.append("VCP")
                breakout = idea.get("breakout", {})
                if breakout and breakout.get("base_number", 99) == 1:
                    rationale.append("1st Base")

                conv_color = "#26a69a" if conv_score >= 60 else "#FF9800" if conv_score >= 40 else "#ef5350"

                st.markdown(
                    f"""<div style="background:#1a1a2e; border:2px solid {conv_color};
                        border-radius:12px; padding:16px; text-align:center;">
                        <div style="font-size:0.8em; color:#999;">#{idx+1}</div>
                        <div style="font-size:1.4em; font-weight:700; margin:4px 0;">{ticker_name}</div>
                        <div style="font-size:0.85em; color:#aaa; margin-bottom:8px;">{sector}</div>
                        <div style="font-size:1.8em; font-weight:800; color:{conv_color};">{conv_score:.0f}</div>
                        <div style="font-size:0.75em; color:#999; margin-bottom:8px;">CONVICTION</div>
                    </div>""",
                    unsafe_allow_html=True,
                )
                if es:
                    st.markdown(
                        f"Entry **{es.get('entry_price', 0):.1f}** | "
                        f"Stop **{es.get('effective_stop', 0):.1f}** | "
                        f"Risk **{es.get('risk_pct', 0):.1f}%**"
                    )
                if pos.get("shares"):
                    st.caption(f"Shares: {pos['shares']} | R:R: {targets.get('reward_risk_ratio', 0):.1f}")
                if rationale:
                    st.caption(" | ".join(rationale))

                # Build human-readable "Why" sentence
                why_parts = []
                sector_rank_pos = next(
                    (i + 1 for i, r in enumerate(sector_rankings)
                     if (r.get("sector") or r.get("name", "")) == sector),
                    None,
                )
                if sector_rank_pos and sector_rank_pos <= 4:
                    why_parts.append(f"#{sector_rank_pos} ranked sector ({sector})")
                s2_score = idea.get("stage", {}).get("s2_score", 0)
                if s2_score >= 6:
                    why_parts.append(f"Stage 2 score {s2_score}/7")
                if vcp and vcp.get("is_vcp"):
                    why_parts.append("VCP breakout")
                risk_pct = es.get("risk_pct", 0) if es else 0
                if risk_pct and 0 < risk_pct < 5:
                    why_parts.append(f"low risk ({risk_pct:.1f}%)")
                accum = idea.get("accumulation_ratio", 0)
                if accum and accum > 1.3:
                    why_parts.append(f"strong accumulation ({accum:.1f}x)")
                if why_parts:
                    st.caption(f"Why: {', '.join(why_parts)}.")
    else:
        st.markdown(
            '<div style="background:#1e1e1e; border-radius:8px; padding:20px; text-align:center;'
            ' color:#888; font-style:italic; margin:10px 0;">'
            'No high-conviction setups today — patience is alpha</div>',
            unsafe_allow_html=True,
        )
else:
    st.caption("Run a scan to generate conviction rankings.")


# ══════════════════════════════════════════════════════════════════
# SECTION 3: Active Positions Summary
# ══════════════════════════════════════════════════════════════════
st.markdown("#### Active Positions")

positions = load_positions()
if positions:
    stock_data = st.session_state.get("stock_data", {})
    pos_summaries = get_positions_summary(stock_data)

    if pos_summaries:
        total_open_pnl = sum(s.get("pnl", 0) for s in pos_summaries)
        pnl_color = "#26a69a" if total_open_pnl >= 0 else "#ef5350"

        pos_col1, pos_col2 = st.columns([1, 4])
        with pos_col1:
            st.metric("Open Positions", len(pos_summaries))
            st.markdown(
                f'<div style="font-size:1.2em; font-weight:600; color:{pnl_color};">'
                f'Open P&L: {total_open_pnl:+,.0f}</div>',
                unsafe_allow_html=True,
            )
        with pos_col2:
            import pandas as pd
            pos_rows = []
            for s in pos_summaries[:5]:
                action = s.get("suggested_action", "HOLD")
                action_icons = {"SELL": "🔴", "PARTIAL SELL": "🟠", "ADD": "🟢", "HOLD": "⚪"}
                pos_rows.append({
                    "Ticker": s["ticker"].replace(".NS", ""),
                    "Entry": f"{s['entry_price']:.1f}",
                    "Current": f"{s.get('current_price', 0):.1f}" if s.get("current_price") else "N/A",
                    "P&L %": f"{s.get('pnl_pct', 0):+.1f}%",
                    "Days": s.get("days_held", 0),
                    "Action": f"{action_icons.get(action, '')} {action}",
                })
            st.dataframe(pd.DataFrame(pos_rows), use_container_width=True, hide_index=True)

        if len(pos_summaries) > 5:
            st.caption(f"+{len(pos_summaries) - 5} more — see Positions page")
else:
    st.caption("No active positions — add positions from the Positions page.")


# ══════════════════════════════════════════════════════════════════
# SECTION 4: Smart Money Dashboard
# ══════════════════════════════════════════════════════════════════
st.markdown("#### Smart Money")

sm_col1, sm_col2 = st.columns(2)

with sm_col1:
    # FII/DII flow cards
    if fii_dii:
        fii_net = fii_dii.get("fii_net", 0)
        dii_net = fii_dii.get("dii_net", 0)
        fii_buy = fii_dii.get("fii_buy", 0)
        fii_sell = fii_dii.get("fii_sell", 0)
        dii_buy = fii_dii.get("dii_buy", 0)
        dii_sell = fii_dii.get("dii_sell", 0)
        fii_c = "#26a69a" if fii_net >= 0 else "#ef5350"
        dii_c = "#26a69a" if dii_net >= 0 else "#ef5350"
        fii_label = "BUYING" if fii_net >= 0 else "SELLING"
        dii_label = "BUYING" if dii_net >= 0 else "SELLING"
        fii_date = fii_dii.get("date", "")

        st.markdown(
            f"""<div style="display:flex; gap:12px;">
                <div style="flex:1; background:#1a1a2e; border:2px solid {fii_c};
                    border-radius:10px; padding:14px; text-align:center;">
                    <div style="color:#999; font-size:0.85em;">FII/FPI Net</div>
                    <div style="font-size:1.5em; font-weight:700; color:{fii_c};">{fii_net:+,.0f} Cr</div>
                    <div style="font-size:0.8em; color:{fii_c};">{fii_label}</div>
                    <div style="font-size:0.7em; color:#666; margin-top:4px;">
                        Buy: {fii_buy:,.0f} | Sell: {fii_sell:,.0f}</div>
                </div>
                <div style="flex:1; background:#1a1a2e; border:2px solid {dii_c};
                    border-radius:10px; padding:14px; text-align:center;">
                    <div style="color:#999; font-size:0.85em;">DII Net</div>
                    <div style="font-size:1.5em; font-weight:700; color:{dii_c};">{dii_net:+,.0f} Cr</div>
                    <div style="font-size:0.8em; color:{dii_c};">{dii_label}</div>
                    <div style="font-size:0.7em; color:#666; margin-top:4px;">
                        Buy: {dii_buy:,.0f} | Sell: {dii_sell:,.0f}</div>
                </div>
            </div>""",
            unsafe_allow_html=True,
        )
        if fii_date:
            st.caption(f"Data as of: {fii_date}")
    else:
        st.caption("FII/DII data unavailable.")

with sm_col2:
    # Recent bulk deals in watchlist stocks (last 7 days)
    st.markdown("**Recent Bulk Deals (Watchlist Stocks)**")
    watchlist_tickers = set()
    for w in watchlist:
        t = w.get("ticker", "").replace(".NS", "").replace(".BO", "").upper()
        if t:
            watchlist_tickers.add(t)

    recent_deals = []
    try:
        from datetime import timedelta
        from_7d = (dt.datetime.now() - timedelta(days=7)).strftime("%d-%m-%Y")
        to_7d = dt.datetime.now().strftime("%d-%m-%Y")
        all_bulk = nse_fetcher.fetch_bulk_deals(from_7d, to_7d)
        recent_deals = [d for d in all_bulk if d.get("symbol", "").upper() in watchlist_tickers]
    except Exception:
        pass

    if recent_deals:
        import pandas as pd
        deal_rows = []
        for d in recent_deals[:10]:
            deal_rows.append({
                "Date": d.get("date", ""),
                "Symbol": d.get("symbol", ""),
                "Client": d.get("client_name", "")[:30],
                "Action": d.get("deal_type", ""),
                "Qty": f"{d.get('quantity', 0):,.0f}",
            })
        st.dataframe(pd.DataFrame(deal_rows), use_container_width=True, hide_index=True)
    else:
        st.caption("No recent bulk deals in watchlist stocks.")


# ── Quick Navigation ──────────────────────────────────────────
st.divider()
st.markdown("""
**Drill deeper via sidebar pages:**
Market Regime | Sector Rotation | Stock Scanner | Stage Analysis | Positions | Stock Deep Dive | Watchlist
""")
