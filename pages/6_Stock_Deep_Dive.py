"""Page 6: Single Stock Deep Dive"""
import streamlit as st
import pandas as pd

from dashboard_helpers import (
    build_candlestick_chart,
    build_rs_line_chart,
    format_large_number,
)
from data_fetcher import fetch_price_data, get_all_stock_tickers
from stage_filter import analyze_stock_stage, detect_bases
from fundamental_veto import fetch_fundamentals, apply_fundamental_veto

st.set_page_config(page_title="Stock Deep Dive", page_icon="📊", layout="wide")
st.title("Stock Deep Dive")

if "nifty_df" not in st.session_state:
    st.info("Run a scan first from the home page (needed for Nifty benchmark data).")
    st.stop()

nifty_df = st.session_state.nifty_df
all_stock_data = st.session_state.get("stock_data", {})

# ── Ticker Selection ────────────────────────────────────────────
all_tickers = sorted(get_all_stock_tickers())
# Also allow free-text entry
ticker_input = st.text_input("Enter ticker (e.g. RELIANCE.NS)", value="")
ticker_select = st.selectbox("Or select from universe", [""] + all_tickers)

ticker = ticker_input.strip().upper() if ticker_input.strip() else ticker_select
if not ticker:
    st.caption("Select or enter a ticker to begin.")
    st.stop()

# ── Fetch Data ──────────────────────────────────────────────────
with st.spinner(f"Loading data for {ticker}..."):
    if ticker in all_stock_data and not all_stock_data[ticker].empty:
        df = all_stock_data[ticker]
    else:
        fetched = fetch_price_data([ticker])
        df = fetched.get(ticker)

    if df is None or df.empty:
        st.error(f"No price data found for {ticker}. Check if the ticker is valid.")
        st.stop()

# ── Stage Analysis ──────────────────────────────────────────────
analysis = analyze_stock_stage(df, ticker)
stage = analysis.get("stage", {})
breakout = analysis.get("breakout")
entry_setup = analysis.get("entry_setup")
vcp = analysis.get("vcp")

# ── Key Metrics Row ─────────────────────────────────────────────
st.subheader(ticker)

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Close", f"{df['Close'].iloc[-1]:.2f}")
c2.metric("Stage", stage.get("stage", "?"))
c3.metric("S2 Score", f"{stage.get('s2_score', 0)}/7")
c4.metric("Confidence", f"{stage.get('confidence', 0):.0%}")
c5.metric("VCP", "Yes" if vcp and vcp.get("is_vcp") else "No")

if breakout and breakout.get("breakout"):
    bc1, bc2, bc3 = st.columns(3)
    bc1.metric("Breakout Price", f"{breakout['breakout_price']:.1f}")
    bc2.metric("Volume Ratio", f"{breakout.get('volume_ratio', 0):.1f}x")
    bc3.metric("Base Depth", f"{breakout.get('base_depth_pct', 0):.1f}%")

if entry_setup:
    ec1, ec2, ec3 = st.columns(3)
    ec1.metric("Entry", f"{entry_setup.get('entry_price', 0):.1f}")
    ec2.metric("Stop", f"{entry_setup.get('effective_stop', 0):.1f}")
    ec3.metric("Risk", f"{entry_setup.get('risk_pct', 0):.1f}%")

# ── Candlestick Chart ──────────────────────────────────────────
st.subheader("Price Chart")
bases = detect_bases(df)
fig = build_candlestick_chart(
    df, ticker,
    mas=[50, 150, 200],
    bases=bases,
    breakout=breakout,
    entry_setup=entry_setup,
    height=600,
)
st.plotly_chart(fig, use_container_width=True)

# ── RS Line Chart ──────────────────────────────────────────────
st.subheader("Relative Strength vs Nifty")
fig = build_rs_line_chart(df["Close"], nifty_df["Close"], ticker)
st.plotly_chart(fig, use_container_width=True)

# ── Stage 2 Checks Detail ──────────────────────────────────────
st.subheader("Stage 2 Checklist")
s2_checks = stage.get("s2_checks", {})
for check_name, passed in s2_checks.items():
    icon = "✅" if passed else "❌"
    st.markdown(f"{icon} {check_name.replace('_', ' ').title()}")

# ── Fundamentals ────────────────────────────────────────────────
st.subheader("Fundamental Data")
with st.spinner("Fetching fundamentals..."):
    fundamentals = fetch_fundamentals(ticker)
    veto_result = apply_fundamental_veto(fundamentals)

if fundamentals.get("data_available"):
    v_col1, v_col2 = st.columns(2)
    with v_col1:
        st.markdown(f"**Company:** {fundamentals.get('company_name', 'N/A')}")
        st.markdown(f"**Sector:** {fundamentals.get('sector', 'N/A')}")
        st.markdown(f"**Industry:** {fundamentals.get('industry', 'N/A')}")
        st.markdown(f"**Market Cap:** {format_large_number(fundamentals.get('market_cap'))}")

    with v_col2:
        metrics = {
            "P/E": fundamentals.get("pe_ratio"),
            "Fwd P/E": fundamentals.get("forward_pe"),
            "PEG": fundamentals.get("peg_ratio"),
            "ROE": f"{fundamentals['roe'] * 100:.1f}%" if fundamentals.get("roe") else None,
            "D/E": fundamentals.get("debt_equity"),
            "Revenue Growth": f"{fundamentals['revenue_growth'] * 100:.1f}%" if fundamentals.get("revenue_growth") else None,
            "Earnings Growth": f"{fundamentals['earnings_growth'] * 100:.1f}%" if fundamentals.get("earnings_growth") else None,
            "Profit Margin": f"{fundamentals['profit_margin'] * 100:.1f}%" if fundamentals.get("profit_margin") else None,
        }
        for label, val in metrics.items():
            if val is not None:
                st.markdown(f"**{label}:** {val}")

    # Veto result
    if veto_result["passes"]:
        st.success(f"Fundamental check: PASS (confidence: {veto_result['confidence']})")
    else:
        st.error(f"Fundamental check: VETOED (confidence: {veto_result['confidence']})")
        for reason in veto_result.get("reasons", []):
            st.markdown(f"- {reason}")
else:
    st.warning(f"No fundamental data available: {fundamentals.get('error', 'Unknown error')}")
