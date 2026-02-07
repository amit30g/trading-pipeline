"""Page 4: Stage 2 Analysis"""
import streamlit as st
import pandas as pd

from dashboard_helpers import build_candlestick_chart
from stage_filter import analyze_stock_stage, detect_bases

st.set_page_config(page_title="Stage Analysis", page_icon="📊", layout="wide")
st.title("Stage Analysis")

if "stage2_candidates" not in st.session_state:
    st.info("Run a scan first from the home page.")
    st.stop()

candidates = st.session_state.stage2_candidates
stock_data = st.session_state.stock_data

if not candidates:
    st.warning("No Stage 2 candidates found in this scan.")
    st.stop()

# ── Summary Metrics ─────────────────────────────────────────────
breakout_count = sum(1 for c in candidates if c.get("breakout") and c["breakout"].get("breakout"))
vcp_count = sum(1 for c in candidates if c.get("vcp") and c["vcp"].get("is_vcp"))

c1, c2, c3 = st.columns(3)
c1.metric("Stage 2 Candidates", len(candidates))
c2.metric("Active Breakouts", breakout_count)
c3.metric("VCP Patterns", vcp_count)

# ── Candidates Table ────────────────────────────────────────────
st.subheader("Stage 2 Candidates")
rows = []
for c in candidates:
    stage = c.get("stage", {})
    breakout = c.get("breakout", {}) or {}
    entry_setup = c.get("entry_setup", {}) or {}
    rows.append({
        "Ticker": c["ticker"],
        "Sector": c.get("sector", ""),
        "Stage": stage.get("stage", "?"),
        "Confidence": f"{stage.get('confidence', 0):.0%}",
        "S2 Score": f"{stage.get('s2_score', 0)}/7",
        "Bases": c.get("bases_found", 0),
        "Base #": c.get("base_count_in_stage2", 0),
        "Breakout": "Yes" if breakout.get("breakout") else "No",
        "BO Price": round(breakout.get("breakout_price", 0), 1) if breakout.get("breakout") else "",
        "Entry": round(entry_setup.get("entry_price", 0), 1) if entry_setup.get("entry_price") else "",
        "Stop": round(entry_setup.get("effective_stop", 0), 1) if entry_setup.get("effective_stop") else "",
        "Risk %": f"{entry_setup.get('risk_pct', 0):.1f}%" if entry_setup.get("risk_pct") else "",
        "VCP": "Yes" if c.get("vcp", {}).get("is_vcp") else "No",
        "RS": round(c.get("rs_vs_nifty", 0), 2),
    })

df = pd.DataFrame(rows)
st.dataframe(df, use_container_width=True, hide_index=True)

# ── Candlestick Charts ─────────────────────────────────────────
st.subheader("Charts")

# Let user pick which stocks to chart
tickers = [c["ticker"] for c in candidates]
selected = st.multiselect("Select stocks to chart", tickers, default=tickers[:3])

for ticker in selected:
    cand = next((c for c in candidates if c["ticker"] == ticker), None)
    if not cand:
        continue
    df_stock = stock_data.get(ticker)
    if df_stock is None or df_stock.empty:
        st.caption(f"No price data for {ticker}")
        continue

    # Get bases and breakout from the candidate
    bases = detect_bases(df_stock)
    breakout = cand.get("breakout")
    entry_setup = cand.get("entry_setup")

    fig = build_candlestick_chart(
        df_stock, ticker,
        mas=[50, 150, 200],
        bases=bases,
        breakout=breakout,
        entry_setup=entry_setup,
        height=550,
    )
    st.plotly_chart(fig, use_container_width=True)
