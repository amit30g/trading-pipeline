"""Page 4: Stage 2 Analysis"""
import streamlit as st
import pandas as pd

from dashboard_helpers import build_candlestick_chart, resample_ohlcv, build_lw_candlestick_html
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

with st.expander("Understanding Stage 2 Analysis"):
    st.markdown("""
**Weinstein Stage Analysis** classifies stocks into 4 stages:

- **Stage 1 (Basing):** Price consolidates sideways after a decline. The 200 MA flattens. Accumulation by smart money.
- **Stage 2 (Advancing):** Price breaks above the base on volume. MAs align bullishly (price > 150 > 200 MA, 200 MA rising). This is the **only stage to buy**.
- **Stage 3 (Topping):** Price stalls, MAs flatten and start curling. Distribution by institutions.
- **Stage 4 (Declining):** Price below falling MAs. Avoid entirely.

**S2 Score (out of 7):** Counts how many Stage 2 criteria are met — price above 150 MA, price above 200 MA, 150 MA above 200 MA, 200 MA rising 20+ days, price 30%+ above 52-week low, price within 25% of 52-week high, and RS > 0.

**VCP (Volatility Contraction Pattern):** A Minervini concept — each successive pullback within a base is shallower than the prior one, showing sellers are exhausted. 2+ contractions with each <60% of the prior = VCP.

**Base Count:** 1st or 2nd base breakouts have the highest success rate. By the 4th+ base, the move is usually mature and risky.
""")

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
        "VCP": "Yes" if (c.get("vcp") or {}).get("is_vcp") else "No",
        "RS": round(c.get("rs_vs_nifty", 0), 2),
    })

df = pd.DataFrame(rows)
st.dataframe(df, use_container_width=True, hide_index=True)

# ── Candlestick Charts ─────────────────────────────────────────
st.subheader("Charts")

# Let user pick which stocks to chart
tickers = [c["ticker"] for c in candidates]
selected = st.multiselect("Select stocks to chart", tickers, default=tickers[:3])

_stage_tf_label = st.radio("Timeframe", ["Daily", "Weekly", "Monthly"], index=1, horizontal=True, key="stage_chart_tf")
_tf_map = {"Weekly": "W", "Daily": "D", "Monthly": "ME"}
_tf = _tf_map[_stage_tf_label]

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

    # Build markers: breakout arrows + base start/end markers
    lw_markers = []
    if breakout and breakout.get("breakout"):
        try:
            bo_date = pd.to_datetime(breakout["breakout_date"]).strftime("%Y-%m-%d")
            lw_markers.append({
                "time": bo_date, "position": "belowBar",
                "color": "#4CAF50", "shape": "arrowUp",
                "text": "Breakout",
            })
        except Exception:
            pass

    if bases:
        for base in bases:
            try:
                bs = pd.to_datetime(base["start_date"]).strftime("%Y-%m-%d")
                be = pd.to_datetime(base["end_date"]).strftime("%Y-%m-%d")
                lw_markers.append({
                    "time": bs, "position": "aboveBar",
                    "color": "#FFD700", "shape": "square",
                    "text": "Base Start",
                })
                lw_markers.append({
                    "time": be, "position": "aboveBar",
                    "color": "#FFD700", "shape": "square",
                    "text": "Base End",
                })
            except Exception:
                pass

    # Build price lines: entry, stop, base high/low
    lw_price_lines = []
    if entry_setup:
        ep = entry_setup.get("entry_price")
        es = entry_setup.get("effective_stop")
        if ep:
            lw_price_lines.append({"price": ep, "color": "#2196F3", "lineStyle": 2, "title": f"Entry {ep:.1f}"})
        if es:
            lw_price_lines.append({"price": es, "color": "#F44336", "lineStyle": 2, "title": f"Stop {es:.1f}"})
    if bases:
        last_base = bases[-1]
        bh = last_base.get("base_high")
        bl = last_base.get("base_low")
        if bh:
            lw_price_lines.append({"price": bh, "color": "#FFD700", "lineStyle": 3, "title": f"Base High {bh:.1f}"})
        if bl:
            lw_price_lines.append({"price": bl, "color": "#FFD700", "lineStyle": 3, "title": f"Base Low {bl:.1f}"})

    chart_df = resample_ohlcv(df_stock, _tf)
    chart_html = build_lw_candlestick_html(
        chart_df, ticker, mas=[50, 150, 200],
        height=550, markers=lw_markers or None, price_lines=lw_price_lines or None,
    )
    st.components.v1.html(chart_html, height=560)
