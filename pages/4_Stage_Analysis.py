"""Page 4: Stage 2 Analysis — Full Universe View"""
import streamlit as st
import pandas as pd

from dashboard_helpers import resample_ohlcv, build_lw_candlestick_html
from stage_filter import detect_bases

st.set_page_config(page_title="Stage Analysis", page_icon="📊", layout="wide")
st.title("Stage Analysis")

# Use broad universe scan if available, fall back to pipeline candidates
all_stage2 = st.session_state.get("all_stage2_stocks", [])
pipeline_candidates = st.session_state.get("stage2_candidates", [])
stock_data = st.session_state.get("stock_data", {})
top_sectors = st.session_state.get("top_sectors", [])

if not all_stage2 and not pipeline_candidates:
    st.info("Run a scan first from the home page.")
    st.stop()

# Build set of pipeline candidate tickers for highlighting
pipeline_tickers = {c["ticker"] for c in pipeline_candidates}

# ── Summary Metrics ─────────────────────────────────────────────
perfect_7 = [s for s in all_stage2 if s.get("stage", {}).get("s2_score", 0) == 7]
score_6 = [s for s in all_stage2 if s.get("stage", {}).get("s2_score", 0) == 6]
score_5 = [s for s in all_stage2 if s.get("stage", {}).get("s2_score", 0) == 5]
score_4 = [s for s in all_stage2 if s.get("stage", {}).get("s2_score", 0) == 4]
breakout_count = sum(1 for s in all_stage2 if s.get("breakout") and s["breakout"].get("breakout"))
vcp_count = sum(1 for s in all_stage2 if s.get("vcp") and s["vcp"].get("is_vcp"))

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("7/7 Perfect", len(perfect_7))
c2.metric("6/7 Strong", len(score_6))
c3.metric("5/7 Solid", len(score_5))
c4.metric("4/7 Emerging", len(score_4))
c5.metric("Active Breakouts", breakout_count)

# ── Explanation ──────────────────────────────────────────────────
with st.expander("Understanding Stage 2 Analysis", expanded=True):
    st.markdown("""
**This page scans the ENTIRE stock universe** (not just top sectors) for Stage 2 setups. The pipeline's stock scanner only looks at top sectors, so it misses Stage 2 stocks in other sectors. This page catches them all.

**S2 Score (out of 7)** checks these criteria:
1. Price > 150 MA
2. Price > 200 MA
3. 50 MA > 150 MA (MA alignment)
4. 150 MA > 200 MA (MA alignment)
5. 200 MA rising (uptrend confirmed)
6. Price 30%+ above 52-week low (not in a hole)
7. Price within 25% of 52-week high (near highs)

**How to use this table:**
- **7/7** = textbook Stage 2 — strongest candidates for immediate entry on breakout
- **6/7** = one criteria slightly off — still very strong, watch for the missing piece to click
- **5/7** = solid but developing — often the 200 MA hasn't turned up yet or price is still building a base
- **4/7** = early transition from Stage 1 to Stage 2 — earliest opportunities, higher risk

**Pipeline column** shows which stocks also passed the sector + RS + accumulation filters. These have the full pipeline's backing.

**Breakout detection:** Looks for price closing above the most recent base's high on volume >= 1.5x average in the last 5 days.
""")

# ── Filters ─────────────────────────────────────────────────────
col_f1, col_f2, col_f3 = st.columns(3)

with col_f1:
    sectors_available = sorted(set(s.get("sector", "Unknown") for s in all_stage2))
    selected_sectors = st.multiselect(
        "Filter by sector", sectors_available, default=sectors_available,
    )

with col_f2:
    min_score = st.selectbox("Min S2 Score", [4, 5, 6, 7], index=0)

with col_f3:
    show_breakouts_only = st.checkbox("Breakouts only", value=False)

# Apply filters
filtered = [
    s for s in all_stage2
    if s.get("sector", "Unknown") in selected_sectors
    and s.get("stage", {}).get("s2_score", 0) >= min_score
    and (not show_breakouts_only or (s.get("breakout") and s["breakout"].get("breakout")))
]

st.metric("Stocks Shown", len(filtered))

# ── Results Table ───────────────────────────────────────────────
st.subheader("Stage 2 Candidates — Full Universe")

rows = []
for s in filtered:
    stage = s.get("stage", {})
    breakout = s.get("breakout", {}) or {}
    entry_setup = s.get("entry_setup", {}) or {}
    vcp = s.get("vcp", {}) or {}
    in_pipeline = s["ticker"] in pipeline_tickers
    in_top_sector = s.get("sector", "") in top_sectors

    rows.append({
        "Ticker": s["ticker"],
        "Sector": s.get("sector", ""),
        "S2 Score": f"{stage.get('s2_score', 0)}/7",
        "Stage": stage.get("stage", "?"),
        "Breakout": "YES" if breakout.get("breakout") else "",
        "VCP": "YES" if vcp.get("is_vcp") else "",
        "Bases": s.get("bases_found", 0),
        "Base #": s.get("base_count_in_stage2", 0),
        "Entry": round(entry_setup.get("entry_price", 0), 1) if entry_setup.get("entry_price") else "",
        "Stop": round(entry_setup.get("effective_stop", 0), 1) if entry_setup.get("effective_stop") else "",
        "Risk %": f"{entry_setup.get('risk_pct', 0):.1f}%" if entry_setup.get("risk_pct") else "",
        "Close": s.get("close", 0),
        "Pipeline": "YES" if in_pipeline else "",
        "Top Sector": "YES" if in_top_sector else "",
    })

df = pd.DataFrame(rows)


def _style_stage_row(row):
    """Color-code by S2 score and pipeline status."""
    styles = ["" for _ in row]
    for col_idx, col_name in enumerate(row.index):
        if col_name == "S2 Score":
            score_str = str(row[col_name])
            if score_str.startswith("7"):
                styles[col_idx] = "color: #4CAF50; font-weight: 700"
            elif score_str.startswith("6"):
                styles[col_idx] = "color: #8BC34A; font-weight: 600"
            elif score_str.startswith("5"):
                styles[col_idx] = "color: #FFD700"
            elif score_str.startswith("4"):
                styles[col_idx] = "color: #FF9800"
        elif col_name == "Breakout" and row[col_name] == "YES":
            styles[col_idx] = "color: #4CAF50; font-weight: 700"
        elif col_name == "VCP" and row[col_name] == "YES":
            styles[col_idx] = "color: #FFD700; font-weight: 600"
        elif col_name == "Pipeline" and row[col_name] == "YES":
            styles[col_idx] = "color: #2196F3; font-weight: 700"
        elif col_name == "Top Sector" and row[col_name] == "YES":
            styles[col_idx] = "color: #26a69a"
    return styles


if not df.empty:
    st.dataframe(
        df.style.apply(_style_stage_row, axis=1),
        use_container_width=True,
        hide_index=True,
        height=min(700, len(rows) * 38 + 40),
    )
else:
    st.warning("No stocks match current filters.")

# ── Sector Distribution ─────────────────────────────────────────
st.subheader("Stage 2 by Sector")
if all_stage2:
    sector_counts = {}
    for s in all_stage2:
        sec = s.get("sector", "Unknown")
        score = s.get("stage", {}).get("s2_score", 0)
        if sec not in sector_counts:
            sector_counts[sec] = {"7/7": 0, "6/7": 0, "5/7": 0, "4/7": 0, "total": 0}
        sector_counts[sec]["total"] += 1
        if score == 7:
            sector_counts[sec]["7/7"] += 1
        elif score == 6:
            sector_counts[sec]["6/7"] += 1
        elif score == 5:
            sector_counts[sec]["5/7"] += 1
        elif score == 4:
            sector_counts[sec]["4/7"] += 1

    dist_rows = []
    for sec, counts in sorted(sector_counts.items(), key=lambda x: x[1]["total"], reverse=True):
        is_top = sec in top_sectors
        dist_rows.append({
            "Sector": sec,
            "Top?": "YES" if is_top else "",
            "Total": counts["total"],
            "7/7": counts["7/7"] or "",
            "6/7": counts["6/7"] or "",
            "5/7": counts["5/7"] or "",
            "4/7": counts["4/7"] or "",
        })
    st.dataframe(pd.DataFrame(dist_rows), use_container_width=True, hide_index=True)

# ── Candlestick Charts ─────────────────────────────────────────
st.subheader("Charts")

tickers = [s["ticker"] for s in filtered[:50]]  # limit chart options
selected = st.multiselect("Select stocks to chart", tickers, default=tickers[:3])

_stage_tf_label = st.radio("Timeframe", ["Daily", "Weekly", "Monthly"], index=1, horizontal=True, key="stage_chart_tf")
_tf_map = {"Weekly": "W", "Daily": "D", "Monthly": "ME"}
_tf = _tf_map[_stage_tf_label]

for ticker in selected:
    cand = next((s for s in filtered if s["ticker"] == ticker), None)
    if not cand:
        continue
    df_stock = stock_data.get(ticker)
    if df_stock is None or df_stock.empty:
        st.caption(f"No price data for {ticker}")
        continue

    bases = detect_bases(df_stock)
    breakout = cand.get("breakout")
    entry_setup = cand.get("entry_setup")

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
