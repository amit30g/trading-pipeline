"""Page 3: Stock Scanner Results"""
import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Stock Scanner", page_icon="📊", layout="wide")
st.title("Stock Scanner")

if "screened_stocks" not in st.session_state:
    st.info("Run a scan first from the home page.")
    st.stop()

screened = st.session_state.screened_stocks
top_sectors = st.session_state.top_sectors

if not screened:
    st.warning("No stocks passed the screening criteria.")
    st.stop()

# Cross-reference with stage2 candidates and watchlist for action signals
stage2_map = {}
for c in st.session_state.get("stage2_candidates", []):
    stage2_map[c["ticker"]] = c
watchlist_map = {}
for w in st.session_state.get("final_watchlist", []):
    watchlist_map[w.get("ticker", "")] = w


def _derive_action_signal(s, stage2_info, watchlist_info):
    """Synthesize all signals into one actionable call.

    The logic follows the pipeline's own decision chain:
    1. Already on watchlist with BUY → "BUY — Setup Ready"
    2. Stage 2 + breakout detected → "BUY — Breakout"
    3. Stage 2 + no breakout yet → "WATCH — Building Base"
    4. RS accelerating (1m > 3m) + accumulation → "WATCH — Accumulating"
    5. RS decelerating (1m < 3m significantly) → "WAIT — Momentum Fading"
    6. Far from high (>15%) → "WAIT — Extended Pullback"
    """
    rs_1m = s.get("rs_1m", 0)
    rs_3m = s.get("rs_3m", 0)
    dist = s.get("dist_from_high_pct", 0)
    accum = s.get("accumulation_ratio", 1.0)

    # Check if this stock has stage2/watchlist data
    if watchlist_info and watchlist_info.get("action") == "BUY":
        es = watchlist_info.get("entry_setup", {})
        if es:
            return "BUY — Setup Ready", "#4CAF50"

    if stage2_info:
        breakout = stage2_info.get("breakout", {})
        if breakout and breakout.get("breakout"):
            return "BUY — Breakout", "#4CAF50"
        stage = stage2_info.get("stage", {})
        if stage.get("stage") == 2:
            vcp = stage2_info.get("vcp", {})
            if vcp and vcp.get("is_vcp"):
                return "WATCH — VCP Forming", "#FFD700"
            return "WATCH — In Base", "#2196F3"

    # No stage2 data — use RS momentum to decide
    if rs_1m > rs_3m and rs_1m > 5 and accum > 1.3:
        return "WATCH — Accumulating", "#2196F3"
    if rs_1m < rs_3m * 0.5 and rs_3m > 10:
        return "WAIT — Momentum Fading", "#FF9800"
    if dist > 15:
        return "WAIT — Far From High", "#FF9800"
    if rs_1m > 0 and accum > 1.1:
        return "WATCH — Improving", "#2196F3"

    return "MONITOR", "#888"


def _rs_trend_label(rs_1m, rs_3m, rs_6m):
    """Describe RS trajectory in plain language."""
    if rs_1m > rs_3m > 0:
        return "Accelerating"
    if rs_1m > 0 and rs_3m > 0 and rs_1m < rs_3m:
        return "Strong, Slowing"
    if rs_1m > 0 and rs_3m <= 0:
        return "Turning Up"
    if rs_1m <= 0 and rs_3m > 0:
        return "Turning Down"
    if rs_1m <= 0 and rs_3m <= 0:
        return "Weak"
    return "Mixed"


# ── Filters ─────────────────────────────────────────────────────
sectors_in_results = sorted(set(s["sector"] for s in screened))
selected_sectors = st.multiselect("Filter by sector", sectors_in_results, default=sectors_in_results)

filtered = [s for s in screened if s["sector"] in selected_sectors]
st.metric("Stocks Shown", len(filtered))

# ── Results Table ───────────────────────────────────────────────
st.subheader("Screener Results")

with st.expander("How to Read This Table"):
    st.markdown("""
**This table answers one question: which stocks should I act on, and when?**

Every stock here has already passed the pipeline's filters (in a top sector, outperforming Nifty, near highs, institutional accumulation). The columns tell you the *timing* story:

| Column | What It Tells You | Action Relevance |
|--------|------------------|------------------|
| **RS 1m / 3m / 6m** | Multi-timeframe relative strength vs Nifty. All rolling trading days from today. | **Key pattern:** 1m > 3m = accelerating (best time to enter). 1m < 3m = decelerating (wait or avoid). |
| **RS Trend** | Plain-language summary of the RS trajectory | "Accelerating" = highest priority. "Strong, Slowing" = already worked, tighten stops. "Turning Up" = early — watch for breakout. |
| **Dist from High** | How close to 52-week high | <5% = near breakout zone. >15% = still building base, needs more time. |
| **Accum** | Up-day volume / down-day volume (50d) | >1.5 = institutions buying aggressively. <1.0 = selling. |
| **Signal** | Synthesized action call from all the above + stage analysis | **BUY** = entry conditions met. **WATCH** = strong but needs trigger. **WAIT** = not ready yet. |

**The decision chain:** Signal = Pipeline verdict. BUY means the stock has a Stage 2 breakout with an entry setup. WATCH means strong but waiting for breakout confirmation. WAIT means either momentum is fading or the stock needs more time in a base.
""")

rows = []
for s in filtered:
    rs_1m = s.get("rs_1m", 0)
    rs_3m = s.get("rs_3m", 0)
    rs_6m = s.get("rs_vs_nifty", 0)
    stage2_info = stage2_map.get(s["ticker"])
    wl_info = watchlist_map.get(s["ticker"])
    signal, signal_color = _derive_action_signal(s, stage2_info, wl_info)
    trend = _rs_trend_label(rs_1m, rs_3m, rs_6m)

    rows.append({
        "Ticker": s["ticker"],
        "Sector": s["sector"],
        "RS 1m": round(rs_1m, 1),
        "RS 3m": round(rs_3m, 1),
        "RS 6m": round(rs_6m, 1),
        "RS Trend": trend,
        "Dist %": round(s.get("dist_from_high_pct", 0), 1),
        "Accum": round(s.get("accumulation_ratio", 0), 2),
        "Signal": signal,
        "Close": round(s.get("close", 0), 2),
    })

df = pd.DataFrame(rows)


def _style_scanner_row(row):
    """Color-code the action signal and RS trend."""
    styles = ["" for _ in row]
    for col_idx, col_name in enumerate(row.index):
        if col_name == "Signal":
            sig = str(row[col_name])
            if sig.startswith("BUY"):
                styles[col_idx] = "color: #4CAF50; font-weight: 700"
            elif sig.startswith("WATCH"):
                styles[col_idx] = "color: #2196F3; font-weight: 600"
            elif sig.startswith("WAIT"):
                styles[col_idx] = "color: #FF9800"
            else:
                styles[col_idx] = "color: #888"
        elif col_name == "RS Trend":
            trend = str(row[col_name])
            if trend == "Accelerating":
                styles[col_idx] = "color: #4CAF50; font-weight: 600"
            elif trend.startswith("Strong"):
                styles[col_idx] = "color: #8BC34A"
            elif trend == "Turning Up":
                styles[col_idx] = "color: #FFD700"
            elif trend in ("Turning Down", "Weak"):
                styles[col_idx] = "color: #FF9800"
        elif col_name in ("RS 1m", "RS 3m", "RS 6m"):
            val = row[col_name]
            if isinstance(val, (int, float)):
                if val > 10:
                    styles[col_idx] = "color: #26a69a"
                elif val < 0:
                    styles[col_idx] = "color: #ef5350"
    return styles


st.dataframe(
    df.style.apply(_style_scanner_row, axis=1),
    use_container_width=True,
    hide_index=True,
    height=min(700, len(rows) * 38 + 40),
)

# ── Action Summary ────────────────────────────────────────────────
buy_count = sum(1 for r in rows if r["Signal"].startswith("BUY"))
watch_count = sum(1 for r in rows if r["Signal"].startswith("WATCH"))
wait_count = sum(1 for r in rows if r["Signal"].startswith("WAIT"))

s1, s2, s3 = st.columns(3)
s1.metric("BUY Signals", buy_count)
s2.metric("WATCH (Waiting for Trigger)", watch_count)
s3.metric("WAIT (Not Ready)", wait_count)

# ── Scatter Plot ────────────────────────────────────────────────
st.subheader("RS Momentum vs Proximity to High")
st.caption("Best candidates: top-right (accelerating RS + near highs). Bubble size = volume, color = accumulation.")
if len(rows) > 0:
    fig = px.scatter(
        df, x="Dist %", y="RS 1m",
        size="Accum", color="RS Trend",
        hover_name="Ticker",
        color_discrete_map={
            "Accelerating": "#4CAF50",
            "Strong, Slowing": "#8BC34A",
            "Turning Up": "#FFD700",
            "Turning Down": "#FF9800",
            "Weak": "#ef5350",
            "Mixed": "#888",
        },
        size_max=25,
        template="plotly_dark",
    )
    fig.update_layout(
        height=500,
        xaxis_title="Distance from 52-week High (%)",
        yaxis_title="1-Month RS vs Nifty (%) — Recent Momentum",
        margin=dict(l=50, r=20, t=30, b=30),
    )
    fig.update_xaxes(autorange="reversed")
    st.plotly_chart(fig, use_container_width=True)
