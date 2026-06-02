"""Page 10: Strength Pullbacks — Stage 2 stocks resting on the 9/21 EMA.

The high reward/risk continuation setup: a confirmed Stage 2 leader that has
rallied and is now pulling back to a rising short-term EMA on quiet volume.
Buy the bounce with a tight stop just below the EMA / recent swing low.
"""
import streamlit as st
import pandas as pd

from dashboard_helpers import resample_ohlcv, build_lw_candlestick_html
from stage_filter import detect_ema_pullback, detect_bases

st.set_page_config(page_title="Strength Pullbacks", page_icon="🎯", layout="wide")
st.title("🎯 Strength Pullbacks")
st.caption(
    "Stage 2 leaders pulling back to a rising 9 or 21 EMA — the high R:R "
    "continuation entry. Buy strength on weakness, stop just below the EMA."
)

if "all_stage2_stocks" not in st.session_state or "all_stock_data" not in st.session_state:
    st.info("Run a scan first from the home page.")
    st.stop()

all_stage2 = st.session_state.get("all_stage2_stocks", [])
all_stock_data = st.session_state.get("all_stock_data", {})
top_sectors = st.session_state.get("top_sectors", [])
pipeline_tickers = {c["ticker"] for c in st.session_state.get("stage2_candidates", [])}

tab1, tab2 = st.tabs(["Today's Pullbacks", "How It Works"])


with tab1:
    # ── Controls ───────────────────────────────────────────────────
    f1, f2, f3, f4 = st.columns([1, 1, 1, 1])
    with f1:
        min_score = st.selectbox(
            "Min Stage 2 score", [4, 5, 6, 7], index=2,
            help="Only consider stocks at least this strong on the 7-point S2 checklist.",
        )
    with f2:
        band_choice = st.radio("EMA band", ["Both", "9 EMA", "21 EMA"], index=0, horizontal=True)
    with f3:
        require_dryup = st.checkbox("Volume drying up", value=False,
                                    help="Pullback volume below the 50-day average (healthy).")
    with f4:
        require_trigger = st.checkbox("Bounce bar today", value=False,
                                      help="Today's candle reversed up off the EMA.")

    f5, f6, f7 = st.columns([2, 1, 1])
    with f5:
        sectors_avail = sorted(set(s.get("sector") or "Unknown" for s in all_stage2))
        sel_sectors = st.multiselect("Filter by sector", sectors_avail, default=sectors_avail)
    with f6:
        top_only = st.checkbox("Top sectors only", value=False)
    with f7:
        sort_by = st.selectbox("Sort by", ["Setup Score", "R:R", "Closest to EMA", "Smallest Risk %"])

    # ── Build candidate list ───────────────────────────────────────
    band_map = {"9 EMA": "9EMA", "21 EMA": "21EMA"}
    candidates = []
    for s in all_stage2:
        if s.get("stage", {}).get("s2_score", 0) < min_score:
            continue
        sector = s.get("sector") or "Unknown"
        if sector not in sel_sectors:
            continue
        if top_only and sector not in top_sectors:
            continue
        df = all_stock_data.get(s["ticker"])
        if df is None or df.empty:
            continue
        pb = detect_ema_pullback(df)
        if not pb or not pb["is_pullback"]:
            continue
        if band_choice != "Both" and pb["ema_band"] != band_map[band_choice]:
            continue
        if require_dryup and pb["vol_ratio"] > 1.0:
            continue
        if require_trigger and not pb["reversal_bar"]:
            continue
        candidates.append({**pb, "ticker": s["ticker"], "sector": sector,
                           "s2_score": s.get("stage", {}).get("s2_score", 0)})

    sort_key = {
        "Setup Score": lambda x: x["setup_score"],
        "R:R": lambda x: x["rr"],
        "Closest to EMA": lambda x: -abs(x["dist9_atr"] if x["ema_band"] == "9EMA" else x["dist21_atr"]),
        "Smallest Risk %": lambda x: -x["risk_pct"],
    }[sort_by]
    candidates.sort(key=sort_key, reverse=True)

    # ── Summary metrics ────────────────────────────────────────────
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Pullback Setups", len(candidates))
    m2.metric("At 9 EMA", sum(1 for c in candidates if c["ema_band"] == "9EMA"))
    m3.metric("At 21 EMA", sum(1 for c in candidates if c["ema_band"] == "21EMA"))
    m4.metric("Bounce Bar Today", sum(1 for c in candidates if c["reversal_bar"]))

    if not candidates:
        st.warning("No Stage 2 pullbacks match the current filters.")
        st.stop()

    # ── Results table ──────────────────────────────────────────────
    rows = []
    for c in candidates:
        rows.append({
            "Ticker": c["ticker"],
            "Sector": c["sector"],
            "S2": f"{c['s2_score']}/7",
            "Band": c["ema_band"],
            "Score": c["setup_score"],
            "Close": c["close"],
            "9EMA": c["ema9"],
            "21EMA": c["ema21"],
            "Dist(ATR)": c["dist9_atr"] if c["ema_band"] == "9EMA" else c["dist21_atr"],
            "Depth%": c["pullback_depth_pct"],
            "Vol": c["vol_ratio"],
            "Bounce": "✓" if c["reversal_bar"] else "",
            "Entry": c["entry"],
            "Stop": c["stop"],
            "Target": c["target"],
            "Risk%": c["risk_pct"],
            "R:R": c["rr"],
            "Pipe": "✓" if c["ticker"] in pipeline_tickers else "",
        })
    df_tbl = pd.DataFrame(rows)

    def _style(row):
        styles = ["" for _ in row]
        for i, col in enumerate(row.index):
            if col == "Score":
                v = row[col]
                if v >= 75:
                    styles[i] = "color: #4CAF50; font-weight: 700"
                elif v >= 60:
                    styles[i] = "color: #8BC34A; font-weight: 600"
                elif v >= 45:
                    styles[i] = "color: #FFD700"
            elif col == "Band":
                styles[i] = "color: #00E5FF; font-weight: 600" if row[col] == "9EMA" else "color: #FFEB3B; font-weight: 600"
            elif col == "Bounce" and row[col] == "✓":
                styles[i] = "color: #4CAF50; font-weight: 700"
            elif col == "Vol":
                styles[i] = "color: #26a69a" if row[col] <= 1.0 else "color: #ef5350"
            elif col == "R:R":
                if row[col] >= 3:
                    styles[i] = "color: #4CAF50; font-weight: 700"
                elif row[col] >= 2:
                    styles[i] = "color: #8BC34A"
            elif col == "Pipe" and row[col] == "✓":
                styles[i] = "color: #2196F3; font-weight: 600"
        return styles

    st.dataframe(
        df_tbl.style.apply(_style, axis=1),
        use_container_width=True, hide_index=True,
        height=min(700, len(rows) * 38 + 40),
    )

    # ── Charts with EMAs + trade levels ───────────────────────────
    st.subheader("Charts")
    tickers = [c["ticker"] for c in candidates]
    selected = st.multiselect("Chart these setups", tickers, default=tickers[:3])
    tf_label = st.radio("Timeframe", ["Daily", "Weekly"], index=0, horizontal=True, key="pb_tf")
    tf = {"Daily": "D", "Weekly": "W"}[tf_label]

    for ticker in selected:
        c = next((x for x in candidates if x["ticker"] == ticker), None)
        df_stock = all_stock_data.get(ticker)
        if c is None or df_stock is None or df_stock.empty:
            continue

        st.markdown(
            f"**{ticker}** · {c['sector']} · S2 {c['s2_score']}/7 · "
            f"resting on **{c['ema_band']}** · Setup score **{c['setup_score']}** · "
            f"R:R **{c['rr']}** · risk **{c['risk_pct']}%**"
            + ("  ·  🟢 bounce bar today" if c["reversal_bar"] else "")
        )

        price_lines = [
            {"price": c["entry"], "color": "#2196F3", "lineStyle": 2, "title": f"Entry {c['entry']}"},
            {"price": c["stop"], "color": "#F44336", "lineStyle": 2, "title": f"Stop {c['stop']}"},
            {"price": c["prior_high"], "color": "#FF9800", "lineStyle": 3, "title": f"Prior High {c['prior_high']}"},
            {"price": c["target"], "color": "#4CAF50", "lineStyle": 2, "title": f"Target {c['target']}"},
        ]
        chart_df = resample_ohlcv(df_stock, tf) if tf != "D" else df_stock
        # Show the last ~9 months of daily action so the EMAs are readable
        if tf == "D" and len(chart_df) > 190:
            chart_df = chart_df.tail(190)
        chart_html = build_lw_candlestick_html(
            chart_df, ticker, mas=[50], emas=[9, 21], height=520, price_lines=price_lines,
        )
        st.components.v1.html(chart_html, height=530)


with tab2:
    st.markdown("""
### What this scans for

A **Strength Pullback** is the lower-risk way to enter an already-strong stock:
instead of chasing a breakout, you wait for a confirmed **Stage 2** leader to
*pull back* to a rising short-term moving average, then buy the bounce with a
tight stop. Less drama, tighter risk, similar upside.

Every name here has already cleared the **Stage 2 checklist** (you set the
minimum score). On top of that, each must pass these pullback gates:

| Test | Why it matters |
|------|----------------|
| **Uptrend intact** | Price above a rising 50 SMA, with 21 EMA > 50 SMA and rising. A "pullback" in a broken trend is just a downtrend. |
| **Resting on an EMA** | Price within ~0.8 ATR of the 9 or 21 EMA. Distance is measured in **ATR units**, not raw %, so volatile and calm stocks are judged fairly. |
| **Healthy depth** | Pulled back a sensible amount (~3–15%) off the recent swing high — enough to reset, not so much it's a failure. |
| **Volume drying up** | Quiet volume on the dip = profit-taking, not distribution. Surging volume on a pullback is a warning. |
| **Bounce bar (trigger)** | Today's candle reversed up off the EMA — an optional but high-value timing signal. |

### The two bands

- **9 EMA** (cyan) — shallow, fast pullback. Strongest momentum names barely
  pause here. Tightest stops, but they can keep running without you.
- **21 EMA** (yellow) — deeper, "value" pullback. More room, slightly looser
  stop, often a better fill on a stock that needed a breather.

### Reading the table

- **Score** — composite (0–100) of trend quality, proximity to the EMA, pullback
  depth, volume dry-up, today's trigger, and R:R. Sort by this to triage.
- **Dist(ATR)** — how far price sits above its EMA in ATR units. Near 0 = right on it.
- **Entry / Stop / Target** — entry at current price, structural stop just below
  the 21 EMA or recent swing low (ATR-buffered), target at the prior swing high.
- **R:R / Risk%** — reward-to-risk and the % you'd lose if stopped. Size positions
  so that Risk% × position never exceeds your per-trade risk budget.
- **Pipe** — also surfaced by the full top-down pipeline (extra confluence).

### How to use it daily

1. Sort by **Setup Score**, scan the top of the list.
2. Favour **bounce bar today ✓** for timing, and **R:R ≥ 2**.
3. Prefer names in **top sectors** and already in the **pipeline**.
4. Enter near the EMA, place the stop where the table says — if it loses the
   EMA / swing low, the setup is wrong and you're out cheap.

*Not financial advice — a scanner to shortlist, not a signal to blindly trade.*
""")
