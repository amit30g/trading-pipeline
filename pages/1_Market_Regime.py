"""Page 1: Market Regime Analysis"""
import streamlit as st
import pandas as pd

from dashboard_helpers import (
    build_candlestick_chart,
    build_regime_gauge,
    build_breadth_chart,
    compute_breadth_timeseries,
    compute_net_new_highs_timeseries,
    compute_all_derivatives,
    build_derivative_chart,
    detect_inflection_points,
    regime_color,
    signal_color,
    resample_ohlcv,
    build_lw_candlestick_html,
    build_lw_area_chart_html,
)
from config import REGIME_CONFIG
import plotly.graph_objects as go

st.set_page_config(page_title="Market Regime", page_icon="📊", layout="wide")
st.title("Market Regime Analysis")

if "regime" not in st.session_state:
    st.info("Run a scan first from the home page.")
    st.stop()

regime = st.session_state.regime
nifty_df = st.session_state.nifty_df
all_stock_data = st.session_state.all_stock_data

# ── Regime Banner ───────────────────────────────────────────────
label = regime["label"]
color = regime_color(label)
raw = regime["raw_score"]
score = regime["regime_score"]

st.markdown(
    f"""
    <div style="background: {color}22; border-left: 5px solid {color};
                padding: 20px 25px; border-radius: 0 8px 8px 0; margin-bottom: 15px;">
        <div style="font-size: 2em; font-weight: 700; color: {color};">{label.upper()}</div>
        <div style="font-size: 1.1em; color: #ccc; margin-top: 5px;">
            Regime Score: {score:+d} (raw: {raw:+d}/5)
            &nbsp;|&nbsp; Breadth Trend: {regime.get('breadth_trend', 'N/A')}
        </div>
        <div style="font-size: 0.95em; color: #aaa; margin-top: 8px;">
            {regime.get('summary', '')}
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.expander("Understanding Market Regime"):
    st.markdown("""
**The regime score combines 5 signals** to determine overall market posture:

1. **Index vs 200 DMA** — Is Nifty 50 above or below its 200-day moving average? Above = bullish (+1), below = bearish (-1), within 2% = neutral (0).
2. **Breadth % > 50 DMA** — What percentage of stocks trade above their 50-day MA? Above 60% = bullish, below 40% = bearish. Measures short-term participation.
3. **Breadth % > 200 DMA** — Same for the 200-day MA. Above 65% = bullish, below 45% = bearish. Measures long-term health.
4. **Net New Highs** — Are more stocks making new 52-week highs than lows? Net > 20 = bullish, Net < -10 = bearish.
5. **Breadth Trend** — Is the 50 DMA breadth rising or falling over the last 20 days?

**Score ranges:** +2 = Aggressive (100% capital), +1 = Normal (80%), 0 = Cautious (50%), -1 = Defensive (20%), -2 = Cash (10%).

Capital allocation and risk-per-trade automatically adjust based on the regime. In defensive/cash regimes, new positions are limited or paused entirely.
""")

# ── Signal Cards ────────────────────────────────────────────────
st.subheader("Signal Breakdown")
signals = regime["signals"]
cols = st.columns(len(signals))

for col, (sig_name, sig_data) in zip(cols, signals.items()):
    sc = sig_data["score"]
    sc_color = signal_color(sc)
    icon = "+" if sc > 0 else ("-" if sc < 0 else "~")
    with col:
        st.markdown(
            f"""
            <div style="background: {sc_color}15; border: 1px solid {sc_color}44;
                        border-radius: 8px; padding: 12px; text-align: center;">
                <div style="font-size: 1.5em; color: {sc_color}; font-weight: 700;">{icon}</div>
                <div style="font-size: 0.85em; color: #ccc; margin-top: 4px;">
                    {sig_name.replace('_', ' ').title()}
                </div>
                <div style="font-size: 0.8em; color: #999; margin-top: 4px;">
                    {sig_data['detail']}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

# ── Nifty Candlestick Chart ────────────────────────────────────
st.subheader("Nifty 50 Index")
_nifty_tf_label = st.radio("Timeframe", ["Daily", "Weekly", "Monthly"], index=1, horizontal=True, key="regime_nifty_tf")
_tf_map = {"Weekly": "W", "Daily": "D", "Monthly": "ME"}
_tf = _tf_map[_nifty_tf_label]
nifty_chart_df = resample_ohlcv(nifty_df, _tf)
nifty_chart_html = build_lw_candlestick_html(nifty_chart_df, "Nifty 50", mas=[50, 200], height=500)
st.components.v1.html(nifty_chart_html, height=510)

# ── Breadth Gauges ──────────────────────────────────────────────
st.subheader("Market Breadth")
g1, g2 = st.columns(2)

breadth_50_val = signals.get("breadth_50dma", {}).get("value", 50)
breadth_200_val = signals.get("breadth_200dma", {}).get("value", 50)

with g1:
    fig = build_regime_gauge(
        breadth_50_val, "% Above 50 DMA",
        (REGIME_CONFIG["breadth_50dma_bearish"], REGIME_CONFIG["breadth_50dma_bullish"]),
    )
    st.plotly_chart(fig, use_container_width=True)

with g2:
    fig = build_regime_gauge(
        breadth_200_val, "% Above 200 DMA",
        (REGIME_CONFIG["breadth_200dma_bearish"], REGIME_CONFIG["breadth_200dma_bullish"]),
    )
    st.plotly_chart(fig, use_container_width=True)

# ── Breadth Timeseries ─────────────────────────────────────────
st.subheader("Breadth Over Time (Last 90 Days)")
with st.spinner("Computing breadth timeseries..."):
    breadth_50_ts = compute_breadth_timeseries(all_stock_data, ma_period=50, lookback=90)
    breadth_200_ts = compute_breadth_timeseries(all_stock_data, ma_period=200, lookback=90)
    area_series = []
    if not breadth_50_ts.empty:
        area_series.append({
            "name": "% > 50 DMA",
            "times": breadth_50_ts.index.strftime("%Y-%m-%d").tolist(),
            "values": breadth_50_ts.values.tolist(),
            "color": "#2196F3",
            "topColor": "rgba(33,150,243,0.3)",
            "bottomColor": "rgba(33,150,243,0.0)",
        })
    if not breadth_200_ts.empty:
        area_series.append({
            "name": "% > 200 DMA",
            "times": breadth_200_ts.index.strftime("%Y-%m-%d").tolist(),
            "values": breadth_200_ts.values.tolist(),
            "color": "#FF9800",
            "topColor": "rgba(255,152,0,0.3)",
            "bottomColor": "rgba(255,152,0,0.0)",
        })
    if area_series:
        breadth_html = build_lw_area_chart_html(area_series, title="Market Breadth Over Time", height=400)
        st.components.v1.html(breadth_html, height=410)
    else:
        st.caption("Insufficient data for breadth chart.")

# ── Net New Highs / Lows ───────────────────────────────────────
st.subheader("Net New Highs / Lows (Last 90 Days)")
with st.spinner("Computing new highs/lows..."):
    nh_df = compute_net_new_highs_timeseries(all_stock_data, lookback=90)
    if not nh_df.empty:
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=nh_df.index, y=nh_df["New Highs"], name="New Highs",
            marker_color="#26a69a",
        ))
        fig.add_trace(go.Bar(
            x=nh_df.index, y=nh_df["New Lows"], name="New Lows",
            marker_color="#ef5350",
        ))
        fig.add_trace(go.Scatter(
            x=nh_df.index, y=nh_df["Net"], name="Net",
            line=dict(color="#2196F3", width=2),
        ))
        fig.update_layout(
            barmode="relative",
            height=400,
            template="plotly_dark",
            yaxis_title="Count",
            margin=dict(l=50, r=20, t=30, b=30),
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.caption("Insufficient data for new highs/lows chart.")

# ── Druckenmiller Derivatives ────────────────────────────────────
st.divider()
st.subheader("Second Derivatives (Momentum Inflection Points)")

with st.expander("Understanding Derivatives"):
    st.markdown("""
**Druckenmiller-style derivative analysis** detects momentum inflections before price confirms:

- **Rate of Change (1st Derivative):** 4-week smoothed ROC of each series. Positive = improving, negative = deteriorating.
- **Acceleration (2nd Derivative):** Change in the ROC itself. This tells you if momentum is speeding up or slowing down.
- **Bullish Inflection:** ROC is negative but acceleration turns positive — "bad but getting less bad." The earliest signal of a turn.
- **Bullish Thrust:** ROC positive AND acceleration positive — confirmed uptrend, momentum strengthening.
- **Bearish Inflection:** ROC positive but acceleration negative — "good but momentum fading." Early warning of distribution.
- **Bearish Breakdown:** ROC negative AND acceleration negative — confirmed downtrend, avoid new longs.
""")

with st.spinner("Computing derivatives..."):
    deriv_results = compute_all_derivatives(nifty_df, all_stock_data)

if deriv_results:
    # Summary inflection signal cards
    deriv_cols = st.columns(len(deriv_results))
    for col, (name, d) in zip(deriv_cols, deriv_results.items()):
        inflection = d.get("inflection", {})
        sig_label = inflection.get("label", "N/A")
        sig_color = inflection.get("color", "#888")
        sig_icon = inflection.get("icon", "--")
        sig_detail = inflection.get("detail", "")
        with col:
            st.markdown(
                f"""<div style="background:{sig_color}15; border:1px solid {sig_color}44;
                    border-radius:8px; padding:12px; text-align:center;">
                    <div style="font-size:1.4em; color:{sig_color}; font-weight:700;">{sig_icon}</div>
                    <div style="font-size:0.8em; color:#ccc; margin-top:4px;">{name}</div>
                    <div style="font-size:0.85em; color:{sig_color}; font-weight:600; margin-top:4px;">{sig_label}</div>
                    <div style="font-size:0.7em; color:#999; margin-top:2px;">{sig_detail}</div>
                </div>""",
                unsafe_allow_html=True,
            )

    # Expandable derivative charts (Plotly 3-panel — complex subplots)
    for name, d in deriv_results.items():
        if d["roc"].empty:
            continue
        with st.expander(f"{name} — Derivative Chart"):
            fig = build_derivative_chart(d["series"], d["roc"], d["accel"], name, lookback=90)
            st.plotly_chart(fig, use_container_width=True)
