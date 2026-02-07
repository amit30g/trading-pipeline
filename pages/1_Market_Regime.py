"""Page 1: Market Regime Analysis"""
import streamlit as st

from dashboard_helpers import (
    build_candlestick_chart,
    build_regime_gauge,
    build_breadth_chart,
    compute_breadth_timeseries,
    compute_net_new_highs_timeseries,
    regime_color,
    signal_color,
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
fig = build_candlestick_chart(nifty_df, "Nifty 50", mas=[50, 200])
st.plotly_chart(fig, use_container_width=True)

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
    fig = build_breadth_chart(breadth_50_ts, breadth_200_ts)
    st.plotly_chart(fig, use_container_width=True)

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
