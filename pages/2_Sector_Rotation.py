"""Page 2: Sector Rotation & Relative Strength"""
import streamlit as st
import pandas as pd

from dashboard_helpers import (
    build_sector_rs_chart,
    build_momentum_heatmap,
    compute_all_sector_rs_timeseries,
    build_lw_line_chart_html,
    compute_derivatives,
    detect_inflection_points,
)

st.set_page_config(page_title="Sector Rotation", page_icon="📊", layout="wide")
st.title("Sector Rotation")

if "sector_rankings" not in st.session_state:
    st.info("Run a scan first from the home page.")
    st.stop()

sector_rankings = st.session_state.sector_rankings
top_sectors = st.session_state.top_sectors
sector_data = st.session_state.sector_data
nifty_df = st.session_state.nifty_df

# ── Rankings Table ──────────────────────────────────────────────
st.subheader("Sector Rankings")

with st.expander("How Sector Ranking Works"):
    st.markdown("""
**Mansfield Relative Strength** measures each sector index's performance vs Nifty 50 over ~52 weeks. A positive RS means the sector outperforms Nifty; negative means it underperforms.

**Ranking factors:**
- **Mansfield RS:** The current RS value. Higher = stronger outperformance vs market.
- **RS Trend:** Whether the RS line is rising (improving) or falling (fading). Sectors with rising RS are gaining leadership.
- **Momentum (1w to 6m):** Rate of change across multiple timeframes. Consistent green across all periods = strongest sectors.
- **Composite Score:** Weighted combination of RS level, trend, and multi-timeframe momentum.

**Why sectors matter:** Institutional money rotates between sectors. Stocks in top-ranked sectors have "wind at their back" — sector tailwinds significantly boost individual stock performance. The pipeline filters stocks only from the top 4 sectors.
""")

rows = []
for i, s in enumerate(sector_rankings):
    mom = s["momentum"]
    rows.append({
        "Rank": i + 1,
        "Sector": s["sector"],
        "Mansfield RS": s["mansfield_rs"],
        "Trend": s["rs_trend"],
        "1m %": mom.get("1m", None),
        "3m %": mom.get("3m", None),
        "6m %": mom.get("6m", None),
        "Score": s["composite_score"],
        "Top": "Yes" if s["sector"] in top_sectors else "",
    })

df_table = pd.DataFrame(rows)
st.dataframe(
    df_table.style.apply(
        lambda row: ["background-color: #26a69a22" if row["Top"] == "Yes" else "" for _ in row],
        axis=1,
    ),
    use_container_width=True,
    hide_index=True,
)

# ── RS Line Chart ───────────────────────────────────────────────
st.subheader("Mansfield RS Over Time")
with st.spinner("Computing sector RS timeseries..."):
    rs_df = compute_all_sector_rs_timeseries(sector_data, nifty_df)
    if not rs_df.empty:
        plot_rs_df = rs_df.iloc[-180:] if len(rs_df) > 180 else rs_df
        lw_series = []
        # Bright colors for top sectors, gray for others
        bright_colors = ["#2196F3", "#FF9800", "#4CAF50", "#E91E63", "#9C27B0", "#00BCD4", "#FFEB3B", "#FF5722"]
        color_idx = 0
        for col in plot_rs_df.columns:
            is_top = col in top_sectors
            if is_top:
                color = bright_colors[color_idx % len(bright_colors)]
                color_idx += 1
                width = 3
            else:
                color = "#555"
                width = 1
            lw_series.append({
                "name": col,
                "times": plot_rs_df.index.strftime("%Y-%m-%d").tolist(),
                "values": plot_rs_df[col].tolist(),
                "color": color,
                "lineWidth": width,
            })
        rs_html = build_lw_line_chart_html(lw_series, title="Sector Mansfield RS (vs Nifty 50)", height=500, zero_line=True)
        st.components.v1.html(rs_html, height=510)
    else:
        st.caption("Insufficient sector data for RS chart.")

# ── Momentum Heatmap ───────────────────────────────────────────
st.subheader("Sector Momentum Heatmap")
fig = build_momentum_heatmap(sector_rankings)
st.plotly_chart(fig, use_container_width=True)

# ── Sector Momentum Derivatives ────────────────────────────────
st.divider()
st.subheader("Sector Momentum Derivatives")
st.caption("Inflection analysis on sector RS timeseries — identifies which top sectors are accelerating vs fading.")

if not rs_df.empty:
    deriv_rows = []
    for sector in top_sectors:
        if sector not in rs_df.columns:
            continue
        rs_series = rs_df[sector].dropna()
        if len(rs_series) < 60:
            continue
        d = compute_derivatives(rs_series)
        inflection = detect_inflection_points(d["roc"], d["accel"])
        latest_roc = d["roc"].iloc[-1] if not d["roc"].empty else 0
        latest_accel = d["accel"].iloc[-1] if not d["accel"].empty else 0
        deriv_rows.append({
            "Sector": sector,
            "Signal": inflection.get("label", "N/A"),
            "ROC": f"{latest_roc:.2f}" if pd.notna(latest_roc) else "N/A",
            "Acceleration": f"{latest_accel:.2f}" if pd.notna(latest_accel) else "N/A",
            "Detail": inflection.get("detail", ""),
        })

    if deriv_rows:
        st.dataframe(pd.DataFrame(deriv_rows), use_container_width=True, hide_index=True)
    else:
        st.caption("Insufficient data for sector derivatives.")
else:
    st.caption("Sector RS data needed for derivative analysis.")
