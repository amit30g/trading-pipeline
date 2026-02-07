"""Page 2: Sector Rotation & Relative Strength"""
import streamlit as st
import pandas as pd

from dashboard_helpers import (
    build_sector_rs_chart,
    build_momentum_heatmap,
    compute_all_sector_rs_timeseries,
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
        fig = build_sector_rs_chart(rs_df, top_sectors)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.caption("Insufficient sector data for RS chart.")

# ── Momentum Heatmap ───────────────────────────────────────────
st.subheader("Sector Momentum Heatmap")
fig = build_momentum_heatmap(sector_rankings)
st.plotly_chart(fig, use_container_width=True)
