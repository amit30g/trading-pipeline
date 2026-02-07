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

# ── Filters ─────────────────────────────────────────────────────
sectors_in_results = sorted(set(s["sector"] for s in screened))
selected_sectors = st.multiselect("Filter by sector", sectors_in_results, default=sectors_in_results)

filtered = [s for s in screened if s["sector"] in selected_sectors]
st.metric("Stocks Shown", len(filtered))

# ── Results Table ───────────────────────────────────────────────
st.subheader("Screener Results")
rows = []
for s in filtered:
    rows.append({
        "Ticker": s["ticker"],
        "Sector": s["sector"],
        "RS vs Nifty": round(s.get("rs_vs_nifty", 0), 2),
        "RS vs Sector": round(s.get("rs_vs_sector", 0), 2),
        "Dist from High %": round(s.get("dist_from_high_pct", 0), 1),
        "Avg Volume": int(s.get("avg_volume", 0)),
        "Accum Ratio": round(s.get("accumulation_ratio", 0), 2),
        "Leadership": round(s.get("leadership_score", 0), 2),
        "Close": round(s.get("close", 0), 2),
    })

df = pd.DataFrame(rows)
st.dataframe(df, use_container_width=True, hide_index=True)

# ── Scatter Plot ────────────────────────────────────────────────
st.subheader("RS vs Distance from High")
if len(rows) > 0:
    fig = px.scatter(
        df, x="Dist from High %", y="RS vs Nifty",
        size="Avg Volume", color="Accum Ratio",
        hover_name="Ticker",
        color_continuous_scale=["#ef5350", "#FFF9C4", "#26a69a"],
        size_max=30,
        template="plotly_dark",
    )
    fig.update_layout(
        height=500,
        xaxis_title="Distance from 52-week High (%)",
        yaxis_title="Relative Strength vs Nifty (%)",
        margin=dict(l=50, r=20, t=30, b=30),
    )
    # Invert x-axis so closer-to-high is on the right
    fig.update_xaxes(autorange="reversed")
    st.plotly_chart(fig, use_container_width=True)
