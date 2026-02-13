"""Page 7: Earnings Season Tracker — aggregate quarterly results by market cap segment."""
import streamlit as st
import pandas as pd

from earnings_season import load_earnings_cache, run_earnings_scan
from data_fetcher import load_universe

st.set_page_config(page_title="Earnings Season", page_icon="📊", layout="wide")
st.title("Earnings Season")

# ── Load data ────────────────────────────────────────────────────
earnings_data = st.session_state.get("earnings_season")
if not earnings_data:
    earnings_data = load_earnings_cache()
    if earnings_data:
        st.session_state.earnings_season = earnings_data

# ── Scan controls ────────────────────────────────────────────────
col_info, col_btn = st.columns([3, 1])
with col_info:
    if earnings_data:
        ts = earnings_data.get("scan_timestamp", "?")
        ql = earnings_data.get("quarter_label", "?")
        st.caption(f"Target quarter: **{ql}** (Consolidated) | Last scan: {ts}")
    else:
        st.caption("No earnings data — click Scan to fetch.")

with col_btn:
    run_scan = st.button("Run Earnings Scan", type="primary", use_container_width=True)

if run_scan:
    with st.status("Running earnings scan...", expanded=True) as status:
        progress = st.progress(0)
        universe_df = load_universe()

        def _progress(current, total, symbol):
            if total > 0:
                progress.progress(min(current / total, 0.99))
            clean = symbol.replace(".NS", "") if isinstance(symbol, str) else symbol
            if current % 50 == 0:
                st.write(f"  Processing {current}/{total} — {clean}")

        st.write(f"Fetching quarterly results for {len(universe_df)} stocks...")
        earnings_data = run_earnings_scan(universe_df, progress_callback=_progress)
        st.session_state.earnings_season = earnings_data
        progress.progress(1.0)
        status.update(
            label=f"Scan complete — {earnings_data.get('reported_count', 0)}/{earnings_data.get('total_universe', 0)} reported",
            state="complete",
        )

if not earnings_data:
    st.info("Click **Run Earnings Scan** to fetch quarterly results for the universe.")
    st.stop()


# ══════════════════════════════════════════════════════════════════
# KPI Cards
# ══════════════════════════════════════════════════════════════════
agg = earnings_data.get("aggregate", {})
gd = earnings_data.get("growth_distribution", {})

c1, c2, c3, c4 = st.columns(4)
with c1:
    reported = earnings_data.get("reported_count", 0)
    total = earnings_data.get("total_universe", 0)
    st.metric("Reported", f"{reported}/{total}", delta=f"{earnings_data.get('reported_pct', 0):.0f}%")
with c2:
    rev = agg.get("revenue_yoy_pct")
    st.metric("Revenue YoY", f"{rev:+.1f}%" if rev is not None else "N/A")
with c3:
    pat = agg.get("pat_yoy_pct")
    st.metric("PAT YoY", f"{pat:+.1f}%" if pat is not None else "N/A")
with c4:
    st.metric("15%+ PAT Growth", f"{gd.get('above_15pct', 0)} stocks", delta=f"{gd.get('above_15pct_pct', 0):.0f}% of universe")

st.markdown("---")


# ══════════════════════════════════════════════════════════════════
# Segment Breakdown
# ══════════════════════════════════════════════════════════════════
st.markdown("#### By Market Cap Segment")

by_seg = earnings_data.get("by_segment", {})
seg_labels = {"large": "Large Cap (Nifty 100)", "mid": "Mid Cap (Midcap 150)", "small": "Small Cap"}
seg_cols = st.columns(3)

for col, (seg_key, seg_label) in zip(seg_cols, seg_labels.items()):
    sd = by_seg.get(seg_key, {})
    with col:
        seg_pat = sd.get("pat_yoy")
        seg_rev = sd.get("revenue_yoy")
        pat_color = "#26a69a" if seg_pat and seg_pat >= 0 else "#ef5350" if seg_pat is not None else "#555"

        st.markdown(
            f'<div style="background:#0f0f1a;border:1px solid #1e1e2e;border-radius:6px;padding:16px;text-align:center;">'
            f'<div style="font-size:0.75em;color:#6a6a8a;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:8px;">{seg_label}</div>'
            f'<div style="font-size:1.8em;font-weight:700;color:{pat_color};font-family:monospace;">'
            f'{seg_pat:+.1f}%' if seg_pat is not None else 'N/A'
            f'</div>'
            f'<div style="font-size:0.65em;color:#555;margin-top:2px;">PAT YoY</div>'
            f'<div style="font-size:0.85em;color:#999;font-family:monospace;margin-top:8px;">'
            f'Rev: {seg_rev:+.1f}%' if seg_rev is not None else 'Rev: N/A'
            f'</div>'
            f'<div style="font-size:0.72em;color:#555;margin-top:4px;">'
            f'{sd.get("reported", 0)}/{sd.get("count", 0)} reported</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

st.markdown("---")


# ══════════════════════════════════════════════════════════════════
# Sector-wise Table
# ══════════════════════════════════════════════════════════════════
st.markdown("#### Sector-wise Breakdown")

sector_breakdown = earnings_data.get("sector_breakdown", {})
if sector_breakdown:
    sector_rows = []
    for industry, sd in sector_breakdown.items():
        sector_rows.append({
            "Industry": industry,
            "Stocks": sd.get("count", 0),
            "Reported": sd.get("reported", 0),
            "Revenue YoY %": sd.get("revenue_yoy"),
            "PAT YoY %": sd.get("pat_yoy"),
        })
    sector_df = pd.DataFrame(sector_rows).sort_values("Reported", ascending=False)
    st.dataframe(
        sector_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Revenue YoY %": st.column_config.NumberColumn(format="%.1f%%"),
            "PAT YoY %": st.column_config.NumberColumn(format="%.1f%%"),
        },
    )
else:
    st.caption("No sector data available.")

st.markdown("---")


# ══════════════════════════════════════════════════════════════════
# Stock Detail Table
# ══════════════════════════════════════════════════════════════════
st.markdown("#### Stock Details")

stock_details = earnings_data.get("stock_details", [])
if stock_details:
    # Filters
    fc1, fc2 = st.columns(2)
    with fc1:
        seg_filter = st.multiselect(
            "Segment",
            options=["large", "mid", "small"],
            default=["large", "mid", "small"],
        )
    with fc2:
        industries = sorted(set(d.get("industry", "Unknown") for d in stock_details))
        ind_filter = st.multiselect("Industry", options=industries, default=[])

    filtered = [
        d for d in stock_details
        if d.get("segment") in seg_filter
        and (not ind_filter or d.get("industry") in ind_filter)
    ]

    if filtered:
        detail_df = pd.DataFrame(filtered).rename(columns={
            "symbol": "Symbol",
            "segment": "Segment",
            "industry": "Industry",
            "revenue_cr": "Revenue (Cr)",
            "revenue_yoy_pct": "Rev YoY %",
            "pat_cr": "PAT (Cr)",
            "pat_yoy_pct": "PAT YoY %",
            "eps": "EPS",
            "opm_pct": "OPM %",
            "npm_pct": "NPM %",
        })
        # Cap extreme YoY for readability (low-base turnarounds)
        for col in ["Rev YoY %", "PAT YoY %"]:
            if col in detail_df.columns:
                detail_df[col] = detail_df[col].clip(lower=-100, upper=500)
        st.dataframe(
            detail_df,
            use_container_width=True,
            hide_index=True,
            height=600,
            column_config={
                "Rev YoY %": st.column_config.NumberColumn(format="%.1f%%"),
                "PAT YoY %": st.column_config.NumberColumn(format="%.1f%%"),
                "Revenue (Cr)": st.column_config.NumberColumn(format="%.1f"),
                "PAT (Cr)": st.column_config.NumberColumn(format="%.1f"),
                "EPS": st.column_config.NumberColumn(format="%.2f"),
                "OPM %": st.column_config.NumberColumn(format="%.1f%%"),
                "NPM %": st.column_config.NumberColumn(format="%.1f%%"),
            },
        )
        st.caption(f"Showing {len(filtered)} of {len(stock_details)} reported stocks (YoY capped at ±500% for readability)")
    else:
        st.caption("No stocks match the selected filters.")
else:
    st.caption("No stock-level data available.")


# ══════════════════════════════════════════════════════════════════
# Not-yet-reported
# ══════════════════════════════════════════════════════════════════
reported_symbols = set(d.get("symbol") for d in stock_details)
all_syms = set()
by_seg_data = earnings_data.get("by_segment", {})
for seg_key, sd in by_seg_data.items():
    all_syms.update([])  # We'll compute from universe

# Use universe to find not-yet-reported
try:
    universe_df = load_universe()
    all_universe_syms = set(universe_df["Symbol"].str.strip().tolist())
    not_reported = sorted(all_universe_syms - reported_symbols)
    if not_reported:
        with st.expander(f"Not Yet Reported ({len(not_reported)} stocks)"):
            nr_cols = 6
            for i in range(0, len(not_reported), nr_cols):
                row = not_reported[i:i + nr_cols]
                st.text("  ".join(row))
except Exception:
    pass

st.markdown("---")

# Growth distribution summary
st.markdown("#### Growth Distribution")
gd = earnings_data.get("growth_distribution", {})
gc1, gc2, gc3 = st.columns(3)
with gc1:
    st.metric("PAT Growth > 15%", f"{gd.get('above_15pct', 0)} stocks")
with gc2:
    st.metric("PAT Growth > 25%", f"{gd.get('above_25pct', 0)} stocks")
with gc3:
    st.metric("Negative PAT Growth", f"{gd.get('negative_growth', 0)} stocks")
