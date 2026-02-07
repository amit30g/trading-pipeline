"""Page 6: Single Stock Deep Dive — Detailed Analysis"""
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from dashboard_helpers import (
    build_candlestick_chart,
    build_rs_line_chart,
    format_large_number,
    regime_color,
)
from data_fetcher import fetch_price_data, get_all_stock_tickers
from stage_filter import analyze_stock_stage, detect_bases
from fundamental_veto import fetch_fundamentals, apply_fundamental_veto

st.set_page_config(page_title="Stock Deep Dive", page_icon="📊", layout="wide")

if "nifty_df" not in st.session_state:
    st.info("Run a scan first from the home page (needed for Nifty benchmark data).")
    st.stop()

nifty_df = st.session_state.nifty_df
all_stock_data = st.session_state.get("stock_data", {})

# ── Ticker Selection ────────────────────────────────────────────
col_input, col_select = st.columns([1, 1])
with col_input:
    ticker_input = st.text_input("Enter ticker (e.g. RELIANCE.NS)", value="")
with col_select:
    all_tickers = sorted(get_all_stock_tickers())
    ticker_select = st.selectbox("Or select from universe", [""] + all_tickers)

ticker = ticker_input.strip().upper() if ticker_input.strip() else ticker_select
if not ticker:
    st.caption("Select or enter a ticker to begin.")
    st.stop()

# ── Fetch Price Data ────────────────────────────────────────────
with st.spinner(f"Loading data for {ticker}..."):
    if ticker in all_stock_data and not all_stock_data[ticker].empty:
        df = all_stock_data[ticker]
    else:
        fetched = fetch_price_data([ticker])
        df = fetched.get(ticker)

    if df is None or df.empty:
        st.error(f"No price data found for {ticker}. Check if the ticker is valid.")
        st.stop()

# ── Fetch yfinance detailed info ────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_yf_info(t):
    """Fetch full yfinance info + quarterly financials."""
    yticker = yf.Ticker(t)
    info = yticker.info or {}
    try:
        qtr_income = yticker.quarterly_income_stmt
    except Exception:
        qtr_income = pd.DataFrame()
    return info, qtr_income

with st.spinner("Fetching detailed info..."):
    info, qtr_income = fetch_yf_info(ticker)

company_name = info.get("longName") or info.get("shortName") or ticker
current_price = df["Close"].iloc[-1]
prev_close = info.get("regularMarketPreviousClose") or info.get("previousClose")
change = current_price - prev_close if prev_close else 0
change_pct = (change / prev_close * 100) if prev_close else 0

# ── Header ──────────────────────────────────────────────────────
chg_color = "#26a69a" if change >= 0 else "#ef5350"
st.markdown(
    f"""
    <div style="margin-bottom: 10px;">
        <span style="font-size: 1.8em; font-weight: 700;">{company_name}</span>
        <span style="font-size: 1.1em; color: #999; margin-left: 10px;">{ticker}</span>
    </div>
    <div>
        <span style="font-size: 2.2em; font-weight: 700;">{current_price:,.2f}</span>
        <span style="font-size: 1.2em; color: {chg_color}; margin-left: 10px;">
            {change:+.2f} ({change_pct:+.2f}%)
        </span>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Performance Returns ─────────────────────────────────────────
def compute_return(df, days):
    if len(df) < days:
        return None
    return (df["Close"].iloc[-1] / df["Close"].iloc[-days] - 1) * 100

periods = {"1D": 1, "5D": 5, "1M": 21, "3M": 63, "6M": 126, "1Y": 252}
returns = {label: compute_return(df, d) for label, d in periods.items()}

cols = st.columns(len(returns))
for col, (label, ret) in zip(cols, returns.items()):
    if ret is not None:
        color = "#26a69a" if ret >= 0 else "#ef5350"
        col.markdown(
            f"""<div style="text-align:center; background:#1e1e1e; border-radius:8px;
                           padding:8px 4px; border: 1px solid #333;">
                <div style="font-size:0.8em; color:#999;">{label}</div>
                <div style="font-size:1.1em; font-weight:600; color:{color};">{ret:+.2f}%</div>
            </div>""",
            unsafe_allow_html=True,
        )
    else:
        col.markdown(
            f"""<div style="text-align:center; background:#1e1e1e; border-radius:8px;
                           padding:8px 4px; border: 1px solid #333;">
                <div style="font-size:0.8em; color:#999;">{label}</div>
                <div style="font-size:1.1em; color:#666;">N/A</div>
            </div>""",
            unsafe_allow_html=True,
        )

st.markdown("<br>", unsafe_allow_html=True)

# ── Key Stats + Fundamentals side by side ───────────────────────
left_col, right_col = st.columns([3, 2])

with left_col:
    # Price chart
    analysis = analyze_stock_stage(df, ticker)
    stage = analysis.get("stage", {})
    breakout = analysis.get("breakout")
    entry_setup = analysis.get("entry_setup")
    vcp = analysis.get("vcp")
    bases = detect_bases(df)

    fig = build_candlestick_chart(
        df, ticker, mas=[50, 150, 200],
        bases=bases, breakout=breakout, entry_setup=entry_setup, height=500,
    )
    st.plotly_chart(fig, use_container_width=True)

with right_col:
    st.markdown("##### Key Fundamentals")

    def _fmt(val, fmt=",.2f", suffix="", pct=False, cr=False):
        if val is None:
            return "N/A"
        if cr:
            return f"{val / 1e7:,.2f} Cr"
        if pct:
            return f"{val * 100:.2f}%"
        return f"{val:{fmt}}{suffix}"

    fundamentals_data = {
        "Market Cap": _fmt(info.get("marketCap"), cr=True),
        "EPS (TTM)": _fmt(info.get("trailingEps")),
        "P/E Ratio": _fmt(info.get("trailingPE")),
        "Forward P/E": _fmt(info.get("forwardPE")),
        "P/B Ratio": _fmt(info.get("priceToBook")),
        "Book Value": _fmt(info.get("bookValue")),
        "EBITDA": _fmt(info.get("ebitda"), cr=True),
        "Dividend Yield": _fmt(info.get("dividendYield"), pct=True),
        "ROE": _fmt(info.get("returnOnEquity"), pct=True),
        "ROA": _fmt(info.get("returnOnAssets"), pct=True),
        "Debt/Equity": _fmt(info.get("debtToEquity")),
        "Current Ratio": _fmt(info.get("currentRatio")),
        "Profit Margin": _fmt(info.get("profitMargins"), pct=True),
        "Revenue Growth": _fmt(info.get("revenueGrowth"), pct=True),
        "Earnings Growth": _fmt(info.get("earningsGrowth"), pct=True),
    }

    fund_html = ""
    for label, val in fundamentals_data.items():
        fund_html += f"""
        <div style="display:flex; justify-content:space-between; padding:5px 0;
                    border-bottom:1px solid #333;">
            <span style="color:#999;">{label}</span>
            <span style="font-weight:600;">{val}</span>
        </div>"""
    st.markdown(fund_html, unsafe_allow_html=True)

    # Industry / Sector
    st.markdown(
        f"""<div style="margin-top:10px; font-size:0.9em;">
            <b>Industry:</b> {info.get('industry', 'N/A')} &nbsp;|&nbsp;
            <b>Sector:</b> {info.get('sector', 'N/A')}
        </div>""",
        unsafe_allow_html=True,
    )

# ── Price Stats Row ─────────────────────────────────────────────
st.divider()
ps1, ps2, ps3, ps4 = st.columns(4)
day_high = info.get("dayHigh") or info.get("regularMarketDayHigh")
day_low = info.get("dayLow") or info.get("regularMarketDayLow")
w52_high = info.get("fiftyTwoWeekHigh")
w52_low = info.get("fiftyTwoWeekLow")
vol = info.get("volume") or info.get("regularMarketVolume")
avg_vol = info.get("averageVolume")

ps1.metric("Day Range", f"{day_low:,.1f} - {day_high:,.1f}" if day_low and day_high else "N/A")
ps2.metric("52W Range", f"{w52_low:,.1f} - {w52_high:,.1f}" if w52_low and w52_high else "N/A")
ps3.metric("Volume", format_large_number(vol))
ps4.metric("Avg Volume", format_large_number(avg_vol))

# ── Analyst Ratings ─────────────────────────────────────────────
st.divider()
analyst_count = info.get("numberOfAnalystOpinions")
rec_key = info.get("recommendationKey", "").replace("_", " ").title()
target_mean = info.get("targetMeanPrice")
target_high = info.get("targetHighPrice")
target_low = info.get("targetLowPrice")

if analyst_count and analyst_count > 0:
    st.subheader("Analyst Ratings & Targets")
    ar1, ar2 = st.columns([1, 2])

    with ar1:
        # Recommendation badge
        rec_colors = {
            "Strong Buy": "#4CAF50", "Buy": "#8BC34A",
            "Hold": "#FF9800", "Sell": "#F44336", "Strong Sell": "#B71C1C",
        }
        rc = rec_colors.get(rec_key, "#2196F3")
        st.markdown(
            f"""<div style="text-align:center; padding:20px;">
                <div style="display:inline-block; border:4px solid {rc}; border-radius:50%;
                            width:100px; height:100px; line-height:100px; text-align:center;">
                    <span style="color:{rc}; font-weight:700; font-size:1.1em;">{rec_key.upper()}</span>
                </div>
                <div style="color:#999; margin-top:8px;">from {analyst_count} analysts</div>
            </div>""",
            unsafe_allow_html=True,
        )

    with ar2:
        # Price targets
        if target_mean:
            upside = (target_mean / current_price - 1) * 100
            upside_color = "#26a69a" if upside >= 0 else "#ef5350"
            st.markdown(
                f"""<div style="padding:10px 0;">
                    <div style="display:flex; justify-content:space-between; margin-bottom:12px;">
                        <div><span style="color:#999;">Target Low</span><br>
                             <span style="font-size:1.2em;">{target_low:,.1f}</span></div>
                        <div style="text-align:center;">
                             <span style="color:#999;">Mean Target</span><br>
                             <span style="font-size:1.4em; font-weight:700;">{target_mean:,.1f}</span><br>
                             <span style="color:{upside_color}; font-size:0.9em;">
                                ({upside:+.1f}% upside)
                             </span></div>
                        <div style="text-align:right;"><span style="color:#999;">Target High</span><br>
                             <span style="font-size:1.2em;">{target_high:,.1f}</span></div>
                    </div>
                </div>""",
                unsafe_allow_html=True,
            )

            # Target range bar
            fig_target = go.Figure()
            fig_target.add_shape(
                type="line", x0=target_low, x1=target_high, y0=0, y1=0,
                line=dict(color="#555", width=6),
            )
            fig_target.add_trace(go.Scatter(
                x=[current_price], y=[0], mode="markers",
                marker=dict(size=16, color="#2196F3", symbol="diamond"),
                name="Current",
            ))
            fig_target.add_trace(go.Scatter(
                x=[target_mean], y=[0], mode="markers",
                marker=dict(size=14, color="#FF9800", symbol="circle"),
                name="Mean Target",
            ))
            fig_target.update_layout(
                height=80, margin=dict(l=0, r=0, t=0, b=0),
                template="plotly_dark", showlegend=True,
                xaxis=dict(showgrid=False),
                yaxis=dict(showgrid=False, showticklabels=False, range=[-0.5, 0.5]),
                legend=dict(orientation="h", yanchor="top", y=1.5),
            )
            st.plotly_chart(fig_target, use_container_width=True)

# ── Quarterly Financials ────────────────────────────────────────
st.divider()
if not qtr_income.empty:
    st.subheader("Quarterly Financials")

    # Extract key rows, convert to Cr
    def get_row(name):
        if name in qtr_income.index:
            return qtr_income.loc[name].sort_index() / 1e7  # Convert to Cr
        return None

    revenue = get_row("Total Revenue")
    ebitda = get_row("EBITDA")
    net_income = get_row("Net Income")
    eps = None
    if "Diluted EPS" in qtr_income.index:
        eps = qtr_income.loc["Diluted EPS"].sort_index()  # EPS already in per-share
    op_margin = None
    if "Operating Income" in qtr_income.index and "Total Revenue" in qtr_income.index:
        op_income = qtr_income.loc["Operating Income"].sort_index()
        total_rev = qtr_income.loc["Total Revenue"].sort_index()
        op_margin = (op_income / total_rev * 100).dropna()

    # Table view
    qtr_rows = {}
    if revenue is not None:
        qtr_rows["Revenue (Cr)"] = {d.strftime("%b %Y"): f"{v:,.0f}" for d, v in revenue.items() if pd.notna(v)}
    if ebitda is not None:
        qtr_rows["EBITDA (Cr)"] = {d.strftime("%b %Y"): f"{v:,.0f}" for d, v in ebitda.items() if pd.notna(v)}
    if net_income is not None:
        qtr_rows["Net Profit (Cr)"] = {d.strftime("%b %Y"): f"{v:,.0f}" for d, v in net_income.items() if pd.notna(v)}
    if op_margin is not None:
        qtr_rows["OPM %"] = {d.strftime("%b %Y"): f"{v:.1f}%" for d, v in op_margin.items() if pd.notna(v)}
    if eps is not None:
        qtr_rows["EPS"] = {d.strftime("%b %Y"): f"{v:.2f}" for d, v in eps.items() if pd.notna(v)}

    if qtr_rows:
        qtr_df = pd.DataFrame(qtr_rows).T
        st.dataframe(qtr_df, use_container_width=True)

    # Revenue + Net Profit trend chart
    if revenue is not None and net_income is not None:
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        fig.add_trace(
            go.Bar(
                x=[d.strftime("%b %Y") for d in revenue.index],
                y=revenue.values,
                name="Revenue (Cr)",
                marker_color="#2196F3",
                opacity=0.7,
            ),
            secondary_y=False,
        )
        fig.add_trace(
            go.Scatter(
                x=[d.strftime("%b %Y") for d in net_income.index],
                y=net_income.values,
                name="Net Profit (Cr)",
                line=dict(color="#26a69a", width=3),
                mode="lines+markers",
            ),
            secondary_y=True,
        )

        fig.update_layout(
            title="Revenue & Net Profit Trend",
            height=400, template="plotly_dark",
            margin=dict(l=50, r=50, t=60, b=30),
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        fig.update_yaxes(title_text="Revenue (Cr)", secondary_y=False)
        fig.update_yaxes(title_text="Net Profit (Cr)", secondary_y=True)
        st.plotly_chart(fig, use_container_width=True)

    # EPS trend chart
    if eps is not None:
        fig_eps = go.Figure()
        eps_clean = eps.dropna().sort_index()
        fig_eps.add_trace(go.Bar(
            x=[d.strftime("%b %Y") for d in eps_clean.index],
            y=eps_clean.values,
            name="EPS",
            marker_color=["#26a69a" if v >= 0 else "#ef5350" for v in eps_clean.values],
        ))
        fig_eps.update_layout(
            title="EPS Trend (Quarterly)",
            height=350, template="plotly_dark",
            yaxis_title="EPS (INR)",
            margin=dict(l=50, r=20, t=60, b=30),
        )
        st.plotly_chart(fig_eps, use_container_width=True)

# ── RS Line Chart ──────────────────────────────────────────────
st.divider()
st.subheader("Relative Strength vs Nifty")
fig = build_rs_line_chart(df["Close"], nifty_df["Close"], ticker)
st.plotly_chart(fig, use_container_width=True)

# ── Stage 2 Checklist ──────────────────────────────────────────
st.divider()
st.subheader("Stage Analysis")
sa1, sa2, sa3, sa4 = st.columns(4)
sa1.metric("Stage", stage.get("stage", "?"))
sa2.metric("S2 Score", f"{stage.get('s2_score', 0)}/7")
sa3.metric("Confidence", f"{stage.get('confidence', 0):.0%}")
sa4.metric("VCP", "Yes" if vcp and vcp.get("is_vcp") else "No")

if breakout and breakout.get("breakout"):
    bo1, bo2, bo3 = st.columns(3)
    bo1.metric("Breakout Price", f"{breakout['breakout_price']:.1f}")
    bo2.metric("Volume Ratio", f"{breakout.get('volume_ratio', 0):.1f}x")
    bo3.metric("Base Depth", f"{breakout.get('base_depth_pct', 0):.1f}%")

if entry_setup:
    en1, en2, en3 = st.columns(3)
    en1.metric("Entry", f"{entry_setup.get('entry_price', 0):.1f}")
    en2.metric("Stop", f"{entry_setup.get('effective_stop', 0):.1f}")
    en3.metric("Risk", f"{entry_setup.get('risk_pct', 0):.1f}%")

st.markdown("**Stage 2 Checklist**")
s2_checks = stage.get("s2_checks", {})
check_cols = st.columns(min(len(s2_checks), 4)) if s2_checks else []
for i, (check_name, passed) in enumerate(s2_checks.items()):
    icon = "✅" if passed else "❌"
    check_cols[i % len(check_cols)].markdown(f"{icon} {check_name.replace('_', ' ').title()}")

# ── Fundamental Veto ───────────────────────────────────────────
st.divider()
st.subheader("Fundamental Veto Check")
with st.spinner("Running fundamental veto..."):
    fundamentals = fetch_fundamentals(ticker)
    veto_result = apply_fundamental_veto(fundamentals)

if veto_result["passes"]:
    st.success(f"PASS (confidence: {veto_result['confidence']})")
else:
    st.error(f"VETOED (confidence: {veto_result['confidence']})")
    for reason in veto_result.get("reasons", []):
        st.markdown(f"- {reason}")

# ── Company Description ────────────────────────────────────────
desc = info.get("longBusinessSummary")
if desc:
    st.divider()
    with st.expander("About the Company"):
        st.write(desc)
