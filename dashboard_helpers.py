"""
Dashboard helper functions: chart builders, timeseries computations, cache wrappers.
"""
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from sector_rs import compute_mansfield_rs


# ── Timeseries Computations ─────────────────────────────────────


def compute_breadth_timeseries(
    all_stock_data: dict[str, pd.DataFrame],
    ma_period: int = 50,
    lookback: int = 90,
) -> pd.Series:
    """Compute daily % of stocks above their N-day MA over the lookback window."""
    above_ma = {}
    for ticker, df in all_stock_data.items():
        if len(df) < ma_period + lookback:
            continue
        ma = df["Close"].rolling(ma_period).mean()
        above = (df["Close"] > ma).astype(int)
        above_ma[ticker] = above

    if not above_ma:
        return pd.Series(dtype=float)

    combined = pd.DataFrame(above_ma)
    breadth = combined.mean(axis=1) * 100  # percentage
    return breadth.iloc[-lookback:]


def compute_net_new_highs_timeseries(
    all_stock_data: dict[str, pd.DataFrame],
    lookback: int = 90,
    high_low_period: int = 52 * 5,
) -> pd.DataFrame:
    """Compute daily new highs, new lows, and net for bar chart."""
    highs_count = {}
    lows_count = {}

    for ticker, df in all_stock_data.items():
        if len(df) < high_low_period:
            continue
        rolling_high = df["Close"].rolling(high_low_period).max()
        rolling_low = df["Close"].rolling(high_low_period).min()
        is_high = (df["Close"] >= rolling_high).astype(int)
        is_low = (df["Close"] <= rolling_low).astype(int)
        highs_count[ticker] = is_high
        lows_count[ticker] = is_low

    if not highs_count:
        return pd.DataFrame()

    new_highs = pd.DataFrame(highs_count).sum(axis=1)
    new_lows = pd.DataFrame(lows_count).sum(axis=1)
    result = pd.DataFrame({
        "New Highs": new_highs,
        "New Lows": -new_lows,
        "Net": new_highs - new_lows,
    })
    return result.iloc[-lookback:]


def compute_all_sector_rs_timeseries(
    sector_data: dict[str, pd.DataFrame],
    nifty_df: pd.DataFrame,
) -> pd.DataFrame:
    """Compute Mansfield RS timeseries for all sectors. Returns DataFrame: date x sector."""
    rs_dict = {}
    nifty_close = nifty_df["Close"]
    for sector_name, sector_df in sector_data.items():
        try:
            rs = compute_mansfield_rs(sector_df["Close"], nifty_close)
            rs_dict[sector_name] = rs
        except Exception:
            continue
    if not rs_dict:
        return pd.DataFrame()
    return pd.DataFrame(rs_dict).dropna(how="all")


# ── Chart Builders ──────────────────────────────────────────────


def build_candlestick_chart(
    df: pd.DataFrame,
    ticker: str,
    mas: list[int] | None = None,
    bases: list[dict] | None = None,
    breakout: dict | None = None,
    entry_setup: dict | None = None,
    height: int = 600,
) -> go.Figure:
    """Build a candlestick chart with optional MA overlays, base shading, breakout markers."""
    if mas is None:
        mas = [50, 150, 200]

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.75, 0.25],
    )

    # Candlestick
    fig.add_trace(
        go.Candlestick(
            x=df.index, open=df["Open"], high=df["High"],
            low=df["Low"], close=df["Close"], name=ticker,
            increasing_line_color="#26a69a", decreasing_line_color="#ef5350",
        ),
        row=1, col=1,
    )

    # Moving averages
    ma_colors = {50: "#2196F3", 150: "#FF9800", 200: "#E91E63"}
    for period in mas:
        if len(df) >= period:
            ma_vals = df["Close"].rolling(period).mean()
            fig.add_trace(
                go.Scatter(
                    x=df.index, y=ma_vals, name=f"{period} MA",
                    line=dict(width=1.5, color=ma_colors.get(period, "#888")),
                ),
                row=1, col=1,
            )

    # Base shading
    if bases:
        for base in bases:
            try:
                start = pd.to_datetime(base["start_date"])
                end = pd.to_datetime(base["end_date"])
                fig.add_shape(
                    type="rect", x0=start, x1=end,
                    y0=base["base_low"], y1=base["base_high"],
                    fillcolor="rgba(255, 193, 7, 0.15)",
                    line=dict(color="rgba(255, 193, 7, 0.5)", width=1),
                    row=1, col=1,
                )
            except Exception:
                pass

    # Breakout marker
    if breakout and breakout.get("breakout"):
        try:
            bo_date = pd.to_datetime(breakout["breakout_date"])
            fig.add_trace(
                go.Scatter(
                    x=[bo_date], y=[breakout["breakout_price"]],
                    mode="markers", name="Breakout",
                    marker=dict(size=14, color="#4CAF50", symbol="triangle-up"),
                ),
                row=1, col=1,
            )
        except Exception:
            pass

    # Entry and stop lines
    if entry_setup:
        entry_price = entry_setup.get("entry_price")
        stop = entry_setup.get("effective_stop")
        if entry_price:
            fig.add_hline(
                y=entry_price, line_dash="dash", line_color="#2196F3",
                annotation_text=f"Entry: {entry_price:.1f}",
                row=1, col=1,
            )
        if stop:
            fig.add_hline(
                y=stop, line_dash="dash", line_color="#F44336",
                annotation_text=f"Stop: {stop:.1f}",
                row=1, col=1,
            )

    # Volume bars
    colors = ["#26a69a" if c >= o else "#ef5350" for c, o in zip(df["Close"], df["Open"])]
    fig.add_trace(
        go.Bar(x=df.index, y=df["Volume"], name="Volume", marker_color=colors, opacity=0.5),
        row=2, col=1,
    )

    # Volume average line
    if len(df) >= 50:
        vol_avg = df["Volume"].rolling(50).mean()
        fig.add_trace(
            go.Scatter(
                x=df.index, y=vol_avg, name="50d Avg Vol",
                line=dict(width=1, color="#FF9800"),
            ),
            row=2, col=1,
        )

    fig.update_layout(
        height=height,
        title=f"{ticker}",
        xaxis_rangeslider_visible=False,
        template="plotly_dark",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=50, r=20, t=60, b=30),
    )
    fig.update_xaxes(type="category", nticks=20, row=1, col=1)
    fig.update_xaxes(type="category", nticks=20, row=2, col=1)

    return fig


def build_regime_gauge(value: float, title: str, thresholds: tuple) -> go.Figure:
    """Build a gauge indicator for breadth values."""
    bearish, bullish = thresholds
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={"text": title},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "#2196F3"},
            "steps": [
                {"range": [0, bearish], "color": "#FFCDD2"},
                {"range": [bearish, bullish], "color": "#FFF9C4"},
                {"range": [bullish, 100], "color": "#C8E6C9"},
            ],
            "threshold": {
                "line": {"color": "white", "width": 2},
                "thickness": 0.75,
                "value": value,
            },
        },
    ))
    fig.update_layout(height=250, margin=dict(l=20, r=20, t=60, b=20), template="plotly_dark")
    return fig


def build_sector_rs_chart(
    rs_df: pd.DataFrame,
    top_sectors: list[str],
    lookback: int = 180,
) -> go.Figure:
    """Multi-line RS chart, top sectors bold."""
    fig = go.Figure()
    plot_df = rs_df.iloc[-lookback:] if len(rs_df) > lookback else rs_df

    for col in plot_df.columns:
        is_top = col in top_sectors
        fig.add_trace(go.Scatter(
            x=plot_df.index, y=plot_df[col], name=col,
            line=dict(width=3 if is_top else 1, dash=None if is_top else "dot"),
            opacity=1.0 if is_top else 0.4,
        ))

    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    fig.update_layout(
        title="Sector Mansfield RS (vs Nifty 50)",
        yaxis_title="RS %",
        height=500,
        template="plotly_dark",
        legend=dict(orientation="h", yanchor="bottom", y=-0.3),
        margin=dict(l=50, r=20, t=60, b=30),
    )
    return fig


def build_momentum_heatmap(sector_rankings: list[dict]) -> go.Figure:
    """Sector momentum heatmap: sectors x 1m/3m/6m."""
    sectors = [s["sector"] for s in sector_rankings]
    periods = ["1m", "3m", "6m"]
    z = []
    for s in sector_rankings:
        row = [s["momentum"].get(p, 0) or 0 for p in periods]
        z.append(row)

    fig = go.Figure(go.Heatmap(
        z=z, x=periods, y=sectors,
        colorscale=[[0, "#ef5350"], [0.5, "#ffffff"], [1, "#26a69a"]],
        zmid=0,
        text=[[f"{v:.1f}%" for v in row] for row in z],
        texttemplate="%{text}",
        textfont={"size": 12},
    ))
    fig.update_layout(
        title="Sector Momentum (RS Rate of Change)",
        height=max(300, len(sectors) * 35 + 100),
        template="plotly_dark",
        margin=dict(l=150, r=20, t=60, b=30),
    )
    return fig


def build_breadth_chart(
    breadth_50_ts: pd.Series,
    breadth_200_ts: pd.Series,
    thresholds_50: tuple = (40, 60),
    thresholds_200: tuple = (45, 65),
) -> go.Figure:
    """Breadth area chart with threshold bands."""
    fig = go.Figure()

    if not breadth_50_ts.empty:
        fig.add_trace(go.Scatter(
            x=breadth_50_ts.index, y=breadth_50_ts.values,
            name="% > 50 DMA", fill="tozeroy",
            line=dict(color="#2196F3"),
            fillcolor="rgba(33, 150, 243, 0.2)",
        ))
        fig.add_hline(y=thresholds_50[0], line_dash="dot", line_color="#ef5350", opacity=0.5,
                      annotation_text=f"50 DMA Bearish ({thresholds_50[0]}%)")
        fig.add_hline(y=thresholds_50[1], line_dash="dot", line_color="#26a69a", opacity=0.5,
                      annotation_text=f"50 DMA Bullish ({thresholds_50[1]}%)")

    if not breadth_200_ts.empty:
        fig.add_trace(go.Scatter(
            x=breadth_200_ts.index, y=breadth_200_ts.values,
            name="% > 200 DMA", fill="tozeroy",
            line=dict(color="#FF9800"),
            fillcolor="rgba(255, 152, 0, 0.2)",
        ))

    fig.update_layout(
        title="Market Breadth Over Time",
        yaxis_title="% of Stocks",
        yaxis_range=[0, 100],
        height=400,
        template="plotly_dark",
        margin=dict(l=50, r=20, t=60, b=30),
    )
    return fig


def build_portfolio_pie(watchlist: list[dict], total_capital: float) -> go.Figure:
    """Capital allocation pie chart for BUY signals."""
    buys = [w for w in watchlist if w.get("action") == "BUY" and w.get("position")]
    if not buys:
        fig = go.Figure()
        fig.add_annotation(text="No BUY signals", xref="paper", yref="paper",
                           x=0.5, y=0.5, showarrow=False, font=dict(size=20))
        fig.update_layout(height=350, template="plotly_dark")
        return fig

    labels = [b["ticker"] for b in buys]
    values = [b["position"].get("position_value", 0) for b in buys]
    allocated = sum(values)
    labels.append("Cash")
    values.append(max(0, total_capital - allocated))

    fig = go.Figure(go.Pie(
        labels=labels, values=values,
        hole=0.4,
        textinfo="label+percent",
        marker=dict(line=dict(color="#1e1e1e", width=2)),
    ))
    fig.update_layout(
        title="Portfolio Allocation",
        height=400,
        template="plotly_dark",
        margin=dict(l=20, r=20, t=60, b=20),
    )
    return fig


def build_rs_line_chart(
    stock_close: pd.Series,
    benchmark_close: pd.Series,
    ticker: str,
    lookback: int = 180,
) -> go.Figure:
    """RS line chart: stock vs benchmark."""
    combined = pd.DataFrame({"stock": stock_close, "bench": benchmark_close}).dropna()
    if combined.empty:
        fig = go.Figure()
        fig.add_annotation(text="No data", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        fig.update_layout(height=300, template="plotly_dark")
        return fig

    rs = combined["stock"] / combined["bench"]
    rs = rs.iloc[-lookback:]
    rs_ma = rs.rolling(min(50, len(rs) - 1)).mean()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=rs.index, y=rs.values, name="RS Line", line=dict(color="#2196F3", width=2)))
    fig.add_trace(go.Scatter(x=rs_ma.index, y=rs_ma.values, name="RS 50 MA", line=dict(color="#FF9800", width=1, dash="dot")))
    fig.update_layout(
        title=f"{ticker} Relative Strength vs Nifty",
        yaxis_title="Relative Strength",
        height=350,
        template="plotly_dark",
        margin=dict(l=50, r=20, t=60, b=30),
    )
    return fig


def build_nifty_sparkline(nifty_df: pd.DataFrame, days: int = 90) -> go.Figure:
    """Small Nifty line chart for home page."""
    recent = nifty_df.iloc[-days:]
    color = "#26a69a" if recent["Close"].iloc[-1] >= recent["Close"].iloc[0] else "#ef5350"
    fig = go.Figure(go.Scatter(
        x=recent.index, y=recent["Close"],
        fill="tozeroy", line=dict(color=color, width=2),
        fillcolor=color.replace(")", ", 0.1)").replace("rgb", "rgba") if "rgb" in color else f"rgba(38,166,154,0.1)" if color == "#26a69a" else "rgba(239,83,80,0.1)",
    ))
    fig.update_layout(
        height=200,
        margin=dict(l=0, r=0, t=10, b=0),
        template="plotly_dark",
        xaxis=dict(showgrid=False, showticklabels=False),
        yaxis=dict(showgrid=False, showticklabels=False),
        showlegend=False,
    )
    return fig


# ── Formatting Helpers ──────────────────────────────────────────


def regime_color(label: str) -> str:
    """Return color for regime label."""
    return {
        "Aggressive": "#4CAF50",
        "Normal": "#8BC34A",
        "Cautious": "#FF9800",
        "Defensive": "#F44336",
        "Cash": "#9E9E9E",
    }.get(label, "#9E9E9E")


def signal_color(score: int) -> str:
    """Return color for individual signal score."""
    if score > 0:
        return "#4CAF50"
    elif score < 0:
        return "#F44336"
    return "#FF9800"


def format_large_number(n: float | int | None) -> str:
    """Format large numbers for display (e.g., 1.5Cr, 250L)."""
    if n is None:
        return "N/A"
    if abs(n) >= 1e7:
        return f"{n / 1e7:.1f} Cr"
    if abs(n) >= 1e5:
        return f"{n / 1e5:.1f} L"
    if abs(n) >= 1e3:
        return f"{n / 1e3:.1f} K"
    return f"{n:.0f}"
