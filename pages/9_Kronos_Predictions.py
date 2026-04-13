"""Page 9: Kronos AI Predictions — Next-Day Forecasts + Accuracy Tracking"""
import streamlit as st
import pandas as pd
import numpy as np
import datetime as dt
import json
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path

from data_fetcher import fetch_price_data, get_all_stock_tickers
from kronos_runner import PRED_DIR, load_all_predictions, run_predictions, backfill_actuals

st.set_page_config(page_title="Kronos Predictions", page_icon="🔮", layout="wide")

st.title("Kronos AI — Next-Day Predictors")
st.caption(
    "Uses the [Kronos](https://github.com/shiyu-coder/Kronos) foundation model "
    "(trained on 45+ global exchanges) to forecast next-day candlestick data. "
    "Predictions are saved daily and compared against actuals to track accuracy."
)

LOOKBACK = 400


@st.cache_resource(show_spinner="Loading Kronos model (~25M params)...")
def _load_kronos():
    from kronos_model import KronosTokenizer, Kronos, KronosPredictor
    tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
    model = Kronos.from_pretrained("NeoQuasar/Kronos-small")
    return KronosPredictor(model, tokenizer, device="cpu", max_context=512)


tab_today, tab_accuracy = st.tabs(["Today's Predictions", "Accuracy Tracker"])

# ══════════════════════════════════════════════════════════════════
# TAB 1: TODAY'S PREDICTIONS
# ══════════════════════════════════════════════════════════════════
with tab_today:
    all_stock_data = st.session_state.get("stock_data", {})

    col1, col2 = st.columns([2, 1])
    with col1:
        mode = st.radio(
            "Stock selection",
            ["From last scan (cached data)", "Pick tickers manually"],
            horizontal=True,
        )
    with col2:
        save_to_disk = st.checkbox("Save predictions to disk (for accuracy tracking)", value=True)

    all_tickers = sorted(get_all_stock_tickers())

    if mode == "Pick tickers manually":
        selected = st.multiselect(
            "Select tickers", all_tickers,
            default=all_tickers[:10] if len(all_tickers) >= 10 else all_tickers,
            max_selections=50,
        )
    else:
        available = {t: df for t, df in all_stock_data.items() if df is not None and not df.empty and len(df) >= 60}
        if not available:
            st.warning("No cached stock data. Run a scan from the home page first, or pick tickers manually.")
            selected = []
        else:
            # Sort by average volume (most liquid first) so the slider picks meaningful stocks
            by_liquidity = sorted(available.keys(), key=lambda t: available[t]["Volume"].tail(20).mean(), reverse=True)
            max_stocks = st.slider("Max stocks to predict (sorted by liquidity)", 10, min(200, len(by_liquidity)), min(50, len(by_liquidity)))
            selected = by_liquidity[:max_stocks]

    if selected and st.button("Run Kronos Predictions", type="primary", use_container_width=True):
        predictor = _load_kronos()

        # Fetch missing data
        missing = [t for t in selected if t not in all_stock_data or all_stock_data.get(t) is None or all_stock_data[t].empty]
        if missing:
            with st.spinner(f"Fetching price data for {len(missing)} tickers..."):
                fetched = fetch_price_data(missing)
                all_stock_data.update(fetched)

        results = []
        progress = st.progress(0, text="Running predictions...")

        for idx, ticker in enumerate(selected):
            progress.progress((idx + 1) / len(selected), text=f"Predicting {ticker} ({idx+1}/{len(selected)})")
            df = all_stock_data.get(ticker)
            if df is None or df.empty or len(df) < 60:
                continue
            try:
                ohlcv = df[["Open", "High", "Low", "Close", "Volume"]].copy()
                ohlcv.columns = ["open", "high", "low", "close", "volume"]
                ohlcv = ohlcv.dropna()

                ctx_len = min(LOOKBACK, len(ohlcv))
                ohlcv = ohlcv.iloc[-ctx_len:].reset_index(drop=True)
                dates = df.index[-ctx_len:]
                x_timestamp = pd.Series(dates).reset_index(drop=True)

                last_date = dates[-1]
                future_dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=1)
                y_timestamp = pd.Series(future_dates)

                pred_df = predictor.predict(
                    df=ohlcv, x_timestamp=x_timestamp, y_timestamp=y_timestamp,
                    pred_len=1, T=0.8, top_p=0.9, sample_count=1, verbose=False,
                )

                last_close = float(ohlcv["close"].iloc[-1])
                pred_close = float(pred_df["close"].iloc[0])
                pred_high = float(pred_df["high"].iloc[0])
                pred_low = float(pred_df["low"].iloc[0])
                pred_open = float(pred_df["open"].iloc[0])
                pred_return = (pred_close - last_close) / last_close * 100

                results.append({
                    "Ticker": ticker,
                    "Last Close": round(last_close, 2),
                    "Pred Open": round(pred_open, 2),
                    "Pred High": round(pred_high, 2),
                    "Pred Low": round(pred_low, 2),
                    "Pred Close": round(pred_close, 2),
                    "Pred Return %": round(pred_return, 2),
                    "Target Date": str(future_dates[0].date()),
                    "_pred_df": pred_df,
                    "_ohlcv": ohlcv,
                    "_dates": dates,
                })
            except Exception as e:
                continue

        progress.empty()

        if not results:
            st.error("No predictions generated.")
        else:
            st.session_state["kronos_results"] = results

            # Save to disk for accuracy tracking
            if save_to_disk:
                today_str = dt.date.today().strftime("%Y-%m-%d")
                pred_file = PRED_DIR / f"pred_{today_str}.json"
                preds_for_disk = []
                for r in results:
                    preds_for_disk.append({
                        "ticker": r["Ticker"],
                        "last_close": r["Last Close"],
                        "pred_open": r["Pred Open"],
                        "pred_high": r["Pred High"],
                        "pred_low": r["Pred Low"],
                        "pred_close": r["Pred Close"],
                        "pred_return_pct": r["Pred Return %"],
                        "target_date": r["Target Date"],
                        "actual_close": None,
                        "actual_return_pct": None,
                    })
                record = {
                    "run_date": today_str,
                    "target_date": results[0]["Target Date"],
                    "stocks_predicted": len(results),
                    "stocks_failed": len(selected) - len(results),
                    "predictions": preds_for_disk,
                }
                PRED_DIR.mkdir(parents=True, exist_ok=True)
                with open(pred_file, "w") as f:
                    json.dump(record, f, indent=2)
                st.success(f"Saved {len(results)} predictions to {pred_file.name}")

    # Display results
    if "kronos_results" in st.session_state:
        results = st.session_state["kronos_results"]
        display_df = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")} for r in results])
        display_df = display_df.sort_values("Pred Return %", ascending=False).reset_index(drop=True)

        bullish = display_df[display_df["Pred Return %"] > 0]
        bearish = display_df[display_df["Pred Return %"] < 0]

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Stocks Predicted", len(display_df))
        c2.metric("Bullish", len(bullish), f"{len(bullish)/len(display_df)*100:.0f}%")
        c3.metric("Bearish", len(bearish), f"{len(bearish)/len(display_df)*100:.0f}%")
        c4.metric("Avg Pred Return", f"{display_df['Pred Return %'].mean():.2f}%")

        st.markdown("---")
        col_bull, col_bear = st.columns(2)

        with col_bull:
            st.subheader("Top Bullish Predictions")
            st.dataframe(
                display_df.head(15).style.applymap(
                    lambda v: "color: green" if isinstance(v, (int, float)) and v > 0 else "color: red" if isinstance(v, (int, float)) and v < 0 else "",
                    subset=["Pred Return %"],
                ),
                use_container_width=True, hide_index=True,
            )

        with col_bear:
            st.subheader("Top Bearish Predictions")
            st.dataframe(
                display_df.tail(15).iloc[::-1].style.applymap(
                    lambda v: "color: green" if isinstance(v, (int, float)) and v > 0 else "color: red" if isinstance(v, (int, float)) and v < 0 else "",
                    subset=["Pred Return %"],
                ),
                use_container_width=True, hide_index=True,
            )

        # Return distribution
        st.markdown("---")
        st.subheader("Predicted Return Distribution")
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(x=display_df["Pred Return %"], nbinsx=40, name="Predicted Returns"))
        fig_hist.update_layout(
            xaxis_title="Predicted Return %", yaxis_title="Count",
            height=300, margin=dict(t=20, b=40), template="plotly_dark",
        )
        st.plotly_chart(fig_hist, use_container_width=True)

        # Individual stock chart
        st.markdown("---")
        st.subheader("Prediction Detail")
        result_map = {r["Ticker"]: r for r in results}
        selected_ticker = st.selectbox(
            "Select ticker",
            [r["Ticker"] for r in sorted(results, key=lambda x: x["Pred Return %"], reverse=True)],
        )
        if selected_ticker and selected_ticker in result_map:
            r = result_map[selected_ticker]
            pred_df = r["_pred_df"]
            ohlcv = r["_ohlcv"]
            dates = r["_dates"]

            show_n = min(60, len(ohlcv))
            hist = ohlcv.iloc[-show_n:].copy()
            hist.index = dates[-show_n:]

            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.05)
            fig.add_trace(go.Candlestick(
                x=hist.index, open=hist["open"], high=hist["high"], low=hist["low"], close=hist["close"],
                name="Historical", increasing_line_color="rgba(0,200,83,0.9)", decreasing_line_color="rgba(255,82,82,0.9)",
            ), row=1, col=1)
            fig.add_trace(go.Candlestick(
                x=pred_df.index, open=pred_df["open"], high=pred_df["high"], low=pred_df["low"], close=pred_df["close"],
                name="Kronos Prediction", increasing_line_color="rgba(0,150,255,0.9)", decreasing_line_color="rgba(255,165,0,0.9)",
            ), row=1, col=1)
            fig.add_trace(go.Bar(x=hist.index, y=hist["volume"], name="Volume", marker_color="rgba(100,100,100,0.4)"), row=2, col=1)
            fig.update_layout(
                title=f"{selected_ticker} — Kronos Next-Day Prediction",
                height=500, template="plotly_dark", xaxis_rangeslider_visible=False,
                margin=dict(t=40, b=20), legend=dict(orientation="h", y=1.08),
            )
            fig.update_xaxes(type="category", row=1, col=1)
            fig.update_xaxes(type="category", row=2, col=1)
            st.plotly_chart(fig, use_container_width=True)

            mc1, mc2, mc3, mc4 = st.columns(4)
            mc1.metric("Last Close", f"{r['Last Close']:.2f}")
            mc2.metric("Pred Close", f"{r['Pred Close']:.2f}", f"{r['Pred Return %']:+.2f}%", delta_color="normal")
            mc3.metric("Pred High", f"{r['Pred High']:.2f}")
            mc4.metric("Pred Low", f"{r['Pred Low']:.2f}")
    else:
        st.info("Click **Run Kronos Predictions** to generate forecasts.")


# ══════════════════════════════════════════════════════════════════
# TAB 2: ACCURACY TRACKER
# ══════════════════════════════════════════════════════════════════
with tab_accuracy:
    st.subheader("Prediction Accuracy Over Time")

    col_bf, col_info = st.columns([1, 3])
    with col_bf:
        if st.button("Backfill Actuals"):
            with st.spinner("Fetching actual prices for past predictions..."):
                backfill_actuals()
            st.success("Backfill complete!")
            st.rerun()

    with col_info:
        pred_files = sorted(PRED_DIR.glob("pred_*.json"))
        st.caption(f"{len(pred_files)} prediction days on disk")

    all_preds = load_all_predictions()

    if all_preds.empty:
        st.info(
            "No prediction history yet. Run predictions from the first tab (with 'Save to disk' checked), "
            "or run `python kronos_runner.py` from the command line daily."
        )
        st.markdown("""
        **Daily usage:**
        ```bash
        # Run predictions (e.g. via cron before market open)
        python kronos_runner.py --max-stocks 200

        # After market close, backfill actuals
        python kronos_runner.py --backfill
        ```
        """)
    else:
        has_actuals = all_preds[all_preds["actual_close"].notna()]
        pending = all_preds[all_preds["actual_close"].isna()]

        m1, m2, m3 = st.columns(3)
        m1.metric("Total Predictions", len(all_preds))
        m2.metric("With Actuals", len(has_actuals))
        m3.metric("Pending Actuals", len(pending))

        if not has_actuals.empty:
            # Direction accuracy
            has_actuals = has_actuals.copy()
            has_actuals["pred_direction"] = np.sign(has_actuals["pred_return_pct"])
            has_actuals["actual_direction"] = np.sign(has_actuals["actual_return_pct"])
            has_actuals["direction_correct"] = has_actuals["pred_direction"] == has_actuals["actual_direction"]
            has_actuals["abs_error"] = (has_actuals["pred_return_pct"] - has_actuals["actual_return_pct"]).abs()

            direction_acc = has_actuals["direction_correct"].mean() * 100
            mae = has_actuals["abs_error"].mean()
            corr = has_actuals[["pred_return_pct", "actual_return_pct"]].corr().iloc[0, 1]

            # Top/bottom quintile accuracy
            n_quintile = max(1, len(has_actuals) // 5)
            top_preds = has_actuals.nlargest(n_quintile, "pred_return_pct")
            bottom_preds = has_actuals.nsmallest(n_quintile, "pred_return_pct")
            top_actual_avg = top_preds["actual_return_pct"].mean()
            bottom_actual_avg = bottom_preds["actual_return_pct"].mean()

            st.markdown("---")
            st.subheader("Overall Accuracy")
            a1, a2, a3, a4 = st.columns(4)
            a1.metric("Direction Accuracy", f"{direction_acc:.1f}%")
            a2.metric("MAE (Return %)", f"{mae:.2f}%")
            a3.metric("Pred vs Actual Corr", f"{corr:.3f}")
            a4.metric("Top Q Avg Actual", f"{top_actual_avg:+.2f}%",
                       help="Average actual return of stocks Kronos predicted most bullish")

            st.markdown("---")
            st.subheader("Does Ranking Work?")
            st.caption("If Kronos is useful, its top-ranked stocks should outperform bottom-ranked ones.")

            # Quintile analysis
            has_actuals_sorted = has_actuals.sort_values("pred_return_pct")
            n = len(has_actuals_sorted)
            quintile_size = n // 5
            quintile_data = []
            for q in range(5):
                start = q * quintile_size
                end = start + quintile_size if q < 4 else n
                chunk = has_actuals_sorted.iloc[start:end]
                quintile_data.append({
                    "Quintile": f"Q{q+1} ({'Most Bearish' if q == 0 else 'Most Bullish' if q == 4 else ''})",
                    "Avg Pred Return %": round(chunk["pred_return_pct"].mean(), 2),
                    "Avg Actual Return %": round(chunk["actual_return_pct"].mean(), 2),
                    "Direction Accuracy %": round(chunk["direction_correct"].mean() * 100, 1),
                    "Count": len(chunk),
                })
            qdf = pd.DataFrame(quintile_data)
            st.dataframe(qdf, use_container_width=True, hide_index=True)

            # Quintile bar chart
            fig_q = go.Figure()
            fig_q.add_trace(go.Bar(
                x=qdf["Quintile"], y=qdf["Avg Pred Return %"],
                name="Predicted", marker_color="rgba(0,150,255,0.7)",
            ))
            fig_q.add_trace(go.Bar(
                x=qdf["Quintile"], y=qdf["Avg Actual Return %"],
                name="Actual", marker_color="rgba(0,200,83,0.7)",
            ))
            fig_q.update_layout(
                barmode="group", height=350, template="plotly_dark",
                yaxis_title="Avg Return %", margin=dict(t=20, b=40),
            )
            st.plotly_chart(fig_q, use_container_width=True)

            # Scatter plot: predicted vs actual
            st.markdown("---")
            st.subheader("Predicted vs Actual Returns")
            fig_scatter = go.Figure()
            fig_scatter.add_trace(go.Scatter(
                x=has_actuals["pred_return_pct"], y=has_actuals["actual_return_pct"],
                mode="markers", marker=dict(size=4, color="rgba(0,150,255,0.5)"),
                text=has_actuals["ticker"], name="Predictions",
            ))
            # Add diagonal line
            rng = max(abs(has_actuals["pred_return_pct"].max()), abs(has_actuals["actual_return_pct"].max()), 5)
            fig_scatter.add_trace(go.Scatter(
                x=[-rng, rng], y=[-rng, rng], mode="lines",
                line=dict(dash="dash", color="gray"), name="Perfect prediction",
            ))
            fig_scatter.update_layout(
                xaxis_title="Predicted Return %", yaxis_title="Actual Return %",
                height=400, template="plotly_dark", margin=dict(t=20, b=40),
            )
            st.plotly_chart(fig_scatter, use_container_width=True)

            # Accuracy over time (by run_date)
            if has_actuals["run_date"].nunique() > 1:
                st.markdown("---")
                st.subheader("Accuracy Over Time")
                daily = has_actuals.groupby("run_date").agg(
                    direction_acc=("direction_correct", "mean"),
                    mae=("abs_error", "mean"),
                    count=("ticker", "count"),
                ).reset_index()
                daily["direction_acc"] *= 100

                fig_time = make_subplots(specs=[[{"secondary_y": True}]])
                fig_time.add_trace(go.Scatter(
                    x=daily["run_date"], y=daily["direction_acc"],
                    mode="lines+markers", name="Direction Accuracy %",
                    line=dict(color="rgba(0,200,83,0.9)"),
                ), secondary_y=False)
                fig_time.add_trace(go.Bar(
                    x=daily["run_date"], y=daily["count"],
                    name="# Predictions", marker_color="rgba(100,100,100,0.3)",
                ), secondary_y=True)
                fig_time.add_hline(y=50, line_dash="dash", line_color="gray",
                                   annotation_text="50% (random)", secondary_y=False)
                fig_time.update_layout(
                    height=350, template="plotly_dark", margin=dict(t=20, b=40),
                )
                fig_time.update_yaxes(title_text="Direction Accuracy %", secondary_y=False)
                fig_time.update_yaxes(title_text="# Predictions", secondary_y=True)
                st.plotly_chart(fig_time, use_container_width=True)

        else:
            st.info("No actuals backfilled yet. Click **Backfill Actuals** after the target dates have passed.")

        # Raw data explorer
        with st.expander("Raw prediction data"):
            st.dataframe(all_preds.sort_values(["run_date", "pred_return_pct"], ascending=[False, False]),
                         use_container_width=True, hide_index=True)

st.markdown("---")
st.caption(
    "**Disclaimer**: Kronos predictions are experimental AI forecasts, not financial advice. "
    "The model was trained on global exchange data and may not perfectly capture NSE-specific dynamics. "
    "Use as one signal among many."
)
