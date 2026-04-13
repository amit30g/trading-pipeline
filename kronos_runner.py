"""
Kronos Daily Runner — Run next-day predictions on the stock universe.

Stores predictions as JSON in scan_cache/kronos_predictions/.
Later, actuals are backfilled so we can track accuracy over time.

Usage:
    python kronos_runner.py              # Run predictions for tomorrow
    python kronos_runner.py --backfill   # Backfill actuals for past predictions
"""
import argparse
import json
import datetime as dt
import logging
from pathlib import Path

import pandas as pd
import numpy as np

from data_fetcher import fetch_price_data, get_all_stock_tickers

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

PRED_DIR = Path(__file__).parent / "scan_cache" / "kronos_predictions"
PRED_DIR.mkdir(parents=True, exist_ok=True)

LOOKBACK = 400  # Kronos context window


def _pred_file(date_str: str) -> Path:
    return PRED_DIR / f"pred_{date_str}.json"


def _load_predictor():
    from kronos_model import KronosTokenizer, Kronos, KronosPredictor
    tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
    model = Kronos.from_pretrained("NeoQuasar/Kronos-small")
    return KronosPredictor(model, tokenizer, device="cpu", max_context=512)


def run_predictions(max_stocks: int = 200):
    """Run Kronos next-day predictions on the stock universe."""
    today = dt.date.today()
    pred_date = today.strftime("%Y-%m-%d")

    # Skip if already ran today
    if _pred_file(pred_date).exists():
        logger.info(f"Predictions for {pred_date} already exist. Skipping.")
        return

    logger.info("Loading Kronos model...")
    predictor = _load_predictor()

    tickers = sorted(get_all_stock_tickers())[:max_stocks]
    logger.info(f"Fetching price data for {len(tickers)} tickers...")
    stock_data = fetch_price_data(tickers)

    predictions = []
    success = 0
    fail = 0

    for ticker, df in stock_data.items():
        if df is None or df.empty or len(df) < 60:
            fail += 1
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
                df=ohlcv,
                x_timestamp=x_timestamp,
                y_timestamp=y_timestamp,
                pred_len=1,
                T=0.8,
                top_p=0.9,
                sample_count=1,
                verbose=False,
            )

            last_close = float(ohlcv["close"].iloc[-1])
            pred_close = float(pred_df["close"].iloc[0])
            pred_high = float(pred_df["high"].iloc[0])
            pred_low = float(pred_df["low"].iloc[0])
            pred_open = float(pred_df["open"].iloc[0])
            pred_return = (pred_close - last_close) / last_close * 100

            predictions.append({
                "ticker": ticker,
                "last_close": round(last_close, 2),
                "pred_open": round(pred_open, 2),
                "pred_high": round(pred_high, 2),
                "pred_low": round(pred_low, 2),
                "pred_close": round(pred_close, 2),
                "pred_return_pct": round(pred_return, 2),
                "target_date": str(future_dates[0].date()),
                "actual_close": None,
                "actual_return_pct": None,
            })
            success += 1
        except Exception as e:
            logger.warning(f"Failed {ticker}: {e}")
            fail += 1

    record = {
        "run_date": pred_date,
        "target_date": predictions[0]["target_date"] if predictions else None,
        "stocks_predicted": success,
        "stocks_failed": fail,
        "predictions": predictions,
    }

    with open(_pred_file(pred_date), "w") as f:
        json.dump(record, f, indent=2)

    logger.info(f"Done: {success} predictions saved, {fail} failed -> {_pred_file(pred_date)}")


def backfill_actuals():
    """Go through past prediction files and fill in actual results where missing."""
    files = sorted(PRED_DIR.glob("pred_*.json"))
    if not files:
        logger.info("No prediction files found.")
        return

    # Collect all tickers and target dates that need actuals
    needs_fill = []
    for f in files:
        with open(f) as fh:
            record = json.load(fh)
        for p in record["predictions"]:
            if p["actual_close"] is None and p["target_date"]:
                needs_fill.append((f, p["ticker"], p["target_date"]))

    if not needs_fill:
        logger.info("All predictions already have actuals.")
        return

    # Group by target_date to batch-fetch
    from collections import defaultdict
    by_date = defaultdict(list)
    for f, ticker, tdate in needs_fill:
        by_date[tdate].append((f, ticker))

    today = dt.date.today()

    for tdate, items in by_date.items():
        target = dt.date.fromisoformat(tdate)
        if target >= today:
            logger.info(f"Target date {tdate} is today or future, skipping.")
            continue

        tickers = list(set(t for _, t in items))
        logger.info(f"Fetching actuals for {len(tickers)} tickers on {tdate}...")

        # Fetch a small window around target date
        fetched = fetch_price_data(tickers, days=10, end_date=target + dt.timedelta(days=5))

        # Build actual close map
        actual_map = {}
        for ticker in tickers:
            df = fetched.get(ticker)
            if df is None or df.empty:
                continue
            # Find the row for target_date
            for idx_date in df.index:
                if idx_date.date() == target:
                    actual_map[ticker] = float(df.loc[idx_date, "Close"])
                    break

        # Update prediction files
        updated_files = set()
        for f, ticker in items:
            if ticker in actual_map:
                updated_files.add(f)

        for f in updated_files:
            with open(f) as fh:
                record = json.load(fh)
            changed = False
            for p in record["predictions"]:
                if p["actual_close"] is None and p["ticker"] in actual_map:
                    p["actual_close"] = round(actual_map[p["ticker"]], 2)
                    p["actual_return_pct"] = round(
                        (actual_map[p["ticker"]] - p["last_close"]) / p["last_close"] * 100, 2
                    )
                    changed = True
            if changed:
                with open(f, "w") as fh:
                    json.dump(record, fh, indent=2)
                logger.info(f"Updated actuals in {f.name}")


def load_all_predictions() -> pd.DataFrame:
    """Load all prediction records into a single DataFrame for analysis."""
    files = sorted(PRED_DIR.glob("pred_*.json"))
    rows = []
    for f in files:
        with open(f) as fh:
            record = json.load(fh)
        run_date = record["run_date"]
        for p in record["predictions"]:
            p["run_date"] = run_date
            rows.append(p)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Kronos daily prediction runner")
    parser.add_argument("--backfill", action="store_true", help="Backfill actuals for past predictions")
    parser.add_argument("--max-stocks", type=int, default=200, help="Max stocks to predict")
    args = parser.parse_args()

    if args.backfill:
        backfill_actuals()
    else:
        run_predictions(max_stocks=args.max_stocks)
