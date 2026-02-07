"""
Data Fetcher — yfinance wrapper for NSE data.
Handles downloading price/volume data for indices, sectors, and individual stocks.
"""
import datetime as dt
import yfinance as yf
import pandas as pd
import numpy as np
from config import (
    NSE_SECTOR_INDICES, NIFTY50_TICKER, LOOKBACK_DAYS
)


# ── Nifty 500 Constituents ─────────────────────────────────────
# Mapping: sector_name -> list of NSE tickers (yfinance format: SYMBOL.NS)
# This is a representative subset. In production, scrape the full list from
# NSE website or load from a CSV.
# You can replace this with a full CSV load — see load_nifty500_from_csv() below.

NIFTY500_SECTOR_MAP = {
    "Nifty IT": [
        "TCS.NS", "INFY.NS", "HCLTECH.NS", "WIPRO.NS", "TECHM.NS",
        "LTI.NS", "MPHASIS.NS", "COFORGE.NS", "PERSISTENT.NS", "LTTS.NS",
        "BIRLASOFT.NS", "HAPPSTMNDS.NS", "SONATSOFTW.NS", "ROUTE.NS",
        "MASTEK.NS", "CYIENT.NS", "ZENSAR.NS", "NIITLTD.NS",
    ],
    "Nifty Bank": [
        "HDFCBANK.NS", "ICICIBANK.NS", "KOTAKBANK.NS", "AXISBANK.NS",
        "SBIN.NS", "BANKBARODA.NS", "PNB.NS", "INDUSINDBK.NS",
        "BANDHANBNK.NS", "FEDERALBNK.NS", "IDFCFIRSTB.NS", "AUBANK.NS",
        "RBLBANK.NS", "CANBK.NS", "UNIONBANK.NS",
    ],
    "Nifty Pharma": [
        "SUNPHARMA.NS", "DRREDDY.NS", "CIPLA.NS", "DIVISLAB.NS",
        "AUROPHARMA.NS", "BIOCON.NS", "LUPIN.NS", "TORNTPHARM.NS",
        "ALKEM.NS", "LAURUSLABS.NS", "IPCALAB.NS", "GLENMARK.NS",
        "NATCOPHARM.NS", "GRANULES.NS", "AJANTPHARM.NS",
    ],
    "Nifty Auto": [
        "M&M.NS", "TATAMOTORS.NS", "MARUTI.NS", "BAJAJ-AUTO.NS",
        "HEROMOTOCO.NS", "EICHERMOT.NS", "ASHOKLEY.NS", "TVSMOTOR.NS",
        "BALKRISIND.NS", "MOTHERSON.NS", "BHARATFORG.NS", "MRF.NS",
        "EXIDEIND.NS", "AMARAJABAT.NS", "BOSCHLTD.NS",
    ],
    "Nifty Metal": [
        "TATASTEEL.NS", "JSWSTEEL.NS", "HINDALCO.NS", "VEDL.NS",
        "COALINDIA.NS", "NMDC.NS", "SAIL.NS", "NATIONALUM.NS",
        "JINDALSTEL.NS", "APLAPOLLO.NS", "RATNAMANI.NS",
    ],
    "Nifty Realty": [
        "DLF.NS", "GODREJPROP.NS", "OBEROIRLTY.NS", "PHOENIXLTD.NS",
        "PRESTIGE.NS", "BRIGADE.NS", "SOBHA.NS", "SUNTECK.NS",
    ],
    "Nifty FMCG": [
        "HINDUNILVR.NS", "ITC.NS", "NESTLEIND.NS", "BRITANNIA.NS",
        "DABUR.NS", "MARICO.NS", "GODREJCP.NS", "COLPAL.NS",
        "TATACONSUM.NS", "VBL.NS", "UBL.NS", "EMAMILTD.NS",
    ],
    "Nifty Energy": [
        "RELIANCE.NS", "NTPC.NS", "POWERGRID.NS", "ONGC.NS",
        "BPCL.NS", "IOC.NS", "GAIL.NS", "ADANIGREEN.NS",
        "TATAPOWER.NS", "NHPC.NS", "SJVN.NS", "IREDA.NS",
    ],
    "Nifty Infra": [
        "LARSEN.NS", "ADANIENT.NS", "ADANIPORTS.NS", "ULTRACEMCO.NS",
        "GRASIM.NS", "SHREECEM.NS", "AMBUJACEM.NS", "ACC.NS",
        "SIEMENS.NS", "ABB.NS", "CUMMINSIND.NS", "THERMAX.NS",
    ],
    "Nifty Healthcare": [
        "APOLLOHOSP.NS", "MAXHEALTH.NS", "FORTIS.NS", "METROPOLIS.NS",
        "LALPATHLAB.NS", "STARHEALTH.NS", "MEDANTA.NS",
    ],
    "Nifty Fin Service": [
        "BAJFINANCE.NS", "BAJAJFINSV.NS", "HDFCLIFE.NS", "SBILIFE.NS",
        "ICICIPRULI.NS", "ICICIGI.NS", "MUTHOOTFIN.NS", "MANAPPURAM.NS",
        "CHOLAFIN.NS", "SHRIRAMFIN.NS", "POONAWALLA.NS",
    ],
    "Nifty Consumption": [
        "TITAN.NS", "ASIANPAINT.NS", "PIDILITIND.NS", "PAGEIND.NS",
        "RELAXO.NS", "BATA.NS", "TRENT.NS", "DMART.NS",
        "JUBLFOOD.NS", "ZOMATO.NS", "NYKAA.NS",
    ],
    "Nifty Media": [
        "ZEEL.NS", "PVR.NS", "SUNTV.NS", "NETWORK18.NS", "TV18BRDCST.NS",
    ],
    "Nifty PSU Bank": [
        "SBIN.NS", "BANKBARODA.NS", "PNB.NS", "CANBK.NS",
        "UNIONBANK.NS", "INDIANB.NS", "BANKINDIA.NS", "MAHABANK.NS",
        "IOB.NS", "CENTRALBK.NS", "UCOBANK.NS",
    ],
}


def get_all_stock_tickers() -> list[str]:
    """Return deduplicated list of all stock tickers across sectors."""
    seen = set()
    tickers = []
    for stocks in NIFTY500_SECTOR_MAP.values():
        for t in stocks:
            if t not in seen:
                seen.add(t)
                tickers.append(t)
    return tickers


def get_sector_for_stock(ticker: str) -> str | None:
    """Return the sector name for a given stock ticker."""
    for sector, stocks in NIFTY500_SECTOR_MAP.items():
        if ticker in stocks:
            return sector
    return None


def fetch_price_data(
    tickers: list[str],
    days: int = LOOKBACK_DAYS,
    end_date: dt.date | None = None,
) -> dict[str, pd.DataFrame]:
    """
    Fetch OHLCV data for a list of tickers.
    Returns dict: ticker -> DataFrame with columns [Open, High, Low, Close, Volume].
    """
    use_max = (days == 0)
    if use_max:
        print(f"  Fetching {len(tickers)} tickers (max history)...")
    else:
        if end_date is None:
            end_date = dt.date.today()
        start_date = end_date - dt.timedelta(days=int(days * 1.5))  # buffer for weekends/holidays
        print(f"  Fetching {len(tickers)} tickers from {start_date} to {end_date}...")

    data = {}

    # Batch download for efficiency
    batch_size = 50
    cols_needed = ["Open", "High", "Low", "Close", "Volume"]

    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i + batch_size]
        batch_str = " ".join(batch)
        try:
            dl_kwargs = dict(
                tickers=batch_str,
                group_by="ticker",
                progress=False,
                threads=True,
            )
            if use_max:
                dl_kwargs["period"] = "max"
            else:
                dl_kwargs["start"] = str(start_date)
                dl_kwargs["end"] = str(end_date)

            raw = yf.download(**dl_kwargs)
            if raw.empty:
                continue

            # yfinance >= 0.2.31 returns MultiIndex columns: (Price, Ticker)
            # Older versions may return flat columns for single tickers.
            if isinstance(raw.columns, pd.MultiIndex):
                for ticker in batch:
                    try:
                        # Try selecting by top-level ticker first (group_by="ticker")
                        if ticker in raw.columns.get_level_values(1):
                            df = raw.xs(ticker, level=1, axis=1)[cols_needed].copy()
                        elif ticker in raw.columns.get_level_values(0):
                            df = raw[ticker][cols_needed].copy()
                        else:
                            continue
                        df.dropna(inplace=True)
                        if len(df) > 0:
                            data[ticker] = df
                    except (KeyError, TypeError):
                        pass
            else:
                # Flat columns — single ticker download
                ticker = batch[0]
                df = raw[cols_needed].copy()
                df.dropna(inplace=True)
                if len(df) > 0:
                    data[ticker] = df
        except Exception as e:
            print(f"  Warning: batch download failed for {batch[:3]}...: {e}")

    print(f"  Successfully fetched {len(data)}/{len(tickers)} tickers.")
    return data


def fetch_index_data(
    days: int = LOOKBACK_DAYS,
    end_date: dt.date | None = None,
) -> pd.DataFrame:
    """Fetch Nifty 50 index data."""
    result = fetch_price_data([NIFTY50_TICKER], days=days, end_date=end_date)
    if NIFTY50_TICKER in result:
        return result[NIFTY50_TICKER]
    raise RuntimeError(f"Could not fetch {NIFTY50_TICKER}")


def fetch_sector_data(
    days: int = LOOKBACK_DAYS,
    end_date: dt.date | None = None,
) -> dict[str, pd.DataFrame]:
    """Fetch all NSE sectoral index data. Returns dict: sector_name -> DataFrame."""
    tickers = list(NSE_SECTOR_INDICES.values())
    raw = fetch_price_data(tickers, days=days, end_date=end_date)

    # Map back to sector names
    ticker_to_name = {v: k for k, v in NSE_SECTOR_INDICES.items()}
    result = {}
    for ticker, df in raw.items():
        name = ticker_to_name.get(ticker, ticker)
        result[name] = df
    return result


def fetch_stock_data_for_sectors(
    sectors: list[str],
    days: int = LOOKBACK_DAYS,
    end_date: dt.date | None = None,
) -> dict[str, pd.DataFrame]:
    """Fetch stock data only for stocks in the given sectors."""
    tickers = []
    for sector in sectors:
        tickers.extend(NIFTY500_SECTOR_MAP.get(sector, []))
    tickers = list(set(tickers))  # deduplicate
    return fetch_price_data(tickers, days=days, end_date=end_date)


def fetch_all_stock_data(
    days: int = LOOKBACK_DAYS,
    end_date: dt.date | None = None,
) -> dict[str, pd.DataFrame]:
    """Fetch data for all stocks in our universe."""
    tickers = get_all_stock_tickers()
    return fetch_price_data(tickers, days=days, end_date=end_date)


def load_nifty500_from_csv(filepath: str) -> dict[str, list[str]]:
    """
    Load Nifty 500 constituents from a CSV file.
    Expected columns: Symbol, Industry (or Sector)
    Download from: https://www.niftyindices.com/reports/nifty-500-702
    """
    df = pd.read_csv(filepath)
    sector_map = {}
    for _, row in df.iterrows():
        sector = row.get("Industry", row.get("Sector", "Unknown"))
        symbol = row["Symbol"].strip() + ".NS"
        sector_map.setdefault(sector, []).append(symbol)
    return sector_map


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Compute Average True Range."""
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()
