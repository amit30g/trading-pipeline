"""
Data Fetcher — yfinance wrapper for NSE data.
Handles downloading price/volume data for indices, sectors, and individual stocks.
Dynamically loads Nifty Total Market constituents from NSE CSV with disk caching.
"""
import datetime as dt
import os
import time
import requests
import yfinance as yf
import pandas as pd
import numpy as np
from pathlib import Path
from config import (
    NSE_SECTOR_INDICES, NIFTY50_TICKER, LOOKBACK_DAYS,
    NSE_TM_CSV_URL, UNIVERSE_CACHE_TTL_HOURS, MACRO_CACHE_TTL_HOURS,
    INDUSTRY_TO_SECTOR, MACRO_TICKERS, SECTOR_CONSTITUENT_URLS,
)

# ── Cache paths ──────────────────────────────────────────────
CACHE_DIR = Path(__file__).parent / "scan_cache"
CONSTITUENTS_CACHE = CACHE_DIR / "nifty_tm_constituents.csv"
MACRO_CACHE = CACHE_DIR / "macro_data.pkl"


# ── Nifty Total Market Universe (dynamic) ─────────────────────


def _cache_age_hours(filepath: Path) -> float:
    """Return age of a file in hours, or infinity if it doesn't exist."""
    if not filepath.exists():
        return float("inf")
    mtime = filepath.stat().st_mtime
    return (time.time() - mtime) / 3600


def _download_nse_csv() -> pd.DataFrame | None:
    """Download the Nifty Total Market constituents CSV from NSE."""
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
            "Accept": "text/csv,text/html,application/xhtml+xml",
        }
        resp = requests.get(NSE_TM_CSV_URL, headers=headers, timeout=30)
        resp.raise_for_status()
        CACHE_DIR.mkdir(exist_ok=True)
        CONSTITUENTS_CACHE.write_bytes(resp.content)
        df = pd.read_csv(CONSTITUENTS_CACHE)
        print(f"  Downloaded {len(df)} constituents from NSE")
        return df
    except Exception as e:
        print(f"  Warning: NSE CSV download failed: {e}")
        return None


def load_universe() -> pd.DataFrame:
    """
    Load Nifty Total Market constituents.
    Downloads from NSE if cache is stale (>24h), falls back to cached file.
    Returns DataFrame with at least columns: Symbol, Industry
    """
    age = _cache_age_hours(CONSTITUENTS_CACHE)

    if age < UNIVERSE_CACHE_TTL_HOURS:
        # Use cached
        df = pd.read_csv(CONSTITUENTS_CACHE)
        print(f"  Using cached universe ({len(df)} stocks, {age:.1f}h old)")
        return df

    # Try downloading fresh
    df = _download_nse_csv()
    if df is not None:
        return df

    # Fallback to stale cache
    if CONSTITUENTS_CACHE.exists():
        df = pd.read_csv(CONSTITUENTS_CACHE)
        print(f"  Using stale cached universe ({len(df)} stocks, {age:.1f}h old)")
        return df

    # No cache at all — return empty
    print("  ERROR: No universe data available. Run with internet connectivity first.")
    return pd.DataFrame(columns=["Symbol", "Industry"])


def _load_sector_constituents() -> dict[str, set[str]]:
    """
    Fetch NSE index constituent CSVs for sectors that can't be reliably
    mapped from industry labels. Returns {sector_name: {SYMBOL.NS, ...}}.
    Caches each CSV to scan_cache/ with 24h TTL.
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
        "Accept": "text/csv,text/html,application/xhtml+xml",
    }
    result: dict[str, set[str]] = {}

    for sector_name, url in SECTOR_CONSTITUENT_URLS.items():
        # Derive cache filename from URL slug
        slug = url.rsplit("/", 1)[-1]  # e.g. ind_niftybanklist.csv
        cache_path = CACHE_DIR / f"sector_{slug}"

        try:
            if _cache_age_hours(cache_path) < UNIVERSE_CACHE_TTL_HOURS:
                df = pd.read_csv(cache_path)
            else:
                resp = requests.get(url, headers=headers, timeout=30)
                resp.raise_for_status()
                CACHE_DIR.mkdir(exist_ok=True)
                cache_path.write_bytes(resp.content)
                df = pd.read_csv(cache_path)
                print(f"  Downloaded {len(df)} constituents for {sector_name}")

            symbols = set()
            for _, row in df.iterrows():
                sym = str(row.get("Symbol", "")).strip()
                if sym:
                    symbols.add(f"{sym}.NS")
            if symbols:
                result[sector_name] = symbols
        except Exception as e:
            print(f"  Warning: {sector_name} constituent CSV failed: {e}")

    return result


def _build_sector_map(universe_df: pd.DataFrame) -> dict[str, list[str]]:
    """Build sector_name -> [tickers] mapping from universe DataFrame.

    1. Map stocks to sectors using INDUSTRY_TO_SECTOR (broad mapping)
    2. Override with actual NSE index constituents for sectors that have CSVs
       (constituent membership wins over industry-based mapping)
    """
    sector_map: dict[str, list[str]] = {}
    if universe_df.empty:
        return sector_map

    # Collect all universe tickers for filtering
    universe_tickers = set()
    for _, row in universe_df.iterrows():
        sym = str(row.get("Symbol", "")).strip()
        if sym:
            universe_tickers.add(f"{sym}.NS")

    # Step 1: Industry-based mapping (existing logic)
    for _, row in universe_df.iterrows():
        symbol = str(row.get("Symbol", "")).strip()
        if not symbol:
            continue
        ticker = f"{symbol}.NS"
        industry = str(row.get("Industry", "")).strip()
        sector = INDUSTRY_TO_SECTOR.get(industry)
        if sector:
            sector_map.setdefault(sector, []).append(ticker)

    # Step 2: Override with NSE index constituent CSVs
    constituent_map = _load_sector_constituents()
    # Process broader indices first, then narrower ones, so specific wins.
    # e.g. "Nifty Fin Service" first, then "Nifty Bank" / "Nifty PSU Bank" override.
    SECTOR_PRIORITY = [
        "Nifty Fin Service", "Nifty Consumption", "Nifty Commodities",
        "Nifty Healthcare",  # broader — process first
        "Nifty Bank", "Nifty PSU Bank", "Nifty Pharma", "Nifty MNC",
    ]
    ordered_sectors = [s for s in SECTOR_PRIORITY if s in constituent_map]
    # Include any sectors not in the priority list at the end
    for s in constituent_map:
        if s not in ordered_sectors:
            ordered_sectors.append(s)

    # Track where each ticker is currently mapped (for removal)
    ticker_to_sector: dict[str, str] = {}
    for sector, tickers in sector_map.items():
        for t in tickers:
            ticker_to_sector[t] = sector

    for sector_name in ordered_sectors:
        constituent_tickers = constituent_map[sector_name]
        for ticker in constituent_tickers:
            if ticker not in universe_tickers:
                continue  # skip stocks outside our TM 750 universe
            current_sector = ticker_to_sector.get(ticker)
            if current_sector == sector_name:
                continue  # already correctly mapped
            # Remove from old sector if mapped elsewhere
            if current_sector and current_sector != sector_name:
                try:
                    sector_map[current_sector].remove(ticker)
                except ValueError:
                    pass
            # Add to constituent sector
            sector_map.setdefault(sector_name, [])
            if ticker not in sector_map[sector_name]:
                sector_map[sector_name].append(ticker)
            ticker_to_sector[ticker] = sector_name

    return sector_map


# Module-level cache for the sector map (rebuilt per-process)
_SECTOR_MAP_CACHE: dict[str, list[str]] | None = None
_UNIVERSE_DF_CACHE: pd.DataFrame | None = None


def get_sector_map() -> dict[str, list[str]]:
    """Get the current sector map, loading/downloading universe if needed."""
    global _SECTOR_MAP_CACHE, _UNIVERSE_DF_CACHE
    if _SECTOR_MAP_CACHE is None:
        _UNIVERSE_DF_CACHE = load_universe()
        _SECTOR_MAP_CACHE = _build_sector_map(_UNIVERSE_DF_CACHE)
    return _SECTOR_MAP_CACHE


def get_universe_df() -> pd.DataFrame:
    """Get the raw universe DataFrame."""
    global _UNIVERSE_DF_CACHE
    if _UNIVERSE_DF_CACHE is None:
        _UNIVERSE_DF_CACHE = load_universe()
    return _UNIVERSE_DF_CACHE


def reload_universe():
    """Force reload of universe data (e.g., after fresh download)."""
    global _SECTOR_MAP_CACHE, _UNIVERSE_DF_CACHE
    _SECTOR_MAP_CACHE = None
    _UNIVERSE_DF_CACHE = None


def get_all_stock_tickers() -> list[str]:
    """Return deduplicated list of all stock tickers across sectors."""
    sector_map = get_sector_map()
    seen = set()
    tickers = []
    for stocks in sector_map.values():
        for t in stocks:
            if t not in seen:
                seen.add(t)
                tickers.append(t)
    # Also include stocks that didn't map to any sector
    universe_df = get_universe_df()
    if not universe_df.empty:
        for _, row in universe_df.iterrows():
            symbol = str(row.get("Symbol", "")).strip()
            if not symbol:
                continue
            ticker = f"{symbol}.NS"
            if ticker not in seen:
                seen.add(ticker)
                tickers.append(ticker)
    return tickers


def get_sector_for_stock(ticker: str) -> str | None:
    """Return the sector name for a given stock ticker."""
    sector_map = get_sector_map()
    for sector, stocks in sector_map.items():
        if ticker in stocks:
            return sector
    return None


# ── Backward compatibility ───────────────────────────────────
# Some modules import NIFTY500_SECTOR_MAP directly. Provide a lazy property.

class _SectorMapProxy(dict):
    """Dict-like proxy that loads the sector map on first access."""
    _loaded = False

    def _ensure_loaded(self):
        if not self._loaded:
            self.update(get_sector_map())
            self._loaded = True

    def __getitem__(self, key):
        self._ensure_loaded()
        return super().__getitem__(key)

    def __contains__(self, key):
        self._ensure_loaded()
        return super().__contains__(key)

    def __iter__(self):
        self._ensure_loaded()
        return super().__iter__()

    def __len__(self):
        self._ensure_loaded()
        return super().__len__()

    def items(self):
        self._ensure_loaded()
        return super().items()

    def values(self):
        self._ensure_loaded()
        return super().values()

    def keys(self):
        self._ensure_loaded()
        return super().keys()

    def get(self, key, default=None):
        self._ensure_loaded()
        return super().get(key, default)


NIFTY500_SECTOR_MAP = _SectorMapProxy()


# ── Price Data Fetching ──────────────────────────────────────


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
            if isinstance(raw.columns, pd.MultiIndex):
                for ticker in batch:
                    try:
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
    sector_map = get_sector_map()
    tickers = []
    for sector in sectors:
        tickers.extend(sector_map.get(sector, []))
    tickers = list(set(tickers))  # deduplicate
    return fetch_price_data(tickers, days=days, end_date=end_date)


def fetch_all_stock_data(
    days: int = LOOKBACK_DAYS,
    end_date: dt.date | None = None,
) -> dict[str, pd.DataFrame]:
    """Fetch data for all stocks in our universe."""
    tickers = get_all_stock_tickers()
    return fetch_price_data(tickers, days=days, end_date=end_date)


# ── Macro Data ───────────────────────────────────────────────


def fetch_macro_data() -> dict[str, dict]:
    """
    Fetch macro dashboard data for all MACRO_TICKERS.
    Returns dict: label -> {price, change, change_pct, week_prices, close_series, dates}
    Also computes synthetic "10Y-5Y Spread" entry.
    """
    import pickle

    # Check cache
    if MACRO_CACHE.exists():
        age = _cache_age_hours(MACRO_CACHE)
        if age < MACRO_CACHE_TTL_HOURS:
            try:
                with open(MACRO_CACHE, "rb") as f:
                    cached = pickle.load(f)
                print(f"  Using cached macro data ({age:.1f}h old)")
                return cached
            except Exception:
                pass

    print("  Fetching macro data...")
    result = {}
    tickers_list = list(MACRO_TICKERS.values())

    try:
        raw = yf.download(
            tickers=" ".join(tickers_list),
            period="1y",
            group_by="ticker",
            progress=False,
            threads=True,
        )

        for label, ticker in MACRO_TICKERS.items():
            try:
                if isinstance(raw.columns, pd.MultiIndex):
                    if ticker in raw.columns.get_level_values(1):
                        df = raw.xs(ticker, level=1, axis=1)
                    elif ticker in raw.columns.get_level_values(0):
                        df = raw[ticker]
                    else:
                        continue
                else:
                    df = raw

                close = df["Close"].dropna()
                if len(close) < 2:
                    continue

                current = float(close.iloc[-1])
                prev = float(close.iloc[-2])
                change = current - prev
                change_pct = (change / prev) * 100 if prev != 0 else 0

                # Last 5 trading days for sparkline
                week_prices = close.iloc[-5:].tolist() if len(close) >= 5 else close.tolist()

                result[label] = {
                    "price": current,
                    "change": change,
                    "change_pct": change_pct,
                    "week_prices": week_prices,
                    "close_series": close.tolist(),
                    "dates": [d.strftime("%Y-%m-%d") for d in close.index],
                }
            except Exception:
                continue
    except Exception as e:
        print(f"  Warning: macro data fetch failed: {e}")

    # Compute yield curve spread: 10Y - 5Y
    if "US 10Y" in result and "US 5Y" in result:
        try:
            spread = result["US 10Y"]["price"] - result["US 5Y"]["price"]
            prev_10y = result["US 10Y"]["price"] - result["US 10Y"]["change"]
            prev_5y = result["US 5Y"]["price"] - result["US 5Y"]["change"]
            prev_spread = prev_10y - prev_5y
            result["10Y-5Y Spread"] = {
                "price": spread,
                "change": spread - prev_spread,
                "change_pct": ((spread - prev_spread) / abs(prev_spread) * 100) if prev_spread != 0 else 0,
                "week_prices": [],
                "close_series": [],
                "dates": [],
            }
        except Exception:
            pass

    # Cache result
    if result:
        CACHE_DIR.mkdir(exist_ok=True)
        with open(MACRO_CACHE, "wb") as f:
            pickle.dump(result, f)

    return result


# ── Utility ──────────────────────────────────────────────────


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
