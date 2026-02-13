"""
NSE India API wrapper using curl_cffi for TLS fingerprint impersonation.

NSE blocks Python's `requests` library via TLS fingerprinting (403).
curl_cffi impersonates a real Chrome browser's TLS handshake to bypass this.

Provides quarterly financials, shareholding patterns, and corporate
announcements from NSE's undocumented API endpoints.
"""
import time
import pickle
import logging
from datetime import datetime, timedelta
from pathlib import Path

from curl_cffi import requests as cf_requests
import pandas as pd

logger = logging.getLogger(__name__)

CACHE_DIR = Path(__file__).parent / "scan_cache"
CACHE_TTL_HOURS = 24


class NSEDataFetcher:
    """NSE India API wrapper with Chrome TLS impersonation and caching."""

    BASE_URL = "https://www.nseindia.com"

    def __init__(self):
        self.session = cf_requests.Session(impersonate="chrome")
        self._last_request_time = 0.0
        self._min_interval = 0.35  # ~3 requests/sec
        self._cookies_valid = False

    def _init_cookies(self):
        """Establish session cookies by visiting the main page."""
        try:
            resp = self.session.get(self.BASE_URL, timeout=10)
            if resp.status_code == 200:
                self._cookies_valid = True
            else:
                logger.warning("NSE cookie init got status %d", resp.status_code)
                self._cookies_valid = False
        except Exception as e:
            logger.warning("NSE cookie init failed: %s", e)
            self._cookies_valid = False

    def _request(self, url, params=None, retries=3):
        """Rate-limited request with retry and cookie refresh."""
        for attempt in range(retries):
            elapsed = time.time() - self._last_request_time
            if elapsed < self._min_interval:
                time.sleep(self._min_interval - elapsed)

            if not self._cookies_valid:
                self._init_cookies()
                if not self._cookies_valid:
                    time.sleep(2 ** attempt)
                    continue

            try:
                self._last_request_time = time.time()
                resp = self.session.get(url, params=params, timeout=15)

                if resp.status_code in (403, 429):
                    logger.info("NSE %d on attempt %d, refreshing cookies", resp.status_code, attempt + 1)
                    self._cookies_valid = False
                    time.sleep(2 ** attempt)
                    continue

                if resp.status_code == 404:
                    return None

                resp.raise_for_status()
                return resp.json()

            except ValueError:
                logger.warning("NSE returned non-JSON on attempt %d", attempt + 1)
                time.sleep(2 ** attempt)
            except Exception as e:
                logger.warning("NSE request failed on attempt %d: %s", attempt + 1, e)
                self._cookies_valid = False
                time.sleep(2 ** attempt)

        return None

    # ── Cache helpers ─────────────────────────────────────────────

    def _cache_path(self, symbol: str, data_type: str) -> Path:
        CACHE_DIR.mkdir(exist_ok=True)
        return CACHE_DIR / f"{symbol}_{data_type}.pkl"

    def _load_cache(self, symbol: str, data_type: str):
        path = self._cache_path(symbol, data_type)
        if not path.exists():
            return None
        try:
            mtime = datetime.fromtimestamp(path.stat().st_mtime)
            if datetime.now() - mtime > timedelta(hours=CACHE_TTL_HOURS):
                return None
            with open(path, "rb") as f:
                return pickle.load(f)
        except Exception:
            return None

    def _save_cache(self, symbol: str, data_type: str, data):
        try:
            path = self._cache_path(symbol, data_type)
            with open(path, "wb") as f:
                pickle.dump(data, f)
        except Exception as e:
            logger.warning("Cache save failed for %s/%s: %s", symbol, data_type, e)

    # ── Clean symbol helper ───────────────────────────────────────

    @staticmethod
    def _clean_symbol(symbol: str) -> str:
        """Convert 'RELIANCE.NS' -> 'RELIANCE'."""
        return symbol.replace(".NS", "").replace(".BO", "").strip().upper()

    # ── Quarterly Results ─────────────────────────────────────────

    def fetch_quarterly_results(self, symbol: str, num_quarters: int = 20) -> pd.DataFrame | None:
        """Fetch quarterly financial results from NSE.

        Uses /api/results-comparision for detailed P&L (5 quarters),
        plus /api/top-corp-info for summary data.

        Returns DataFrame with columns:
            date, revenue, operating_income, net_income, diluted_eps,
            opm_pct, npm_pct, depreciation, tax
        All monetary values in lakhs (divide by 100 for Cr).
        """
        clean = self._clean_symbol(symbol)
        cached = self._load_cache(clean, "quarterly_results")
        if cached is not None:
            return cached

        # Primary: detailed P&L from results-comparision
        data = self._request(f"{self.BASE_URL}/api/results-comparision?symbol={clean}")

        if not data:
            return None

        try:
            records = data.get("resCmpData", [])
            if not records:
                return None

            rows = []
            for r in records[:num_quarters]:
                date_str = r.get("re_to_dt")
                if not date_str:
                    continue
                try:
                    date = pd.to_datetime(date_str, dayfirst=True)
                except Exception:
                    continue

                revenue = self._parse_num(r.get("re_net_sale"))
                total_income = self._parse_num(r.get("re_total_inc"))
                other_income = self._parse_num(r.get("re_oth_inc_new"))
                total_expense = self._parse_num(r.get("re_oth_tot_exp"))
                depreciation = self._parse_num(r.get("re_depr_und_exp"))
                pbt = self._parse_num(r.get("re_pro_loss_bef_tax"))
                tax = self._parse_num(r.get("re_tax"))
                net_income = self._parse_num(r.get("re_net_profit") or r.get("re_con_pro_loss"))
                eps = self._parse_num(r.get("re_dilut_eps_for_cont_dic_opr") or r.get("re_diluted_eps"))

                # Operating income = revenue - total expenses + depreciation + tax
                # Or: PBT + interest
                interest = self._parse_num(r.get("re_int_new"))
                operating_income = None
                if pbt is not None and interest is not None:
                    operating_income = pbt + interest
                elif revenue is not None and total_expense is not None:
                    operating_income = revenue - total_expense + (depreciation or 0) + (tax or 0)

                # Margins computed before unit conversion (ratios are unitless)
                opm = (operating_income / revenue * 100) if revenue and operating_income else None
                npm = (net_income / revenue * 100) if revenue and net_income else None

                # NSE reports monetary values in lakhs; convert to rupees
                # so _fmt_cr (which divides by 1e7) displays correctly.
                LAKHS = 1e5
                rows.append({
                    "date": date,
                    "revenue": revenue * LAKHS if revenue else None,
                    "operating_income": operating_income * LAKHS if operating_income else None,
                    "net_income": net_income * LAKHS if net_income else None,
                    "diluted_eps": eps,  # EPS is already per-share, no conversion
                    "opm_pct": opm,
                    "npm_pct": npm,
                    "depreciation": depreciation * LAKHS if depreciation else None,
                    "tax": tax * LAKHS if tax else None,
                    "pbt": pbt * LAKHS if pbt else None,
                    "other_income": other_income * LAKHS if other_income else None,
                })

            if not rows:
                return None

            # Detect and fix unit inconsistency: NSE sometimes returns
            # the oldest quarter in crores while others are in lakhs,
            # causing a ~100x difference in monetary values.
            self._fix_unit_outliers(rows)

            df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
            self._save_cache(clean, "quarterly_results", df)
            return df

        except Exception as e:
            logger.warning("Failed to parse quarterly results for %s: %s", clean, e)
            return None

    # ── Annual Results ────────────────────────────────────────────

    def fetch_annual_results(self, symbol: str, num_years: int = 10) -> pd.DataFrame | None:
        """Fetch annual financial results from NSE.

        NSE doesn't have a separate annual endpoint — we return None and
        let the caller fall back to yfinance for annual data.
        """
        # NSE's results-comparision only returns recent quarterly data.
        # No separate annual endpoint exists. Return None to trigger yfinance fallback.
        return None

    # ── Shareholding Pattern ──────────────────────────────────────

    def fetch_shareholding_pattern(self, symbol: str, num_quarters: int = 20) -> list[dict] | None:
        """Fetch shareholding pattern from NSE.

        Uses /api/top-corp-info which provides Promoter vs Public split
        for the last 5 quarters.

        Returns list of dicts: {date, promoter_pct, public_pct}
        """
        clean = self._clean_symbol(symbol)
        cached = self._load_cache(clean, "shareholding")
        if cached is not None:
            return cached

        data = self._request(
            f"{self.BASE_URL}/api/top-corp-info",
            params={"symbol": clean, "market": "equities"},
        )
        if not data:
            return None

        try:
            sh_data = data.get("shareholdings_patterns", {}).get("data", {})
            if not sh_data:
                return None

            results = []
            for date_str, holdings in sh_data.items():
                try:
                    date = pd.to_datetime(date_str, dayfirst=True)
                except Exception:
                    continue

                entry = {"date": date, "promoter_pct": None, "public_pct": None}
                for h in holdings:
                    for key, val in h.items():
                        pct = self._parse_num(val)
                        key_lower = key.lower()
                        if "promoter" in key_lower:
                            entry["promoter_pct"] = pct
                        elif "public" in key_lower:
                            entry["public_pct"] = pct

                if entry["promoter_pct"] is not None:
                    results.append(entry)

            if not results:
                return None

            results.sort(key=lambda x: x["date"])
            self._save_cache(clean, "shareholding", results)
            return results

        except Exception as e:
            logger.warning("Failed to parse shareholding for %s: %s", clean, e)
            return None

    # ── Announcements ─────────────────────────────────────────────

    def fetch_announcements(self, symbol: str, months: int = 3) -> list[dict] | None:
        """Fetch corporate announcements from NSE.

        Uses /api/corporate-announcements which returns full history.
        We filter to the requested months.

        Returns list of dicts: {date, subject, category}
        """
        clean = self._clean_symbol(symbol)
        cached = self._load_cache(clean, "announcements")
        if cached is not None:
            return cached

        data = self._request(
            f"{self.BASE_URL}/api/corporate-announcements",
            params={"index": "equities", "symbol": clean},
        )
        if not data:
            return None

        try:
            records = data if isinstance(data, list) else data.get("data", [])
            if not records:
                return None

            cutoff = datetime.now() - timedelta(days=months * 30)
            results = []
            for r in records:
                date_str = r.get("an_dt") or r.get("sort_date")
                if not date_str:
                    continue
                try:
                    date = pd.to_datetime(date_str)
                except Exception:
                    continue

                if date < cutoff:
                    continue

                # Use attchmntText (full description) if available, else desc (short)
                subject = r.get("attchmntText") or r.get("desc") or ""
                # Use desc as category (it's usually a short label like "Updates", "Board Meeting")
                category = r.get("desc") or r.get("smIndustry") or "General"

                results.append({
                    "date": date,
                    "subject": subject[:300],
                    "category": category,
                })

            if not results:
                return None

            results.sort(key=lambda x: x["date"], reverse=True)
            self._save_cache(clean, "announcements", results)
            return results

        except Exception as e:
            logger.warning("Failed to parse announcements for %s: %s", clean, e)
            return None

    # ── Smart Money Data ──────────────────────────────────────────

    def fetch_fii_dii_data(self) -> dict | None:
        """Fetch FII/DII daily trading activity from NSE.

        Returns dict: {date, fii_buy, fii_sell, fii_net, dii_buy, dii_sell, dii_net}
        Values in Crores.
        """
        cached = self._load_cache("MARKET", "fii_dii")
        if cached is not None:
            return cached

        data = self._request(f"{self.BASE_URL}/api/fiidiiTrading")
        if not data:
            return None

        try:
            result = {"date": None, "fii_buy": 0, "fii_sell": 0, "fii_net": 0,
                      "dii_buy": 0, "dii_sell": 0, "dii_net": 0}

            records = data if isinstance(data, list) else [data]
            for r in records:
                category = (r.get("category") or "").upper()
                buy_val = self._parse_num(r.get("buyValue"))
                sell_val = self._parse_num(r.get("sellValue"))
                net_val = self._parse_num(r.get("netValue"))
                date_str = r.get("date")

                if date_str and result["date"] is None:
                    result["date"] = date_str

                if "FII" in category or "FPI" in category:
                    result["fii_buy"] = buy_val or 0
                    result["fii_sell"] = sell_val or 0
                    result["fii_net"] = net_val or 0
                elif "DII" in category:
                    result["dii_buy"] = buy_val or 0
                    result["dii_sell"] = sell_val or 0
                    result["dii_net"] = net_val or 0

            self._save_cache("MARKET", "fii_dii", result)
            return result

        except Exception as e:
            logger.warning("Failed to parse FII/DII data: %s", e)
            return None

    def fetch_fii_dii_historical(self, days: int = 1825) -> pd.DataFrame:
        """Fetch and persist historical FII/DII daily data.

        Strategy:
        1. Load existing history from scan_cache/fii_dii_history.csv
        2. Fetch missing days from NSE (tries date-range params, then single-day)
        3. Merge, deduplicate, save, return

        Args:
            days: How far back to try fetching (default 5 years = 1825 days).

        Returns:
            DataFrame with columns: date, fii_buy, fii_sell, fii_net,
            dii_buy, dii_sell, dii_net. Sorted by date ascending.
            Empty DataFrame if no data available.
        """
        csv_path = CACHE_DIR / "fii_dii_history.csv"
        CACHE_DIR.mkdir(exist_ok=True)

        # Load existing CSV
        existing = None
        if csv_path.exists():
            try:
                existing = pd.read_csv(csv_path, parse_dates=["date"])
            except Exception:
                existing = None

        # Determine what we need to fetch
        today = datetime.now().date()
        new_rows = []

        if existing is not None and not existing.empty:
            last_date = existing["date"].max().date()
            if last_date >= today - timedelta(days=1):
                return existing.sort_values("date").reset_index(drop=True)
            fetch_from = last_date + timedelta(days=1)
        else:
            fetch_from = today - timedelta(days=days)

        # Try fetching historical data from NSE in chunks
        chunk_days = 90
        current = fetch_from
        while current <= today:
            chunk_end = min(current + timedelta(days=chunk_days - 1), today)
            from_str = current.strftime("%d-%m-%Y")
            to_str = chunk_end.strftime("%d-%m-%Y")

            data = self._request(
                f"{self.BASE_URL}/api/fiidiiTrading",
                params={"from": from_str, "to": to_str},
            )

            if data:
                parsed = self._parse_fii_dii_records(data)
                new_rows.extend(parsed)

            current = chunk_end + timedelta(days=1)

            # If first chunk returned no multi-day data, NSE likely doesn't
            # support date params — just persist today's single-day data
            if not new_rows and current > fetch_from + timedelta(days=chunk_days):
                break

        # If no historical data fetched, try current-day endpoint
        if not new_rows:
            data = self._request(f"{self.BASE_URL}/api/fiidiiTrading")
            if data:
                new_rows = self._parse_fii_dii_records(data)

        # Merge with existing
        if new_rows:
            new_df = pd.DataFrame(new_rows)
            if existing is not None and not existing.empty:
                combined = pd.concat([existing, new_df], ignore_index=True)
            else:
                combined = new_df

            # Deduplicate by date (keep latest)
            combined["date"] = pd.to_datetime(combined["date"])
            combined = combined.sort_values("date").drop_duplicates(
                subset=["date"], keep="last"
            ).reset_index(drop=True)

            # Save to CSV
            try:
                combined.to_csv(csv_path, index=False)
            except Exception as e:
                logger.warning("Failed to save FII/DII history: %s", e)

            return combined

        if existing is not None and not existing.empty:
            return existing.sort_values("date").reset_index(drop=True)

        return pd.DataFrame(
            columns=["date", "fii_buy", "fii_sell", "fii_net",
                      "dii_buy", "dii_sell", "dii_net"]
        )

    def _parse_fii_dii_records(self, data) -> list[dict]:
        """Parse NSE FII/DII API response into row dicts.

        Handles both single-day (2 records: FII + DII) and
        multi-day (many records with dates) response formats.
        """
        rows_by_date: dict[str, dict] = {}
        records = data if isinstance(data, list) else [data]

        for r in records:
            category = (r.get("category") or "").upper()
            date_str = r.get("date")
            if not date_str:
                continue

            # Normalize date
            try:
                dt_obj = pd.to_datetime(date_str, dayfirst=True)
                date_key = dt_obj.strftime("%Y-%m-%d")
            except Exception:
                continue

            if date_key not in rows_by_date:
                rows_by_date[date_key] = {
                    "date": date_key,
                    "fii_buy": 0, "fii_sell": 0, "fii_net": 0,
                    "dii_buy": 0, "dii_sell": 0, "dii_net": 0,
                }

            buy_val = self._parse_num(r.get("buyValue")) or 0
            sell_val = self._parse_num(r.get("sellValue")) or 0
            net_val = self._parse_num(r.get("netValue")) or 0

            if "FII" in category or "FPI" in category:
                rows_by_date[date_key]["fii_buy"] = buy_val
                rows_by_date[date_key]["fii_sell"] = sell_val
                rows_by_date[date_key]["fii_net"] = net_val
            elif "DII" in category:
                rows_by_date[date_key]["dii_buy"] = buy_val
                rows_by_date[date_key]["dii_sell"] = sell_val
                rows_by_date[date_key]["dii_net"] = net_val

        return list(rows_by_date.values())

    def fetch_bulk_deals(self, from_date: str = None, to_date: str = None) -> list[dict]:
        """Fetch bulk deals from NSE.

        Args:
            from_date: DD-MM-YYYY format. Defaults to 90 days ago.
            to_date: DD-MM-YYYY format. Defaults to today.

        Returns list of {date, symbol, client_name, deal_type, quantity, price}.
        """
        if from_date is None:
            from_date = (datetime.now() - timedelta(days=90)).strftime("%d-%m-%Y")
        if to_date is None:
            to_date = datetime.now().strftime("%d-%m-%Y")

        data = self._request(
            f"{self.BASE_URL}/api/historical/bulk-deals",
            params={"from": from_date, "to": to_date},
        )
        if not data:
            return []

        try:
            records = data if isinstance(data, list) else data.get("data", [])
            results = []
            for r in records:
                results.append({
                    "date": r.get("mTd") or r.get("date") or "",
                    "symbol": (r.get("symbol") or "").strip(),
                    "client_name": r.get("clientName") or r.get("clientname") or "",
                    "deal_type": r.get("buySell") or r.get("buysell") or "",
                    "quantity": self._parse_num(r.get("quantity") or r.get("quantityTraded")) or 0,
                    "price": self._parse_num(r.get("wAvgPrice") or r.get("wapc") or r.get("price")) or 0,
                })
            return results
        except Exception as e:
            logger.warning("Failed to parse bulk deals: %s", e)
            return []

    def fetch_block_deals(self, from_date: str = None, to_date: str = None) -> list[dict]:
        """Fetch block deals from NSE.

        Args:
            from_date: DD-MM-YYYY format. Defaults to 90 days ago.
            to_date: DD-MM-YYYY format. Defaults to today.

        Returns list of {date, symbol, client_name, deal_type, quantity, price}.
        """
        if from_date is None:
            from_date = (datetime.now() - timedelta(days=90)).strftime("%d-%m-%Y")
        if to_date is None:
            to_date = datetime.now().strftime("%d-%m-%Y")

        data = self._request(
            f"{self.BASE_URL}/api/historical/block-deals",
            params={"from": from_date, "to": to_date},
        )
        if not data:
            return []

        try:
            records = data if isinstance(data, list) else data.get("data", [])
            results = []
            for r in records:
                results.append({
                    "date": r.get("mTd") or r.get("date") or "",
                    "symbol": (r.get("symbol") or "").strip(),
                    "client_name": r.get("clientName") or r.get("clientname") or "",
                    "deal_type": r.get("buySell") or r.get("buysell") or "",
                    "quantity": self._parse_num(r.get("quantity") or r.get("quantityTraded")) or 0,
                    "price": self._parse_num(r.get("wAvgPrice") or r.get("wapc") or r.get("price")) or 0,
                })
            return results
        except Exception as e:
            logger.warning("Failed to parse block deals: %s", e)
            return []

    def fetch_delivery_data(self, symbol: str) -> dict | None:
        """Fetch delivery percentage data for a symbol from NSE quote.

        Returns: {symbol, delivery_qty, traded_qty, delivery_pct}
        """
        clean = self._clean_symbol(symbol)
        cached = self._load_cache(clean, "delivery")
        if cached is not None:
            return cached

        data = self._request(
            f"{self.BASE_URL}/api/quote-equity",
            params={"symbol": clean},
        )
        if not data:
            return None

        try:
            sec_info = data.get("securityWiseDP") or data.get("preOpenMarket") or {}
            delivery_qty = self._parse_num(sec_info.get("deliveryQuantity") or
                                           data.get("deliveryQuantity"))
            traded_qty = self._parse_num(sec_info.get("quantityTraded") or
                                         data.get("totalTradedVolume"))
            delivery_pct = self._parse_num(sec_info.get("deliveryToTradedQuantity") or
                                           data.get("deliveryToTradedQuantity"))

            if delivery_pct is None and delivery_qty and traded_qty and traded_qty > 0:
                delivery_pct = (delivery_qty / traded_qty) * 100

            result = {
                "symbol": clean,
                "delivery_qty": delivery_qty or 0,
                "traded_qty": traded_qty or 0,
                "delivery_pct": delivery_pct or 0,
            }

            self._save_cache(clean, "delivery", result)
            return result

        except Exception as e:
            logger.warning("Failed to parse delivery data for %s: %s", clean, e)
            return None

    # ── Helpers ────────────────────────────────────────────────────

    @staticmethod
    def _fix_unit_outliers(rows: list[dict]):
        """Fix unit inconsistency in quarterly results.

        NSE sometimes returns the oldest quarter's monetary values in crores
        while the rest are in lakhs (~100x difference). Detect outliers and
        scale them up to match the majority.
        """
        monetary_keys = [
            "revenue", "operating_income", "net_income",
            "depreciation", "tax", "pbt", "other_income",
        ]
        # Use revenue as the reference for detection
        rev_by_idx = []
        for i, r in enumerate(rows):
            rv = r.get("revenue")
            if rv is not None and rv > 0:
                rev_by_idx.append((i, rv))

        if len(rev_by_idx) < 3:
            return

        rev_values = sorted([rv for _, rv in rev_by_idx])
        median_rev = rev_values[len(rev_values) // 2]

        for idx, rv in rev_by_idx:
            ratio = median_rev / rv
            if 30 < ratio < 300:
                # This row's monetary values are ~100x too small (crores vs lakhs)
                scale = round(ratio / 100) * 100  # snap to nearest 100x
                if scale < 50:
                    continue
                for key in monetary_keys:
                    val = rows[idx].get(key)
                    if val is not None:
                        rows[idx][key] = val * scale
                # Margins are ratios — they stay correct, no need to fix

    @staticmethod
    def _parse_num(val) -> float | None:
        """Safely parse a numeric value from NSE response."""
        if val is None:
            return None
        if isinstance(val, (int, float)):
            return float(val)
        try:
            cleaned = str(val).replace(",", "").replace(" ", "").strip()
            if cleaned in ("", "-", "NA", "N/A"):
                return None
            return float(cleaned)
        except (ValueError, TypeError):
            return None


# Module-level singleton
_fetcher = None


def get_nse_fetcher() -> NSEDataFetcher:
    """Get or create the module-level NSE data fetcher."""
    global _fetcher
    if _fetcher is None:
        _fetcher = NSEDataFetcher()
    return _fetcher


def compute_fii_dii_flows(history_df: pd.DataFrame) -> dict:
    """Compute cumulative FII/DII net flows for multiple timeframes.

    Args:
        history_df: DataFrame from fetch_fii_dii_historical() with columns:
                    date, fii_net, dii_net (+ buy/sell columns).

    Returns:
        Dict of {timeframe_label: {fii_net, dii_net, days_available}}.
        timeframe_label in: 1w, 2w, 1m, 3m, 6m, 1y, 2y, 5y.
        Returns empty dict if insufficient data.
    """
    if history_df is None or history_df.empty:
        return {}

    df = history_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")

    latest_date = df["date"].max()

    timeframes = {
        "1w": 7,
        "2w": 14,
        "1m": 30,
        "3m": 91,
        "6m": 182,
        "1y": 365,
        "2y": 730,
        "5y": 1825,
    }

    result = {}
    for label, cal_days in timeframes.items():
        cutoff = latest_date - timedelta(days=cal_days)
        period_df = df[df["date"] > cutoff]

        if period_df.empty:
            result[label] = {"fii_net": None, "dii_net": None, "days_available": 0}
            continue

        result[label] = {
            "fii_net": round(period_df["fii_net"].sum(), 1),
            "dii_net": round(period_df["dii_net"].sum(), 1),
            "days_available": len(period_df),
        }

    return result
