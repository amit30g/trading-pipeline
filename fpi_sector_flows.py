"""
FPI Sector-wise Investment Data — scrapes NSDL fortnightly reports.

Source: https://www.fpi.nsdl.co.in/web/StaticReports/Fortnightly_Sector_wise_FII_Investment_Data/
Reports available every fortnight (15th and last day of month).
"""
import datetime
import logging
import os
import pickle
import re
from pathlib import Path

import pandas as pd
import requests

logger = logging.getLogger(__name__)

CACHE_DIR = Path(__file__).parent / "scan_cache"
FPI_CACHE_FILE = CACHE_DIR / "fpi_sector_flows.pkl"
FPI_CACHE_TTL_HOURS = 12  # re-scrape after 12 hours

BASE_URL = "https://www.fpi.nsdl.co.in/web/StaticReports/Fortnightly_Sector_wise_FII_Investment_Data/FIIInvestSector_{date_str}.html"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
}

# Sectors we care about (map NSDL names to our shorter names)
SECTOR_NAME_MAP = {
    "Automobile and Auto Components": "Auto",
    "Capital Goods": "Capital Goods",
    "Chemicals": "Chemicals",
    "Construction": "Construction",
    "Construction Materials": "Const. Materials",
    "Consumer Durables": "Consumer Durables",
    "Consumer Services": "Consumer Services",
    "Diversified": "Diversified",
    "Fast Moving Consumer Goods": "FMCG",
    "Financial Services": "Financial Services",
    "Forest Materials": "Forest Materials",
    "Healthcare": "Healthcare",
    "Information Technology": "IT",
    "Media, Entertainment & Publication": "Media",
    "Media, Entertainment &amp; Publication": "Media",
    "Metals & Mining": "Metals & Mining",
    "Metals &amp; Mining": "Metals & Mining",
    "Oil, Gas & Consumable Fuels": "Oil & Gas",
    "Oil, Gas &amp; Consumable Fuels": "Oil & Gas",
    "Power": "Power",
    "Realty": "Realty",
    "Services": "Services",
    "Telecommunication": "Telecom",
    "Textiles": "Textiles",
    "Utilities": "Utilities",
}


def _generate_report_dates(months_back: int = 6) -> list[datetime.date]:
    """Generate fortnightly report dates for the last N months."""
    today = datetime.date.today()
    dates = []
    for mb in range(months_back + 1):
        m = today.month - mb
        y = today.year
        while m <= 0:
            m += 12
            y -= 1
        # 15th
        d15 = datetime.date(y, m, 15)
        if d15 <= today:
            dates.append(d15)
        # Last day
        if m == 12:
            last = datetime.date(y + 1, 1, 1) - datetime.timedelta(days=1)
        else:
            last = datetime.date(y, m + 1, 1) - datetime.timedelta(days=1)
        if last <= today:
            dates.append(last)
    dates.sort()
    return dates


def _parse_number(s: str) -> float:
    """Parse Indian number format like '1,23,456' or '-1,234' to float."""
    s = s.strip().replace(",", "").replace(" ", "")
    if not s or s == "-" or s == "—":
        return 0.0
    try:
        return float(s)
    except ValueError:
        return 0.0


def _scrape_single_report(report_date: datetime.date) -> dict | None:
    """Scrape a single fortnightly report. Returns dict of sector -> net equity investment (INR Cr)."""
    date_str = report_date.strftime("%b%d%Y")
    url = BASE_URL.format(date_str=date_str)

    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        if resp.status_code != 200:
            logger.debug("FPI report %s: HTTP %d", date_str, resp.status_code)
            return None
        html = resp.text
    except Exception as e:
        logger.debug("FPI report %s fetch failed: %s", date_str, e)
        return None

    # Parse HTML table rows
    rows = re.findall(r'<tr[^>]*>(.*?)</tr>', html, re.DOTALL)
    if len(rows) < 5:
        return None

    # Determine which columns contain the net investment equity data
    # The structure has multiple column groups. We want the fortnightly net investment in equity (INR Cr).
    # From the header analysis:
    # Row 0: period headers - "Net Investment {period}" columns
    # Row 3: detailed headers with "Equity", "Debt", etc.
    # Data rows start at row 4
    #
    # For the report FIIInvestSector_Feb282026.html:
    # Columns layout (0-indexed after Sr.No and Sector):
    # Cols 0-11: AUC as on start date (INR Cr: Equity, Debt GL, Debt VRR, Debt-FAR, Hybrid, ...)
    # Cols 12-23: AUC as on start date (USD Mn)
    # Cols 24-35: Net Investment 1st fortnight (INR Cr)
    # Cols 36-47: Net Investment 1st fortnight (USD Mn)
    # Cols 48-59: Net Investment 2nd fortnight (INR Cr)
    # Cols 60-71: Net Investment 2nd fortnight (USD Mn)
    # Cols 72-83: AUC as on end date (INR Cr)
    # Cols 84-95: AUC as on end date (USD Mn)
    #
    # We want: Net Investment equity for each fortnight
    # 1st fortnight equity = col index 24 (after removing Sr.No and Sector)
    # 2nd fortnight equity = col index 48

    # Detect the period from header
    header_row = rows[0] if rows else ""
    cells_h = re.findall(r'<td[^>]*>(.*?)</td>', header_row, re.DOTALL)
    cells_h = [re.sub(r'<[^>]+>', '', c).strip() for c in cells_h]

    # Find which fortnight periods are in this report
    periods = []
    for c in cells_h:
        if "Net Investment" in c:
            periods.append(c)

    result = {}
    for row in rows[4:]:  # skip header rows
        cells = re.findall(r'<td[^>]*>(.*?)</td>', row, re.DOTALL)
        cleaned = [re.sub(r'<[^>]+>', '', c).strip() for c in cells]
        if len(cleaned) < 50:
            continue
        sr_no = cleaned[0].strip()
        sector_raw = cleaned[1].strip()

        if not sr_no or not sr_no.isdigit():
            continue
        if sector_raw in ("Sovereign", "Others"):
            continue

        sector = SECTOR_NAME_MAP.get(sector_raw, sector_raw)

        # Extract net equity investment for both fortnights
        # Column positions (0-indexed from cleaned):
        # Index 2-13: AUC start (INR) - Equity at index 2
        # Index 14-25: AUC start (USD)
        # Index 26-37: Net Inv fortnight 1 (INR) - Equity at index 26
        # Index 38-49: Net Inv fortnight 1 (USD)
        # Index 50-61: Net Inv fortnight 2 (INR) - Equity at index 50
        # Index 62-73: Net Inv fortnight 2 (USD)
        # Index 74-85: AUC end (INR) - Equity at index 74
        # Index 86-97: AUC end (USD)

        # But the actual column count varies. Let's use a simpler approach:
        # Each group has 12 columns (Equity, Debt GL, Debt VRR, Debt-FAR, Hybrid,
        # Equity(MF), Debt GL(MF), Hybrid(MF), Solution, Other, AIF, Total)
        # So total = 2 + 8 groups * 12 = 98 columns

        # Let me count from the data we've seen:
        # Row 4 example has 98 cells after Sr.No and Sector
        # Groups of 12: AUC_INR(12) + AUC_USD(12) + Net1_INR(12) + Net1_USD(12) +
        #               Net2_INR(12) + Net2_USD(12) + AUC_end_INR(12) + AUC_end_USD(12)
        # Net1 equity = cell at position 2 + 12 + 12 = 26
        # Net2 equity = cell at position 2 + 12 + 12 + 12 + 12 = 50

        if len(cleaned) >= 52:
            net1_equity = _parse_number(cleaned[26])
            net2_equity = _parse_number(cleaned[50])
            total_net = net1_equity + net2_equity
            auc_equity = _parse_number(cleaned[74]) if len(cleaned) > 74 else 0
            result[sector] = {
                "net_fortnight_1": net1_equity,
                "net_fortnight_2": net2_equity,
                "net_total": total_net,
                "auc_equity": auc_equity,
            }

    return result if result else None


def fetch_fpi_sector_flows(months_back: int = 6, force_refresh: bool = False) -> pd.DataFrame:
    """Fetch FPI sector-wise flows for the last N months.

    Returns DataFrame with columns: date, sector, net_equity_cr, auc_equity_cr
    Each row = one sector for one fortnightly report.
    Cached to disk.
    """
    os.makedirs(CACHE_DIR, exist_ok=True)

    # Check cache
    if not force_refresh and FPI_CACHE_FILE.exists():
        try:
            age_hours = (datetime.datetime.now().timestamp() - FPI_CACHE_FILE.stat().st_mtime) / 3600
            if age_hours < FPI_CACHE_TTL_HOURS:
                with open(FPI_CACHE_FILE, "rb") as f:
                    cached = pickle.load(f)
                if isinstance(cached, pd.DataFrame) and not cached.empty:
                    logger.info("FPI sector flows loaded from cache (%d rows)", len(cached))
                    return cached
        except Exception:
            pass

    report_dates = _generate_report_dates(months_back)
    all_rows = []

    for rd in report_dates:
        data = _scrape_single_report(rd)
        if data:
            for sector, vals in data.items():
                all_rows.append({
                    "date": rd,
                    "sector": sector,
                    "net_equity_cr": vals["net_total"],
                    "net_f1_cr": vals["net_fortnight_1"],
                    "net_f2_cr": vals["net_fortnight_2"],
                    "auc_equity_cr": vals["auc_equity"],
                })
            logger.info("FPI report %s: %d sectors parsed", rd, len(data))
        else:
            logger.debug("FPI report %s: no data", rd)

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    df["date"] = pd.to_datetime(df["date"])

    # Save cache
    try:
        with open(FPI_CACHE_FILE, "wb") as f:
            pickle.dump(df, f)
    except Exception as e:
        logger.warning("Failed to cache FPI data: %s", e)

    return df


def get_fpi_sector_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize FPI flows by sector: cumulative net, latest AUC, trend direction.

    Returns DataFrame sorted by cumulative net equity flow.
    """
    if df.empty:
        return pd.DataFrame()

    summary = []
    for sector in df["sector"].unique():
        sdf = df[df["sector"] == sector].sort_values("date")
        cum_net = sdf["net_equity_cr"].sum()
        latest_auc = sdf["auc_equity_cr"].iloc[-1] if not sdf.empty else 0
        # Last 3 reports trend
        last3 = sdf["net_equity_cr"].tail(3)
        trend = "buying" if last3.mean() > 0 else "selling"
        # Acceleration: is the most recent stronger than prior?
        if len(last3) >= 2:
            if last3.iloc[-1] > last3.iloc[-2]:
                trend += " (accelerating)"
            elif last3.iloc[-1] < last3.iloc[-2]:
                trend += " (decelerating)"
        summary.append({
            "sector": sector,
            "cum_net_cr": cum_net,
            "latest_auc_cr": latest_auc,
            "trend": trend,
            "n_reports": len(sdf),
        })

    return pd.DataFrame(summary).sort_values("cum_net_cr", ascending=False)
