"""
Layer 5: Fundamental Veto + Position Sizing + Final Output
Fetches basic fundamentals via yfinance and vetoes stocks that fail quality checks.
"""
import yfinance as yf
import pandas as pd
import numpy as np
from config import FUNDAMENTAL_CONFIG, POSITION_CONFIG, PROFIT_CONFIG


def fetch_fundamentals(ticker: str) -> dict:
    """
    Fetch fundamental data for a single stock via yfinance.
    Returns dict with key financial metrics.
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info

        return {
            "ticker": ticker,
            "company_name": info.get("longName", info.get("shortName", ticker)),
            "sector": info.get("sector", "N/A"),
            "industry": info.get("industry", "N/A"),
            "market_cap": info.get("marketCap", 0),
            "pe_ratio": info.get("trailingPE", None),
            "forward_pe": info.get("forwardPE", None),
            "peg_ratio": info.get("pegRatio", None),
            "roe": info.get("returnOnEquity", None),  # as decimal
            "debt_equity": info.get("debtToEquity", None),  # as percentage
            "revenue_growth": info.get("revenueGrowth", None),  # as decimal
            "earnings_growth": info.get("earningsGrowth", None),  # as decimal
            "profit_margin": info.get("profitMargins", None),
            "current_ratio": info.get("currentRatio", None),
            "book_value": info.get("bookValue", None),
            "dividend_yield": info.get("dividendYield", None),
            "data_available": True,
        }
    except Exception as e:
        return {
            "ticker": ticker,
            "data_available": False,
            "error": str(e),
        }


def apply_fundamental_veto(fundamentals: dict) -> dict:
    """
    Apply fundamental quality checks. Returns veto decision with reasons.

    The veto is intentionally lenient — we only reject clearly bad companies.
    A stock that passes is "not fundamentally broken", not "fundamentally great".
    """
    cfg = FUNDAMENTAL_CONFIG
    reasons = []
    passes = True

    if not fundamentals.get("data_available"):
        return {
            "passes": True,  # don't veto if we can't get data; flag it instead
            "reasons": ["Fundamental data unavailable — manual check needed"],
            "confidence": "low",
        }

    # EPS / Earnings Growth
    eg = fundamentals.get("earnings_growth")
    if eg is not None:
        eg_pct = eg * 100
        if eg_pct < cfg["min_eps_growth_yoy_pct"]:
            reasons.append(f"Earnings growth {eg_pct:.1f}% < {cfg['min_eps_growth_yoy_pct']}% threshold")
            if eg_pct < 0:
                passes = False  # negative earnings growth is a hard veto

    # Revenue Growth
    rg = fundamentals.get("revenue_growth")
    if rg is not None:
        rg_pct = rg * 100
        if rg_pct < cfg["min_revenue_growth_yoy_pct"]:
            reasons.append(f"Revenue growth {rg_pct:.1f}% < {cfg['min_revenue_growth_yoy_pct']}% threshold")
            if rg_pct < 0:
                passes = False  # declining revenue is a hard veto

    # ROE
    roe = fundamentals.get("roe")
    if roe is not None:
        roe_pct = roe * 100
        if roe_pct < cfg["min_roe_pct"]:
            reasons.append(f"ROE {roe_pct:.1f}% < {cfg['min_roe_pct']}% threshold")
            if roe_pct < 5:
                passes = False  # very low ROE is a hard veto

    # Debt/Equity
    de = fundamentals.get("debt_equity")
    if de is not None:
        de_ratio = de / 100  # yfinance returns as percentage
        if de_ratio > cfg["max_debt_equity"]:
            reasons.append(f"D/E {de_ratio:.2f} > {cfg['max_debt_equity']} threshold")
            if de_ratio > 3:
                passes = False  # very high debt is a hard veto

    # PEG Ratio
    peg = fundamentals.get("peg_ratio")
    if peg is not None:
        if peg > cfg["max_peg_ratio"]:
            reasons.append(f"PEG {peg:.2f} > {cfg['max_peg_ratio']} (expensive relative to growth)")
        if peg < 0:
            reasons.append("PEG negative (declining earnings)")
            passes = False

    confidence = "high" if not reasons else ("medium" if passes else "low")

    return {
        "passes": passes,
        "reasons": reasons if reasons else ["All fundamental checks passed"],
        "confidence": confidence,
    }


def compute_position_size(
    entry_price: float,
    stop_loss: float,
    regime_posture: dict,
    total_capital: float = None,
    current_open_risk_pct: float = 0,
) -> dict:
    """
    Compute position size using fixed fractional risk model.

    Position Size = (Capital * Risk%) / Risk per Share
    """
    if total_capital is None:
        total_capital = POSITION_CONFIG["total_capital"]

    risk_per_trade_pct = regime_posture["risk_per_trade_pct"]
    max_capital_pct = regime_posture["max_capital_pct"]

    risk_per_share = entry_price - stop_loss
    if risk_per_share <= 0:
        return {"shares": 0, "error": "Stop loss >= entry price"}

    # Capital allocated based on regime
    available_capital = total_capital * (max_capital_pct / 100)

    # Risk amount for this trade
    risk_amount = total_capital * (risk_per_trade_pct / 100)

    # Check portfolio-level risk limit
    remaining_risk = POSITION_CONFIG["max_portfolio_risk_pct"] - current_open_risk_pct
    if remaining_risk <= 0:
        return {
            "shares": 0,
            "error": f"Portfolio risk limit reached ({current_open_risk_pct:.1f}%)",
        }
    risk_amount = min(risk_amount, total_capital * remaining_risk / 100)

    # Position size from risk
    shares = int(risk_amount / risk_per_share)

    # Cap by max single position
    max_position_value = total_capital * (POSITION_CONFIG["max_single_position_pct"] / 100)
    max_shares_by_value = int(max_position_value / entry_price)
    shares = min(shares, max_shares_by_value)

    # Cap by available capital
    max_shares_by_capital = int(available_capital / entry_price)
    shares = min(shares, max_shares_by_capital)

    position_value = shares * entry_price
    position_risk = shares * risk_per_share
    position_risk_pct = position_risk / total_capital * 100

    return {
        "shares": shares,
        "entry_price": entry_price,
        "stop_loss": stop_loss,
        "risk_per_share": round(risk_per_share, 2),
        "position_value": round(position_value, 2),
        "position_pct_of_capital": round(position_value / total_capital * 100, 2),
        "risk_amount": round(position_risk, 2),
        "risk_pct_of_capital": round(position_risk_pct, 2),
        "regime": regime_posture["label"],
    }


def compute_profit_targets(entry_price: float, stop_loss: float) -> dict:
    """Compute profit-taking levels and reward/risk ratio."""
    cfg = PROFIT_CONFIG
    risk = entry_price - stop_loss

    first_target = round(entry_price * (1 + cfg["first_target_gain_pct"] / 100), 2)
    r_multiple_at_target = round((first_target - entry_price) / risk, 1)

    return {
        "first_target": first_target,
        "first_target_gain_pct": cfg["first_target_gain_pct"],
        "partial_sell_pct": cfg["partial_sell_pct"],
        "reward_risk_ratio": r_multiple_at_target,
        "hold_if_fast_gain_weeks": cfg["hold_min_weeks_if_fast"],
    }


def generate_final_watchlist(
    candidates: list[dict],
    regime: dict,
    total_capital: float = None,
) -> list[dict]:
    """
    Apply fundamental veto and position sizing to all Stage 2 candidates.
    Returns the final actionable watchlist.
    """
    if total_capital is None:
        total_capital = POSITION_CONFIG["total_capital"]

    posture = regime["posture"]
    max_new = posture["max_new_positions"]
    current_risk = 0  # track cumulative risk

    watchlist = []

    for candidate in candidates:
        ticker = candidate["ticker"]

        # Fetch fundamentals
        print(f"  Checking fundamentals for {ticker}...")
        fundamentals = fetch_fundamentals(ticker)
        veto = apply_fundamental_veto(fundamentals)

        candidate["fundamentals"] = fundamentals
        candidate["veto"] = veto

        # Informational flag instead of hard veto
        if veto["passes"]:
            candidate["fundamental_flag"] = "CLEAN"
            candidate["fundamental_reasons"] = []
        else:
            candidate["fundamental_flag"] = "CAUTION"
            candidate["fundamental_reasons"] = veto.get("reasons", [])

        # Position sizing for all stocks with breakout entry (no hard veto gate)
        entry_setup = candidate.get("entry_setup")
        if entry_setup:
            position = compute_position_size(
                entry_price=entry_setup["entry_price"],
                stop_loss=entry_setup["effective_stop"],
                regime_posture=posture,
                total_capital=total_capital,
                current_open_risk_pct=current_risk,
            )
            targets = compute_profit_targets(
                entry_setup["entry_price"],
                entry_setup["effective_stop"],
            )

            candidate["position"] = position
            candidate["targets"] = targets
            current_risk += position.get("risk_pct_of_capital", 0)

            if len([w for w in watchlist if w.get("action") == "BUY"]) < max_new:
                candidate["action"] = "BUY"
            else:
                candidate["action"] = "WATCHLIST"
        else:
            candidate["action"] = "WATCH"

        watchlist.append(candidate)

    return watchlist


def print_final_watchlist(watchlist: list[dict], regime: dict) -> None:
    """Pretty-print the final actionable watchlist."""
    print("\n" + "=" * 110)
    print("  FINAL OUTPUT: ACTIONABLE WATCHLIST")
    print("=" * 110)
    print(f"\n  Market Regime: {regime['label']} | "
          f"Max New Positions: {regime['posture']['max_new_positions']} | "
          f"Risk/Trade: {regime['posture']['risk_per_trade_pct']}%")

    buys = [w for w in watchlist if w.get("action") == "BUY"]
    watches = [w for w in watchlist if w.get("action") in ("WATCH", "WATCHLIST")]
    cautions = [w for w in watchlist if w.get("fundamental_flag") == "CAUTION"]

    if buys:
        print(f"\n  BUY SIGNALS ({len(buys)}):")
        print(
            f"  {'Ticker':<14} {'Entry':>8} {'Stop':>8} {'Risk%':>6} "
            f"{'Shares':>7} {'Value':>10} {'Target':>8} {'R:R':>5} "
            f"{'Fundamentals'}"
        )
        print("  " + "-" * 100)

        for b in buys:
            es = b.get("entry_setup", {})
            pos = b.get("position", {})
            tgt = b.get("targets", {})
            fund_flag = b.get("fundamental_flag", "CLEAN")
            fund_status = "OK" if fund_flag == "CLEAN" else f"CAUTION"
            print(
                f"  {b['ticker']:<14} "
                f"{es.get('entry_price', ''):>8} "
                f"{es.get('effective_stop', ''):>8} "
                f"{es.get('risk_pct', ''):>5}% "
                f"{pos.get('shares', ''):>7} "
                f"Rs{pos.get('position_value', 0):>9,.0f} "
                f"{tgt.get('first_target', ''):>8} "
                f"{tgt.get('reward_risk_ratio', ''):>4}R "
                f"{fund_status}"
            )

    if watches:
        print(f"\n  WATCHLIST ({len(watches)}):")
        for w in watches:
            reason = "Awaiting breakout" if w.get("action") == "WATCH" else "Max positions reached"
            flag = w.get("fundamental_flag", "CLEAN")
            flag_str = f" [{flag}]" if flag == "CAUTION" else ""
            print(f"  {w['ticker']:<14} Stage 2 | {reason} | RS: {w.get('rs_vs_nifty', 0):.1f}{flag_str}")

    if cautions:
        print(f"\n  FUNDAMENTAL CAUTIONS ({len(cautions)}):")
        for c in cautions:
            reasons = c.get("fundamental_reasons", [])
            print(f"  {c['ticker']:<14} {'; '.join(reasons[:2])}")

    print()
