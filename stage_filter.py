"""
Layer 4: Weinstein Stage Analysis + Entry Filter
Classifies stocks into Stages 1-4, detects base breakouts, counts bases.
"""
import pandas as pd
import numpy as np
from config import STAGE_CONFIG, STOP_CONFIG
from data_fetcher import compute_atr


def classify_stage(df: pd.DataFrame) -> dict:
    """
    Classify a stock into Weinstein Stage 1-4.

    Stage 1 (Basing): Price sideways around flat 150 MA
    Stage 2 (Advancing): Price > rising 150 MA, MAs aligned bullishly
    Stage 3 (Topping): Price churning around flattening 150 MA
    Stage 4 (Declining): Price < falling 150 MA

    Returns dict with stage number, confidence, and details.
    """
    cfg = STAGE_CONFIG

    if len(df) < cfg["ma_long"] + 30:
        return {"stage": 0, "confidence": 0, "detail": "Insufficient data"}

    close = df["Close"]
    ma50 = close.rolling(cfg["ma_short"]).mean()
    ma150 = close.rolling(cfg["ma_mid"]).mean()
    ma200 = close.rolling(cfg["ma_long"]).mean()

    latest = close.iloc[-1]
    latest_ma50 = ma50.iloc[-1]
    latest_ma150 = ma150.iloc[-1]
    latest_ma200 = ma200.iloc[-1]

    # MA slopes (over last 20 days)
    ma150_slope = (ma150.iloc[-1] - ma150.iloc[-20]) / ma150.iloc[-20] * 100
    ma200_slope = (ma200.iloc[-1] - ma200.iloc[-20]) / ma200.iloc[-20] * 100

    # 52-week high/low
    high_52w = df["High"].tail(252).max() if len(df) >= 252 else df["High"].max()
    low_52w = df["Low"].tail(252).min() if len(df) >= 252 else df["Low"].min()
    pct_above_52w_low = (latest / low_52w - 1) * 100
    pct_below_52w_high = (1 - latest / high_52w) * 100

    # ── Stage 2 Criteria (Weinstein + Minervini) ────────────────
    s2_checks = {
        "price_above_ma150": latest > latest_ma150,
        "price_above_ma200": latest > latest_ma200,
        "ma50_above_ma150": latest_ma50 > latest_ma150,
        "ma150_above_ma200": latest_ma150 > latest_ma200,
        "ma200_rising": ma200_slope > 0,
        "above_52w_low_30pct": pct_above_52w_low >= cfg["price_above_52w_low_pct"],
        "within_52w_high_25pct": pct_below_52w_high <= cfg["price_within_52w_high_pct"],
    }
    s2_score = sum(s2_checks.values())

    # ── Stage Classification ────────────────────────────────────
    if s2_score >= 6:
        stage = 2
        confidence = s2_score / 7
    elif s2_score >= 4 and ma150_slope > -0.5:
        # Transitioning — could be late Stage 1 entering Stage 2
        # or early Stage 3
        if latest > latest_ma150 and ma150_slope > 0:
            stage = 2
            confidence = s2_score / 7
        else:
            stage = 1  # basing
            confidence = 0.6
    elif latest < latest_ma150 and latest < latest_ma200 and ma150_slope < 0:
        stage = 4  # declining
        confidence = 0.8 if ma200_slope < 0 else 0.6
    elif latest > latest_ma200 and ma150_slope < 0.3 and ma150_slope > -0.3:
        stage = 3  # topping
        confidence = 0.6
    else:
        # Default to stage 1 (basing) if unclear
        stage = 1
        confidence = 0.4

    return {
        "stage": stage,
        "confidence": round(confidence, 2),
        "s2_checks": s2_checks,
        "s2_score": s2_score,
        "ma_slopes": {"ma150": round(ma150_slope, 2), "ma200": round(ma200_slope, 2)},
        "detail": f"Stage {stage} (confidence {confidence:.0%})",
    }


def detect_bases(df: pd.DataFrame) -> list[dict]:
    """
    Detect consolidation bases (flat price ranges with declining volume).

    A base is identified as a period where:
    - Price range (high-low) is within X% (the base depth)
    - Duration is between min and max days
    - Volume tends to contract

    Returns list of bases with their properties.
    """
    cfg = STAGE_CONFIG
    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    volume = df["Volume"]

    bases = []
    i = len(df) - 1  # start from most recent

    while i > cfg["base_min_days"]:
        # Look backwards to find a consolidation range
        base_high = high.iloc[i]
        base_low = low.iloc[i]

        j = i - 1
        while j >= max(0, i - cfg["base_max_days"]):
            # Expand the range
            base_high = max(base_high, high.iloc[j])
            base_low = min(base_low, low.iloc[j])

            depth = (base_high - base_low) / base_high * 100
            if depth > cfg["base_max_depth_pct"]:
                break

            j -= 1

        base_length = i - j
        depth = (base_high - base_low) / base_high * 100

        if (
            cfg["base_min_days"] <= base_length <= cfg["base_max_days"]
            and depth <= cfg["base_max_depth_pct"]
        ):
            # Check for volume contraction within base
            base_vol = volume.iloc[j:i+1]
            first_half_vol = base_vol.iloc[:len(base_vol)//2].mean()
            second_half_vol = base_vol.iloc[len(base_vol)//2:].mean()
            vol_contracting = second_half_vol < first_half_vol

            bases.append({
                "start_idx": j,
                "end_idx": i,
                "start_date": str(df.index[j].date()) if hasattr(df.index[j], 'date') else str(df.index[j]),
                "end_date": str(df.index[i].date()) if hasattr(df.index[i], 'date') else str(df.index[i]),
                "length_days": base_length,
                "base_high": round(float(base_high), 2),
                "base_low": round(float(base_low), 2),
                "depth_pct": round(depth, 1),
                "volume_contracting": vol_contracting,
            })

            # Jump past this base to look for earlier ones
            i = j - 5
        else:
            i -= 10  # step back and try again

    bases.reverse()  # chronological order
    return bases


def count_bases_in_stage2(df: pd.DataFrame, bases: list[dict]) -> int:
    """Count how many base breakouts have occurred in the current Stage 2 advance."""
    if not bases:
        return 0

    ma150 = df["Close"].rolling(150).mean()
    count = 0
    for base in bases:
        idx = base["end_idx"]
        if idx < len(ma150) and df["Close"].iloc[idx] > ma150.iloc[idx]:
            count += 1
    return count


def detect_breakout(df: pd.DataFrame, bases: list[dict]) -> dict | None:
    """
    Check if the most recent price action is a breakout from the latest base.

    Breakout = price closes above base_high on volume >= 1.5x average.
    """
    cfg = STAGE_CONFIG
    if not bases:
        return None

    latest_base = bases[-1]
    base_high = latest_base["base_high"]

    # Check last 5 trading days for a breakout
    recent = df.tail(5)
    avg_vol = df["Volume"].tail(50).mean()

    for idx in range(len(recent)):
        row = recent.iloc[idx]
        if (
            row["Close"] > base_high
            and row["Volume"] >= avg_vol * cfg["volume_surge_multiple"]
        ):
            return {
                "breakout": True,
                "breakout_date": str(recent.index[idx].date()) if hasattr(recent.index[idx], 'date') else str(recent.index[idx]),
                "breakout_price": round(float(row["Close"]), 2),
                "base_high": base_high,
                "volume_ratio": round(float(row["Volume"] / avg_vol), 1),
                "base_depth_pct": latest_base["depth_pct"],
                "base_length_days": latest_base["length_days"],
            }

    return None


def detect_vcp(df: pd.DataFrame, base: dict) -> dict:
    """
    Detect Volatility Contraction Pattern within a base.
    VCP = successive tightenings of the price range.
    """
    cfg = STAGE_CONFIG
    start = base["start_idx"]
    end = base["end_idx"]

    if end - start < 20:
        return {"is_vcp": False, "contractions": 0}

    # Divide the base into segments and measure range of each
    segment_len = max(5, (end - start) // 4)
    ranges = []
    for k in range(start, end, segment_len):
        segment = df.iloc[k:k + segment_len]
        if len(segment) > 2:
            r = (segment["High"].max() - segment["Low"].min()) / segment["High"].max() * 100
            ranges.append(r)

    if len(ranges) < 2:
        return {"is_vcp": False, "contractions": 0}

    # Count contractions (each range smaller than the previous)
    contractions = 0
    for k in range(1, len(ranges)):
        if ranges[k] < ranges[k - 1] * cfg["vcp_contraction_ratio"]:
            contractions += 1

    return {
        "is_vcp": contractions >= cfg["vcp_contractions_min"],
        "contractions": contractions,
        "ranges": [round(r, 1) for r in ranges],
    }


def compute_entry_and_stop(
    df: pd.DataFrame,
    breakout: dict,
    base: dict,
) -> dict:
    """
    Compute entry price, initial stop loss, and risk per share.
    """
    entry_price = breakout["breakout_price"]

    # Initial stop: just below base low
    buffer_pct = STOP_CONFIG["initial_stop_buffer_pct"]
    stop_loss = round(base["base_low"] * (1 - buffer_pct / 100), 2)

    # ATR-based stop alternative
    atr = compute_atr(df, STOP_CONFIG["atr_period"])
    atr_val = float(atr.iloc[-1]) if len(atr.dropna()) > 0 else 0
    atr_stop = round(entry_price - STOP_CONFIG["atr_multiple"] * atr_val, 2)

    # Use the tighter of the two (higher stop = less risk)
    effective_stop = max(stop_loss, atr_stop)
    risk_per_share = round(entry_price - effective_stop, 2)
    risk_pct = round(risk_per_share / entry_price * 100, 2)

    return {
        "entry_price": entry_price,
        "base_stop": stop_loss,
        "atr_stop": atr_stop,
        "effective_stop": effective_stop,
        "risk_per_share": risk_per_share,
        "risk_pct": risk_pct,
        "atr": round(atr_val, 2),
    }


def analyze_stock_stage(df: pd.DataFrame, ticker: str) -> dict:
    """
    Full stage analysis for a single stock.
    Returns comprehensive analysis dict.
    """
    stage = classify_stage(df)
    bases = detect_bases(df)
    base_count = count_bases_in_stage2(df, bases)
    breakout = detect_breakout(df, bases) if bases else None

    result = {
        "ticker": ticker,
        "stage": stage,
        "bases_found": len(bases),
        "base_count_in_stage2": base_count,
        "breakout": breakout,
        "entry_setup": None,
        "vcp": None,
    }

    # Only generate entry if Stage 2 with a valid breakout
    if stage["stage"] == 2 and breakout and base_count <= STAGE_CONFIG["max_base_count"]:
        latest_base = bases[-1]
        vcp = detect_vcp(df, latest_base)
        entry_stop = compute_entry_and_stop(df, breakout, latest_base)

        result["entry_setup"] = entry_stop
        result["vcp"] = vcp

    return result


def filter_stage2_candidates(
    stock_data: dict[str, pd.DataFrame],
    screened_stocks: list[dict],
) -> list[dict]:
    """
    Apply stage analysis to all screened stocks.
    Returns only Stage 2 stocks with valid breakout setups.
    """
    candidates = []

    for stock_info in screened_stocks:
        ticker = stock_info["ticker"]
        if ticker not in stock_data:
            continue

        df = stock_data[ticker]
        analysis = analyze_stock_stage(df, ticker)

        if (
            analysis["stage"]["stage"] == 2
            and analysis["stage"]["confidence"] >= 0.7
            and analysis["base_count_in_stage2"] <= STAGE_CONFIG["max_base_count"]
        ):
            # Merge screening data with stage analysis
            combined = {**stock_info, **analysis}
            candidates.append(combined)

    # Sort by: has breakout first, then by leadership score
    candidates.sort(
        key=lambda x: (
            1 if x["breakout"] else 0,
            x.get("leadership_score", 0),
        ),
        reverse=True,
    )
    return candidates


def print_stage_results(candidates: list[dict], max_show: int = 20) -> None:
    """Pretty-print Stage 2 candidates."""
    print("\n" + "=" * 100)
    print("  LAYER 4: STAGE 2 CANDIDATES")
    print("=" * 100)

    breakout_candidates = [c for c in candidates if c.get("breakout")]
    watchlist_candidates = [c for c in candidates if not c.get("breakout")]

    if breakout_candidates:
        print(f"\n  ACTIVE BREAKOUTS ({len(breakout_candidates)}):")
        print(
            f"  {'Ticker':<14} {'Stage':>6} {'Bases':>6} {'Entry':>8} {'Stop':>8} "
            f"{'Risk%':>6} {'VolRatio':>9} {'VCP':>5} {'Score':>7}"
        )
        print("  " + "-" * 87)

        for c in breakout_candidates[:max_show]:
            bo = c["breakout"]
            es = c.get("entry_setup", {})
            vcp = c.get("vcp", {})
            print(
                f"  {c['ticker']:<14} "
                f"{'S2':>6} "
                f"{c['base_count_in_stage2']:>6} "
                f"{es.get('entry_price', 'N/A'):>8} "
                f"{es.get('effective_stop', 'N/A'):>8} "
                f"{es.get('risk_pct', 'N/A'):>5}% "
                f"{bo.get('volume_ratio', 'N/A'):>8}x "
                f"{'Y' if vcp and vcp.get('is_vcp') else 'N':>5} "
                f"{c.get('leadership_score', 0):>7.2f}"
            )

    if watchlist_candidates:
        print(f"\n  WATCHLIST — Stage 2, no breakout yet ({len(watchlist_candidates)}):")
        print(f"  {'Ticker':<14} {'Conf':>6} {'Bases':>6} {'%frHigh':>8} {'RS/Nifty':>9} {'Score':>7}")
        print("  " + "-" * 58)

        for c in watchlist_candidates[:max_show]:
            print(
                f"  {c['ticker']:<14} "
                f"{c['stage']['confidence']:>5.0%} "
                f"{c['base_count_in_stage2']:>6} "
                f"{c.get('dist_from_high_pct', 0):>7.1f}% "
                f"{c.get('rs_vs_nifty', 0):>9.2f} "
                f"{c.get('leadership_score', 0):>7.2f}"
            )

    print(f"\n  Total Stage 2 candidates: {len(candidates)}")
    print()
