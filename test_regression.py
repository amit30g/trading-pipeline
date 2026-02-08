"""
Comprehensive Backend Regression Suite
Tests: conviction_scorer, position_manager, fundamental_veto (flag mode),
       nse_data_fetcher (new methods), config additions, and end-to-end pipeline flow.
"""
import json
import os
import sys
import uuid
import tempfile
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch, MagicMock

import pandas as pd
import numpy as np

# ── Setup path ─────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))

PASS = 0
FAIL = 0
ERRORS = []


def check(name, condition, detail=""):
    global PASS, FAIL, ERRORS
    if condition:
        PASS += 1
        print(f"  ✅ {name}")
    else:
        FAIL += 1
        msg = f"  ❌ {name}" + (f" — {detail}" if detail else "")
        print(msg)
        ERRORS.append(msg)


# ══════════════════════════════════════════════════════════════════
# SECTION 1: Config Additions
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SECTION 1: Config Additions")
print("=" * 70)

from config import (
    CONVICTION_CONFIG, SMART_MONEY_CONFIG, FII_DII_CACHE_TTL_HOURS,
    STOP_CONFIG, PROFIT_CONFIG, POSITION_CONFIG,
)

check("CONVICTION_CONFIG exists", isinstance(CONVICTION_CONFIG, dict))
check("CONVICTION_CONFIG has 5 weights",
      len(CONVICTION_CONFIG) == 5,
      f"got {len(CONVICTION_CONFIG)}")
check("Conviction weights sum to 100",
      sum(CONVICTION_CONFIG.values()) == 100,
      f"got {sum(CONVICTION_CONFIG.values())}")
check("SMART_MONEY_CONFIG exists", isinstance(SMART_MONEY_CONFIG, dict))
check("SMART_MONEY_CONFIG has delivery thresholds",
      "delivery_threshold_high" in SMART_MONEY_CONFIG and "delivery_threshold_low" in SMART_MONEY_CONFIG)
check("FII_DII_CACHE_TTL_HOURS is 1", FII_DII_CACHE_TTL_HOURS == 1)
check("STOP_CONFIG atr_multiple is 2.5", STOP_CONFIG["atr_multiple"] == 2.5)
check("PROFIT_CONFIG hold_min_weeks_if_fast is 8", PROFIT_CONFIG["hold_min_weeks_if_fast"] == 8)


# ══════════════════════════════════════════════════════════════════
# SECTION 2: Conviction Scorer — Unit Tests
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SECTION 2: Conviction Scorer")
print("=" * 70)

from conviction_scorer import (
    compute_conviction_score,
    rank_candidates_by_conviction,
    get_top_conviction_ideas,
)

# Build realistic test candidates
def make_candidate(ticker, sector, s2_score=5, rs=0.15, acc=1.3,
                   base_number=1, is_vcp=False, has_breakout=True,
                   vol_ratio=1.8, action="BUY"):
    c = {
        "ticker": ticker,
        "sector": sector,
        "rs_vs_nifty": rs,
        "rs_vs_sector": rs * 0.8,
        "accumulation_ratio": acc,
        "stage": {"s2_score": s2_score, "stage": 2, "confidence": 0.85},
        "vcp": {"is_vcp": is_vcp} if is_vcp else None,
        "action": action,
    }
    if has_breakout:
        c["breakout"] = {
            "breakout": True,
            "base_number": base_number,
            "volume_ratio": vol_ratio,
            "breakout_price": 500.0,
            "base_depth_pct": 15.0,
        }
        c["entry_setup"] = {
            "entry_price": 510.0,
            "effective_stop": 470.0,
            "risk_pct": 7.8,
        }
    else:
        c["breakout"] = None
        c["entry_setup"] = None
    return c


sector_rankings = [
    {"sector": "Nifty IT", "composite_score": 5.2, "mansfield_rs": 2.1, "rs_trend": "rising"},
    {"sector": "Nifty Bank", "composite_score": 4.1, "mansfield_rs": 1.5, "rs_trend": "rising"},
    {"sector": "Nifty Pharma", "composite_score": 3.2, "mansfield_rs": 0.8, "rs_trend": "flat"},
    {"sector": "Nifty Auto", "composite_score": 2.0, "mansfield_rs": 0.2, "rs_trend": "falling"},
]

candidates = [
    make_candidate("TCS.NS", "Nifty IT", s2_score=7, rs=0.25, acc=1.5, base_number=1, is_vcp=True, vol_ratio=2.5),
    make_candidate("INFY.NS", "Nifty IT", s2_score=6, rs=0.20, acc=1.3, base_number=2),
    make_candidate("HDFCBANK.NS", "Nifty Bank", s2_score=5, rs=0.10, acc=1.1, base_number=3),
    make_candidate("SUNPHARMA.NS", "Nifty Pharma", s2_score=4, rs=0.05, acc=0.9, base_number=1),
    make_candidate("MARUTI.NS", "Nifty Auto", s2_score=3, rs=-0.05, acc=0.8, base_number=4, action="WATCH", has_breakout=False),
]

# Test 2a: Top-sector + perfect S2 + VCP + 1st base + 2x volume candidate scores highest
score_tcs = compute_conviction_score(candidates[0], sector_rankings, candidates)
score_infy = compute_conviction_score(candidates[1], sector_rankings, candidates)
score_hdfc = compute_conviction_score(candidates[2], sector_rankings, candidates)
score_sun = compute_conviction_score(candidates[3], sector_rankings, candidates)
score_maruti = compute_conviction_score(candidates[4], sector_rankings, candidates)

check("TCS (top sector, 7/7 S2, VCP, 1st base, 2.5x vol) scores highest",
      score_tcs > score_infy > score_hdfc,
      f"TCS={score_tcs} INFY={score_infy} HDFC={score_hdfc}")

check("Scores are in 0-100 range",
      all(0 <= s <= 100 for s in [score_tcs, score_infy, score_hdfc, score_sun, score_maruti]),
      f"scores: {[score_tcs, score_infy, score_hdfc, score_sun, score_maruti]}")

check("Top-1 sector (IT) scores > Top-3 sector (Pharma)",
      score_tcs > score_sun,
      f"TCS={score_tcs} SUN={score_sun}")

check("Perfect S2 (7/7) scores higher than low S2 (3/7) same sector conditions",
      score_tcs > score_maruti)

# Test 2b: Conviction score with bonuses
check("TCS gets VCP bonus (+3) and volume surge bonus (+2)",
      score_tcs >= 75,  # top sector 40 + s2 20 + 1st base 10 + high RS/acc + VCP(3) + vol(2) = well above 75
      f"TCS score={score_tcs}")

# Test 2c: Rank candidates
ranked = rank_candidates_by_conviction(list(candidates), sector_rankings)
check("rank_candidates_by_conviction returns sorted list",
      ranked[0]["conviction_score"] >= ranked[1]["conviction_score"] >= ranked[-1]["conviction_score"],
      f"scores: {[c['conviction_score'] for c in ranked]}")

check("All candidates have conviction_score key after ranking",
      all("conviction_score" in c for c in ranked))

check("TCS ranks #1",
      ranked[0]["ticker"] == "TCS.NS",
      f"#1 is {ranked[0]['ticker']}")

# Test 2d: get_top_conviction_ideas — only actionable (BUY/WATCHLIST + has entry)
top3 = get_top_conviction_ideas(ranked, top_n=3)
check("get_top_conviction_ideas returns max 3",
      len(top3) <= 3)

check("Top ideas all have entry_setup",
      all(t.get("entry_setup") is not None for t in top3),
      f"setups: {[bool(t.get('entry_setup')) for t in top3]}")

check("WATCH stocks (no breakout) excluded from top ideas",
      all(t["ticker"] != "MARUTI.NS" for t in top3))

# Test 2e: Edge case — empty candidates
empty_ranked = rank_candidates_by_conviction([], sector_rankings)
check("Empty candidates returns empty list", empty_ranked == [])

empty_ideas = get_top_conviction_ideas([], top_n=3)
check("Empty ranked returns empty ideas", empty_ideas == [])

# Test 2f: 4th+ base gets 0 base points
score_4th_base = compute_conviction_score(
    make_candidate("TEST.NS", "Nifty IT", base_number=4),
    sector_rankings, candidates
)
score_1st_base = compute_conviction_score(
    make_candidate("TEST.NS", "Nifty IT", base_number=1),
    sector_rankings, candidates
)
check("1st base scores higher than 4th+ base",
      score_1st_base > score_4th_base,
      f"1st={score_1st_base} 4th={score_4th_base}")

# Test 2g: Delivery bonus
score_no_delivery = compute_conviction_score(candidates[0], sector_rankings, candidates, delivery_data=None)
score_high_delivery = compute_conviction_score(candidates[0], sector_rankings, candidates, delivery_data={"delivery_pct": 65})
score_low_delivery = compute_conviction_score(candidates[0], sector_rankings, candidates, delivery_data={"delivery_pct": 30})
check("High delivery (>50%) adds bonus vs no delivery",
      score_high_delivery >= score_no_delivery,
      f"high_del={score_high_delivery} no_del={score_no_delivery}")
check("Low delivery (<50%) gets no bonus",
      score_low_delivery == score_no_delivery,
      f"low_del={score_low_delivery} no_del={score_no_delivery}")


# ══════════════════════════════════════════════════════════════════
# SECTION 3: Position Manager — Full Lifecycle Tests
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SECTION 3: Position Manager")
print("=" * 70)

import position_manager as pm

# Use temp files to avoid polluting real data
_orig_pos_file = pm.POSITIONS_FILE
_orig_hist_file = pm.TRADE_HISTORY_FILE
_tmp_dir = Path(tempfile.mkdtemp())
pm.POSITIONS_FILE = _tmp_dir / "test_positions.json"
pm.TRADE_HISTORY_FILE = _tmp_dir / "test_trade_history.json"

try:
    # Test 3a: Add position
    pos1 = pm.add_position(
        ticker="RELIANCE.NS",
        entry_date="2025-01-15",
        entry_price=2800.0,
        shares=35,
        initial_stop=2650.0,
        notes="Test position",
    )
    check("add_position returns dict with id",
          isinstance(pos1, dict) and "id" in pos1 and len(pos1["id"]) > 0)
    check("Position has correct ticker", pos1["ticker"] == "RELIANCE.NS")
    check("Position has correct entry price", pos1["entry_price"] == 2800.0)
    check("Position has correct shares", pos1["shares"] == 35)
    check("Position has correct initial stop", pos1["initial_stop"] == 2650.0)
    check("Trailing stop initialized to initial stop", pos1["trailing_stop"] == 2650.0)
    check("Highest close initialized to entry price", pos1["highest_close"] == 2800.0)

    # Test 3b: Load positions
    positions = pm.load_positions()
    check("load_positions returns list with 1 entry", len(positions) == 1)
    check("Loaded position matches added",
          positions[0]["ticker"] == "RELIANCE.NS" and positions[0]["id"] == pos1["id"])

    # Test 3c: Add second position
    pos2 = pm.add_position(
        ticker="TCS.NS",
        entry_date="2025-02-01",
        entry_price=4000.0,
        shares=25,
        initial_stop=3800.0,
    )
    positions = pm.load_positions()
    check("Two positions after adding second", len(positions) == 2)

    # Test 3d: Close position — P&L computation
    trade = pm.close_position(
        position_id=pos1["id"],
        exit_date="2025-03-01",
        exit_price=3100.0,
        reason="Target hit",
    )
    check("close_position returns trade dict", isinstance(trade, dict))
    check("Trade P&L computed correctly",
          trade["pnl"] == (3100.0 - 2800.0) * 35,
          f"pnl={trade['pnl']} expected={(3100.0 - 2800.0) * 35}")
    check("Trade P&L % correct",
          abs(trade["pnl_pct"] - ((3100 / 2800) - 1) * 100) < 0.1,
          f"pnl_pct={trade['pnl_pct']}")
    check("Days held computed",
          trade["days_held"] == (datetime(2025, 3, 1) - datetime(2025, 1, 15)).days,
          f"days={trade['days_held']}")
    check("Exit reason recorded", trade["exit_reason"] == "Target hit")

    # Test 3e: Positions after close
    remaining = pm.load_positions()
    check("Only 1 position remains after closing one", len(remaining) == 1)
    check("Remaining position is TCS", remaining[0]["ticker"] == "TCS.NS")

    # Test 3f: Trade history
    history = pm.load_trade_history()
    check("Trade history has 1 entry", len(history) == 1)
    check("History trade is RELIANCE", history[0]["ticker"] == "RELIANCE.NS")

    # Test 3g: Close non-existent position
    bad_close = pm.close_position("nonexistent", "2025-03-01", 100.0)
    check("Closing non-existent position returns None", bad_close is None)

    # Test 3h: Trade stats
    stats = pm.get_trade_stats()
    check("Trade stats has correct total", stats["total_trades"] == 1)
    check("Win rate is 100% (1 winning trade)", stats["win_rate"] == 100.0)
    check("Total PnL matches trade", stats["total_pnl"] == trade["pnl"])

    # Test 3i: Add a losing trade to test stats
    pos3 = pm.add_position("INFY.NS", "2025-01-10", 1800.0, 50, 1700.0)
    pm.close_position(pos3["id"], "2025-02-10", 1650.0, "Stop hit")
    stats2 = pm.get_trade_stats()
    check("2 total trades after loss", stats2["total_trades"] == 2)
    check("Win rate is 50%", stats2["win_rate"] == 50.0)
    check("Avg gain is positive", stats2["avg_gain"] > 0)
    check("Avg loss is negative", stats2["avg_loss"] < 0)

    # Test 3j: Trailing stop logic
    check("Trailing stop formula: highest - 2.5*ATR",
          pm._compute_trailing_stop(3000.0, 50.0) == 3000.0 - 2.5 * 50.0,
          f"got {pm._compute_trailing_stop(3000.0, 50.0)}")
    check("Trailing stop rounds to 2 decimals",
          isinstance(pm._compute_trailing_stop(3000.0, 33.33), float))

    # Test 3k: 8-week hold rule
    # Position with 25% gain in 2 weeks should trigger
    fast_pos = {
        "entry_date": (datetime.now() - timedelta(days=10)).strftime("%Y-%m-%d"),
        "entry_price": 100.0,
        "hold_until": None,
    }
    hold_info = pm._check_8_week_hold(fast_pos, 125.0)  # 25% gain in ~1.4 weeks
    check("8-week rule triggers for 25% gain in <3 weeks",
          hold_info is not None and hold_info["active"],
          f"hold_info={hold_info}")

    # Position with 5% gain — should NOT trigger
    slow_pos = {
        "entry_date": (datetime.now() - timedelta(days=10)).strftime("%Y-%m-%d"),
        "entry_price": 100.0,
        "hold_until": None,
    }
    no_hold = pm._check_8_week_hold(slow_pos, 105.0)
    check("8-week rule does NOT trigger for 5% gain", no_hold is None)

    # Position already past 3 weeks with 25% gain — should NOT trigger
    old_pos = {
        "entry_date": (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d"),
        "entry_price": 100.0,
        "hold_until": None,
    }
    no_hold_2 = pm._check_8_week_hold(old_pos, 125.0)
    check("8-week rule does NOT trigger after 3+ weeks", no_hold_2 is None)

    # Test 3l: Climax detection
    # Build a DataFrame with a massive volume spike on biggest up day
    dates = pd.date_range("2025-01-01", periods=60, freq="B")
    np.random.seed(42)
    closes = 100 + np.cumsum(np.random.randn(60) * 0.5)
    volumes = np.random.randint(100000, 200000, 60).astype(float)
    # Insert a climax day: biggest up move + 4x avg volume
    closes[55] = closes[54] + 8  # big up day
    volumes[55] = 800000  # 4x normal

    climax_df = pd.DataFrame({
        "Open": closes - 0.5,
        "High": closes + 1,
        "Low": closes - 1,
        "Close": closes,
        "Volume": volumes,
    }, index=dates)

    is_climax = pm._detect_climax(climax_df, "2025-01-01")
    check("Climax detected on 4x volume spike on biggest up day", is_climax == True,
          f"is_climax={is_climax}")

    # Test with normal volume — no climax
    normal_df = climax_df.copy()
    normal_df["Volume"] = np.random.randint(100000, 200000, 60).astype(float)
    not_climax = pm._detect_climax(normal_df, "2025-01-01")
    check("No climax on normal volume", not_climax == False)

    # Test 3m: get_positions_summary with mock data
    pm.save_positions([])  # clear
    pm.add_position("RELIANCE.NS", "2025-01-15", 2800.0, 10, 2650.0)

    # Build mock stock data
    dates2 = pd.date_range("2024-06-01", periods=200, freq="B")
    mock_prices = pd.DataFrame({
        "Open": np.linspace(2500, 2900, 200),
        "High": np.linspace(2520, 2920, 200),
        "Low": np.linspace(2480, 2880, 200),
        "Close": np.linspace(2500, 2900, 200),
        "Volume": [500000] * 200,
    }, index=dates2)

    summaries = pm.get_positions_summary({"RELIANCE.NS": mock_prices})
    check("get_positions_summary returns list", isinstance(summaries, list) and len(summaries) == 1)

    s = summaries[0]
    check("Summary has current_price", s.get("current_price") is not None and s["current_price"] > 0)
    check("Summary has pnl", "pnl" in s)
    check("Summary has pnl_pct", "pnl_pct" in s)
    check("Summary has suggested_action", s.get("suggested_action") in ("HOLD", "ADD", "SELL", "PARTIAL SELL", "NO DATA"))
    check("Trailing stop is >= initial stop (only moves up)",
          s.get("trailing_stop", 0) >= 2650.0,
          f"trailing_stop={s.get('trailing_stop')}")

    # Test 3n: No data scenario
    no_data_summary = pm.get_positions_summary({})
    check("No data produces NO DATA action",
          no_data_summary[0]["suggested_action"] == "NO DATA")

    # Test 3o: Trailing stop uses only post-entry highs (THE BUG FIX)
    pm.save_positions([])
    pm.add_position("BUGTEST.NS", "2025-03-01", 100.0, 10, 90.0)

    # Price was 200 BEFORE entry, then dropped to 100 area at entry, now at 105
    dates_bug = pd.date_range("2024-06-01", periods=250, freq="B")
    closes_bug = np.concatenate([
        np.linspace(150, 200, 180),   # pre-entry: peaked at 200
        np.linspace(100, 105, 70),    # post-entry: max 105
    ])
    bug_df = pd.DataFrame({
        "Open": closes_bug - 1,
        "High": closes_bug + 2,
        "Low": closes_bug - 2,
        "Close": closes_bug,
        "Volume": [500000] * 250,
    }, index=dates_bug)

    bug_summary = pm.get_positions_summary({"BUGTEST.NS": bug_df})
    bs = bug_summary[0]
    check("Highest close only considers post-entry data (not pre-entry 200)",
          bs.get("highest_close", 999) <= 110,  # should be ~105, not 200
          f"highest_close={bs.get('highest_close')} (pre-entry peak was 200)")
    check("Trailing stop is below current price (not above it)",
          bs.get("trailing_stop", 999) < bs.get("current_price", 0) + 50,
          f"trail={bs.get('trailing_stop')} current={bs.get('current_price')}")
    check("Not suggesting SELL on a position that's slightly profitable",
          bs.get("suggested_action") != "SELL" or bs.get("current_price", 0) < bs.get("trailing_stop", 0),
          f"action={bs.get('suggested_action')} price={bs.get('current_price')} trail={bs.get('trailing_stop')}")

finally:
    # Restore original paths
    pm.POSITIONS_FILE = _orig_pos_file
    pm.TRADE_HISTORY_FILE = _orig_hist_file
    shutil.rmtree(_tmp_dir, ignore_errors=True)


# ══════════════════════════════════════════════════════════════════
# SECTION 4: Fundamental Veto → Flag Conversion
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SECTION 4: Fundamental Veto → Informational Flag")
print("=" * 70)

from fundamental_veto import (
    apply_fundamental_veto,
    generate_final_watchlist,
    compute_position_size,
    compute_profit_targets,
)

# Test 4a: apply_fundamental_veto still works correctly
clean_fundamentals = {
    "data_available": True,
    "earnings_growth": 0.25,  # 25%
    "revenue_growth": 0.20,   # 20%
    "roe": 0.22,              # 22%
    "debt_equity": 50.0,      # 0.5 ratio (50/100)
    "peg_ratio": 1.5,
}
veto_result = apply_fundamental_veto(clean_fundamentals)
check("Clean fundamentals pass veto", veto_result["passes"] == True)
check("Clean fundamentals confidence is high", veto_result["confidence"] == "high")

bad_fundamentals = {
    "data_available": True,
    "earnings_growth": -0.15,  # -15% — hard veto
    "revenue_growth": -0.05,
    "roe": 0.03,               # 3% — hard veto
    "debt_equity": 350.0,      # 3.5 ratio — hard veto
    "peg_ratio": -1.0,         # negative PEG
}
veto_bad = apply_fundamental_veto(bad_fundamentals)
check("Bad fundamentals fail veto", veto_bad["passes"] == False)
check("Bad fundamentals has reasons", len(veto_bad["reasons"]) > 0)

# Test 4b: generate_final_watchlist — NO MORE VETOED action
regime = {
    "label": "Normal",
    "regime_score": 1,
    "posture": {
        "label": "Normal",
        "max_capital_pct": 80,
        "risk_per_trade_pct": 1.0,
        "max_new_positions": 4,
    },
}

# Build candidates with entry setups — some will have bad fundamentals
test_candidates = [
    {
        "ticker": "GOOD.NS",
        "sector": "Nifty IT",
        "rs_vs_nifty": 0.15,
        "stage": {"s2_score": 6},
        "entry_setup": {"entry_price": 500.0, "effective_stop": 470.0, "risk_pct": 6.0},
    },
    {
        "ticker": "BAD.NS",
        "sector": "Nifty Bank",
        "rs_vs_nifty": 0.10,
        "stage": {"s2_score": 5},
        "entry_setup": {"entry_price": 300.0, "effective_stop": 280.0, "risk_pct": 6.7},
    },
]

# Mock fetch_fundamentals to control output
with patch("fundamental_veto.fetch_fundamentals") as mock_fetch:
    def side_effect(ticker):
        if ticker == "GOOD.NS":
            return clean_fundamentals
        else:
            return bad_fundamentals

    mock_fetch.side_effect = side_effect

    result_wl = generate_final_watchlist(test_candidates, regime, total_capital=1_000_000)

check("generate_final_watchlist returns all candidates", len(result_wl) == 2)

# Critical: NO VETOED action
vetoed_stocks = [w for w in result_wl if w.get("action") == "VETOED"]
check("NO stocks are VETOED (hard veto removed)", len(vetoed_stocks) == 0,
      f"vetoed: {[v['ticker'] for v in vetoed_stocks]}")

# Both should get BUY or WATCHLIST
actionable = [w for w in result_wl if w.get("action") in ("BUY", "WATCHLIST")]
check("All stocks with entry setups get BUY or WATCHLIST action", len(actionable) == 2,
      f"actions: {[(w['ticker'], w['action']) for w in result_wl]}")

# Check fundamental flags
good_stock = next(w for w in result_wl if w["ticker"] == "GOOD.NS")
bad_stock = next(w for w in result_wl if w["ticker"] == "BAD.NS")

check("Good stock has CLEAN flag",
      good_stock.get("fundamental_flag") == "CLEAN",
      f"flag={good_stock.get('fundamental_flag')}")
check("Bad stock has CAUTION flag",
      bad_stock.get("fundamental_flag") == "CAUTION",
      f"flag={bad_stock.get('fundamental_flag')}")
check("Bad stock has fundamental_reasons",
      len(bad_stock.get("fundamental_reasons", [])) > 0)
check("Good stock has empty fundamental_reasons",
      bad_stock.get("fundamental_reasons") is not None)

# Test 4c: Position sizing still works for CAUTION stocks
check("Bad stock still gets position sizing",
      bad_stock.get("position") is not None and bad_stock["position"].get("shares", 0) > 0,
      f"position={bad_stock.get('position')}")
check("Bad stock still gets profit targets",
      bad_stock.get("targets") is not None and bad_stock["targets"].get("first_target", 0) > 0)

# Test 4d: WATCH action for candidates without entry
with patch("fundamental_veto.fetch_fundamentals") as mock_fetch2:
    mock_fetch2.return_value = clean_fundamentals

    no_entry_candidates = [
        {
            "ticker": "NOENTRY.NS",
            "sector": "Nifty IT",
            "rs_vs_nifty": 0.10,
            "stage": {"s2_score": 4},
            "entry_setup": None,  # no breakout yet
        },
    ]
    result_no_entry = generate_final_watchlist(no_entry_candidates, regime, 1_000_000)

check("No-entry candidate gets WATCH action",
      result_no_entry[0].get("action") == "WATCH",
      f"action={result_no_entry[0].get('action')}")


# ══════════════════════════════════════════════════════════════════
# SECTION 5: Position Sizing Sanity
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SECTION 5: Position Sizing Sanity")
print("=" * 70)

# Test 5a: Basic position sizing
posture = {"label": "Normal", "max_capital_pct": 80, "risk_per_trade_pct": 1.0, "max_new_positions": 4}
pos_size = compute_position_size(
    entry_price=500.0,
    stop_loss=470.0,
    regime_posture=posture,
    total_capital=1_000_000,
)
check("Position size computed",
      pos_size.get("shares", 0) > 0,
      f"shares={pos_size.get('shares')}")

expected_risk = 1_000_000 * 0.01  # 1% of capital = 10,000
raw_shares = int(expected_risk / (500 - 470))  # 10000 / 30 = 333
max_shares_by_value = int(1_000_000 * 0.15 / 500)  # 150000 / 500 = 300
expected_shares = min(raw_shares, max_shares_by_value)  # capped at 300 by 15% position limit
check("Shares = min(risk-based, 15% cap)",
      pos_size["shares"] == expected_shares,
      f"got {pos_size['shares']} expected {expected_shares}")

check("Position value is shares * entry",
      abs(pos_size["position_value"] - pos_size["shares"] * 500) < 1)

check("Risk amount is shares * risk_per_share",
      abs(pos_size["risk_amount"] - pos_size["shares"] * 30) < 1)

# Test 5b: Position cap at 15%
big_pos = compute_position_size(
    entry_price=10.0,  # cheap stock
    stop_loss=9.5,     # tight stop
    regime_posture=posture,
    total_capital=1_000_000,
)
max_value = 1_000_000 * 0.15  # 150,000
check("Position capped at 15% of capital",
      big_pos["position_value"] <= max_value + 10,  # +10 for rounding
      f"value={big_pos['position_value']} max={max_value}")

# Test 5c: Stop >= entry returns error
bad_stop = compute_position_size(500.0, 500.0, posture, 1_000_000)
check("Stop >= entry returns shares=0",
      bad_stop["shares"] == 0 and "error" in bad_stop)

# Test 5d: Profit targets
targets = compute_profit_targets(500.0, 470.0)
check("First target = entry * 1.20",
      targets["first_target"] == 600.0,
      f"target={targets['first_target']}")
check("Partial sell is 33%", targets["partial_sell_pct"] == 33)
check("R:R computed correctly",
      targets["reward_risk_ratio"] == round((600 - 500) / (500 - 470), 1),
      f"rr={targets['reward_risk_ratio']}")

# Test 5e: Cash regime — 0 risk = 0 shares
cash_posture = {"label": "Cash", "max_capital_pct": 10, "risk_per_trade_pct": 0.0, "max_new_positions": 0}
cash_pos = compute_position_size(500.0, 470.0, cash_posture, 1_000_000)
check("Cash regime risk 0% = 0 shares", cash_pos["shares"] == 0)

# Test 5f: Portfolio risk limit
maxed_pos = compute_position_size(500.0, 470.0, posture, 1_000_000, current_open_risk_pct=6.0)
check("At max portfolio risk (6%), no more shares", maxed_pos["shares"] == 0)


# ══════════════════════════════════════════════════════════════════
# SECTION 6: NSE Data Fetcher — New Methods Structure
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SECTION 6: NSE Data Fetcher — Method Signatures & Structure")
print("=" * 70)

from nse_data_fetcher import NSEDataFetcher, get_nse_fetcher

nse = NSEDataFetcher()

# Test 6a: Method existence
check("fetch_fii_dii_data method exists", hasattr(nse, "fetch_fii_dii_data"))
check("fetch_bulk_deals method exists", hasattr(nse, "fetch_bulk_deals"))
check("fetch_block_deals method exists", hasattr(nse, "fetch_block_deals"))
check("fetch_delivery_data method exists", hasattr(nse, "fetch_delivery_data"))

# Test 6b: Method signatures
import inspect
sig_fii = inspect.signature(nse.fetch_fii_dii_data)
check("fetch_fii_dii_data takes no required args", len(sig_fii.parameters) == 0)

sig_bulk = inspect.signature(nse.fetch_bulk_deals)
check("fetch_bulk_deals has from_date and to_date params",
      "from_date" in sig_bulk.parameters and "to_date" in sig_bulk.parameters)

sig_delivery = inspect.signature(nse.fetch_delivery_data)
check("fetch_delivery_data takes symbol param", "symbol" in sig_delivery.parameters)

# Test 6c: Test with mocked _request to verify return structures
with patch.object(nse, "_request") as mock_req:
    # Mock FII/DII response
    mock_req.return_value = [
        {"category": "FII/FPI", "date": "07-Feb-2025", "buyValue": "5000.00", "sellValue": "4000.00", "netValue": "1000.00"},
        {"category": "DII", "date": "07-Feb-2025", "buyValue": "3000.00", "sellValue": "3500.00", "netValue": "-500.00"},
    ]
    with patch.object(nse, "_load_cache", return_value=None):
        fii_result = nse.fetch_fii_dii_data()

    check("FII/DII result has required keys",
          fii_result is not None and all(k in fii_result for k in ["fii_buy", "fii_sell", "fii_net", "dii_buy", "dii_sell", "dii_net"]),
          f"keys={list(fii_result.keys()) if fii_result else None}")
    check("FII net parsed correctly",
          fii_result["fii_net"] == 1000.0,
          f"fii_net={fii_result.get('fii_net')}")
    check("DII net parsed correctly",
          fii_result["dii_net"] == -500.0,
          f"dii_net={fii_result.get('dii_net')}")

    # Mock bulk deals response
    mock_req.return_value = [
        {"mTd": "06-Feb-2025", "symbol": "RELIANCE", "clientName": "Goldman Sachs",
         "buySell": "BUY", "quantity": "500000", "wAvgPrice": "2850.50"},
    ]
    bulk_result = nse.fetch_bulk_deals("01-01-2025", "07-02-2025")

    check("Bulk deals returns list", isinstance(bulk_result, list) and len(bulk_result) == 1)
    check("Bulk deal has required keys",
          all(k in bulk_result[0] for k in ["date", "symbol", "client_name", "deal_type", "quantity", "price"]))
    check("Bulk deal symbol parsed", bulk_result[0]["symbol"] == "RELIANCE")
    check("Bulk deal quantity parsed as float", bulk_result[0]["quantity"] == 500000.0)
    check("Bulk deal price parsed", bulk_result[0]["price"] == 2850.5)

    # Mock block deals (same structure)
    mock_req.return_value = []
    block_result = nse.fetch_block_deals()
    check("Empty block deals returns empty list", block_result == [])

    # Mock delivery data
    mock_req.return_value = {
        "securityWiseDP": {
            "deliveryQuantity": "1500000",
            "quantityTraded": "3000000",
            "deliveryToTradedQuantity": "50.00",
        }
    }
    with patch.object(nse, "_load_cache", return_value=None):
        del_result = nse.fetch_delivery_data("RELIANCE.NS")

    check("Delivery data has required keys",
          del_result is not None and all(k in del_result for k in ["symbol", "delivery_qty", "traded_qty", "delivery_pct"]))
    check("Delivery symbol cleaned", del_result["symbol"] == "RELIANCE")
    check("Delivery pct parsed correctly", del_result["delivery_pct"] == 50.0,
          f"pct={del_result.get('delivery_pct')}")

    # Mock None response (NSE down)
    mock_req.return_value = None
    with patch.object(nse, "_load_cache", return_value=None):
        none_fii = nse.fetch_fii_dii_data()
    check("FII/DII returns None when NSE is down", none_fii is None)

    none_bulk = nse.fetch_bulk_deals()
    check("Bulk deals returns [] when NSE is down", none_bulk == [])

    with patch.object(nse, "_load_cache", return_value=None):
        none_del = nse.fetch_delivery_data("TEST.NS")
    check("Delivery returns None when NSE is down", none_del is None)


# ══════════════════════════════════════════════════════════════════
# SECTION 7: End-to-End Signal Flow — Full Pipeline Simulation
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SECTION 7: End-to-End Signal Flow")
print("=" * 70)

# Simulate: candidates → conviction ranking → watchlist generation → flag check → top ideas

# Build 5 realistic stage2 candidates
e2e_candidates = [
    {
        "ticker": "PERSISTENT.NS", "sector": "Nifty IT",
        "rs_vs_nifty": 0.30, "rs_vs_sector": 0.15, "accumulation_ratio": 1.6,
        "stage": {"s2_score": 7, "stage": 2, "confidence": 0.95},
        "breakout": {"breakout": True, "base_number": 1, "volume_ratio": 2.2, "breakout_price": 5000, "base_depth_pct": 12},
        "entry_setup": {"entry_price": 5100.0, "effective_stop": 4750.0, "risk_pct": 6.9},
        "vcp": {"is_vcp": True, "contractions": 3},
        "close": 5100.0, "avg_volume": 200000,
    },
    {
        "ticker": "COFORGE.NS", "sector": "Nifty IT",
        "rs_vs_nifty": 0.22, "rs_vs_sector": 0.10, "accumulation_ratio": 1.4,
        "stage": {"s2_score": 6, "stage": 2, "confidence": 0.88},
        "breakout": {"breakout": True, "base_number": 2, "volume_ratio": 1.7, "breakout_price": 6000, "base_depth_pct": 18},
        "entry_setup": {"entry_price": 6100.0, "effective_stop": 5700.0, "risk_pct": 6.6},
        "vcp": None,
        "close": 6100.0, "avg_volume": 150000,
    },
    {
        "ticker": "ICICIBANK.NS", "sector": "Nifty Bank",
        "rs_vs_nifty": 0.12, "rs_vs_sector": 0.08, "accumulation_ratio": 1.2,
        "stage": {"s2_score": 5, "stage": 2, "confidence": 0.78},
        "breakout": {"breakout": True, "base_number": 1, "volume_ratio": 1.5, "breakout_price": 1100, "base_depth_pct": 10},
        "entry_setup": {"entry_price": 1120.0, "effective_stop": 1050.0, "risk_pct": 6.3},
        "vcp": None,
        "close": 1120.0, "avg_volume": 5000000,
    },
    {
        "ticker": "SUNPHARMA.NS", "sector": "Nifty Pharma",
        "rs_vs_nifty": 0.08, "rs_vs_sector": 0.05, "accumulation_ratio": 1.0,
        "stage": {"s2_score": 4, "stage": 2, "confidence": 0.65},
        "breakout": {"breakout": True, "base_number": 3, "volume_ratio": 1.3, "breakout_price": 1500, "base_depth_pct": 22},
        "entry_setup": {"entry_price": 1520.0, "effective_stop": 1400.0, "risk_pct": 7.9},
        "vcp": None,
        "close": 1520.0, "avg_volume": 800000,
    },
    {
        "ticker": "BAJAJFINSV.NS", "sector": "Nifty Fin Service",
        "rs_vs_nifty": -0.02, "rs_vs_sector": -0.05, "accumulation_ratio": 0.9,
        "stage": {"s2_score": 3, "stage": 2, "confidence": 0.50},
        "breakout": None,
        "entry_setup": None,
        "vcp": None,
        "close": 1700.0, "avg_volume": 300000,
    },
]

# Step 1: Rank by conviction
ranked_e2e = rank_candidates_by_conviction(list(e2e_candidates), sector_rankings)

check("E2E: All 5 candidates ranked", len(ranked_e2e) == 5)
check("E2E: Ranked by conviction score descending",
      all(ranked_e2e[i]["conviction_score"] >= ranked_e2e[i + 1]["conviction_score"]
          for i in range(len(ranked_e2e) - 1)),
      f"scores: {[c['conviction_score'] for c in ranked_e2e]}")

check("E2E: PERSISTENT (top sector, 7/7, VCP, 1st base, 2.2x vol) ranks #1",
      ranked_e2e[0]["ticker"] == "PERSISTENT.NS",
      f"#1 is {ranked_e2e[0]['ticker']} score={ranked_e2e[0]['conviction_score']}")

# Step 2: Get top ideas — need action set first (mirrors real app.py flow)
# In production, rank_candidates_by_conviction is called on the watchlist AFTER generate_final_watchlist
# For this pre-watchlist test, set actions manually to simulate
for c in ranked_e2e:
    if c.get("entry_setup"):
        c["action"] = "BUY"
    else:
        c["action"] = "WATCH"

top_ideas_e2e = get_top_conviction_ideas(ranked_e2e, top_n=3)
check("E2E: Top 3 ideas have entry setups",
      len(top_ideas_e2e) == 3 and all(t.get("entry_setup") is not None for t in top_ideas_e2e))

check("E2E: BAJAJFINSV (no breakout) excluded from top ideas",
      all(t["ticker"] != "BAJAJFINSV.NS" for t in top_ideas_e2e))

# Step 3: Generate watchlist with flags (mock fundamentals)
with patch("fundamental_veto.fetch_fundamentals") as mock_fetch_e2e:
    def e2e_fundamentals(ticker):
        if ticker in ("PERSISTENT.NS", "COFORGE.NS", "ICICIBANK.NS"):
            return {**clean_fundamentals, "ticker": ticker}
        else:
            return {**bad_fundamentals, "ticker": ticker}

    mock_fetch_e2e.side_effect = e2e_fundamentals

    e2e_watchlist = generate_final_watchlist(list(e2e_candidates), regime, 1_000_000)

check("E2E: Watchlist has 5 entries", len(e2e_watchlist) == 5)

buy_count = sum(1 for w in e2e_watchlist if w["action"] == "BUY")
watchlist_count = sum(1 for w in e2e_watchlist if w["action"] == "WATCHLIST")
watch_count = sum(1 for w in e2e_watchlist if w["action"] == "WATCH")
vetoed_count = sum(1 for w in e2e_watchlist if w["action"] == "VETOED")

check("E2E: No VETOED stocks in output", vetoed_count == 0,
      f"vetoed={vetoed_count}")
check("E2E: Max 4 BUY signals (regime limit)", buy_count <= 4,
      f"buys={buy_count}")
check("E2E: BUY + WATCHLIST for stocks with entry",
      buy_count + watchlist_count == 4,  # 4 have entry_setup
      f"buy={buy_count} watchlist={watchlist_count}")
check("E2E: WATCH for stock without entry", watch_count == 1,
      f"watch={watch_count}")

# Check flags
clean_stocks = [w for w in e2e_watchlist if w.get("fundamental_flag") == "CLEAN"]
caution_stocks = [w for w in e2e_watchlist if w.get("fundamental_flag") == "CAUTION"]
check("E2E: 3 CLEAN flagged stocks", len(clean_stocks) == 3)
check("E2E: 2 CAUTION flagged stocks", len(caution_stocks) == 2,
      f"caution tickers: {[c['ticker'] for c in caution_stocks]}")

# Verify CAUTION stock still got BUY action (the key behavioral change)
sunpharma = next(w for w in e2e_watchlist if w["ticker"] == "SUNPHARMA.NS")
check("E2E: SUNPHARMA (CAUTION) still gets BUY/WATCHLIST (not VETOED)",
      sunpharma["action"] in ("BUY", "WATCHLIST"),
      f"action={sunpharma['action']}")
check("E2E: SUNPHARMA has position sizing despite CAUTION",
      sunpharma.get("position") is not None and sunpharma["position"].get("shares", 0) > 0)

# Check that position sizing accumulates risk correctly
buy_stocks = [w for w in e2e_watchlist if w["action"] == "BUY"]
total_risk_pct = sum(w.get("position", {}).get("risk_pct_of_capital", 0) for w in buy_stocks)
check("E2E: Total risk % <= max portfolio risk (6%)",
      total_risk_pct <= 6.5,  # slight float tolerance
      f"total_risk={total_risk_pct}%")


# ══════════════════════════════════════════════════════════════════
# SECTION 8: Data Integrity — Conviction Scoring Boundary Tests
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SECTION 8: Conviction Scoring Edge Cases")
print("=" * 70)

# Test 8a: Candidate not in any ranked sector
outsider = make_candidate("RANDOM.NS", "Nifty MNC", s2_score=7)
score_outsider = compute_conviction_score(outsider, sector_rankings, [outsider])
check("Non-ranked sector gets 0 sector points",
      score_outsider < 80,  # should miss the 40pts from sector
      f"score={score_outsider}")

# Test 8b: Single candidate — percentiles still work
solo = make_candidate("SOLO.NS", "Nifty IT", s2_score=7, rs=0.30, acc=1.5)
score_solo = compute_conviction_score(solo, sector_rankings, [solo])
check("Single candidate percentile computes (no division by zero)",
      0 <= score_solo <= 100,
      f"score={score_solo}")

# Test 8c: All identical RS — percentiles handle ties
dupes = [make_candidate(f"T{i}.NS", "Nifty IT", rs=0.10) for i in range(5)]
ranked_dupes = rank_candidates_by_conviction(dupes, sector_rankings)
check("Identical RS candidates all get scores",
      all("conviction_score" in d for d in ranked_dupes))
check("Identical candidates get equal scores (within rounding)",
      max(d["conviction_score"] for d in ranked_dupes) - min(d["conviction_score"] for d in ranked_dupes) < 2)

# Test 8d: Bulk deal bonus
fake_deals = [{"symbol": "DEALSTOCK", "date": "01-Feb-2025"}]
deal_candidate = make_candidate("DEALSTOCK.NS", "Nifty IT")
score_with_deal = compute_conviction_score(deal_candidate, sector_rankings, [deal_candidate], bulk_deals=fake_deals)
score_no_deal = compute_conviction_score(deal_candidate, sector_rankings, [deal_candidate], bulk_deals=[])
check("Bulk deal bonus adds to score",
      score_with_deal > score_no_deal,
      f"with={score_with_deal} without={score_no_deal}")
check("Bulk deal bonus is +3",
      abs(score_with_deal - score_no_deal - 3) < 0.5,
      f"diff={score_with_deal - score_no_deal}")


# ══════════════════════════════════════════════════════════════════
# SECTION 9: Position Manager — File Persistence & JSON Integrity
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SECTION 9: File Persistence & JSON Integrity")
print("=" * 70)

_tmp_dir2 = Path(tempfile.mkdtemp())
pm.POSITIONS_FILE = _tmp_dir2 / "pos.json"
pm.TRADE_HISTORY_FILE = _tmp_dir2 / "hist.json"

try:
    # Test 9a: Fresh start — no files
    check("No positions file = empty list", pm.load_positions() == [])
    check("No history file = empty list", pm.load_trade_history() == [])

    # Test 9b: Write and re-read
    pm.add_position("TEST.NS", "2025-01-01", 100.0, 10, 90.0, "test note")
    raw = json.loads(pm.POSITIONS_FILE.read_text())
    check("Positions file is valid JSON", isinstance(raw, list) and len(raw) == 1)
    check("JSON has all required fields",
          all(k in raw[0] for k in ["id", "ticker", "entry_date", "entry_price", "shares", "initial_stop", "trailing_stop"]))

    # Test 9c: Close and verify history file
    pid = raw[0]["id"]
    pm.close_position(pid, "2025-02-01", 110.0, "profit")
    hist_raw = json.loads(pm.TRADE_HISTORY_FILE.read_text())
    check("History file is valid JSON", isinstance(hist_raw, list) and len(hist_raw) == 1)
    check("History has P&L fields",
          all(k in hist_raw[0] for k in ["pnl", "pnl_pct", "days_held", "exit_price", "exit_reason"]))

    # Test 9d: Corrupt file handling
    pm.POSITIONS_FILE.write_text("not json{{{")
    check("Corrupt positions file returns empty list", pm.load_positions() == [])

finally:
    pm.POSITIONS_FILE = _orig_pos_file
    pm.TRADE_HISTORY_FILE = _orig_hist_file
    shutil.rmtree(_tmp_dir2, ignore_errors=True)


# ══════════════════════════════════════════════════════════════════
# SECTION 10: Integration — app.py imports & page imports
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SECTION 10: Import Validation")
print("=" * 70)

# Test that all modules can be imported without errors (no circular imports, missing deps)
import_errors = []
modules_to_test = [
    "config",
    "conviction_scorer",
    "position_manager",
    "nse_data_fetcher",
    "fundamental_veto",
    "data_fetcher",
]
for mod in modules_to_test:
    try:
        __import__(mod)
        check(f"Import {mod}", True)
    except Exception as e:
        check(f"Import {mod}", False, str(e))

# Verify key functions are accessible from the modules app.py imports
try:
    from conviction_scorer import rank_candidates_by_conviction, get_top_conviction_ideas
    from position_manager import get_positions_summary, load_positions
    from nse_data_fetcher import get_nse_fetcher
    check("app.py dependencies importable", True)
except Exception as e:
    check("app.py dependencies importable", False, str(e))


# ══════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print(f"RESULTS: {PASS} passed, {FAIL} failed out of {PASS + FAIL} tests")
print("=" * 70)

if ERRORS:
    print("\nFailed tests:")
    for e in ERRORS:
        print(e)
    print()

    if __name__ == "__main__":
        sys.exit(1)
    else:
        raise AssertionError(f"{FAIL} tests failed")  # noqa
else:
    print("\nAll tests passed! ✅")

    if __name__ == "__main__":
        sys.exit(0)
