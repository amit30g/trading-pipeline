"""
Conviction Scoring Engine — ranks Stage 2 candidates by composite score (0-100).
Combines sector rank, stage score, base count, RS percentile, accumulation,
and bonus factors (VCP, bulk deals, volume surge, delivery %).
"""
from datetime import datetime, timedelta
from config import CONVICTION_CONFIG


def compute_conviction_score(
    candidate: dict,
    sector_rankings: list[dict],
    all_candidates: list[dict],
    bulk_deals: list[dict] | None = None,
    delivery_data: dict | None = None,
) -> float:
    """Compute a 0-100 conviction score for a single candidate.

    Args:
        candidate: Stage 2 candidate dict with keys like ticker, sector,
                   stage (with s2_score), rs_vs_nifty, accumulation_ratio,
                   breakout, entry_setup, vcp.
        sector_rankings: List of sector ranking dicts from sector_rs module.
        all_candidates: All candidates (for percentile computation).
        bulk_deals: Recent bulk deals list from NSE.
        delivery_data: Delivery data dict for this stock {delivery_pct, ...}.

    Returns:
        Float score 0-100.
    """
    cfg = CONVICTION_CONFIG
    score = 0.0

    # ── Factor 1: Sector Rank (40%) ────────────────────────────
    sector = candidate.get("sector", "")
    top_sectors = [r.get("sector") or r.get("name", "") for r in sector_rankings]
    if sector in top_sectors:
        rank = top_sectors.index(sector) + 1
        sector_pts_map = {1: 40, 2: 35, 3: 25, 4: 15}
        score += sector_pts_map.get(rank, 0) * (cfg["sector_rank_weight"] / 40)
    # If sector not in top rankings, 0 points

    # ── Factor 2: Stage 2 Score (20%) ──────────────────────────
    s2_score = candidate.get("stage", {}).get("s2_score", 0)
    score += (s2_score / 7) * cfg["stage2_score_weight"]

    # ── Factor 3: Base Count (10%) ─────────────────────────────
    base_count = candidate.get("breakout", {}).get("base_number", 1) if candidate.get("breakout") else 1
    base_pts_map = {1: 10, 2: 7, 3: 4}
    score += base_pts_map.get(base_count, 0) * (cfg["base_count_weight"] / 10)

    # ── Factor 4: RS Percentile (15%) ──────────────────────────
    rs_val = candidate.get("rs_vs_nifty", 0)
    all_rs = sorted([c.get("rs_vs_nifty", 0) for c in all_candidates])
    if all_rs:
        rs_rank = sum(1 for v in all_rs if v <= rs_val)
        rs_pctl = rs_rank / len(all_rs)
        score += rs_pctl * cfg["rs_percentile_weight"]

    # ── Factor 5: Accumulation Ratio Percentile (15%) ──────────
    acc_val = candidate.get("accumulation_ratio", 1.0)
    all_acc = sorted([c.get("accumulation_ratio", 1.0) for c in all_candidates])
    if all_acc:
        acc_rank = sum(1 for v in all_acc if v <= acc_val)
        acc_pctl = acc_rank / len(all_acc)
        score += acc_pctl * cfg["accumulation_weight"]

    # ── Bonuses (up to +10) ────────────────────────────────────
    bonus = 0.0

    # VCP pattern detected: +3
    vcp = candidate.get("vcp")
    if vcp and vcp.get("is_vcp"):
        bonus += 3

    # Bulk deal in last 30 days: +3
    if bulk_deals:
        ticker_clean = candidate.get("ticker", "").replace(".NS", "").replace(".BO", "").upper()
        cutoff = (datetime.now() - timedelta(days=30)).strftime("%d-%b-%Y")
        for deal in bulk_deals:
            if deal.get("symbol", "").upper() == ticker_clean:
                bonus += 3
                break

    # Volume surge >2x on breakout: +2
    breakout = candidate.get("breakout")
    if breakout and breakout.get("breakout"):
        vol_ratio = breakout.get("volume_ratio", 0)
        if vol_ratio >= 2.0:
            bonus += 2

    # Delivery % >50%: +2
    if delivery_data and delivery_data.get("delivery_pct", 0) > 50:
        bonus += 2

    score += min(bonus, 10)  # cap bonus at 10

    return round(min(score, 100), 1)


def rank_candidates_by_conviction(
    candidates: list[dict],
    sector_rankings: list[dict],
    bulk_deals: list[dict] | None = None,
) -> list[dict]:
    """Score and rank all candidates by conviction. Returns sorted list (highest first).

    Adds 'conviction_score' key to each candidate dict.
    """
    for candidate in candidates:
        candidate["conviction_score"] = compute_conviction_score(
            candidate=candidate,
            sector_rankings=sector_rankings,
            all_candidates=candidates,
            bulk_deals=bulk_deals,
        )

    return sorted(candidates, key=lambda c: c.get("conviction_score", 0), reverse=True)


def get_top_conviction_ideas(
    ranked_candidates: list[dict],
    top_n: int = 3,
) -> list[dict]:
    """Return top N actionable ideas (BUY action + has entry setup).

    Args:
        ranked_candidates: Candidates already sorted by conviction_score.
        top_n: Number of top ideas to return.

    Returns:
        List of up to top_n candidates with BUY action and entry setups.
    """
    ideas = []
    for c in ranked_candidates:
        action = c.get("action", "")
        entry_setup = c.get("entry_setup")
        if action in ("BUY", "WATCHLIST") and entry_setup:
            ideas.append(c)
            if len(ideas) >= top_n:
                break
    return ideas
