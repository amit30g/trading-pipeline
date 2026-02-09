"""
AI Market Summary — Ollama LLM with template fallback.
No pip dependencies beyond stdlib (uses subprocess for Ollama).
"""
import subprocess
import json

from dashboard_helpers import generate_template_summary


def _build_prompt(macro_data: dict, regime: dict, fii_dii: dict | None,
                  fii_dii_flows: dict, sector_rankings: list[dict]) -> str:
    """Build a structured prompt for the LLM."""
    lines = ["You are a senior macro strategist writing a morning briefing for an Indian equity investor.",
             "Write 4-5 sentences. Be direct and opinionated. No bullet points.",
             "", "=== DATA ===", ""]

    # Global indices
    for label in ["S&P 500", "Nasdaq", "Dow Jones", "FTSE 100", "DAX", "Nikkei 225", "Hang Seng", "Shanghai"]:
        d = macro_data.get(label)
        if d:
            lines.append(f"{label}: {d['price']:.1f} ({d['change_pct']:+.2f}%)")

    # Risk gauges
    lines.append("")
    for label in ["VIX", "India VIX", "Dollar Index", "US 10Y", "US 5Y", "US 30Y"]:
        d = macro_data.get(label)
        if d:
            lines.append(f"{label}: {d['price']:.2f}")

    spread = macro_data.get("10Y-5Y Spread", {})
    if spread:
        lines.append(f"10Y-5Y Spread: {spread['price']:.2f}")

    # Commodities / currencies
    lines.append("")
    for label in ["USD/INR", "EUR/USD", "USD/JPY", "Crude Oil", "Gold", "Copper"]:
        d = macro_data.get(label)
        if d:
            lines.append(f"{label}: {d['price']:.2f} ({d['change_pct']:+.2f}%)")

    # India
    lines.append("")
    lines.append(f"India Regime: {regime.get('label', 'N/A')} (score {regime.get('regime_score', 0):+d})")
    lines.append(f"Breadth trend: {regime.get('breadth_trend', 'N/A')}")

    if fii_dii:
        lines.append(f"FII today: {fii_dii.get('fii_net', 0):+,.0f} Cr | DII today: {fii_dii.get('dii_net', 0):+,.0f} Cr")
    fii_1m = fii_dii_flows.get("1m", {}).get("fii_net")
    if fii_1m is not None:
        lines.append(f"FII 1-month cumulative: {fii_1m:+,.0f} Cr")

    if sector_rankings:
        top3 = [s.get("sector", "") for s in sector_rankings[:3]]
        bottom3 = [s.get("sector", "") for s in sector_rankings[-3:]]
        lines.append(f"Top sectors: {', '.join(top3)}")
        lines.append(f"Lagging sectors: {', '.join(bottom3)}")

    lines.append("")
    lines.append("Write a concise morning briefing for today. Focus on actionable macro context for Indian equity investing.")

    return "\n".join(lines)


def _try_ollama(prompt: str, model: str = "llama3.1:8b", timeout: int = 60) -> str | None:
    """Try to get a summary from local Ollama. Returns None if unavailable."""
    try:
        result = subprocess.run(
            ["ollama", "run", model],
            input=prompt,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        pass
    return None


def generate_market_summary(
    macro_data: dict,
    regime: dict,
    fii_dii: dict | None = None,
    fii_dii_flows: dict | None = None,
    sector_rankings: list[dict] | None = None,
) -> tuple[str, str]:
    """
    Generate market summary. Tries Ollama first, falls back to template.
    Returns (summary_text, source_label).
    """
    if fii_dii_flows is None:
        fii_dii_flows = {}
    if sector_rankings is None:
        sector_rankings = []

    # Try Ollama
    prompt = _build_prompt(macro_data, regime, fii_dii, fii_dii_flows, sector_rankings)
    ai_text = _try_ollama(prompt)
    if ai_text:
        return ai_text, "AI-generated (Llama 3.1)"

    # Fallback to template
    template_text = generate_template_summary(macro_data, regime, fii_dii, fii_dii_flows, sector_rankings)
    return template_text, "Data-driven summary"
