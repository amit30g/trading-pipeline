"""Page 7: Final Watchlist with Position Sizing"""
import streamlit as st
import pandas as pd
import io

from dashboard_helpers import build_portfolio_pie, format_large_number
from config import POSITION_CONFIG

st.set_page_config(page_title="Watchlist", page_icon="📊", layout="wide")
st.title("Watchlist & Position Sizing")

if "final_watchlist" not in st.session_state:
    st.info("Run a scan first from the home page.")
    st.stop()

watchlist = st.session_state.final_watchlist
regime = st.session_state.regime
capital = st.session_state.get("capital", POSITION_CONFIG["total_capital"])

if not watchlist:
    st.warning("No watchlist entries. No Stage 2 candidates found.")
    st.stop()

# Separate by action
buys = [w for w in watchlist if w.get("action") == "BUY"]
watches = [w for w in watchlist if w.get("action") in ("WATCH", "WATCHLIST")]

# ── Tabs (no more VETOED tab) ─────────────────────────────────
tab_buy, tab_watch = st.tabs([
    f"BUY SIGNALS ({len(buys)})",
    f"WATCHLIST ({len(watches)})",
])

with tab_buy:
    if not buys:
        st.caption("No BUY signals in this scan.")
    else:
        rows = []
        for b in buys:
            pos = b.get("position", {})
            targets = b.get("targets", {})
            entry_setup = b.get("entry_setup", {}) or {}
            fund_flag = b.get("fundamental_flag", "CLEAN")
            fund_icon = "✅" if fund_flag == "CLEAN" else "⚠️"
            rows.append({
                "Ticker": b["ticker"],
                "Sector": b.get("sector", ""),
                "Entry": round(entry_setup.get("entry_price", 0), 1),
                "Stop": round(entry_setup.get("effective_stop", 0), 1),
                "Risk %": f"{entry_setup.get('risk_pct', 0):.1f}%",
                "Shares": pos.get("shares", 0),
                "Value": format_large_number(pos.get("position_value", 0)),
                "Risk Amt": format_large_number(pos.get("risk_amount", 0)),
                "Target 1": round(targets.get("first_target", 0), 1),
                "R:R": f"{targets.get('reward_risk_ratio', 0):.1f}",
                "Fund": f"{fund_icon} {fund_flag}",
            })
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True)

        # Expanders for CAUTION stocks
        caution_buys = [b for b in buys if b.get("fundamental_flag") == "CAUTION"]
        if caution_buys:
            with st.expander(f"Fundamental Cautions ({len(caution_buys)} stocks)"):
                for b in caution_buys:
                    reasons = b.get("fundamental_reasons", [])
                    st.markdown(f"**{b['ticker']}**: {'; '.join(reasons)}")

        # Portfolio summary
        total_value = sum(b.get("position", {}).get("position_value", 0) for b in buys)
        total_risk = sum(b.get("position", {}).get("risk_amount", 0) for b in buys)
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Allocated", format_large_number(total_value))
        c2.metric("Total Risk", format_large_number(total_risk))
        c3.metric("Risk % of Capital", f"{total_risk / capital * 100:.1f}%" if capital > 0 else "N/A")

with tab_watch:
    if not watches:
        st.caption("No watchlist entries.")
    else:
        rows = []
        for w in watches:
            entry_setup = w.get("entry_setup", {}) or {}
            fund_flag = w.get("fundamental_flag", "CLEAN")
            fund_icon = "✅" if fund_flag == "CLEAN" else "⚠️"
            rows.append({
                "Ticker": w["ticker"],
                "Sector": w.get("sector", ""),
                "Entry": round(entry_setup.get("entry_price", 0), 1) if entry_setup.get("entry_price") else "Pending",
                "Stop": round(entry_setup.get("effective_stop", 0), 1) if entry_setup.get("effective_stop") else "",
                "Stage Score": f"{w.get('stage', {}).get('s2_score', 0)}/7",
                "RS": round(w.get("rs_vs_nifty", 0), 2),
                "Action": w.get("action", ""),
                "Fund": f"{fund_icon} {fund_flag}",
            })
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True)

        # Expanders for CAUTION stocks
        caution_watches = [w for w in watches if w.get("fundamental_flag") == "CAUTION"]
        if caution_watches:
            with st.expander(f"Fundamental Cautions ({len(caution_watches)} stocks)"):
                for w in caution_watches:
                    reasons = w.get("fundamental_reasons", [])
                    st.markdown(f"**{w['ticker']}**: {'; '.join(reasons)}")

# ── Portfolio Pie ───────────────────────────────────────────────
st.subheader("Portfolio Allocation")
fig = build_portfolio_pie(watchlist, capital)
st.plotly_chart(fig, use_container_width=True)

# ── CSV Export ──────────────────────────────────────────────────
st.divider()

all_rows = []
for w in watchlist:
    entry_setup = w.get("entry_setup", {}) or {}
    pos = w.get("position", {})
    targets = w.get("targets", {})
    all_rows.append({
        "Ticker": w["ticker"],
        "Sector": w.get("sector", ""),
        "Action": w.get("action", ""),
        "Fund Flag": w.get("fundamental_flag", ""),
        "Entry": entry_setup.get("entry_price", ""),
        "Stop": entry_setup.get("effective_stop", ""),
        "Shares": pos.get("shares", ""),
        "Value": pos.get("position_value", ""),
        "Target": targets.get("first_target", ""),
        "R:R": targets.get("reward_risk_ratio", ""),
        "RS vs Nifty": w.get("rs_vs_nifty", ""),
    })
csv_df = pd.DataFrame(all_rows)
csv_buffer = io.StringIO()
csv_df.to_csv(csv_buffer, index=False)

st.download_button(
    label="Download Watchlist as CSV",
    data=csv_buffer.getvalue(),
    file_name=f"watchlist_{st.session_state.get('scan_date', 'scan').replace(' ', '_').replace(':', '')}.csv",
    mime="text/csv",
)
