"""Demo app for the trading design system.

Run with: streamlit run demo.py
"""

import streamlit as st
import numpy as np
import pandas as pd

from design_system import *

apply_theme()

# ── Top accent + metrics row ────────────────────────────────────────────────

st.html(
    f"""
    <div style="display:flex;gap:1rem;align-items:flex-start;flex-wrap:wrap;">
        {metric_card("Total PnL", "+$12,345", ACCENT_GREEN)}
        {metric_card("Sharpe", "1.87", ACCENT_CYAN)}
        {metric_card("Win Rate", "62.4%", ACCENT_AMBER)}
        {metric_card("Open Trades", "7", ACCENT_VIOLET)}
        {metric_card("Drawdown", "-3.2%", ACCENT_RED)}
    </div>
    """,
)

# ── Regime badges ───────────────────────────────────────────────────────────

st.html(
    f"""
    <div style="display:flex;gap:0.75rem;align-items:center;flex-wrap:wrap;margin-top:0.5rem;">
        Regime: {regime_badge("Low Vol", 87.3)}
        {regime_badge("Medium Vol", 42.1)}
        {regime_badge("High Vol", 15.8)}
        {regime_badge("Uncertain", 23.0)}
    </div>
    """,
)

# ── Status indicators ───────────────────────────────────────────────────────

st.html(
    f"""
    <div style="display:flex;gap:1.5rem;align-items:center;margin-top:0.5rem;">
        Data Feed: {status_dot("connected")} Connected
        &nbsp;&nbsp;|&nbsp;&nbsp;
        Strategy: {status_dot("active")} Active
        &nbsp;&nbsp;|&nbsp;&nbsp;
        Risk: {status_dot("warning")} Warning
    </div>
    """,
)

# ── Section: Chart ──────────────────────────────────────────────────────────

st.html(section_header("Equity Curve"))

st.html(
    f'<div style="color:{TEXT_MUTED};font-size:12px;">2025 performance — daily equity</div>',
)

np.random.seed(42)
dates = pd.date_range("2025-01-01", periods=100, freq="B")
equity = 10000 + np.cumsum(np.random.randn(100) * 150)

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=dates, y=equity,
    mode="lines",
    name="Equity",
    line=dict(color=ACCENT_CYAN, width=2),
))

# shaded drawdown zones
dd = equity - np.maximum.accumulate(equity)
fig.add_trace(go.Scatter(
    x=dates, y=dd,
    mode="lines",
    fill="tozeroy",
    name="Drawdown",
    line=dict(color=ACCENT_RED, width=1),
    fillcolor="rgba(255,23,68,0.1)",
))

fig.update_layout(get_plotly_layout(title=None))
st.plotly_chart(fig, width="stretch")

# ── Section: DataFrame ─────────────────────────────────────────────────────

st.html(section_header("Recent Trades"))

np.random.seed(123)
n = 10
trades = pd.DataFrame({
    "Ticker": np.random.choice(["AAPL", "MSFT", "GOOG", "NVDA", "TSLA"], n),
    "Side": np.random.choice(["LONG", "SHORT"], n),
    "Entry": np.round(np.random.uniform(100, 500, n), 2),
    "Exit": np.round(np.random.uniform(100, 500, n), 2),
    "PnL": np.round(np.random.uniform(-5000, 5000, n), 2),
})

styled_df = style_dataframe(trades)
styled_df = styled_df.format({
    "Entry": "${:,.2f}",
    "Exit": "${:,.2f}",
    "PnL": lambda v: f"{'${:,.2f}'.format(v)}" if v >= 0 else f"${abs(v):,.2f}",
})

# color PnL column
st.dataframe(
    styled_df.map(
        lambda v: f"color:{pnl_color(v)}" if isinstance(v, str) and v.startswith("$") else "",
        subset=["PnL"],
    ),
    width="stretch",
    hide_index=True,
)

# ── Section: Mixed metrics ──────────────────────────────────────────────────

st.html(section_header("Risk Metrics"))

col1, col2 = st.columns(2)

with col1:
    st.html(
        f"""
        <div style="display:flex;flex-direction:column;align-items:center;padding:1rem;">
            <div style="font-family:DM Sans,sans-serif;font-size:11px;text-transform:uppercase;letter-spacing:3px;color:{TEXT_MUTED};margin-bottom:4px;">VaR (95%)</div>
            <div style="font-family:JetBrains Mono,monospace;font-size:24px;font-weight:700;color:{ACCENT_RED};">-${284.5}</div>
        </div>
        """,
    )

with col2:
    st.html(
        f"""
        <div style="display:flex;flex-direction:column;align-items:center;padding:1rem;">
            <div style="font-family:DM Sans,sans-serif;font-size:11px;text-transform:uppercase;letter-spacing:3px;color:{TEXT_MUTED};margin-bottom:4px;">Max Drawdown</div>
            <div style="font-family:JetBrains Mono,monospace;font-size:24px;font-weight:700;color:{pnl_color(-1.8)};">-3.2%</div>
        </div>
        """,
    )

# ── Section: Strategy breakdown ─────────────────────────────────────────────

st.html(section_header("Strategy Breakdown"))

strategy_df = pd.DataFrame({
    "Strategy": ["Momentum", "Mean Reversion", "Stat Arb", "Event-Driven"],
    "Capital": [2_500_000, 1_800_000, 3_200_000, 1_500_000],
    "Return": [0.142, -0.023, 0.087, 0.061],
    "Sharpe": [1.6, 0.4, 2.1, 1.3],
})

styled_strategy = style_dataframe(strategy_df)
styled_strategy = styled_strategy.format({
    "Capital": lambda v: f"${v:,.0f}",
    "Return": lambda v: f"{v*100:+.1f}%" if v >= 0 else f"{v*100:+.1f}%",
    "Sharpe": "{:.1f}",
})

st.dataframe(styled_strategy, width="stretch", hide_index=True)
