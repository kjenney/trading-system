"""Correlation Break Detector — rolling pairwise correlations, z-score alerts, historical context."""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import datetime
import json
import streamlit as st
import streamlit.components.v1 as components
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import yfinance as yf

from design_system import (
    ACCENT_CYAN, ACCENT_GREEN, ACCENT_RED, ACCENT_AMBER,
    BG_CARD, BG_CARD_HOVER, BG_PRIMARY, BORDER, TEXT_MUTED, TEXT_SECONDARY, TEXT_PRIMARY,
    apply_theme, metric_card, section_header, get_plotly_layout,
)

apply_theme()

# ── Constants ──────────────────────────────────────────────────────────────────

DEFAULT_PAIRS = [
    ("SPY", "QQQ"),      # large cap equity
    ("GLD", "TLT"),      # safe haven
    ("SPY", "IWM"),      # large cap vs small cap
    ("BTC-USD", "ETH-USD"),  # crypto majors
    ("SPY", "EEM"),      # US vs emerging markets
]

Z_SCORE_SEVERITY = [
    (-2.5, "Extreme", ACCENT_RED, "pulsing-red"),
    (-2.0, "Significant", ACCENT_AMBER, "glow-orange"),
    (-1.5, "Notable", ACCENT_AMBER, "glow-amber"),
    (np.inf, "Normal", TEXT_MUTED, "normal"),
]

ALERT_LOG_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "correlation_alerts.json")

# ── CSS Animations ─────────────────────────────────────────────────────────────

CORRELATION_BREAK_CSS = """
<style>
@keyframes pulse-red {
    0%, 100% { box-shadow: 0 0 8px rgba(255,23,68,0.4); }
    50% { box-shadow: 0 0 20px rgba(255,23,68,0.8); }
}
@keyframes glow-orange {
    0%, 100% { box-shadow: 0 0 6px rgba(255,193,7,0.3); }
    50% { box-shadow: 0 0 14px rgba(255,193,7,0.6); }
}
@keyframes glow-amber {
    0%, 100% { box-shadow: 0 0 4px rgba(255,193,7,0.2); }
    50% { box-shadow: 0 0 10px rgba(255,193,7,0.4); }
}
.correlation-card {
    display: inline-block;
    background: """ + BG_CARD + """;
    border: 1px solid """ + BORDER + """;
    border-radius: 12px;
    padding: 1rem 1.25rem;
    min-width: 200px;
    text-align: center;
    transition: all 0.3s ease;
}
.correlation-card:hover {
    background: """ + BG_CARD_HOVER + """;
}
.correlation-card.normal {
    border-color: """ + BORDER + """;
    box-shadow: none;
}
.correlation-card.notable {
    border-color: """ + ACCENT_AMBER + """40;
    animation: glow-amber 2s ease-in-out infinite;
}
.correlation-card.significant {
    border-color: """ + ACCENT_AMBER + """80;
    animation: glow-orange 2s ease-in-out infinite;
}
.correlation-card.extreme {
    border-color: """ + ACCENT_RED + """80;
    animation: pulse-red 1.5s ease-in-out infinite;
}
.status-pill {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 14px;
    font-family: DM Sans, sans-serif;
    font-size: 11px;
    font-weight: 500;
    margin-top: 6px;
}
.status-pill.normal {
    background: """ + BORDER + """;
    color: """ + TEXT_MUTED + """;
}
.status-pill.notable {
    background: """ + ACCENT_AMBER + """20;
    color: """ + ACCENT_AMBER + """;
    border: 1px solid """ + ACCENT_AMBER + """40;
}
.status-pill.significant {
    background: """ + ACCENT_AMBER + """25;
    color: """ + ACCENT_AMBER + """;
    border: 1px solid """ + ACCENT_AMBER + """60;
}
.status-pill.extreme {
    background: """ + ACCENT_RED + """20;
    color: """ + ACCENT_RED + """;
    border: 1px solid """ + ACCENT_RED + """40;
}
.correlation-value {
    font-family: JetBrains Mono, monospace;
    font-size: 32px;
    font-weight: 700;
    color: """ + TEXT_PRIMARY + """;
}
.z-score-text {
    font-family: JetBrains Mono, monospace;
    font-size: 13px;
    color: """ + TEXT_MUTED + """;
    margin-top: 2px;
}
</style>
"""

# ── Data helpers ────────────────────────────────────────────────────────────────

@st.cache_data(ttl=300)
def download_pair_data(tickers: list[str], days: int = 3 * 365) -> pd.DataFrame:
    """Download daily Close prices for a list of tickers."""
    try:
        raw = yf.download(tickers, period=f"{days}d", progress=False)
    except Exception:
        # Fallback: try using start date instead of period
        end_date = datetime.datetime.now()
        start_date = end_date - datetime.timedelta(days=days)
        raw = yf.download(tickers, start=start_date, end=end_date, progress=False)

    if raw.empty:
        return pd.DataFrame()

    # Flatten MultiIndex columns if present
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    # Use Close prices only
    if "Close" in raw.columns.names if hasattr(raw.columns, "names") else False:
        prices = raw["Close"]
    else:
        prices = raw

    return prices.dropna()

@st.cache_data(ttl=300)
def compute_rolling_correlation(prices1: pd.Series, prices2: pd.Series,
                                window: int) -> pd.Series:
    """Rolling correlation between two price series at a given window."""
    returns1 = prices1.pct_change().dropna()
    returns2 = prices2.pct_change().dropna()

    # Align dates
    common_dates = returns1.index.intersection(returns2.index)
    returns1 = returns1[common_dates]
    returns2 = returns2[common_dates]

    rolling_corr = returns1.rolling(window).corr(returns2)
    return rolling_corr.dropna()

@st.cache_data(ttl=300)
def compute_correlation_stats(rolling_corr: pd.Series) -> dict:
    """Compute historical mean and std of a rolling correlation series."""
    if len(rolling_corr) < 60:
        return {"mean": 0.0, "std": 1.0}  # avoid division by zero

    return {
        "mean": rolling_corr.mean(),
        "std": rolling_corr.std(),
    }

# ── Z-score & severity ─────────────────────────────────────────────────────────

def compute_z_score(rolling_corr: pd.Series, stats: dict) -> float:
    """Z-score of current correlation against historical distribution."""
    current = rolling_corr.iloc[-1]
    if stats["std"] == 0:
        return 0.0
    return (current - stats["mean"]) / stats["std"]

def get_severity(z_score: float) -> tuple[str, str, str]:
    """Return (severity_label, color, css_class) for a z-score."""
    for threshold, label, color, css_class in Z_SCORE_SEVERITY:
        if z_score < threshold:
            return label, color, css_class
    return "Normal", TEXT_MUTED, "normal"

# ── Alert logging ───────────────────────────────────────────────────────────────

def load_alerts() -> list[dict]:
    """Load alert log from JSON file."""
    try:
        with open(ALERT_LOG_FILE, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return []

def save_alert(pair: str, z_score: float, severity: str):
    """Append alert to JSON log file."""
    os.makedirs(os.path.dirname(ALERT_LOG_FILE), exist_ok=True)
    alerts = load_alerts()
    alerts.append({
        "timestamp": datetime.datetime.now().isoformat(),
        "pair": pair,
        "z_score": round(z_score, 3),
        "severity": severity,
    })
    with open(ALERT_LOG_FILE, "w") as f:
        json.dump(alerts, f, indent=2)

# ── Historical context ──────────────────────────────────────────────────────────

@st.cache_data(ttl=600)
def find_historical_breaks(prices1: pd.Series, prices2: pd.Series,
                            target_z: float, window: int = 60,
                            lookback_days: int = 3 * 365) -> pd.DataFrame:
    """Find historical periods where z-score was similar to target,
    and compute forward returns for each asset at 5/10/20 day horizons."""
    rolling_60 = compute_rolling_correlation(prices1, prices2, 60)
    rolling_20 = compute_rolling_correlation(prices1, prices2, 20)
    stats = compute_correlation_stats(rolling_60)

    if stats["std"] == 0:
        return pd.DataFrame()

    z_scores_60 = (rolling_60 - stats["mean"]) / stats["std"]

    # Find indices where z-score was near the target (within 0.5)
    tolerance = 0.5
    break_mask = z_scores_60.between(target_z - tolerance, target_z + tolerance)
    break_indices = z_scores_60[break_mask].index

    if break_indices.empty:
        return pd.DataFrame()

    # For each break, compute forward returns
    returns1 = prices1.pct_change().dropna()
    returns2 = prices2.pct_change().dropna()
    common_dates = returns1.index.intersection(returns2.index)
    returns1 = returns1[common_dates]
    returns2 = returns2[common_dates]

    rows = []
    for break_date in break_indices:
        try:
            break_idx = common_dates.get_loc(break_date)
        except KeyError:
            continue

        # Forward returns for each horizon
        horizons = {"5d": 5, "10d": 10, "20d": 20}
        row = {"date": break_date, "z_score": round(z_scores_60[break_date], 3)}

        for horizon_label, horizon_days in horizons.items():
            forward_idx = min(break_idx + horizon_days, len(common_dates) - 1)
            if forward_idx > break_idx:
                ret1 = (prices1.iloc[forward_idx] / prices1.iloc[break_idx]) - 1
                ret2 = (prices2.iloc[forward_idx] / prices2.iloc[break_idx]) - 1
            else:
                ret1 = np.nan
                ret2 = np.nan
            row[f"asset1_return_{horizon_label}"] = ret1 if not np.isnan(ret1) else None
            row[f"asset2_return_{horizon_label}"] = ret2 if not np.isnan(ret2) else None

        rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # Average returns at each horizon
    avg_row = {"date": "AVERAGE", "z_score": round(df["z_score"].mean(), 3)}
    for horizon_label, horizon_days in horizons.items():
        for asset_label in ["asset1", "asset2"]:
            col = f"{asset_label}_return_{horizon_label}"
            avg_row[col] = round(df[col].mean(), 4) if not df[col].isna().all() else None

    df = pd.concat([df, pd.DataFrame([avg_row])], ignore_index=True)
    return df

# ── Build status cards ──────────────────────────────────────────────────────────

def build_status_cards(pair_results: dict, selected_pair: str) -> str:
    """Build HTML row of status cards for all pairs."""
    cards = []
    for pair, result in pair_results.items():
        pair_name = f"{pair[0]} / {pair[1]}"
        corr = result["correlation_60"]
        z_score = result["z_score"]
        severity, color, css_class = result["severity"]

        is_selected = (pair == selected_pair)
        card_border = f"border: 2px solid {color};" if is_selected else ""

        cards.append(f"""
        <div class="correlation-card {css_class}"
             style="cursor:pointer;{card_border}"
             onclick="parent.location.href='?pair={pair[0]}+{pair[1]}'"
             onmouseover="this.style.background='{BG_CARD_HOVER}'"
             onmouseout="this.style.background='{BG_CARD}'">
            <div style="font-family:DM Sans,sans-serif;font-size:13px;font-weight:700;
                         color:{TEXT_PRIMARY};margin-bottom:6px;">{pair_name}</div>
            <div class="correlation-value">{corr:.3f}</div>
            <div class="z-score-text">z = {z_score:+.2f}</div>
            <span class="status-pill {css_class}">{severity}</span>
        </div>
        """)

    return "<div style='display:flex;flex-wrap:wrap;gap:1rem;'>" + "".join(cards) + "</div>"

# ── Build main correlation chart ────────────────────────────────────────────────

def build_correlation_chart(pair: tuple, rolling_corr_60: pd.Series,
                            rolling_corr_20: pd.Series, stats: dict) -> go.Figure:
    """Main chart with rolling 60-day correlation, mean line, threshold, and break shading."""
    dates = rolling_corr_60.index.tolist()
    corr_60 = rolling_corr_60.values
    corr_20 = rolling_corr_20.values

    mean_line = stats["mean"]
    threshold = mean_line - 2 * stats["std"]

    # Build area fill between correlation and threshold
    # Above threshold: green tint; below threshold: red tint
    area_above_x = []
    area_above_y = []
    area_below_x = []
    area_below_y = []

    for i in range(len(dates)):
        if corr_60[i] >= threshold:
            area_above_x.extend([dates[i], dates[i]])
            area_above_y.extend([threshold, corr_60[i]])
        else:
            area_below_x.extend([dates[i], dates[i]])
            area_below_y.extend([threshold, corr_60[i]])

    # Find contiguous regions for efficient area rendering
    area_traces = []

    # Green area (above threshold)
    if area_above_x:
        area_traces.append(go.Scatter(
            x=area_above_x,
            y=area_above_y,
            mode="none",
            fillcolor="rgba(0,230,118,0.08)",
            fill="toself",
            showlegend=False,
            hoverinfo="skip",
        ))

    # Red area (below threshold)
    if area_below_x:
        area_traces.append(go.Scatter(
            x=area_below_x,
            y=area_below_y,
            mode="none",
            fillcolor="rgba(255,23,68,0.08)",
            fill="toself",
            showlegend=False,
            hoverinfo="skip",
        ))

    # Historical break periods (low red opacity)
    threshold_indices = np.where(corr_60 < threshold)[0]
    if len(threshold_indices) > 0:
        # Group consecutive indices into regions
        regions = []
        start = threshold_indices[0]
        for i in range(1, len(threshold_indices)):
            if threshold_indices[i] != threshold_indices[i-1] + 1:
                regions.append((start, threshold_indices[i-1]))
                start = threshold_indices[i]
        regions.append((start, threshold_indices[-1]))

        for r_start, r_end in regions:
            area_traces.append(go.Scatter(
                x=dates[r_start:r_end+1],
                y=[threshold] * (r_end - r_start + 1),
                mode="lines",
                fill="tozeroy",
                fillcolor="rgba(255,23,68,0.05)",
                line=dict(width=0),
                showlegend=False,
                hoverinfo="skip",
            ))

    # 60-day correlation line
    area_traces.append(go.Scatter(
        x=dates,
        y=corr_60,
        mode="lines",
        name="60-day",
        line=dict(color=ACCENT_CYAN, width=2),
        hovertemplate="Correlation: %{y:.3f}<br>Date: %{x|%Y-%m-%d}<extra></extra>",
    ))

    # 20-day correlation line
    area_traces.append(go.Scatter(
        x=dates,
        y=corr_20,
        mode="lines",
        name="20-day",
        line=dict(color=ACCENT_VIOLET, width=1.5, dash="dot"),
        hovertemplate="Correlation: %{y:.3f}<br>Date: %{x|%Y-%m-%d}<extra></extra>",
    ))

    # Mean line
    area_traces.append(go.Scatter(
        x=dates,
        y=[mean_line] * len(dates),
        mode="lines",
        name="Mean",
        line=dict(color=ACCENT_CYAN, width=1, dash="dash"),
        opacity=0.6,
        hoverinfo="skip",
        showlegend=True,
    ))

    # Threshold line (-2 std)
    area_traces.append(go.Scatter(
        x=dates,
        y=[threshold] * len(dates),
        mode="lines",
        name="-2σ Threshold",
        line=dict(color=ACCENT_RED, width=1, dash="dash"),
        opacity=0.6,
        hoverinfo="skip",
        showlegend=True,
    ))

    # Current correlation marker
    area_traces.append(go.Scatter(
        x=[dates[-1]],
        y=[corr_60[-1]],
        mode="markers",
        marker=dict(
            size=10,
            color=ACCENT_CYAN,
            line=dict(color=BG_PRIMARY, width=2),
            symbol="circle",
        ),
        name="Current",
        hovertext=f"{corr_60[-1]:.3f}",
        hoverinfo="text",
        showlegend=True,
    ))

    layout = get_plotly_layout(
        height=450,
        xaxis=dict(
            type="date",
            showspikes=True,
            spikemode="across",
        ),
        yaxis=dict(
            title="Correlation",
            showspikes=True,
            spikemode="across",
            range=[-1.0, 1.0],
        ),
    )

    fig = go.Figure(data=area_traces, layout=layout)
    return fig

# ── Build 20d/60d comparison ────────────────────────────────────────────────────

def build_comparison_chart(rolling_corr_20: pd.Series, rolling_corr_60: pd.Series,
                            stats: dict) -> go.Figure:
    """Two small line charts side by side showing both rolling windows."""
    dates = rolling_corr_60.index.tolist()

    fig = go.Figure()

    # 20-day
    fig.add_trace(go.Scatter(
        x=dates,
        y=rolling_corr_20.values,
        mode="lines",
        name="20-day",
        line=dict(color=ACCENT_VIOLET, width=1.5),
        hovertemplate="Correlation: %{y:.3f}<br>Date: %{x|%Y-%m-%d}<extra></extra>",
    ))

    # 60-day
    fig.add_trace(go.Scatter(
        x=dates,
        y=rolling_corr_60.values,
        mode="lines",
        name="60-day",
        line=dict(color=ACCENT_CYAN, width=1.5),
        hovertemplate="Correlation: %{y:.3f}<br>Date: %{x|%Y-%m-%d}<extra></extra>",
    ))

    # Mean line
    fig.add_trace(go.Scatter(
        x=dates,
        y=[stats["mean"]] * len(dates),
        mode="lines",
        name="Mean",
        line=dict(color=ACCENT_CYAN, width=1, dash="dash"),
        opacity=0.6,
        hoverinfo="skip",
        showlegend=True,
    ))

    layout = get_plotly_layout(
        height=250,
        xaxis=dict(
            type="date",
            showspikes=True,
            spikemode="across",
        ),
        yaxis=dict(
            title="Correlation",
            showspikes=True,
            spikemode="across",
            range=[-1.0, 1.0],
        ),
    )

    fig.update_layout(layout)
    return fig

# ── Build historical context table ──────────────────────────────────────────────

def build_historical_context_table(df: pd.DataFrame, pair: tuple) -> str:
    """Build HTML table of historical break instances with forward returns."""
    if df.empty:
        return f"""
        <div style="font-family:DM Sans,sans-serif;font-size:13px;color:{TEXT_MUTED};
                     text-align:center;padding:2rem;">
            No historical breaks matching this z-score found.
        </div>
        """

    # Asset names
    asset1, asset2 = pair

    # Build header
    header = f"""
    <table style="width:100%;border-collapse:collapse;margin-bottom:0.5rem;">
    <thead>
    <tr style="border-bottom:1px solid {BORDER};">
        <th style="text-align:left;padding:8px 12px;font-size:10px;
                   text-transform:uppercase;letter-spacing:1px;
                   font-family:DM Sans,sans-serif;color:{TEXT_MUTED};">Date</th>
        <th style="text-align:left;padding:8px 12px;font-size:10px;
                   text-transform:uppercase;letter-spacing:1px;
                   font-family:DM Sans,sans-serif;color:{TEXT_MUTED};">Z-Score</th>
        <th style="text-align:right;padding:8px 12px;font-size:10px;
                   text-transform:uppercase;letter-spacing:1px;
                   font-family:DM Sans,sans-serif;color:{TEXT_MUTED};">
            {asset1} 5d</th>
        <th style="text-align:right;padding:8px 12px;font-size:10px;
                   text-transform:uppercase;letter-spacing:1px;
                   font-family:DM Sans,sans-serif;color:{TEXT_MUTED};">
            {asset1} 10d</th>
        <th style="text-align:right;padding:8px 12px;font-size:10px;
                   text-transform:uppercase;letter-spacing:1px;
                   font-family:DM Sans,sans-serif;color:{TEXT_MUTED};">
            {asset1} 20d</th>
        <th style="text-align:right;padding:8px 12px;font-size:10px;
                   text-transform:uppercase;letter-spacing:1px;
                   font-family:DM Sans,sans-serif;color:{TEXT_MUTED};">
            {asset2} 5d</th>
        <th style="text-align:right;padding:8px 12px;font-size:10px;
                   text-transform:uppercase;letter-spacing:1px;
                   font-family:DM Sans,sans-serif;color:{TEXT_MUTED};">
            {asset2} 10d</th>
        <th style="text-align:right;padding:8px 12px;font-size:10px;
                   text-transform:uppercase;letter-spacing:1px;
                   font-family:DM Sans,sans-serif;color:{TEXT_MUTED};">
            {asset2} 20d</th>
    </tr>
    </thead>
    <tbody>
    """

    # Build rows
    rows_html = ""
    for _, row in df.iterrows():
        is_avg = row["date"] == "AVERAGE"
        row_style = f"background:{BG_CARD_HOVER};font-weight:700;" if is_avg else ""

        cell_color = lambda x: (ACCENT_GREEN if x > 0.01 else (ACCENT_RED if x < -0.01 else TEXT_MUTED)) if x is not None else TEXT_MUTED
        cell_fmt = lambda x: f"{x:+.2%}" if x is not None else "—"

        cells = []
        for col, label in [("z_score", "")]:
            cells.append(f'<td style="padding:6px 12px;font-size:11px;font-family:JetBrains Mono,monospace;color:{TEXT_SECONDARY};{row_style}">{row[col]:.2f}</td>')

        for horizon in ["5d", "10d", "20d"]:
            for asset_label, asset_name in [("asset1", asset1), ("asset2", asset2)]:
                val = row.get(f"{asset_label}_return_{horizon}")
                if val is not None:
                    color = cell_color(val)
                    cells.append(f'<td style="padding:6px 12px;font-size:11px;font-family:JetBrains Mono,monospace;text-align:right;color:{color};{row_style}">{cell_fmt(val)}</td>')
                else:
                    cells.append(f'<td style="padding:6px 12px;font-size:11px;font-family:JetBrains Mono,monospace;text-align:right;color:{TEXT_MUTED};{row_style}">—</td>')

        date_display = row["date"] if is_avg else row["date"].strftime("%Y-%m-%d")

        rows_html += f"""
        <tr style="{row_style}">
            <td style="padding:6px 12px;font-size:11px;font-family:JetBrains Mono,monospace;color:{TEXT_SECONDARY};{row_style}">{date_display}</td>
            {" ".join(cells)}
        </tr>
        """

    table_html = header + rows_html + "</tbody></table>"

    # Caveat
    caveat = f"""
    <div style="margin-top:0.75rem;padding:0.75rem 1rem;
                background:{BG_CARD};border:1px solid {BORDER};border-radius:8px;
                font-family:DM Sans,sans-serif;font-size:11px;color:{TEXT_MUTED};
                font-style:italic;">
        ⚠ Past correlation breaks do not guarantee similar outcomes. Correlations can drift without signaling directional moves.
    </div>
    """

    return table_html + caveat

# ── Build alert log ─────────────────────────────────────────────────────────────

def build_alert_log() -> str:
    """Build HTML for alert log section."""
    alerts = load_alerts()

    if not alerts:
        return f"""
        <div style="font-family:DM Sans,sans-serif;font-size:13px;color:{TEXT_MUTED};
                     text-align:center;padding:2rem;">
            No alerts yet. Breaks will be logged here.
        </div>
        """

    # Limit to most recent 50
    alerts = alerts[-50:]

    header = f"""
    <table style="width:100%;border-collapse:collapse;">
    <thead>
    <tr style="border-bottom:1px solid {BORDER};">
        <th style="text-align:left;padding:8px 12px;font-size:10px;
                   text-transform:uppercase;letter-spacing:1px;
                   font-family:DM Sans,sans-serif;color:{TEXT_MUTED};">Timestamp</th>
        <th style="text-align:left;padding:8px 12px;font-size:10px;
                   text-transform:uppercase;letter-spacing:1px;
                   font-family:DM Sans,sans-serif;color:{TEXT_MUTED};">Pair</th>
        <th style="text-align:right;padding:8px 12px;font-size:10px;
                   text-transform:uppercase;letter-spacing:1px;
                   font-family:DM Sans,sans-serif;color:{TEXT_MUTED};">Z-Score</th>
        <th style="text-align:left;padding:8px 12px;font-size:10px;
                   text-transform:uppercase;letter-spacing:1px;
                   font-family:DM Sans,sans-serif;color:{TEXT_MUTED};">Severity</th>
    </tr>
    </thead>
    <tbody>
    """

    rows_html = ""
    for alert in alerts:
        severity, color, css_class = get_severity(alert["z_score"])
        timestamp = datetime.datetime.fromisoformat(alert["timestamp"])
        time_str = timestamp.strftime("%Y-%m-%d %H:%M")

        rows_html += f"""
        <tr style="border-bottom:1px solid {BORDER};">
            <td style="padding:6px 12px;font-size:11px;font-family:JetBrains Mono,monospace;color:{TEXT_SECONDARY};">
                {time_str}
            </td>
            <td style="padding:6px 12px;font-size:11px;font-family:JetBrains Mono,monospace;color:{TEXT_PRIMARY};">
                {alert['pair']}
            </td>
            <td style="padding:6px 12px;font-size:11px;font-family:JetBrains Mono,monospace;
                       text-align:right;color:{color};">
                {alert['z_score']:+.2f}
            </td>
            <td style="padding:6px 12px;font-size:11px;">
                <span class="status-pill {css_class}">{severity}</span>
            </td>
        </tr>
        """

    return header + rows_html + "</tbody></table>"

# ── Sidebar ─────────────────────────────────────────────────────────────────────

def render_sidebar():
    """Sidebar with pair management, window selector, z-score threshold, date range."""
    with st.sidebar:
        st.subheader("Pair Management")

        # Default pairs
        st.session_state.setdefault("pairs", [list(p) for p in DEFAULT_PAIRS])

        # Show current pairs
        for i, pair in enumerate(st.session_state.pairs):
            pair_name = f"{pair[0]} / {pair[1]}"
            col1, col2 = st.columns([4, 1])
            with col1:
                col3, col4 = st.columns(2)
                with col3:
                    ticker1 = st.text_input(
                        f"Ticker 1 — {i}", value=pair[0],
                        key=f"pair1_{i}", label_visibility="collapsed",
                    )
                with col4:
                    ticker2 = st.text_input(
                        f"Ticker 2 — {i}", value=pair[1],
                        key=f"pair2_{i}", label_visibility="collapsed",
                    )
            with col2:
                if st.button("✕", key=f"remove_{i}", use_container_width=True):
                    st.session_state.pairs.pop(i)
                    st.rerun()

            # Update pair if changed
            if ticker1 != pair[0] or ticker2 != pair[1]:
                st.session_state.pairs[i] = [ticker1.strip().upper(), ticker2.strip().upper()]

        # Add pair
        with st.expander("Add Pair", expanded=False):
            new_col1, new_col2, new_col3 = st.columns([3, 1, 3])
            with new_col1:
                new_ticker1 = st.text_input("Ticker 1", key="new_ticker1",
                                            label_visibility="collapsed",
                                            placeholder="e.g. SPY")
            with new_col2:
                st.markdown("**/**")
            with new_col3:
                new_ticker2 = st.text_input("Ticker 2", key="new_ticker2",
                                            label_visibility="collapsed",
                                            placeholder="e.g. QQQ")
            if new_ticker1.strip() and new_ticker2.strip():
                if st.button("Add", key="add_pair", use_container_width=True):
                    t1 = new_ticker1.strip().upper()
                    t2 = new_ticker2.strip().upper()
                    if t1 and t2 and [t1, t2] not in st.session_state.pairs:
                        st.session_state.pairs.append([t1, t2])
                        st.rerun()

        # Correlation window
        st.subheader("Settings")
        st.session_state.setdefault("correlation_window", 60)
        window = st.selectbox(
            "Correlation Window",
            options=[20, 60, 120],
            index=[20, 60, 120].index(st.session_state.correlation_window),
            key="correlation_window",
            help="Rolling window for correlation calculation",
        )
        if window != st.session_state.correlation_window:
            st.session_state.correlation_window = window
            st.rerun()

        # Z-score threshold
        st.session_state.setdefault("z_threshold", -2.0)
        threshold = st.slider(
            "Z-Score Alert Threshold",
            min_value=-4.0,
            max_value=-1.0,
            value=st.session_state.z_threshold,
            step=0.1,
            key="z_threshold",
            help="Alert when z-score drops below this value",
        )
        if threshold != st.session_state.z_threshold:
            st.session_state.z_threshold = threshold
            st.rerun()

        # Date range
        st.subheader("Date Range")
        col1, col2 = st.sidebar.columns(2)
        with col1:
            st.session_state.setdefault("start_date",
                                        datetime.datetime.now() - datetime.timedelta(days=3*365))
            start_date = st.date_input("From", value=st.session_state.start_date)
            if start_date != st.session_state.start_date:
                st.session_state.start_date = start_date
        with col2:
            st.session_state.setdefault("end_date", datetime.datetime.now())
            end_date = st.date_input("To", value=st.session_state.end_date)
            if end_date != st.session_state.end_date:
                st.session_state.end_date = end_date

        # Check button
        st.markdown(f'<hr style="border:1px solid {BORDER};border-radius:2px;" '
                    'noshade="noshade" style="width:100%;margin:1rem 0;">',
                    unsafe_allow_html=True)
        if st.button("Check Correlations", type="primary", use_container_width=True):
            st.session_state._run_clicked = True
            st.rerun()

        # Refresh
        if st.button("Refresh", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

# ── Main ────────────────────────────────────────────────────────────────────────

def main():
    st.set_page_config(
        page_title="Correlation Break Detector",
        page_icon="📊",
        layout="wide",
    )

    st.html(CORRELATION_BREAK_CSS)

    # ── Sidebar ─────────────────────────────────────────────────────────────
    render_sidebar()

    # ── Get pairs ───────────────────────────────────────────────────────────
    pairs = st.session_state.pairs
    if not pairs:
        st.markdown(
            '<div style="display:flex;align-items:center;justify-content:center;'
            'height:60vh;font-family:DM Sans,sans-serif;color:'
            + TEXT_MUTED + ';font-size:16px;">'
            "Add pairs in the sidebar to begin</div>",
            unsafe_allow_html=True,
        )
        return

    # ── Compute correlation for all pairs ───────────────────────────────────
    pair_results = {}
    pair_data = {}
    pair_stats = {}

    for pair in pairs:
        ticker1, ticker2 = pair
        pair_name = f"{ticker1} / {ticker2}"

        try:
            with st.spinner(f"Computing correlation for {pair_name}..."):
                prices = download_pair_data([ticker1, ticker2])
                if prices.empty or len(prices) < 60:
                    st.warning(f"Not enough data for {pair_name}")
                    continue

                rolling_60 = compute_rolling_correlation(prices[ticker1], prices[ticker2], 60)
                rolling_20 = compute_rolling_correlation(prices[ticker1], prices[ticker2], 20)
                stats = compute_correlation_stats(rolling_60)
                z_score = compute_z_score(rolling_60, stats)
                severity, color, css_class = get_severity(z_score)

                pair_results[pair] = {
                    "correlation_60": rolling_60.iloc[-1],
                    "correlation_20": rolling_20.iloc[-1],
                    "z_score": z_score,
                    "severity": (severity, color, css_class),
                    "rolling_60": rolling_60,
                    "rolling_20": rolling_20,
                    "stats": stats,
                }
                pair_data[pair] = prices
                pair_stats[pair] = stats

        except Exception as e:
            st.warning(f"Error computing correlation for {pair_name}: {e}")
            continue

    if not pair_results:
        st.error("No correlation data computed. Check ticker symbols and date range.")
        return

    # ── Check for alerts ────────────────────────────────────────────────────
    z_threshold = st.session_state.z_threshold
    for pair, result in pair_results.items():
        if result["z_score"] < z_threshold:
            severity = result["severity"][0]
            pair_name = f"{pair[0]} / {pair[1]}"
            save_alert(pair_name, result["z_score"], severity)

    # ── Determine selected pair ─────────────────────────────────────────────
    qp = st.query_params
    selected_pair_tickers = qp.get("pair", None)
    if selected_pair_tickers:
        selected_pair = tuple(selected_pair_tickers.split("+"))
    else:
        # Default to first pair with a break
        for pair in pairs:
            if pair in pair_results and pair_results[pair]["z_score"] < -1.5:
                selected_pair = pair
                break
        else:
            selected_pair = pairs[0] if pairs in pair_results else pairs[0]

    # ── Status overview cards ───────────────────────────────────────────────
    st.markdown(
        '<div style="font-family:DM Sans,sans-serif;font-size:11px;'
        'text-transform:uppercase;letter-spacing:3px;color:'
        + TEXT_MUTED + ';margin-bottom:0.5rem;">Pair Status</div>',
        unsafe_allow_html=True,
    )
    st.markdown(build_status_cards(pair_results, selected_pair), unsafe_allow_html=True)

    # ── Selected pair analysis ──────────────────────────────────────────────
    if selected_pair not in pair_results:
        selected_pair = pairs[0]

    ticker1, ticker2 = selected_pair
    pair_name = f"{ticker1} / {ticker2}"
    result = pair_results[selected_pair]
    severity, severity_color, css_class = result["severity"]
    stats = result["stats"]
    rolling_60 = result["rolling_60"]
    rolling_20 = result["rolling_20"]

    # ── Main correlation chart ──────────────────────────────────────────────
    st.markdown(section_header(f"Correlation — {pair_name}"))
    corr_fig = build_correlation_chart(selected_pair, rolling_60, rolling_20, stats)
    st.plotly_chart(corr_fig, use_container_width=True, config={"displayModeBar": False})

    # ── 20d/60d comparison ──────────────────────────────────────────────────
    st.markdown(section_header("20-Day vs 60-Day Comparison"))
    comp_fig = build_comparison_chart(rolling_20, rolling_60, stats)
    st.plotly_chart(comp_fig, use_container_width=True, config={"displayModeBar": False})

    # ── Current values ──────────────────────────────────────────────────────
    current_corr_60 = result["correlation_60"]
    current_corr_20 = result["correlation_20"]
    z_score = result["z_score"]

    st.markdown(section_header("Current Values"))
    html_row(
        f'<span style="font-family:JetBrains Mono,monospace;font-size:14px;color:{TEXT_SECONDARY};">'
        f'60-day correlation: <b style="color:{TEXT_PRIMARY};">{current_corr_60:.3f}</b></span>',
        f'<span style="font-family:JetBrains Mono,monospace;font-size:14px;color:{TEXT_SECONDARY};">'
        f'20-day correlation: <b style="color:{ACCENT_VIOLET};">{current_corr_20:.3f}</b></span>',
        f'<span style="font-family:JetBrains Mono,monospace;font-size:14px;color:{TEXT_SECONDARY};">'
        f'Z-score: <b style="color:{severity_color};">{z_score:+.2f}</b></span>',
        f'<span style="font-family:DM Sans,sans-serif;font-size:14px;color:{severity_color};">'
        f'{severity}</span>',
    )

    # ── Historical context (only when notable or worse) ─────────────────────
    if z_score < -1.5:
        st.markdown(section_header("Historical Context"))

        with st.spinner("Computing historical context..."):
            hist_df = find_historical_breaks(
                pair_data[selected_pair][ticker1],
                pair_data[selected_pair][ticker2],
                target_z=z_score,
                window=60,
            )

        table_html = build_historical_context_table(hist_df, selected_pair)
        st.markdown(table_html, unsafe_allow_html=True)

    # ── Alert log ───────────────────────────────────────────────────────────
    st.markdown(section_header("Alert Log"))
    alert_html = build_alert_log()
    st.markdown(alert_html, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
