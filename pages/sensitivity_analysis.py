"""Sensitivity Analysis — parameter robustness for MA-crossover strategy on SPY."""

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

import yfinance as yf

from design_system import (
    ACCENT_CYAN, ACCENT_GREEN, ACCENT_RED, ACCENT_AMBER,
    BG_CARD, BG_CARD_HOVER, BG_PRIMARY, BORDER, TEXT_MUTED, TEXT_SECONDARY, TEXT_PRIMARY,
    metric_card, section_header, get_plotly_layout, pnl_color, GRID_LINE,
)

# ── Page setup ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Sensitivity",
    page_icon="🔬",
    layout="wide",
)

# ── Constants ───────────────────────────────────────────────────────────────

ROBUST_THRESHOLD = 70
FRAGILE_THRESHOLD = 40
STABLE_ZONE_PCT = 0.20  # 20% stable zone

HEATMAP_COLORS = {
    "deep_green": "#00c853",
    "cyan": "#00d4ff",
    "amber": "#ffc107",
    "red": "#ff1744",
}

# ── Demo strategy: MA crossover ─────────────────────────────────────────────

PARAM_CONFIG = {
    "fast_ma": {
        "base": 10,
        "range_min": 5,
        "range_max": 30,
        "step": 1,
    },
    "slow_ma": {
        "base": 50,
        "range_min": 20,
        "range_max": 100,
        "step": 5,
    },
    "stop_loss_pct": {
        "base": 2.0,
        "range_min": 0.5,
        "range_max": 5.0,
        "step": 0.5,
    },
    "take_profit_pct": {
        "base": 4.0,
        "range_min": 1.0,
        "range_max": 10.0,
        "step": 0.5,
    },
}

def run_ma_crossover_backtest(
    df: pd.DataFrame,
    fast_ma: int,
    slow_ma: int,
    stop_loss_pct: float,
    take_profit_pct: float,
) -> dict:
    """Run a simple MA-crossover backtest with stop-loss and take-profit.

    Buy: fast MA crosses above slow MA
    Sell: fast MA crosses below slow MA or stop-loss/take-profit hit
    """
    data = df.copy()
    data["fast_ma"] = data["Close"].rolling(int(fast_ma)).mean()
    data["slow_ma"] = data["Close"].rolling(int(slow_ma)).mean()
    data.dropna(inplace=True)
    data = data.reset_index(drop=True)

    position = 0
    entry_price = 0.0
    trades = []
    equity = 1.0
    equity_curve = [1.0]

    for i in range(1, len(data)):
        prev_fast = data["fast_ma"].iloc[i - 1]
        prev_slow = data["slow_ma"].iloc[i - 1]
        curr_fast = data["fast_ma"].iloc[i]
        curr_slow = data["slow_ma"].iloc[i]
        price = data["Close"].iloc[i]

        # Buy signal
        if position == 0 and prev_fast <= prev_slow and curr_fast > curr_slow:
            position = 1
            entry_price = price

        # Sell signals
        if position == 1:
            sell = False
            pnl_pct = 0.0

            if prev_fast >= prev_slow and curr_fast < curr_slow:
                # MA cross down
                sell = True
                pnl_pct = (price - entry_price) / entry_price

            if pnl_pct <= -stop_loss_pct / 100:
                sell = True

            if pnl_pct >= take_profit_pct / 100:
                sell = True

            if sell:
                trades.append({
                    "entry": entry_price,
                    "exit": price,
                    "pnl_pct": pnl_pct,
                    "duration": i,
                })
                equity *= (1 + pnl_pct)
                position = 0
                entry_price = 0.0

        equity_curve.append(equity)

    trades_df = pd.DataFrame(trades) if trades else pd.DataFrame(columns=["entry", "exit", "pnl_pct", "duration"])

    total_return = equity - 1.0
    sharpe = 0.0
    max_dd = 0.0
    win_rate = 0.0

    if len(equity_curve) > 1:
        daily_rets = np.diff(equity_curve) / np.array(equity_curve[:-1])
        if daily_rets.std() > 0:
            sharpe = (daily_rets.mean() / daily_rets.std()) * np.sqrt(252)

        peak = 1.0
        for val in equity_curve:
            peak = max(peak, val)
            dd = (peak - val) / peak
            max_dd = max(max_dd, dd)

    if len(trades_df) > 0:
        win_rate = (trades_df["pnl_pct"] > 0).mean()

    return {
        "total_return": total_return,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "win_rate": win_rate,
        "num_trades": len(trades_df),
        "equity_curve": equity_curve,
    }

# ── Sensitivity sweep ────────────────────────────────────────────────────────

def sweep_parameter(
    df: pd.DataFrame,
    param_name: str,
    param_range: np.ndarray,
    other_params: dict,
) -> dict:
    """Sweep a single parameter, return results dict."""
    results = {}
    for val in param_range:
        params = {**other_params, param_name: val}
        result = run_ma_crossover_backtest(
            df,
            fast_ma=params["fast_ma"],
            slow_ma=params["slow_ma"],
            stop_loss_pct=params["stop_loss_pct"],
            take_profit_pct=params["take_profit_pct"],
        )
        results[val] = result

    return results

# ── Robustness scoring ───────────────────────────────────────────────────────

def compute_robustness_score(base_result: dict, results: dict, metric: str) -> float:
    """Score 0-100 based on coefficient of variation of performance across range.

    Lower CV = more robust = higher score.
    When base value is 0 (no trades), score all points equally — no degradation.
    """
    values = [results[v][metric] for v in sorted(results.keys())]
    base_val = base_result[metric]

    # If base has no trades (0), no degradation possible — score is perfect
    if base_val == 0 and all(v == 0 for v in values):
        return 100.0

    if len(values) < 2:
        return 100.0

    std_val = np.std(values)
    mean_val = np.mean(values)

    if abs(mean_val) < 1e-10:
        # Mean is effectively 0 — use range-based scoring instead
        max_val = max(values)
        min_val = min(values)
        range_pct = abs(max_val - min_val) / (abs(base_val) + 1e-10)
        score = max(0, min(100, 100 * (1 - range_pct)))
        return score

    cv = std_val / abs(mean_val)

    # Map CV to 0-100 scale: CV=0 → 100, CV=1 → 50, CV=2 → 0
    score = max(0, min(100, 100 * (1 - cv)))
    return score

def classify_robustness(score: float) -> tuple[str, str]:
    """Return (label, color) for robustness score."""
    if score >= ROBUST_THRESHOLD:
        return "Robust", ACCENT_GREEN
    elif score >= FRAGILE_THRESHOLD:
        return "Moderate", ACCENT_AMBER
    else:
        return "Fragile", ACCENT_RED

# ── Chart builders ───────────────────────────────────────────────────────────

def build_robustness_gauge(score: float) -> go.Figure:
    """Circular gauge showing overall robustness score."""
    color = ACCENT_GREEN if score >= ROBUST_THRESHOLD else (ACCENT_AMBER if score >= FRAGILE_THRESHOLD else ACCENT_RED)

    # Arc: 0-100 mapped to 0-180 degrees
    start_angle = 180
    end_angle = 0
    score_angle = start_angle - (score / 100) * 180

    fig = go.Figure()

    # Background arc
    fig.add_trace(go.Scatter(
        x=[np.cos(np.radians(a)) for a in range(start_angle, end_angle - 1, -1)],
        y=[np.sin(np.radians(a)) for a in range(start_angle, end_angle - 1, -1)],
        mode="lines",
        line=dict(color=TEXT_MUTED, width=20),
        hoverinfo="skip",
        showlegend=False,
    ))

    # Colored arc for score
    if score > 0:
        score_angle_int = int(score_angle)
        fig.add_trace(go.Scatter(
            x=[np.cos(np.radians(a)) for a in range(start_angle, score_angle_int - 1, -1)],
            y=[np.sin(np.radians(a)) for a in range(start_angle, score_angle_int - 1, -1)],
            mode="lines",
            line=dict(color=color, width=20),
            hoverinfo="skip",
            showlegend=False,
        ))

    # Score text in center
    fig.add_annotation(
        x=0, y=0,
        text=f'<span style="font-family:JetBrains Mono,monospace;font-size:64px;font-weight:700;color:{color};">{score:.0f}</span>',
        showarrow=False,
        font=dict(size=64, color=color),
    )
    fig.add_annotation(
        x=0, y=0.35,
        text="STRATEGY ROBUSTNESS",
        showarrow=False,
        xref="x",
        yref="y",
        font=dict(
            family="DM Sans, sans-serif",
            size=14,
            color=TEXT_MUTED,
            textcase="upper",
        ),
    )

    fig.update_layout(
        paper_bgcolor=BG_PRIMARY,
        plot_bgcolor=BG_PRIMARY,
        margin=dict(l=20, r=20, t=20, b=20),
        xaxis=dict(showgrid=False, showticklabels=False, zeroline=False, range=[-1.3, 1.3]),
        yaxis=dict(showgrid=False, showticklabels=False, zeroline=False, range=[-1.3, 1.3]),
        height=380,
    )

    return fig

def build_heatmap(
    param_results: dict,
    base_params: dict,
    df: pd.DataFrame,
) -> go.Figure:
    """Heatmap: parameters on Y, metrics on X, colored by degradation from base."""
    metrics = ["total_return", "sharpe", "max_drawdown", "win_rate"]
    param_names = list(PARAM_CONFIG.keys())

    # Base results for each parameter (at base value)
    base_results = {}
    for pname in param_names:
        other = {k: v for k, v in base_params.items() if k != pname}
        base_results[pname] = sweep_parameter(
            df, pname, np.array([base_params[pname]]), other,
        )

    cell_values = np.zeros((len(param_names), len(metrics)))
    cell_colors = []
    cell_texts = []

    for i, pname in enumerate(param_names):
        for j, metric in enumerate(metrics):
            base_val = base_results[pname][base_params[pname]][metric]
            if base_val == 0:
                cell_values[i, j] = 0
                cell_colors.append(HEATMAP_COLORS["amber"])
                cell_texts.append(f"{base_val:.3f}")
            else:
                sweep_res = param_results.get(pname, {})
                vals = [sweep_res[v][metric] for v in sorted(sweep_res.keys())]
                avg_val = np.mean(vals)
                degradation = abs(base_val - avg_val) / abs(base_val)

                cell_values[i, j] = degradation
                if degradation <= 0.10:
                    cell_colors.append(HEATMAP_COLORS["deep_green"])
                elif degradation <= 0.20:
                    cell_colors.append(HEATMAP_COLORS["cyan"])
                elif degradation <= 0.40:
                    cell_colors.append(HEATMAP_COLORS["amber"])
                else:
                    cell_colors.append(HEATMAP_COLORS["red"])
                cell_texts.append(f"{degradation:.1%}")

    metric_labels = ["Return", "Sharpe", "MaxDD", "Win%"]

    fig = go.Figure(data=go.Heatmap(
        z=cell_values,
        x=metric_labels,
        y=[n.replace("_", " ").title() for n in param_names],
        colorscale=[[0, HEATMAP_COLORS["deep_green"]],
                    [0.25, HEATMAP_COLORS["cyan"]],
                    [0.5, HEATMAP_COLORS["amber"]],
                    [1, HEATMAP_COLORS["red"]]],
        zmin=0,
        zmax=0.40,
        text=cell_texts,
        texttemplate="%{text}",
        textfont={"family": "JetBrains Mono, monospace", "size": 12, "color": TEXT_PRIMARY},
        hovertemplate="Parameter: %{y}<br>Metric: %{x}<br>Degradation: %{z:.1%}<extra></extra>",
        showscale=False,
    ))

    fig.update_layout(
        paper_bgcolor=BG_PRIMARY,
        plot_bgcolor=BG_PRIMARY,
        margin=dict(l=80, r=20, t=30, b=60),
        xaxis=dict(
            title=dict(text="", font=dict(size=11)),
            tickfont={"family": "DM Sans, sans-serif", "size": 11, "color": TEXT_SECONDARY},
        ),
        yaxis=dict(
            title=dict(text="", font=dict(size=11)),
            tickfont={"family": "DM Sans, sans-serif", "size": 11, "color": TEXT_SECONDARY},
            autorange="reversed",
        ),
        height=280,
    )

    return fig

def build_param_detail_chart(
    param_name: str,
    param_results: dict,
    base_val: float,
    robustness_score: float,
) -> go.Figure:
    """Line chart: parameter value vs performance metric, with stable zone band."""
    fig = go.Figure()

    # Sort results
    sorted_vals = sorted(param_results.keys())
    x_vals = list(sorted_vals)

    # Determine which metric to show — total_return is primary
    y_vals = [param_results[v]["total_return"] for v in sorted_vals]

    # Base return is at the base parameter value
    base_return = param_results[base_val]["total_return"]

    # Add stable zone band (within 20% of base, use abs for direction-agnostic)
    stable_low = base_return - abs(base_return) * STABLE_ZONE_PCT
    stable_high = base_return + abs(base_return) * STABLE_ZONE_PCT

    fig.add_trace(go.Scatter(
        x=[x_vals[0], x_vals[-1], x_vals[-1], x_vals[0]],
        y=[stable_low, stable_low, stable_high, stable_high],
        fill="toself",
        fillcolor="rgba(0, 200, 83, 0.06)",
        line=dict(width=0),
        showlegend=False,
        hoverinfo="skip",
    ))

    # Performance line
    fig.add_trace(go.Scatter(
        x=x_vals,
        y=y_vals,
        mode="lines+markers",
        name="Total Return",
        line=dict(color=TEXT_PRIMARY, width=2),
        marker=dict(size=5, color=TEXT_PRIMARY),
        hovertemplate="Parameter: %{x:.1f}<br>Return: %{y:.1%}<extra></extra>",
    ))

    # Base value dashed line
    fig.add_shape(
        type="line",
        x0=base_val, x1=base_val,
        y0=min(y_vals) * 1.1, y1=max(y_vals) * 1.1,
        yref="y",
        line=dict(color=ACCENT_CYAN, width=1.5, dash="dash"),
    )

    # Base value annotation
    fig.add_annotation(
        x=base_val,
        y=max(y_vals) * 1.05,
        text=f"Base={base_val:.0f}" if isinstance(base_val, int) else f"Base={base_val:.1f}",
        showarrow=False,
        font=dict(size=10, color=ACCENT_CYAN),
        xref="x",
        yref="y",
    )

    param_display = param_name.replace("_", " ").title()
    label, color = classify_robustness(robustness_score)

    fig.update_layout(
        paper_bgcolor=BG_PRIMARY,
        plot_bgcolor=BG_PRIMARY,
        margin=dict(l=50, r=20, t=40, b=40),
        title=dict(
            text=f"{param_display} · {label} ({robustness_score:.0f}/100)",
            font=dict(
                family="DM Sans, sans-serif",
                size=13,
                color=TEXT_PRIMARY,
            ),
        ),
        xaxis=dict(
            title=dict(text=param_display, font=dict(size=11, color=TEXT_SECONDARY)),
            tickfont={"family": "JetBrains Mono, monospace", "size": 10, "color": TEXT_MUTED},
            gridcolor=GRID_LINE,
        ),
        yaxis=dict(
            title=dict(text="Total Return", font=dict(size=11, color=TEXT_SECONDARY)),
            tickfont={"family": "JetBrains Mono, monospace", "size": 10, "color": TEXT_MUTED},
            tickformat=".0%",
            gridcolor=GRID_LINE,
        ),
        height=280,
        showlegend=False,
    )

    return fig

# ── Interpretation text ──────────────────────────────────────────────────────

def generate_interpretation(param_name: str, score: float) -> str:
    """Generate human-readable interpretation."""
    label, _ = classify_robustness(score)
    config = PARAM_CONFIG[param_name]
    range_str = f"{config['range_min']}-{config['range_max']}"

    if label == "Robust":
        return (
            f"{param_name} is Robust ({score:.0f}/100) — performance stays stable "
            f"across {range_str} range"
        )
    elif label == "Moderate":
        return (
            f"{param_name} is Moderate ({score:.0f}/100) — some sensitivity to parameter "
            f"changes, consider tightening the range"
        )
    else:
        return (
            f"{param_name} is Fragile ({score:.0f}/100) — small changes cause large "
            f"performance swings, possible overfitting"
        )

# ── UI helpers ───────────────────────────────────────────────────────────────

def format_pct(value: float) -> str:
    return f"{value:.1%}"

def format_return(value: float) -> str:
    return f"{value:+.1%}"

# ── URL param parsing ───────────────────────────────────────────────────────

def parse_config():
    """Read analysis config from URL query params, fall back to sidebar inputs."""
    qp = st.query_params
    ticker = qp.get("ticker", "SPY")
    start_date = qp.get("start_date", None)
    end_date = qp.get("end_date", None)

    # Parse dates from URL — format: YYYY-MM-DD
    if start_date:
        try:
            start_date = pd.Timestamp(start_date)
        except ValueError:
            start_date = pd.Timestamp.now() - pd.DateOffset(years=5)
    else:
        start_date = pd.Timestamp.now() - pd.DateOffset(years=5)

    if end_date:
        try:
            end_date = pd.Timestamp(end_date)
        except ValueError:
            end_date = pd.Timestamp.now()
    else:
        end_date = pd.Timestamp.now()

    return ticker, start_date, end_date

# ── Main app ────────────────────────────────────────────────────────────────

def main():
    st.html("""
        <style>
        .stApp::before {
            content: "";
            display: block;
            height: 2px;
            background: """ + ACCENT_CYAN + """;
        }
        [data-testid="stSidebar"] {
            background: """ + BG_CARD + """ !important;
            border-right: 1px solid """ + BORDER + """ !important;
            padding: 1rem !important;
            width: 280px !important;
        }
        [data-testid="stSidebar"] .stSidebarNav {
            display: none !important;
        }
        [data-testid="stSidebar"] label {
            color: """ + TEXT_MUTED + """ !important;
            font-family: DM Sans, sans-serif !important;
            font-size: 11px !important;
            text-transform: uppercase !important;
            letter-spacing: 1px !important;
        }
        [data-testid="stSidebar"] .stSelectbox > div[data-baseweb="select"] {
            background: """ + BG_CARD + """ !important;
            border: 1px solid """ + BORDER + """ !important;
            border-radius: 8px !important;
            color: """ + TEXT_PRIMARY + """ !important;
        }
        [data-testid="stSidebar"] .stButton > button {
            width: 100%;
            font-family: DM Sans, sans-serif !important;
            font-size: 13px !important;
            text-transform: uppercase !important;
            letter-spacing: 1px !important;
            color: """ + TEXT_PRIMARY + """ !important;
            background: """ + BG_CARD + """ !important;
            border: 1px solid """ + BORDER + """ !important;
            border-radius: 8px !important;
        }
        [data-testid="stSidebar"] .stButton > button:hover {
            background: """ + BG_CARD_HOVER + """ !important;
            border-color: """ + ACCENT_CYAN + """ !important;
        }
        [data-testid="stSidebar"] .stSlider > div > div > div {
            background: """ + ACCENT_CYAN + """ !important;
        }
        [data-testid="stSidebar"] .stSlider > div > div {
            background: """ + TEXT_MUTED + """ !important;
        }
        [data-testid="stSidebar"] .stSlider > div > div > div > div {
            background: """ + ACCENT_CYAN + """ !important;
        }
        .st-emotion-cache-1kyxreq {
            background: """ + BG_CARD + """ !important;
            border: 1px solid """ + BORDER + """ !important;
            border-radius: 12px !important;
            padding: 1rem !important;
        }
        #MainMenu { visibility: hidden; }
        footer { visibility: hidden; }
        </style>
    """)

    # ── Path-based routing: parse URL params ────────────────────────────────
    url_ticker, url_start, url_end = parse_config()
    has_url = any(st.query_params.get(k) for k in ("ticker", "start_date", "end_date"))

    # ── Sidebar ────────────────────────────────────────────────────────

    with st.sidebar:
        st.markdown(
            f"""<div style='font-family:DM Sans,sans-serif; font-size:14px;
            font-weight:700; color:{TEXT_PRIMARY}; text-transform:uppercase;
            letter-spacing:2px; margin-bottom:1.5rem;'>🔬 Sensitivity</div>""",
            unsafe_allow_html=True,
        )

        if has_url:
            # URL params exist — render URL-param widgets, auto-run
            ticker = st.selectbox(
                "Ticker",
                options=["SPY", "QQQ", "IWM", "DIA"],
                index=0 if url_ticker == "SPY" else 1 if url_ticker == "QQQ" else 2 if url_ticker == "IWM" else 3,
                key="ticker_url",
            )
            col1, col2 = st.columns(2)
            with col1:
                start_date = st.date_input(
                    "From",
                    value=url_start,
                    key="start_url",
                )
            with col2:
                end_date = st.date_input(
                    "To",
                    value=url_end,
                    key="end_url",
                )
            run_btn = True
        else:
            # No URL params — render normal sidebar widgets
            ticker = st.selectbox(
                "Ticker",
                options=["SPY", "QQQ", "IWM", "DIA"],
                index=0,
            )

            col1, col2 = st.columns(2)
            with col1:
                start_date = st.date_input(
                    "From",
                    value=pd.Timestamp.now() - pd.DateOffset(years=5),
                )
            with col2:
                end_date = st.date_input(
                    "To",
                    value=pd.Timestamp.now(),
                )
            if st.sidebar.button("▶ Run Analysis", type="primary", use_container_width=True):
                st.session_state._run_clicked = True
                st.rerun()

        st.markdown(f"<div style='height:8px'></div>", unsafe_allow_html=True)
        st.markdown(
            f"""<div style='font-family:DM Sans,sans-serif; font-size:10px;
            text-transform:uppercase; letter-spacing:2px;
            color:{TEXT_MUTED}; margin-bottom:8px;'>Custom Ranges</div>""",
            unsafe_allow_html=True,
        )

        use_custom = st.checkbox("Use custom ranges", value=False)

        if use_custom:
            for pname, config in PARAM_CONFIG.items():
                st.markdown(
                    f"<div style='font-family:DM Sans,sans-serif; font-size:11px; "
                    f"color:{TEXT_PRIMARY}; margin:8px 0 4px;'>"
                    f"{pname.replace('_', ' ').title()}</div>",
                    unsafe_allow_html=True,
                )
                cmin, cmax = st.columns(2)
                with cmin:
                    config["range_min"] = st.number_input(
                        "Min", value=config["range_min"], step=config["step"],
                        key=f"custom_{pname}_min",
                    )
                with cmax:
                    config["range_max"] = st.number_input(
                        "Max", value=config["range_max"], step=config["step"],
                        key=f"custom_{pname}_max",
                    )
        else:
            # Show base values as reference
            for pname, config in PARAM_CONFIG.items():
                st.markdown(
                    f"<div style='font-family:DM Sans,sans-serif; font-size:11px; "
                    f"color:{TEXT_SECONDARY}; margin:4px 0;'>"
                    f"{pname.replace('_', ' ').title()}: base={config['base']}</div>",
                    unsafe_allow_html=True,
                )

        st.markdown(f"<div style='height:12px'></div>", unsafe_allow_html=True)

    # ── Main content ────────────────────────────────────────────────────

    st.title("🔬 Sensitivity Analysis")
    st.markdown(
        f'<div style="font-family:DM Sans,sans-serif;font-size:14px;'
        f'color:{TEXT_SECONDARY};margin-bottom:1.5rem;">'
        f"Test strategy robustness by sweeping parameters and scoring stability.</div>",
        unsafe_allow_html=True,
    )

    # ── Persist button click ────────────────────────────────────────────────
    # Button click returns True only for one rerun; persist via session state
    if st.session_state.get("_run_clicked", False):
        st.session_state._run_clicked = False
        has_url = True  # force past empty-state check

    if not has_url and not st.session_state.get("_run_clicked", False):
        st.markdown(
            f"""<div style="background:{BG_CARD}; border:1px solid {BORDER};
            border-radius:12px; padding:3rem; text-align:center;">
            <div style="font-size:48px; margin-bottom:1rem;">🔬</div>
            <div style="font-family:DM Sans,sans-serif; font-size:20px;
            color:{TEXT_PRIMARY}; font-weight:700; margin-bottom:0.5rem;">
            Sensitivity Analysis</div>
            <div style="font-family:DM Sans,sans-serif; font-size:14px;
            color:{TEXT_SECONDARY};">
            Test your strategy's robustness by sweeping parameters.<br>
            Click <strong style="color:{ACCENT_CYAN};">Run Analysis</strong> in the sidebar to begin.
            </div>
            </div>""",
            unsafe_allow_html=True,
        )
        return

    # ── Update URL params to reflect current values ─────────────────────────
    qp = st.query_params
    qp["ticker"] = ticker
    qp["start_date"] = start_date.strftime("%Y-%m-%d")
    qp["end_date"] = end_date.strftime("%Y-%m-%d")

    # ── Download data ──────────────────────────────────────────────────

    with st.spinner("Downloading data..."):
        raw = yf.download(ticker, start=start_date, end=end_date, progress=False)

    if raw.empty:
        st.error(f"No data found for {ticker}")
        st.session_state._run_clicked = False
        return

    # Flatten MultiIndex columns and handle yfinance prefix on column names
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    # Find the Close column — yfinance may prefix it with the ticker
    close_col = None
    if "Close" in raw.columns:
        close_col = "Close"
    else:
        # Try to find it by suffix — e.g. "SPY Close"
        for col in raw.columns:
            if col.endswith("Close"):
                close_col = col
                break
    if close_col is None:
        st.error(f"No Close column found for {ticker}")
        st.session_state._run_clicked = False
        return

    df = pd.DataFrame({
        "Date": raw.index,
        "Close": raw[close_col].astype(float),
    }).dropna()

    # ── Run sensitivity sweep ──────────────────────────────────────────

    base_params = {
        "fast_ma": PARAM_CONFIG["fast_ma"]["base"],
        "slow_ma": PARAM_CONFIG["slow_ma"]["base"],
        "stop_loss_pct": PARAM_CONFIG["stop_loss_pct"]["base"],
        "take_profit_pct": PARAM_CONFIG["take_profit_pct"]["base"],
    }

    param_results = {}
    param_robustness = {}
    param_labels = {}

    for pname, config in PARAM_CONFIG.items():
        other_params = {k: v for k, v in base_params.items() if k != pname}
        param_range = np.arange(config["range_min"], config["range_max"] + config["step"], config["step"])
        param_results[pname] = sweep_parameter(df, pname, param_range, other_params)

        base_result = param_results[pname][config["base"]]

        # Score each metric
        metric_scores = {}
        for metric in ["total_return", "sharpe", "max_drawdown", "win_rate"]:
            score = compute_robustness_score(base_result, param_results[pname], metric)
            metric_scores[metric] = score

        # Average score across metrics
        avg_score = np.mean(list(metric_scores.values()))
        param_robustness[pname] = avg_score
        param_labels[pname] = classify_robustness(avg_score)

    overall_score = np.mean(list(param_robustness.values()))
    overall_label, overall_color = classify_robustness(overall_score)

    # ── Gauge ──────────────────────────────────────────────────────────

    gauge_fig = build_robustness_gauge(overall_score)
    col_gauge, _ = st.columns([1, 3])
    with col_gauge:
        st.plotly_chart(gauge_fig, use_container_width=True, key="gauge")

    # ── Metric cards ───────────────────────────────────────────────────

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(metric_card("Overall", f"{overall_score:.0f}/100", overall_color), unsafe_allow_html=True)
    with col2:
        st.markdown(
            metric_card(
                "Strategy",
                f"{base_params['fast_ma']}/{base_params['slow_ma']}",
                TEXT_PRIMARY,
            ),
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            metric_card("Data", f"{ticker} · {(end_date - start_date).days // 365}yr", TEXT_SECONDARY),
            unsafe_allow_html=True,
        )
    with col4:
        base_result = run_ma_crossover_backtest(
            df,
            fast_ma=base_params["fast_ma"],
            slow_ma=base_params["slow_ma"],
            stop_loss_pct=base_params["stop_loss_pct"],
            take_profit_pct=base_params["take_profit_pct"],
        )
        st.markdown(
            metric_card(
                "Base Return",
                format_return(base_result["total_return"]),
                pnl_color(base_result["total_return"]),
            ),
            unsafe_allow_html=True,
        )

    # ── Heatmap ────────────────────────────────────────────────────────

    st.markdown(section_header("Degradation Heatmap"), unsafe_allow_html=True)
    heatmap_fig = build_heatmap(param_results, base_params, df)
    st.plotly_chart(heatmap_fig, use_container_width=True, key="heatmap")

    # ── Per-parameter detail charts ────────────────────────────────────

    st.markdown(section_header("Parameter Sensitivity"), unsafe_allow_html=True)

    for pname in PARAM_CONFIG:
        score = param_robustness[pname]
        chart_fig = build_param_detail_chart(pname, param_results[pname], PARAM_CONFIG[pname]["base"], score)
        st.plotly_chart(chart_fig, use_container_width=True, key=f"detail_{pname}")

    # ── Interpretation cards ───────────────────────────────────────────

    st.markdown(section_header("Interpretation"), unsafe_allow_html=True)

    for pname in PARAM_CONFIG:
        score = param_robustness[pname]
        label, color = param_labels[pname]
        text = generate_interpretation(pname, score)

        st.markdown(
            f"""<div style="background:{BG_CARD}; border:1px solid {color}40;
            border-radius:12px; padding:1.25rem; margin-bottom:0.75rem;
            border-left:4px solid {color};">
            <div style="font-family:DM Sans,sans-serif; font-size:14px;
            color:{TEXT_PRIMARY}; line-height:1.5;">{text}</div>
            </div>""",
            unsafe_allow_html=True,
        )

if __name__ == "__main__":
    main()
