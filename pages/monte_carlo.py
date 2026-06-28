"""Monte Carlo Simulation — equity curve probability cloud."""

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from design_system import (
    ACCENT_CYAN, ACCENT_GREEN, ACCENT_RED, ACCENT_AMBER,
    BG_CARD, BG_CARD_HOVER, BG_PRIMARY, BORDER, TEXT_MUTED, TEXT_SECONDARY, TEXT_PRIMARY,
    metric_card, section_header, get_plotly_layout, pnl_color,
)

# ── Page setup ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Monte Carlo",
    page_icon="🎲",
    layout="wide",
)

# ── Constants ───────────────────────────────────────────────────────────────

DEFAULT_CAPITAL = 100_000
DEFAULT_SIMS = 1_000
MAX_SIMS = 5_000
NOISE_AMPLITUDE = 0.003  # +/- 0.3%
DRAWDOWN_20 = 0.20
DRAWDOWN_30 = 0.30
OVERFIT_THRESHOLD = 0.90

# ── Sample data generation ──────────────────────────────────────────────────

def generate_sample_data(n_trades: int = 200) -> pd.DataFrame:
    """Generate realistic sample backtest data with slight positive edge and loss clustering."""
    np.random.seed(42)
    dates = pd.date_range(start="2023-01-01", periods=n_trades, freq="B")

    # Base distribution: slightly positive edge
    returns = np.random.normal(loc=0.0015, scale=0.015, size=n_trades)

    # Add loss clustering: pick a few contiguous blocks and make them negative
    cluster_size = np.random.randint(5, 12)
    n_clusters = np.random.randint(2, 5)
    for _ in range(n_clusters):
        start_idx = np.random.randint(0, n_trades - cluster_size)
        end_idx = start_idx + cluster_size
        returns[start_idx:end_idx] = np.random.normal(loc=-0.02, scale=0.01, size=cluster_size)

    # Add a few large outlier losses
    outlier_indices = np.random.choice(np.arange(20, n_trades), size=3, replace=False)
    returns[outlier_indices] = np.random.uniform(-0.06, -0.03, size=3)

    df = pd.DataFrame({"date": dates, "trade_return": returns})
    return df

# ── Simulation engine ───────────────────────────────────────────────────────

def run_simulation(trade_returns: pd.Series, n_sims: int, capital: float):
    """Run Monte Carlo simulation. Returns dicts of equity curves and max drawdowns."""
    equity_curves = np.zeros((n_sims, len(trade_returns) + 1))
    max_drawdowns = np.zeros(n_sims)

    for i in range(n_sims):
        # Shuffle + noise
        noise = np.random.uniform(-NOISE_AMPLITUDE, NOISE_AMPLITUDE, size=len(trade_returns))
        shuffled = np.random.permutation(trade_returns.values) + noise
        equity = np.zeros(len(trade_returns) + 1)
        equity[0] = capital

        for t in range(len(trade_returns)):
            equity[t + 1] = equity[t] * (1 + shuffled[t])

        equity_curves[i] = equity

        # Max drawdown
        peak = capital
        for t in range(1, len(equity)):
            peak = max(peak, equity[t])
            drawdown = (peak - equity[t]) / peak
            max_drawdowns[i] = max(max_drawdowns[i], drawdown)

    return equity_curves, max_drawdowns

# ── Statistics ──────────────────────────────────────────────────────────────

def compute_statistics(equity_curves, max_drawdowns, original_equity, capital):
    """Compute all required statistics from simulation results."""
    n_sims = len(equity_curves)

    # Final values
    final_values = equity_curves[:, -1]

    # Median, percentiles
    median_final = float(np.median(final_values))
    p5_final = float(np.percentile(final_values, 5))
    p95_final = float(np.percentile(final_values, 95))

    # Probability metrics
    prob_loss = float(np.mean(final_values < capital))
    prob_dd20 = float(np.mean(max_drawdowns >= DRAWDOWN_20))
    prob_dd30 = float(np.mean(max_drawdowns >= DRAWDOWN_30))

    # Max drawdown distribution
    dd_5 = float(np.percentile(max_drawdowns, 5))
    dd_25 = float(np.percentile(max_drawdowns, 25))
    dd_50 = float(np.percentile(max_drawdowns, 50))
    dd_75 = float(np.percentile(max_drawdowns, 75))
    dd_95 = float(np.percentile(max_drawdowns, 95))

    # Original backtest percentile
    original_final = original_equity[-1]
    original_pctile = float(np.mean(final_values < original_final))

    # Overfitting check
    overfitting = original_pctile > OVERFIT_THRESHOLD
    if overfitting:
        overfitting_risk = "HIGH"
        overfit_color = ACCENT_RED
    elif original_pctile > 0.80:
        overfitting_risk = "MEDIUM"
        overfit_color = ACCENT_AMBER
    else:
        overfitting_risk = "LOW"
        overfit_color = ACCENT_GREEN

    # Percentile for display (1-based)
    original_pctile_display = int(round(original_pctile * 100))

    stats = {
        "median_final": median_final,
        "p5_final": p5_final,
        "p95_final": p95_final,
        "prob_loss": prob_loss,
        "prob_dd20": prob_dd20,
        "prob_dd30": prob_dd30,
        "dd_5": dd_5,
        "dd_25": dd_25,
        "dd_50": dd_50,
        "dd_75": dd_75,
        "dd_95": dd_95,
        "original_pctile": original_pctile,
        "original_pctile_display": original_pctile_display,
        "overfitting": overfitting,
        "overfitting_risk": overfitting_risk,
        "overfit_color": overfit_color,
    }

    return stats, final_values

# ── Chart builders ──────────────────────────────────────────────────────────

def build_fan_chart(equity_curves, original_equity, n_sims, n_steps):
    """Build the cinematic fan chart — probability cloud with percentile bands."""
    dates = np.arange(n_steps)

    # Individual curves (low opacity)
    traces = []

    # Plot in chunks to avoid browser crash — plot all but in groups
    # Split into 4 groups of ~250 curves each
    chunk_size = n_sims // 4
    for chunk in range(4):
        start = chunk * chunk_size
        end = start + chunk_size if chunk < 3 else n_sims
        trace = go.Scatter(
            x=dates,
            y=equity_curves[start:end].T,
            mode="lines",
            line=dict(color=ACCENT_CYAN, width=0.5, opacity=0.015),
            showlegend=False,
            hoverinfo="skip",
        )
        traces.append(trace)

    # Percentile bands
    # Calculate percentiles at each time step
    p5 = np.percentile(equity_curves, 5, axis=0)
    p25 = np.percentile(equity_curves, 25, axis=0)
    p50 = np.percentile(equity_curves, 50, axis=0)
    p75 = np.percentile(equity_curves, 75, axis=0)
    p95 = np.percentile(equity_curves, 95, axis=0)

    # 5th-95th band
    traces.append(go.Scatter(
        x=np.concatenate([dates, dates[::-1]]),
        y=np.concatenate([p5, p95[::-1]]),
        fill="toself",
        fillcolor="rgba(0, 180, 255, 0.08)",
        line=dict(color="rgba(0, 180, 255, 0.15)", width=1),
        name="5th–95th",
        showlegend=True,
        hoverinfo="skip",
    ))

    # 25th-75th band
    traces.append(go.Scatter(
        x=np.concatenate([dates, dates[::-1]]),
        y=np.concatenate([p25, p75[::-1]]),
        fill="toself",
        fillcolor="rgba(0, 180, 255, 0.15)",
        line=dict(color="rgba(0, 180, 255, 0.25)", width=1),
        name="25th–75th",
        showlegend=True,
        hoverinfo="skip",
    ))

    # Median line
    traces.append(go.Scatter(
        x=dates,
        y=p50,
        mode="lines",
        line=dict(color=ACCENT_CYAN, width=2.5),
        name="Median",
        showlegend=True,
        hovertemplate="Median: $%{y:,.0f}<extra></extra>",
    ))

    # Original backtest
    traces.append(go.Scatter(
        x=dates,
        y=original_equity,
        mode="lines",
        line=dict(color=TEXT_PRIMARY, width=2.5, dash="solid"),
        name="Original",
        showlegend=True,
        hovertemplate="Original: $%{y:,.0f}<extra></extra>",
    ))

    layout = get_plotly_layout(
        xaxis=dict(
            title=dict(text="Trade Number", font=dict(size=11)),
            tickmode="linear",
            dtick=20,
        ),
        yaxis=dict(
            title=dict(text="Portfolio Value", font=dict(size=11)),
            tickformat="$,.0f",
        ),
        legend=dict(
            orientation="h",
            y=-0.1,
            yanchor="top",
            xanchor="center",
            x=0.5,
        ),
        height=450,
    )

    fig = go.Figure(data=traces, layout=layout)
    return fig

def build_final_value_histogram(final_values, original_final, capital):
    """Histogram of final portfolio values with original backtest line."""
    fig = go.Figure()

    fig.add_trace(go.Histogram(
        x=final_values,
        nbinsx=60,
        marker=dict(color=BG_CARD, line=dict(color=ACCENT_CYAN, width=1)),
        opacity=0.85,
        hovertemplate="Final Value: $%{x:,.0f}<br>Count: %{y}<extra></extra>",
    ))

    # Original backtest line
    fig.add_shape(
        type="line",
        x0=original_final,
        x1=original_final,
        y0=0,
        y1=1,
        yref="paper",
        line=dict(color=TEXT_PRIMARY, width=2, dash="dash"),
    )

    # Capital line
    fig.add_shape(
        type="line",
        x0=capital,
        x1=capital,
        y0=0,
        y1=1,
        yref="paper",
        line=dict(color=TEXT_MUTED, width=1.5, dash="dot"),
    )

    layout = get_plotly_layout(
        title=dict(
            text="Final Portfolio Values",
            font=dict(size=12, color=TEXT_SECONDARY),
        ),
        xaxis=dict(
            title=dict(text="Final Value", font=dict(size=11)),
            tickformat="$,.0f",
        ),
        yaxis=dict(title=dict(text="Count", font=dict(size=11))),
        height=320,
        legend=dict(
            orientation="h",
            y=-0.15,
            yanchor="top",
            xanchor="center",
            x=0.5,
            font=dict(size=10),
            itemsizing="constant",
        ),
    )
    fig.update_layout(layout)

    return fig

def build_drawdown_histogram(max_drawdowns):
    """Histogram of max drawdowns."""
    fig = go.Figure()

    fig.add_trace(go.Histogram(
        x=max_drawdowns,
        nbinsx=50,
        marker=dict(color=BG_CARD, line=dict(color=ACCENT_RED, width=1)),
        opacity=0.85,
        hovertemplate="Max Drawdown: %{x:.1%}<br>Count: %{y}<extra></extra>",
    ))

    # 20% drawdown threshold
    fig.add_shape(
        type="line",
        x0=DRAWDOWN_20,
        x1=DRAWDOWN_20,
        y0=0,
        y1=1,
        yref="paper",
        line=dict(color=ACCENT_AMBER, width=1.5, dash="dot"),
    )

    layout = get_plotly_layout(
        title=dict(
            text="Max Drawdown Distribution",
            font=dict(size=12, color=TEXT_SECONDARY),
        ),
        xaxis=dict(
            title=dict(text="Max Drawdown", font=dict(size=11)),
            tickformat=".1%",
        ),
        yaxis=dict(title=dict(text="Count", font=dict(size=11))),
        height=320,
    )
    fig.update_layout(layout)

    return fig

# ── UI helpers ──────────────────────────────────────────────────────────────

def format_pct(value: float) -> str:
    return f"{value:.1%}"

def format_currency(value: float) -> str:
    if abs(value) >= 1_000_000:
        return f"${value/1_000_000:.1f}M"
    elif abs(value) >= 1_000:
        return f"${value/1_000:.1f}K"
    else:
        return f"${value:,.0f}"

def format_return(value: float) -> str:
    """Format return as percentage."""
    return f"{value:.1%}"

# ── Main app ────────────────────────────────────────────────────────────────

def main():
    # Apply design theme
    st.html("""
        <style>
        .stApp::before {
            content: "";
            display: block;
            height: 2px;
            background: """ + ACCENT_CYAN + """;
        }
        /* Make the main area not have sidebar */
        [data-testid="stSidebar"] {
            background: """ + BG_CARD + """ !important;
            border-right: 1px solid """ + BORDER + """ !important;
            padding: 1rem !important;
            width: 280px !important;
        }
        [data-testid="stSidebar"] .stSidebarNav {
            display: none !important;
        }
        /* Sidebar labels */
        [data-testid="stSidebar"] label {
            color: """ + TEXT_MUTED + """ !important;
            font-family: DM Sans, sans-serif !important;
            font-size: 11px !important;
            text-transform: uppercase !important;
            letter-spacing: 1px !important;
        }
        [data-testid="stSidebar"] .stFileUploader > div {
            background: """ + BG_CARD + """ !important;
            border: 1px dashed """ + BORDER + """ !important;
            border-radius: 8px !important;
        }
        [data-testid="stSidebar"] .stFileUploader > div:hover {
            border-color: """ + ACCENT_CYAN + """ !important;
        }
        /* Sidebar buttons */
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
        /* Main content area */
        .st-emotion-cache-1kyxreq {
            background: """ + BG_CARD + """ !important;
            border: 1px solid """ + BORDER + """ !important;
            border-radius: 12px !important;
            padding: 1rem !important;
        }
        /* Remove main sidebar nav */
        #MainMenu { visibility: hidden; }
        footer { visibility: hidden; }
        </style>
    """)

    # ── Sidebar ────────────────────────────────────────────────────────

    with st.sidebar:
        st.markdown(
            f"""<div style='font-family:DM Sans,sans-serif; font-size:14px;
            font-weight:700; color:{TEXT_PRIMARY}; text-transform:uppercase;
            letter-spacing:2px; margin-bottom:1.5rem;'>🎲 Monte Carlo</div>""",
            unsafe_allow_html=True,
        )

        uploaded_file = st.file_uploader(
            "Upload CSV",
            type=["csv"],
            help="CSV with columns: date, trade_return",
        )

        st.markdown(f"<div style='height:12px'></div>", unsafe_allow_html=True)

        capital = st.number_input(
            "Starting Capital",
            min_value=10_000,
            max_value=10_000_000,
            value=DEFAULT_CAPITAL,
            step=10_000,
            format="%d",
        )

        n_sims = st.slider(
            "Simulations",
            min_value=100,
            max_value=MAX_SIMS,
            value=DEFAULT_SIMS,
            step=100,
        )

        if st.button("Run Simulation", use_container_width=True):
            st.session_state.run_simulation = True

    # ── Main content ────────────────────────────────────────────────────

    st.title("🎲 Monte Carlo Simulation")

    # Load or generate data
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        if "date" not in df.columns or "trade_return" not in df.columns:
            st.error("CSV must contain columns: 'date' and 'trade_return'")
            st.stop()
        trade_returns = df["trade_return"].astype(float)
    else:
        trade_returns = generate_sample_data().set_index("date")["trade_return"]

    n_trades = len(trade_returns)

    # ── Run simulation ─────────────────────────────────────────────────

    if st.session_state.get("run_simulation", False):
        # Run
        equity_curves, max_drawdowns = run_simulation(trade_returns, n_sims, capital)

        # Original equity curve
        original_equity = np.zeros(n_trades + 1)
        original_equity[0] = capital
        for t in range(n_trades):
            original_equity[t + 1] = original_equity[t] * (1 + trade_returns.iloc[t])

        # Stats
        stats, final_values = compute_statistics(
            equity_curves, max_drawdowns, original_equity, capital
        )

        # ── Metric cards ───────────────────────────────────────────────

        # Probability of loss
        if stats["prob_loss"] > 0.30:
            loss_color = ACCENT_RED
        elif stats["prob_loss"] > 0.10:
            loss_color = ACCENT_AMBER
        else:
            loss_color = ACCENT_GREEN

        # Median return
        median_return = (stats["median_final"] / capital) - 1
        median_color = ACCENT_GREEN if median_return >= 0 else ACCENT_RED

        # Worst 5% drawdown
        worst5_dd = stats["dd_5"]

        # Overfitting card
        overfit_color = stats["overfit_color"]

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(
                metric_card(
                    "Probability of Loss",
                    format_pct(stats["prob_loss"]),
                    loss_color,
                ),
                unsafe_allow_html=True,
            )
        with col2:
            st.markdown(
                metric_card("Median Return", format_return(median_return), median_color),
                unsafe_allow_html=True,
            )
        with col3:
            st.markdown(
                metric_card(
                    "Worst 5% Max Drawdown",
                    format_pct(worst5_dd),
                    ACCENT_RED,
                ),
                unsafe_allow_html=True,
            )
        with col4:
            st.markdown(
                metric_card(
                    f"Overfitting Risk: {stats['overfitting_risk']}",
                    "",
                    overfit_color,
                ),
                unsafe_allow_html=True,
            )

        # ── Fan chart ──────────────────────────────────────────────────

        st.markdown(section_header("Equity Curve — Probability Cloud"), unsafe_allow_html=True)
        fan_fig = build_fan_chart(equity_curves, original_equity, n_sims, n_trades + 1)
        st.plotly_chart(fan_fig, use_container_width=True, key="fan_chart")

        # ── Histograms ─────────────────────────────────────────────────

        col_hist1, col_hist2 = st.columns(2)
        with col_hist1:
            st.markdown(
                section_header("Final Portfolio Values"),
                unsafe_allow_html=True,
            )
            hist1 = build_final_value_histogram(final_values, original_equity[-1], capital)
            st.plotly_chart(hist1, use_container_width=True, key="hist_final")

        with col_hist2:
            st.markdown(
                section_header("Max Drawdown Distribution"),
                unsafe_allow_html=True,
            )
            hist2 = build_drawdown_histogram(max_drawdowns)
            st.plotly_chart(hist2, use_container_width=True, key="hist_dd")

        # ── Drawdown distribution detail ───────────────────────────────

        st.markdown(section_header("Drawdown Distribution"), unsafe_allow_html=True)

        dd_col1, dd_col2, dd_col3, dd_col4, dd_col5 = st.columns(5)
        with dd_col1:
            st.markdown(f"<div style='text-align:center;'><div style='font-family:DM Sans,sans-serif; font-size:10px; text-transform:uppercase; letter-spacing:2px; color:{TEXT_MUTED};'>5th %</div><div style='font-family:JetBrains Mono,monospace; font-size:18px; color:{ACCENT_RED};'>{format_pct(stats['dd_5'])}</div></div>", unsafe_allow_html=True)
        with dd_col2:
            st.markdown(f"<div style='text-align:center;'><div style='font-family:DM Sans,sans-serif; font-size:10px; text-transform:uppercase; letter-spacing:2px; color:{TEXT_MUTED};'>25th %</div><div style='font-family:JetBrains Mono,monospace; font-size:18px; color:{ACCENT_RED};'>{format_pct(stats['dd_25'])}</div></div>", unsafe_allow_html=True)
        with dd_col3:
            st.markdown(f"<div style='text-align:center;'><div style='font-family:DM Sans,sans-serif; font-size:10px; text-transform:uppercase; letter-spacing:2px; color:{TEXT_MUTED};'>50th %</div><div style='font-family:JetBrains Mono,monospace; font-size:18px; color:{ACCENT_AMBER};'>{format_pct(stats['dd_50'])}</div></div>", unsafe_allow_html=True)
        with dd_col4:
            st.markdown(f"<div style='text-align:center;'><div style='font-family:DM Sans,sans-serif; font-size:10px; text-transform:uppercase; letter-spacing:2px; color:{TEXT_MUTED};'>75th %</div><div style='font-family:JetBrains Mono,monospace; font-size:18px; color:{ACCENT_AMBER};'>{format_pct(stats['dd_75'])}</div></div>", unsafe_allow_html=True)
        with dd_col5:
            st.markdown(f"<div style='text-align:center;'><div style='font-family:DM Sans,sans-serif; font-size:10px; text-transform:uppercase; letter-spacing:2px; color:{TEXT_MUTED};'>95th %</div><div style='font-family:JetBrains Mono,monospace; font-size:18px; color:{ACCENT_RED};'>{format_pct(stats['dd_95'])}</div></div>", unsafe_allow_html=True)

        # ── Interpretation ─────────────────────────────────────────────

        st.markdown(section_header("Analysis"), unsafe_allow_html=True)

        original_return_pct = ((original_equity[-1] / capital) - 1) * 100

        interpretation = (
            f"Based on {n_sims:,} simulations, there is a "
            f"**{format_pct(stats['prob_loss'])}** chance of losing money, "
            f"a **{format_pct(stats['prob_dd20'])}** chance of a 20%+ drawdown, "
            f"and a **{format_pct(stats['prob_dd30'])}** chance of a 30%+ drawdown. "
            f"The median outcome is a **{format_return(median_return)}** return "
            f"({format_currency(stats['median_final'])}). "
            f"Your original backtest returned **{original_return_pct:+.1f}%** "
            f"({format_currency(original_equity[-1])}), which falls in the "
            f"**{stats['original_pctile_display']}th percentile** of outcomes."
        )

        st.markdown(
            f'<div style="background:{BG_CARD}; border:1px solid {BORDER}; '
            f'border-radius:12px; padding:1.5rem; font-family:DM Sans,sans-serif; '
            f'font-size:14px; color:{TEXT_SECONDARY}; line-height:1.6;">'
            f'{interpretation}'
            f'</div>',
            unsafe_allow_html=True,
        )

        # Overfitting warning
        if stats["overfitting"]:
            st.markdown(
                f'<div style="background:rgba(255,193,7,0.08); border:1px solid {ACCENT_AMBER}40; '
                f'border-radius:12px; padding:1.5rem; margin-top:1rem; '
                f'font-family:DM Sans,sans-serif; font-size:14px; color:{ACCENT_AMBER}; '
                f'line-height:1.6;">'
                f'⚠️ <strong>Overfitting Warning:</strong> Your original backtest result '
                f'({format_currency(original_equity[-1])}) falls in the '
                f'**{stats["original_pctile_display"]}th percentile** of simulated outcomes. '
                f'This suggests the strategy may be overfit to the historical data — '
                f'the result looks too good compared to random permutations. '
                f'The original return of {original_return_pct:+.1f}% is better than '
                f'{100 - stats["original_pctile_display"]}% of shuffled outcomes.'
                f'</div>',
                unsafe_allow_html=True,
            )

        # ── Additional stats ───────────────────────────────────────────

        st.markdown(section_header("Summary Statistics"), unsafe_allow_html=True)

        stats_col1, stats_col2, stats_col3 = st.columns(3)

        with stats_col1:
            st.markdown(
                f"""<div style="background:{BG_CARD}; border:1px solid {BORDER};
                border-radius:12px; padding:1.25rem;">
                <div style="font-family:DM Sans,sans-serif; font-size:10px;
                text-transform:uppercase; letter-spacing:2px;
                color:{TEXT_MUTED}; margin-bottom:6px;">Median Final</div>
                <div style="font-family:JetBrains Mono,monospace; font-size:22px;
                font-weight:700; color:{ACCENT_CYAN};">{format_currency(stats['median_final'])}</div>
                <div style="font-family:JetBrains Mono,monospace; font-size:13px;
                color:{pnl_color((stats['median_final'] / capital) - 1)};
                margin-top:2px;">{format_return((stats['median_final'] / capital) - 1)}</div>
                </div>""",
                unsafe_allow_html=True,
            )

        with stats_col2:
            st.markdown(
                f"""<div style="background:{BG_CARD}; border:1px solid {BORDER};
                border-radius:12px; padding:1.25rem;">
                <div style="font-family:DM Sans,sans-serif; font-size:10px;
                text-transform:uppercase; letter-spacing:2px;
                color:{TEXT_MUTED}; margin-bottom:6px;">5th–95th Range</div>
                <div style="font-family:JetBrains Mono,monospace; font-size:22px;
                font-weight:700; color:{TEXT_PRIMARY};">{format_currency(stats['p5_final'])} – {format_currency(stats['p95_final'])}</div>
                <div style="font-family:JetBrains Mono,monospace; font-size:13px;
                color:{TEXT_MUTED}; margin-top:2px;">90% probability range</div>
                </div>""",
                unsafe_allow_html=True,
            )

        with stats_col3:
            st.markdown(
                f"""<div style="background:{BG_CARD}; border:1px solid {BORDER};
                border-radius:12px; padding:1.25rem;">
                <div style="font-family:DM Sans,sans-serif; font-size:10px;
                text-transform:uppercase; letter-spacing:2px;
                color:{TEXT_MUTED}; margin-bottom:6px;">Original Result</div>
                <div style="font-family:JetBrains Mono,monospace; font-size:22px;
                font-weight:700; color:{pnl_color(original_return_pct)};">{format_currency(original_equity[-1])}</div>
                <div style="font-family:JetBrains Mono,monospace; font-size:13px;
                color:{pnl_color(original_return_pct)}; margin-top:2px;">
                {original_return_pct:+.1f}% · {stats['original_pctile_display']}th percentile
                </div>
                </div>""",
                unsafe_allow_html=True,
            )

    else:
        # No simulation yet — show sample data preview
        st.markdown(
            f"""<div style="background:{BG_CARD}; border:1px solid {BORDER};
            border-radius:12px; padding:3rem; text-align:center;">
            <div style="font-size:48px; margin-bottom:1rem;">🎲</div>
            <div style="font-family:DM Sans,sans-serif; font-size:20px;
            color:{TEXT_PRIMARY}; font-weight:700; margin-bottom:0.5rem;">
            Monte Carlo Simulation</div>
            <div style="font-family:DM Sans,sans-serif; font-size:14px;
            color:{TEXT_SECONDARY};">
            Upload a CSV with backtest results or use sample data.<br>
            Click <strong style="color:{ACCENT_CYAN};">Run Simulation</strong> in the sidebar to begin.
            </div>
            <div style="font-family:JetBrains Mono,monospace; font-size:12px;
            color:{TEXT_MUTED}; margin-top:1rem;">
            Sample data: {n_trades} trades · Edge: +{(trade_returns.mean() * 100):.2f}% per trade · σ: {(trade_returns.std() * 100):.2f}%
            </div>
            </div>""",
            unsafe_allow_html=True,
        )

if __name__ == "__main__":
    main()
