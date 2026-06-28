"""Multi-Asset Regime Backtester — walk-forward HMM, stress tests, regime strips."""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import streamlit.components.v1 as components
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import yfinance as yf
from hmmlearn import hmm

from design_system import *

apply_theme()

# ── Constants ──────────────────────────────────────────────────────────────────

DEFAULT_ASSETS = ["SPY", "BTC-USD", "GLD", "TLT"]
ASSET_COLORS = {
    "SPY": ACCENT_CYAN,
    "BTC-USD": ACCENT_AMBER,
    "GLD": "#FFD700",
    "TLT": ACCENT_VIOLET,
}
REGIME_NAMES = ["Low Vol", "Medium Vol", "High Vol"]
REGIME_ORDER = {"Low Vol": 0, "Medium Vol": 1, "High Vol": 2, "Uncertain": 3}

CRISIS_WINDOWS = {
    "2008 Crisis": ("2008-09-01", "2009-03-31"),
    "2020 Covid": ("2020-02-01", "2020-04-30"),
    "2022 Rate Hikes": ("2022-01-01", "2022-10-31"),
}

# ── Feature engineering ────────────────────────────────────────────────────────

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return DataFrame with log returns, vol, volume ratio, H-L range."""
    df = df.copy()
    df["log_ret"] = np.log(df["Close"] / df["Close"].shift(1))
    df["realized_vol"] = df["log_ret"].rolling(20).std()
    df["vol_ratio"] = df["Volume"] / df["Volume"].rolling(20).mean()
    df["hl_range"] = (df["High"] - df["Low"]) / df["Close"]
    df.dropna(inplace=True)
    return df.reset_index(drop=True)

# ── Forward-filtered regime labels (no look-ahead) ─────────────────────────────

def forward_filter_labels(model, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Online forward-filtered regime labels — label at T uses ONLY data 1..T.

    Uses forward algorithm (alpha) to compute posterior marginals gamma[t],
    then argmax for label. Confidence = max(gamma[t]) as softmax probability.
    NO look-ahead bias — never calls model.predict().

    Returns (labels, confidences) — both length N.
    """
    T = X.shape[0]
    n_states = model.n_components
    labels = np.zeros(T, dtype=int)
    confidences = np.zeros(T)

    log_alpha = np.full((T, n_states), -1e10)

    log_pi = np.log(np.maximum(model.startprob_, 1e-300))
    log_emission = model._compute_log_likelihood(X[0:1])
    log_alpha[0] = log_pi + log_emission[0]

    for t in range(1, T):
        log_trans = np.log(np.maximum(model.transmat_, 1e-300))
        log_emission_t = model._compute_log_likelihood(X[t:t+1])
        for j in range(n_states):
            log_alpha[t, j] = (
                np.log(np.sum(np.exp(np.minimum(log_alpha[t - 1], 0) + log_trans[:, j])))
                + log_emission_t[0, j]
            )

    for t in range(T):
        max_val = log_alpha[t].max()
        exp_alpha = np.exp(np.minimum(log_alpha[t] - max_val, 0))
        gamma_t = exp_alpha / exp_alpha.sum()
        labels[t] = np.argmax(gamma_t)
        confidences[t] = gamma_t.max()

    return labels, confidences

# ── BIC calculation ────────────────────────────────────────────────────────────

def compute_bic(model, X: np.ndarray) -> float:
    """Bayesian Information Criterion for diagonal Gaussian HMM."""
    n_samples, n_features = X.shape
    n = model.n_components
    n_params = n * n_features + n * n_features + n * n - n + n - 1
    try:
        log_likelihood = model.score(X)
    except Exception:
        return np.inf
    bic = -2 * log_likelihood + n_params * np.log(n_samples)
    return bic

# ── Stability filter ───────────────────────────────────────────────────────────

def apply_stability_filter(labels: np.ndarray) -> np.ndarray:
    """Regime must persist 3 consecutive bars.

    If flickering >4 times in 20-bar window, flag as 999 (Uncertain).
    """
    filtered = np.copy(labels)
    n = len(labels)

    i = 0
    while i < n - 2:
        if labels[i] == labels[i + 1] == labels[i + 2]:
            i += 3
        else:
            window = labels[i : i + 3]
            unique, counts = np.unique(window, return_counts=True)
            dominant = unique[np.argmax(counts)]
            for j in range(3):
                filtered[i + j] = dominant
            i += 3

    flicker_threshold = 4
    window_size = 20
    uncertain = np.zeros(n, dtype=bool)
    for start in range(0, n - window_size + 1):
        window = filtered[start : start + window_size]
        transitions = np.sum(window[1:] != window[:-1])
        if transitions > flicker_threshold:
            uncertain[start : start + window_size] = True

    filtered[uncertain] = 999

    return filtered

# ── Regime naming ──────────────────────────────────────────────────────────────

def label_regimes(means: np.ndarray, n_regimes: int) -> list[str]:
    """Sort regimes by mean volatility, assign labels."""
    vol_means = means[:, 1]
    sorted_idx = np.argsort(vol_means)

    labels = []
    for rank, idx in enumerate(sorted_idx):
        if rank == 0:
            labels.append("Low Vol")
        elif rank == n_regimes - 1:
            labels.append("High Vol")
        else:
            labels.append(f"Medium Vol")
    return labels

# ── Allocation rules ───────────────────────────────────────────────────────────

def compute_allocation(labels: np.ndarray, regime_names: list[str]) -> np.ndarray:
    """95% allocation in low-vol, 60% in high-vol, linear scale between."""
    allocation = np.zeros(len(labels))
    for i, label in enumerate(labels):
        if label == 999:
            allocation[i] = 0.5  # uncertain → neutral
        elif label < len(regime_names):
            regime_name = regime_names[label]
            if regime_name == "Low Vol":
                allocation[i] = 0.95
            elif regime_name == "High Vol":
                allocation[i] = 0.60
            else:
                # Medium Vol — linear interpolation
                low_alloc = 0.95
                high_alloc = 0.60
                allocation[i] = low_alloc - (low_alloc - high_alloc) * (1.0 / 2.0)
        else:
            allocation[i] = 0.5
    return allocation

# ── Walk-forward backtest ──────────────────────────────────────────────────────

def walk_forward_backtest(df: pd.DataFrame, regime_names: list[str],
                          train_years: int, test_months: int) -> pd.DataFrame:
    """Walk-forward backtest with HMM regime allocation.

    Train on train_years, test on test_months, roll forward.
    Returns DataFrame with equity curves for strategy and benchmarks.
    """
    dates = pd.DatetimeIndex(df.index)
    n = len(dates)
    strategy_returns = pd.Series(0.0, index=dates)
    benchmark_returns = pd.Series(0.0, index=dates)
    sma_returns = pd.Series(0.0, index=dates)

    # 200-day SMA trend
    sma = df["Close"].rolling(200).mean()

    # Walk-forward window
    train_days = train_years * 252
    test_days = test_months * 30

    i = 0
    while i + train_days + test_days <= n:
        train_end = i + train_days
        test_end = min(train_end + test_days, n)

        train_data = df.iloc[i:train_end]
        test_data = df.iloc[train_end:test_end]

        X_train = train_data[["log_ret", "realized_vol", "vol_ratio", "hl_range"]].values
        X_test = test_data[["log_ret", "realized_vol", "vol_ratio", "hl_range"]].values

        # Train HMM on training window
        model = hmm.GaussianHMM(
            n_components=3,
            covariance_type="diag",
            n_iter=200,
            random_state=42,
        )
        model.fit(X_train)

        # Forward-filter labels for test period
        labels, _ = forward_filter_labels(model, X_test)
        labels = apply_stability_filter(labels)

        # Allocation series
        alloc = compute_allocation(labels, regime_names)

        # Strategy returns: weighted by allocation
        for j in range(len(test_data)):
            strategy_returns.iloc[train_end + j] = alloc[j] * test_data["log_ret"].iloc[j]

        # Benchmark: buy and hold
        for j in range(len(test_data)):
            benchmark_returns.iloc[train_end + j] = test_data["log_ret"].iloc[j]

        # Benchmark: 200-day SMA trend following
        for j in range(len(test_data)):
            test_idx = train_end + j
            if test_idx >= 200 and test_data["Close"].iloc[j] > sma.iloc[test_idx]:
                sma_returns.iloc[test_idx] = test_data["log_ret"].iloc[j]

        i = train_end

    # Cumulative equity curves
    strategy_equity = (1 + strategy_returns).cumprod()
    benchmark_equity = (1 + benchmark_returns).cumprod()
    sma_equity = (1 + sma_returns).cumprod()

    # NaN fill with 1.0 for equity curves
    strategy_equity = strategy_equity.fillna(1.0)
    benchmark_equity = benchmark_equity.fillna(1.0)
    sma_equity = sma_equity.fillna(1.0)

    return pd.DataFrame({
        "date": dates,
        "strategy": strategy_equity,
        "buy_hold": benchmark_equity,
        "sma_trend": sma_equity,
    })

# ── Performance metrics ────────────────────────────────────────────────────────

def calc_ann_return(returns: pd.Series, dates: pd.Series | None = None) -> float:
    """Annualized return from cumulative equity curve."""
    total = returns.iloc[-1] / returns.iloc[0]
    if dates is not None and len(dates) > 0:
        try:
            days = (dates.iloc[-1] - dates.iloc[0]).days
        except AttributeError:
            days = 0
    else:
        try:
            days = (returns.index[-1] - returns.index[0]).days
        except AttributeError:
            days = 0
    if days <= 0:
        return 0.0
    return total ** (252 / days) - 1

def calc_sharpe(returns: pd.Series) -> float:
    """Annualized Sharpe ratio."""
    daily = returns.pct_change().dropna()
    if daily.std() == 0 or len(daily) < 2:
        return 0.0
    return daily.mean() / daily.std() * np.sqrt(252)

def calc_max_drawdown(equity: pd.Series) -> float:
    """Maximum drawdown from equity curve."""
    peak = equity.expanding().max()
    drawdown = (equity - peak) / peak
    return drawdown.min()

# ── Stress test ────────────────────────────────────────────────────────────────

def stress_test(equity: pd.Series, dates: pd.Series, start: str, end: str) -> float:
    """Drawdown during a crisis window."""
    mask = (dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))
    if not mask.any():
        return 0.0
    window = equity[mask]
    peak = window.expanding().max()
    drawdown = (window - peak) / peak
    return drawdown.min()

# ── Build equity curve chart ───────────────────────────────────────────────────

def build_equity_chart(df: pd.DataFrame, asset_color: str) -> go.Figure:
    """Equity curve: strategy (colored) vs buy-and-hold (muted gray dashed)."""
    traces = []

    # Buy-and-hold (muted gray dashed)
    traces.append(
        go.Scatter(
            x=df["date"],
            y=df["buy_hold"],
            mode="lines",
            name="Buy & Hold",
            line=dict(color=TEXT_MUTED, width=1.5, dash="dash"),
            hovertemplate="Buy & Hold<br>%{y:.4f}<br>%{x|%Y-%m-%d}",
        )
    )

    # 200-day SMA trend (faint)
    traces.append(
        go.Scatter(
            x=df["date"],
            y=df["sma_trend"],
            mode="lines",
            name="200-Day SMA",
            line=dict(color=TEXT_MUTED, width=1.0, dash="dot"),
            opacity=0.5,
            hovertemplate="SMA Trend<br>%{y:.4f}<br>%{x|%Y-%m-%d}",
        )
    )

    # Strategy (asset color, solid)
    traces.append(
        go.Scatter(
            x=df["date"],
            y=df["strategy"],
            mode="lines",
            name="Regime Strategy",
            line=dict(color=asset_color, width=2.5),
            hovertemplate="Strategy<br>%{y:.4f}<br>%{x|%Y-%m-%d}",
        )
    )

    layout = get_plotly_layout(
        height=450,
        xaxis=dict(
            type="date",
            showspikes=True,
            spikemode="across",
        ),
        yaxis=dict(
            title="Equity",
            showspikes=True,
            spikemode="across",
            tickformat=",.0f",
        ),
    )

    fig = go.Figure(data=traces, layout=layout)
    fig.update_layout(
        legend=dict(
            x=0.01,
            y=0.99,
            bgcolor="rgba(0,0,0,0)",
            font=dict(color=TEXT_SECONDARY),
        ),
    )
    return fig

# ── Build regime timeline strips ───────────────────────────────────────────────

def build_regime_strips(regime_data: dict) -> go.Figure:
    """All assets stacked vertically as horizontal regime bars.

    Each bar divided into segments colored by regime.
    """
    fig = go.Figure()

    # Sort assets by date range alignment
    assets = sorted(regime_data.keys(), key=lambda k: regime_data[k]["dates"].min())

    # Y positions: top to bottom = first asset at top
    y_positions = {}
    for i, asset in enumerate(assets):
        y_positions[asset] = len(assets) - i - 1

    for asset in assets:
        data = regime_data[asset]
        dates = data["dates"]
        labels = data["labels"]
        regime_names = data["regime_names"]
        colors = data["colors"]

        n = len(labels)
        y = y_positions[asset]

        # Build segments
        x_segments = []
        y_segments = []
        text_segments = []
        color_segments = []

        seg_start = 0
        prev_label = labels[0]
        prev_regime_name = regime_names[0] if labels[0] < len(regime_names) else "Uncertain"
        prev_color = REGIME_COLORS.get(prev_regime_name, TEXT_MUTED)

        for i in range(1, n):
            curr = labels[i]
            curr_regime_name = regime_names[curr] if curr < len(regime_names) else "Uncertain"
            curr_color = REGIME_COLORS.get(curr_regime_name, TEXT_MUTED)

            if curr_regime_name != prev_regime_name:
                x_segments.append([dates.iloc[seg_start], dates.iloc[i - 1]])
                y_segments.append([y - 0.35, y + 0.35])
                text_segments.append([
                    f"{prev_regime_name}<br>{dates.iloc[seg_start].strftime('%Y-%m-%d')} — {dates.iloc[i-1].strftime('%Y-%m-%d')}",
                    f"{prev_regime_name}<br>{dates.iloc[seg_start].strftime('%Y-%m-%d')} — {dates.iloc[i-1].strftime('%Y-%m-%d')}",
                ])
                color_segments.append(prev_color)
                seg_start = i
                prev_regime_name = curr_regime_name
                prev_color = curr_color

        # Last segment
        x_segments.append([dates.iloc[seg_start], dates.iloc[-1]])
        y_segments.append([y - 0.35, y + 0.35])
        text_segments.append([
            f"{prev_regime_name}<br>{dates.iloc[seg_start].strftime('%Y-%m-%d')} — {dates.iloc[-1].strftime('%Y-%m-%d')}",
            f"{prev_regime_name}<br>{dates.iloc[seg_start].strftime('%Y-%m-%d')} — {dates.iloc[-1].strftime('%Y-%m-%d')}",
        ])
        color_segments.append(prev_color)

        # Draw bar segments
        for idx in range(len(x_segments)):
            fig.add_trace(go.Scatter(
                x=x_segments[idx],
                y=y_segments[idx],
                mode="lines",
                line=dict(width=8, color=color_segments[idx]),
                opacity=0.9,
                hovertext=text_segments[idx],
                hoverinfo="text",
                showlegend=False,
                hoverlabel=dict(
                    bgcolor=BG_CARD,
                    bordercolor=ACCENT_CYAN,
                    font=dict(family="JetBrains Mono, monospace", size=11, color=TEXT_PRIMARY),
                ),
            ))

    # Y-axis: asset labels
    fig.update_yaxes(
        range=[-0.5, len(assets) - 0.5],
        ticktext=assets,
        tickvals=[y_positions[a] for a in assets],
        tickfont=dict(family="DM Sans, sans-serif", size=12, color=TEXT_PRIMARY),
        showgrid=False,
        showline=False,
        zeroline=False,
    )

    layout = get_plotly_layout(
        height=80 + len(assets) * 40,
        xaxis=dict(
            type="date",
            showspikes=True,
            spikemode="across",
        ),
        yaxis=dict(
            showticklabels=True,
            showgrid=False,
            showline=False,
        ),
        showlegend=False,
    )

    fig = go.Figure(layout=layout)
    return fig

# ── Build comparison table ─────────────────────────────────────────────────────

def build_comparison_table(results: dict) -> pd.DataFrame:
    """Dark styled table with all assets side by side."""
    rows = []
    for asset, data in results.items():
        rows.append({
            "Asset": asset,
            "Strategy Return (ann.)": f"{data['strat_return']:+.2%}",
            "Buy&Hold Return (ann.)": f"{data['bh_return']:+.2%}",
            "Strategy Max DD": f"{data['strat_max_dd']:+.2%}",
            "Buy&Hold Max DD": f"{data['bh_max_dd']:+.2%}",
            "Strategy Sharpe": f"{data['strat_sharpe']:+.2f}",
            "Buy&Hold Sharpe": f"{data['bh_sharpe']:+.2f}",
            "Sharpe Improvement": f"{data['sharpe_improvement']:+.2f}",
        })

    df = pd.DataFrame(rows)
    return df.style

# ── Build stress test mini bar charts ──────────────────────────────────────────

def build_stress_charts(results: dict, crisis_key: str) -> go.Figure:
    """Mini bar chart: strategy drawdown vs buy-and-hold drawdown per asset."""
    fig = go.Figure()

    assets = sorted(results.keys())
    strategies = []
    benchmarks = []

    for asset in assets:
        data = results[asset]
        strat_dd = data["crisis_drawdowns"].get(crisis_key, 0.0)
        bh_dd = data["bh_crisis_drawdowns"].get(crisis_key, 0.0)
        strategies.append(strat_dd)
        benchmarks.append(bh_dd)

    # Strategy bars
    fig.add_trace(go.Bar(
        x=assets,
        y=strategies,
        name="Strategy",
        marker_color=[ASSET_COLORS.get(a, TEXT_MUTED) for a in assets],
        hovertemplate="Strategy<br>%{y:.2%}<br>%{x}",
    ))

    # Buy-and-hold bars
    fig.add_trace(go.Bar(
        x=assets,
        y=benchmarks,
        name="Buy & Hold",
        marker_color=TEXT_MUTED,
        hovertemplate="Buy & Hold<br>%{y:.2%}<br>%{x}",
    ))

    layout = get_plotly_layout(
        height=200,
        xaxis=dict(showgrid=False, showline=False),
        yaxis=dict(
            title="Drawdown",
            tickformat=".0%",
            showgrid=True,
            gridcolor=GRID_LINE,
        ),
        legend=dict(
            x=0.01,
            y=0.99,
            bgcolor="rgba(0,0,0,0)",
            font=dict(color=TEXT_SECONDARY),
        ),
    )

    fig = go.Figure(layout=layout)
    return fig

# ── Build asset tab summary ────────────────────────────────────────────────────

def asset_tab_summary(asset: str, data: dict) -> str:
    """Small summary for asset tab: return, Sharpe improvement, thumbs icon."""
    thumb = "👍" if data["sharpe_improvement"] >= 0 else "👎"
    return (
        f'<div style="margin-top:4px;font-family:JetBrains Mono,monospace;font-size:11px;color:{TEXT_SECONDARY};">'
        f'{data["strat_return"]:+.2%} · Sharpe +{data["sharpe_improvement"]:.2f} {thumb}'
        f'</div>'
    )

# ── URL param parsing ────────────────────────────────────────────────────────

def parse_config():
    """Read analysis config from URL query params, fall back to sidebar inputs."""
    qp = st.query_params
    selected = qp.get("selected", "SPY")

    start_date = qp.get("start_date", None)
    end_date = qp.get("end_date", None)
    train_years = qp.get("train_years", "1")
    test_months = qp.get("test_months", "6")

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

    try:
        train_years = int(train_years)
    except ValueError:
        train_years = 1

    try:
        test_months = int(test_months)
    except ValueError:
        test_months = 6

    return selected, start_date, end_date, train_years, test_months

# ── Sidebar ────────────────────────────────────────────────────────────────────

def render_sidebar(selected, start_date, end_date, train_years, test_months, has_url):
    """Sidebar with asset list, date range, window sizes, run button."""
    if "assets" not in st.session_state:
        st.session_state.assets = list(DEFAULT_ASSETS)

    st.sidebar.subheader("Assets")

    # Remove assets
    for asset in list(st.session_state.assets):
        if st.sidebar.button(f"Remove {asset}", key=f"remove_{asset}", use_container_width=True):
            st.session_state.assets.remove(asset)

    # Add asset
    with st.sidebar.expander("Add Asset"):
        new_ticker = st.text_input("Ticker", key="new_ticker_input")
        if new_ticker and st.button("Add", key="add_ticker_btn", use_container_width=True):
            t = new_ticker.strip().upper()
            if t and t not in st.session_state.assets:
                st.session_state.assets.append(t)
            st.rerun()

    # Show current assets
    st.sidebar.write(", ".join(st.session_state.assets))

    st.sidebar.subheader("Date Range")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        if has_url:
            start_date = st.date_input("From", value=start_date, key="start_url")
        else:
            start_date = st.date_input("From", value=start_date)
    with col2:
        if has_url:
            end_date = st.date_input("To", value=end_date, key="end_url")
        else:
            end_date = st.date_input("To", value=end_date)

    st.sidebar.subheader("Walk-Forward")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        if has_url:
            train_years = st.number_input("Train Period (years)", min_value=1, max_value=5, value=train_years, step=1, key="train_years_url")
        else:
            train_years = st.number_input("Train Period (years)", min_value=1, max_value=5, value=train_years, step=1)
    with col2:
        if has_url:
            test_months = st.number_input("Test Period (months)", min_value=1, max_value=12, value=test_months, step=1, key="test_months_url")
        else:
            test_months = st.number_input("Test Period (months)", min_value=1, max_value=12, value=test_months, step=1)

    if has_url:
        run_btn = True
    else:
        st.sidebar.subheader("Actions")
        if st.sidebar.button("▶ Run Backtest", type="primary", use_container_width=True):
            st.session_state._run_clicked = True
            st.rerun()
        run_btn = False

    return start_date, end_date, train_years, test_months

# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    st.session_state.setdefault("assets", list(DEFAULT_ASSETS))
    st.session_state.setdefault("selected_asset", "SPY")
    st.session_state.setdefault("results", {})
    st.session_state.setdefault("regime_data", {})
    st.session_state.setdefault("run_date", None)

    # ── Path-based routing: parse URL params ────────────────────────────────
    url_selected, url_start, url_end, url_train, url_test = parse_config()
    has_url = any(st.query_params.get(k) for k in ("selected", "start_date", "end_date", "train_years", "test_months"))

    # ── Sidebar ────────────────────────────────────────────────────────
    start_date, end_date, train_years, test_months = render_sidebar(
        url_selected, url_start, url_end, url_train, url_test, has_url
    )

    # ── Persist button click ────────────────────────────────────────────────
    if st.session_state.get("_run_clicked", False):
        st.session_state._run_clicked = False
        has_url = True  # force past empty-state check

    if not has_url:
        components.html(
            '<div style="display:flex;align-items:center;justify-content:center;'
            'height:60vh;font-family:DM Sans,sans-serif;color:'
            + TEXT_MUTED + ';font-size:16px;">'
            "Click Run Backtest to start"
            "</div>",
            height=200,
        )
        return

    # ── Update URL params to reflect current values ─────────────────────────
    qp = st.query_params
    qp["start_date"] = start_date.strftime("%Y-%m-%d")
    qp["end_date"] = end_date.strftime("%Y-%m-%d")
    qp["train_years"] = str(train_years)
    qp["test_months"] = str(test_months)

    # Run backtest
    with st.spinner("Downloading data..."):
        raw_data = {}
        feature_data = {}
        for asset in st.session_state.assets:
            try:
                raw = yf.download(asset, start=start_date, end=end_date, progress=False)
                if raw.empty:
                    st.warning(f"No data for {asset}, skipping.")
                    continue
                if isinstance(raw.columns, pd.MultiIndex):
                    raw.columns = raw.columns.get_level_values(0)
                feature_data[asset] = build_features(raw)
                raw_data[asset] = raw
            except Exception as e:
                st.warning(f"Error downloading {asset}: {e}")
                continue

    if not feature_data:
        st.error("No data downloaded.")
        return

    # Train HMMs and run walk-forward
    results = {}
    regime_data = {}

    for asset in st.session_state.assets:
        df = feature_data[asset]
        if df.empty:
            continue

        # Train HMM on full period for regime labels
        feature_cols = ["log_ret", "realized_vol", "vol_ratio", "hl_range"]
        X = df[feature_cols].values

        model = hmm.GaussianHMM(
            n_components=3,
            covariance_type="diag",
            n_iter=200,
            random_state=42,
        )
        model.fit(X)

        labels, confidences = forward_filter_labels(model, X)
        labels = apply_stability_filter(labels)

        means = model.means_
        regime_names = label_regimes(means, 3)

        # Walk-forward backtest
        bf = walk_forward_backtest(df, regime_names, train_years, test_months)

        # Metrics
        strat_ann = calc_ann_return(bf["strategy"], bf["date"])
        bh_ann = calc_ann_return(bf["buy_hold"], bf["date"])
        sma_ann = calc_ann_return(bf["sma_trend"], bf["date"])

        strat_sharpe = calc_sharpe(bf["strategy"])
        bh_sharpe = calc_sharpe(bf["buy_hold"])
        sma_sharpe = calc_sharpe(bf["sma_trend"])

        strat_dd = calc_max_drawdown(bf["strategy"])
        bh_dd = calc_max_drawdown(bf["buy_hold"])
        sma_dd = calc_max_drawdown(bf["sma_trend"])

        sharpe_improvement = strat_sharpe - bh_sharpe

        # Stress test draws
        crisis_drawdowns = {}
        bh_crisis_drawdowns = {}
        for crisis_name, (cs, ce) in CRISIS_WINDOWS.items():
            crisis_drawdowns[crisis_name] = stress_test(bf["strategy"], bf["date"], cs, ce)
            bh_crisis_drawdowns[crisis_name] = stress_test(bf["buy_hold"], bf["date"], cs, ce)

        results[asset] = {
            "strat_return": strat_ann,
            "bh_return": bh_ann,
            "sma_return": sma_ann,
            "strat_sharpe": strat_sharpe,
            "bh_sharpe": bh_sharpe,
            "sma_sharpe": sma_sharpe,
            "strat_max_dd": strat_dd,
            "bh_max_dd": bh_dd,
            "sma_max_dd": sma_dd,
            "sharpe_improvement": sharpe_improvement,
            "crisis_drawdowns": crisis_drawdowns,
            "bh_crisis_drawdowns": bh_crisis_drawdowns,
        }

        regime_data[asset] = {
            "dates": bf["date"],
            "labels": labels,
            "regime_names": regime_names,
            "colors": [ASSET_COLORS.get(asset, TEXT_MUTED)],
        }

        # Store equity curves for chart rendering
        st.session_state[f"_equity_{asset}"] = bf

    if not results:
        st.error("No backtest results.")
        return

    # Store in session state
    st.session_state.results = results
    st.session_state.regime_data = regime_data
    st.session_state.run_date = pd.Timestamp.now()

    render_results()

def render_results():
    """Render the full dashboard layout with existing results."""
    results = st.session_state.results
    regime_data = st.session_state.regime_data
    assets = st.session_state.assets

    # ── Top: Asset Tabs ──────────────────────────────────────────────────────
    tab_items = []
    for asset in assets:
        if asset not in results:
            continue
        data = results[asset]
        color = ASSET_COLORS.get(asset, TEXT_MUTED)
        tab_items.append(
            f'<div style="display:inline-block;text-align:center;cursor:pointer;'
            f'padding:8px 16px;border-radius:8px;border:1px solid {color}40;'
            f'margin-right:8px;transition:all 0.2s;"'
            f'onmouseover="this.style.background=\'{color}20\'" '
            f'onmouseout="this.style.background=\'transparent\'" '
            f'onclick="parent.location.href=\'?selected={asset}\''
            f'">{asset}'
            f'{asset_tab_summary(asset, data)}'
            f'</div>'
        )

    # Selected asset from URL
    selected = st.query_params.get("selected", None)
    if selected is None:
        selected = assets[0] if assets else None
    if selected not in assets or selected not in results:
        selected = assets[0] if assets else None

    if selected:
        components.html(
            '<div style="font-family:DM Sans,sans-serif;font-size:28px;font-weight:700;'
            f'color:{ASSET_COLORS.get(selected, TEXT_PRIMARY)};margin-bottom:0.5rem;">'
            f'{selected}</div>',
            height=0,
        )

    components.html(
        '<div style="display:flex;flex-wrap:wrap;gap:8px;margin-bottom:1.5rem;">'
        + "".join(tab_items) + "</div>",
        height=0,
    )

    if not selected or selected not in results:
        return

    # ── Hero: Equity Curves ──────────────────────────────────────────────────
    asset_color = ASSET_COLORS.get(selected, TEXT_MUTED)
    data = results[selected]

    components.html(section_header("Equity Curve"), height=0)

    # Get equity curve from session state
    equity_df = st.session_state.get(f"_equity_{selected}")
    if equity_df is not None:
        fig_eq = build_equity_chart(equity_df, asset_color)
        st.plotly_chart(fig_eq, key=f"equity_{selected}", use_container_width=True, config={"displayModeBar": False})

    # Metric cards
    strat_ret = f"{data['strat_return']:+.2%}"
    bh_ret = f"{data['bh_return']:+.2%}"
    sharpe_imp = f"{data['sharpe_improvement']:+.2f}"
    sharpe_color = ACCENT_GREEN if data['sharpe_improvement'] >= 0 else ACCENT_RED
    strat_dd = f"{data['strat_max_dd']:+.2%}"
    bh_dd = f"{data['bh_max_dd']:+.2%}"

    components.html(
        f'<div style="display:flex;gap:2rem;flex-wrap:wrap;margin-bottom:1.5rem;">'
        f'{metric_card("Strategy Return", strat_ret, asset_color)}'
        f'{metric_card("Buy & Hold", bh_ret, TEXT_MUTED)}'
        f'{metric_card("Sharpe Imp.", sharpe_imp, sharpe_color)}'
        f'{metric_card("Strategy Max DD", strat_dd, asset_color)}'
        f'{metric_card("B&H Max DD", bh_dd, TEXT_MUTED)}'
        f'</div>',
        height=0,
    )

    # ── Middle: Regime Timeline Strips ───────────────────────────────────────
    components.html(section_header("Regime Timeline"), height=0)

    fig_strips = build_regime_strips(regime_data)
    st.plotly_chart(fig_strips, key="regime_strips", use_container_width=True, config={"displayModeBar": False})

    # ── Bottom: Comparison Table ─────────────────────────────────────────────
    components.html(section_header("Asset Comparison"), height=0)

    best_asset = max(results.keys(), key=lambda a: results[a]["sharpe_improvement"])

    # Build styled comparison table
    rows = []
    for asset in assets:
        if asset not in results:
            continue
        data = results[asset]
        imp_color = ACCENT_GREEN if data["sharpe_improvement"] >= 0 else ACCENT_RED
        is_best = asset == best_asset
        rows.append({
            "Asset": asset,
            "Strategy Return (ann.)": f"{data['strat_return']:+.2%}",
            "Buy&Hold Return (ann.)": f"{data['bh_return']:+.2%}",
            "Strategy Max DD": f"{data['strat_max_dd']:+.2%}",
            "Buy&Hold Max DD": f"{data['bh_max_dd']:+.2%}",
            "Strategy Sharpe": f"{data['strat_sharpe']:+.2f}",
            "Buy&Hold Sharpe": f"{data['bh_sharpe']:+.2f}",
            "Sharpe Improvement": f'{data["sharpe_improvement"]:+.2f}',
        })

    comp_df = pd.DataFrame(rows)

    # Style: color Sharpe Improvement green/red, highlight best row
    def color_sharpe_improvement(s):
        return s.apply(lambda v: f"color: {ACCENT_GREEN if float(v) >= 0 else ACCENT_RED};")

    def highlight_best_row(row):
        if row["Asset"] == best_asset:
            return [f"background-color: rgba(0,212,255,0.08); border-left: 2px solid {ASSET_COLORS.get(best_asset, ACCENT_CYAN)};"] * len(row)
        return [""] * len(row)

    styled = comp_df.style.format(
        {"Sharpe Improvement": lambda x: x}  # already formatted, just need color
    ).apply(color_sharpe_improvement, subset=["Sharpe Improvement"]).apply(highlight_best_row, axis=1)

    try:
        styled = styled.set_table_styles([
            {"selector": "th", "props": [
                ("background-color", BG_PRIMARY),
                ("color", TEXT_PRIMARY),
                ("font-family", "DM Sans, sans-serif"),
                ("font-weight", "500"),
                ("font-size", "11px"),
                ("text-transform", "uppercase"),
                ("letter-spacing", "1px"),
                ("border-color", BORDER),
                ("padding", "8px 12px"),
            ]},
            {"selector": "tbody tr", "props": [
                ("border-bottom", f"1px solid {BORDER}"),
            ]},
            {"selector": "tbody tr:hover", "props": [
                ("background-color", BG_CARD_HOVER),
            ]},
            {"selector": "td", "props": [
                ("background-color", BG_CARD),
                ("color", TEXT_SECONDARY),
                ("border-color", BORDER),
                ("font-family", "JetBrains Mono, monospace"),
                ("font-size", "12px"),
            ]},
        ])
    except Exception:
        pass

    st.dataframe(styled, use_container_width=True, hide_index=True)

    # ── Bottom: Stress Tests ─────────────────────────────────────────────────
    components.html(section_header("Stress Test Results"), height=0)

    cols = st.columns(3)
    for idx, (crisis_name, _) in enumerate(CRISIS_WINDOWS.items()):
        with cols[idx]:
            components.html(
                f'<div style="font-family:DM Sans,sans-serif;font-size:14px;font-weight:600;'
                f'color:{TEXT_PRIMARY};margin-bottom:0.5rem;">{crisis_name}</div>',
                height=0,
            )
            fig_stress = build_stress_charts(results, crisis_name)
            st.plotly_chart(fig_stress, key=f"stress_{crisis_name}", use_container_width=True, config={"displayModeBar": False})

    # ── Summary ──────────────────────────────────────────────────────────────
    best_improve = max(results.keys(), key=lambda a: results[a]["sharpe_improvement"])
    worst_improve = min(results.keys(), key=lambda a: results[a]["sharpe_improvement"])
    best_val = results[best_improve]["sharpe_improvement"]
    worst_val = results[worst_improve]["sharpe_improvement"]

    components.html(
        f'<div style="margin-top:2rem;padding:1rem;background:{BG_CARD};'
        f'border:1px solid {BORDER};border-radius:12px;">'
        f'<div style="font-family:DM Sans,sans-serif;font-size:14px;color:{TEXT_SECONDARY};">'
        f'Regime detection added the most value for <b>{best_improve}</b> with a Sharpe improvement of '
        f'<span style="color:{ACCENT_GREEN};">{best_val:+.2f}</span>. '
        f'It struggled most with <b>{worst_improve}</b>, suggesting regime detection may not suit this asset class as well.'
        f'</div></div>',
        height=0,
    )

# ── Unused helper (removed) ────────────────────────────────────────────────────
# build_comparison_table() and results_to_dataframe() replaced by inline styled
# table rendering in render_results().

if __name__ == "__main__":
    main()
