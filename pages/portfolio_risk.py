"""Portfolio Risk Dashboard — regime overlay, correlation, stress testing."""

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

DEFAULT_POSITIONS = [
    {"ticker": "SPY", "shares": 100, "entry": 540.0},
    {"ticker": "QQQ", "shares": 50, "entry": 480.0},
    {"ticker": "AAPL", "shares": 75, "entry": 210.0},
    {"ticker": "GLD", "shares": 40, "entry": 235.0},
    {"ticker": "TLT", "shares": 60, "entry": 88.0},
]

STRESS_SCENARIOS = {
    "2008 Crisis": {
        "SPY": -0.56, "QQQ": -0.54, "AAPL": -0.61, "GLD": 0.21, "TLT": 0.33,
    },
    "2020 Covid": {
        "SPY": -0.34, "QQQ": -0.28, "AAPL": -0.31, "GLD": -0.03, "TLT": 0.21,
    },
    "2022 Rate Hikes": {
        "SPY": -0.25, "QQQ": -0.33, "AAPL": -0.30, "GLD": -0.04, "TLT": -0.31,
    },
}

FAVORABLE_REGIMES = {"Low Vol", "Bull"}

# ── Session state helpers ──────────────────────────────────────────────────────

def get_positions():
    """Return positions from session state or defaults."""
    if "positions" not in st.session_state:
        st.session_state.positions = [dict(p) for p in DEFAULT_POSITIONS]
    return st.session_state.positions

def set_positions(positions):
    st.session_state.positions = positions

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

# ── Run regime detection for a ticker ──────────────────────────────────────────

@st.cache_data(ttl=300)
def run_regime(ticker: str, days: int = 365) -> dict | None:
    """Run HMM regime detection for a single ticker. Returns regime info dict or None."""
    try:
        raw = yf.download(ticker, period=f"{days}d", progress=False)
        if raw.empty:
            return None
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        df = build_features(raw)
        if len(df) < 60:
            return None

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

        current_label = labels[-1]
        current_conf = confidences[-1]

        if current_label == 999:
            regime_display = "Uncertain"
            stability = "Flickering"
        elif current_label < 3:
            regime_display = regime_names[current_label]
            window = labels[-20:] if len(labels) >= 20 else labels
            transitions = np.sum(window[1:] != window[:-1])
            stability = "Stable" if transitions <= 4 else "Flickering"
        else:
            regime_display = "Unknown"
            stability = "N/A"

        # Days in current regime
        days_in_regime = 0
        for i in range(len(labels) - 1, -1, -1):
            if labels[i] == current_label:
                days_in_regime += 1
            else:
                break

        return {
            "ticker": ticker,
            "regime": regime_display,
            "confidence": current_conf,
            "days_in_regime": days_in_regime,
            "stability": stability,
            "regime_names": regime_names,
            "labels": labels,
            "current_label": current_label,
        }
    except Exception as e:
        st.warning(f"Regime detection failed for {ticker}: {e}")
        return None

# ── Download current prices ────────────────────────────────────────────────────

@st.cache_data(ttl=60)
def download_current_prices(tickers: list[str]) -> dict[str, float]:
    """Return {ticker: current_price} for a list of tickers."""
    prices = {}
    raw = yf.download(tickers, period="1d", progress=False)
    for ticker in tickers:
        if isinstance(raw.columns, pd.MultiIndex):
            # MultiIndex: columns are (Close, AAPL), (High, AAPL), etc.
            key = ("Close", ticker)
            if key in raw.columns and not raw[key].dropna().empty:
                prices[ticker] = float(raw[key].dropna().iloc[-1])
        else:
            if ticker in raw.columns and not raw[ticker].dropna().empty:
                prices[ticker] = float(raw[ticker].dropna().iloc[-1])
    return prices

# ── Correlation matrix ─────────────────────────────────────────────────────────

@st.cache_data(ttl=300)
def compute_correlation_matrix(tickers: list[str], window: int = 60) -> pd.DataFrame:
    """60-day rolling correlation between all positions."""
    raw = yf.download(tickers, period="2y", progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    returns = raw["Close"].pct_change().dropna()
    # Use the last `window` days of returns to compute the correlation matrix
    return returns.tail(window).corr()

# ── Alpaca connection ──────────────────────────────────────────────────────────

def get_alpaca_positions():
    """Try to fetch positions from Alpaca API if keys are configured."""
    api_key = st.session_state.get("alpaca_api_key", "")
    api_secret = st.session_state.get("alpaca_api_secret", "")
    if not api_key or not api_secret:
        return None

    try:
        import alpaca_trade_api as api
        a = api.REST(api_key, api_secret, base_url="https://api.alpaca.markets")
        pos = a.list_positions()
        positions = []
        for p in pos:
            positions.append({
                "ticker": p.symbol,
                "shares": int(float(p.qty)),
                "entry": float(p.avg_entry_price),
                "current": float(p.current_price),
            })
        return positions
    except Exception as e:
        st.warning(f"Alpaca connection failed: {e}")
        return None

# ── Build top bar ──────────────────────────────────────────────────────────────

def build_top_bar(positions: list[dict], positions_regimes: dict[str, dict], market_open: bool):
    """Full-width premium top bar with portfolio metrics."""
    total_value = 0
    total_cost = 0
    favorable_count = 0
    total_positions = len(positions)

    for pos in positions:
        price = pos.get("current", 0) or 0
        current_value = price * pos["shares"]
        total_value += current_value
        total_cost += pos["entry"] * pos["shares"]

        # Check if favorable regime
        ticker = pos["ticker"]
        regime_info = positions_regimes.get(ticker)
        if regime_info and regime_info["regime"] in FAVORABLE_REGIMES:
            favorable_count += 1

    pnl = total_value - total_cost
    pnl_pct = (pnl / total_cost * 100) if total_cost > 0 else 0

    # Market status
    market_status_html = (
        f'<span style="display:inline-block; width:10px; height:10px; '
        f'background:{ACCENT_GREEN}; border-radius:50%; '
        f'box-shadow:0 0 6px {ACCENT_GREEN}80; margin-right:6px;"></span>'
        if market_open else
        f'<span style="display:inline-block; width:10px; height:10px; '
        f'background:{ACCENT_RED}; border-radius:50%; '
        f'box-shadow:0 0 6px {ACCENT_RED}80; margin-right:6px;"></span>'
    )
    market_label = "Market Open" if market_open else "Market Closed"

    # Regime health
    regime_health = f"{favorable_count} of {total_positions} positions in favorable regimes"

    st.markdown(
        f'<div style="font-family:DM Sans,sans-serif;font-size:52px;'
        f'font-weight:700;color:{TEXT_PRIMARY};margin-bottom:0.25rem;'
        f'text-shadow:0 0 20px {ACCENT_CYAN}40;">'
        f'${total_value:,.0f}</div>',
        unsafe_allow_html=True,
    )

    html_row(
        f'<span style="font-family:JetBrains Mono,monospace;font-size:24px;'
        f'font-weight:700;color:{pnl_color(pnl)};">'
        f'{"+" if pnl >= 0 else ""}${pnl:,.0f} ({pnl_pct:+.1f}%)</span>',
        f'<span style="font-family:DM Sans,sans-serif;font-size:13px;color:'
        f'{TEXT_SECONDARY};margin-left:1.5rem;">{total_positions} positions</span>',
        f'<span style="font-family:DM Sans,sans-serif;font-size:13px;color:'
        f'{ACCENT_CYAN};margin-left:1.5rem;">{regime_health}</span>',
        f'<span style="font-family:DM Sans,sans-serif;font-size:13px;color:'
        f'{TEXT_SECONDARY};margin-left:1.5rem;">{market_status_html} {market_label}</span>',
    )

# ── Build position card ────────────────────────────────────────────────────────

def build_position_card(pos: dict, regime_info: dict | None):
    """Dark card/row for a single position."""
    ticker = pos["ticker"]
    shares = pos["shares"]
    entry = pos["entry"]
    current = pos.get("current", 0) or 0

    pnl = (current - entry) * shares
    pnl_pct = (current - entry) / entry * 100 if entry > 0 else 0

    # P&L bar: max extent based on max expected move (e.g., 30% of position value)
    max_move = entry * shares * 0.3
    bar_max = max(max_move, 1)  # avoid zero-length bars
    pnl_bar_pct = pnl / bar_max if bar_max > 0 else 0

    # Center the bar: range from -50% to +50% of card width
    bar_center = 0.5  # 50% of width is center
    bar_start = max(0.05, 0.5 + pnl_bar_pct * 0.5)
    bar_width = abs(pnl_bar_pct) * 0.5
    bar_width = max(bar_width, 0.01)  # minimum width

    card_color = ACCENT_GREEN if pnl >= 0 else ACCENT_RED

    # Regime badge
    if regime_info:
        regime_badge_html = regime_badge(regime_info["regime"], confidence=regime_info["confidence"] * 100)
        days_in_regime_text = f"{regime_info['days_in_regime']}d in regime"
        stability_text = f" · {regime_info['stability']}"
    else:
        regime_badge_html = regime_badge("N/A")
        days_in_regime_text = ""
        stability_text = ""

    bar_left = '50%' if pnl >= 0 else str(100 - bar_width * 100) + '%'
    return f"""
    <span style="display:block;background:{BG_CARD};border:1px solid {BORDER};
                border-radius:12px;padding:1rem 1.5rem;margin-bottom:0.75rem;
                border-left:4px solid {card_color};">
        <span style="display:flex;align-items:center;justify-content:space-between;">
            <span style="display:flex;align-items:center;gap:1rem;">
                <span style="font-family:DM Sans,sans-serif;font-size:18px;
                             font-weight:700;color:{TEXT_PRIMARY};">{ticker}</span>
                {regime_badge_html}
            </span>
            <span style="text-align:right;">
                <span style="font-family:JetBrains Mono,monospace;font-size:16px;
                             font-weight:700;color:{card_color};">
                    {"+" if pnl >= 0 else ""}${abs(pnl):,.0f} ({pnl_pct:+.1f}%)
                </span>
            </span>
        </span>
        <span style="display:flex;align-items:center;gap:1.5rem;margin-top:0.75rem;">
            <span style="font-family:JetBrains Mono,monospace;font-size:12px;color:{TEXT_SECONDARY};">
                Entry: ${entry:,.0f}
            </span>
            <span style="font-family:JetBrains Mono,monospace;font-size:12px;color:{TEXT_SECONDARY};">
                Current: ${current:,.0f}
            </span>
            <span style="font-family:JetBrains Mono,monospace;font-size:12px;color:{TEXT_SECONDARY};">
                Shares: {shares}
            </span>
            <span style="font-family:JetBrains Mono,monospace;font-size:12px;color:{TEXT_MUTED};">
                {days_in_regime_text}{stability_text}
            </span>
        </span>
        <span style="margin-top:0.5rem;position:relative;height:6px;
                    background:{BG_PRIMARY};border-radius:3px;overflow:visible;">
            <span style="position:absolute;left:50%;top:0;height:100%;
                        width:1px;background:{BORDER};z-index:1;"></span>
            <span style="position:absolute;top:0;height:100%;
                        width:{bar_width * 100}%;background:{card_color};
                        border-radius:3px;
                        left:{bar_left};
                        opacity:0.8;">
            </span>
        </span>
    </span>
    """

# ── Build correlation heatmap ──────────────────────────────────────────────────

def build_correlation_heatmap(tickers: list[str]) -> go.Figure:
    """Square heatmap with dark cells, cyan scale, red warnings."""
    corr_matrix = compute_correlation_matrix(tickers, window=60)
    n = len(tickers)
    # corr_matrix is now a square DataFrame indexed by ticker
    corr_values = [[corr_matrix.iloc[i].iloc[j] for j in range(n)] for i in range(n)]

    n = len(tickers)
    # Color scale: deep navy (low) → ACCENT_CYAN (moderate) → white (high)
    colors = []
    for i in range(n):
        row = []
        for j in range(n):
            val = corr_values[i][j]
            colors.append(val)

    # Custom colorscale: 0=navy, 0.5=cyan, 1=white
    colorscale = [
        [0, "#0a1628"],      # deep navy
        [0.3, "#0a2844"],    # mid navy
        [0.5, ACCENT_CYAN],  # cyan
        [0.7, "#80eaff"],    # light cyan
        [1, "#ffffff"],      # white
    ]

    fig = go.Figure()
    fig.add_trace(go.Heatmap(
        z=colors,
        x=tickers,
        y=tickers,
        colorscale=colorscale,
        zmin=0,
        zmax=1,
        text=[[f"{corr_values[i][j]:.2f}" for j in range(n)] for i in range(n)],
        texttemplate="%{text}",
        textfont={"family": "JetBrains Mono, monospace", "size": 11, "color": "#ffffff"},
        hovertemplate="%(y)s vs %(x)s<br>Correlation: %{z:.3f}<extra></extra>",
        hoverongaps=False,
    ))

    # Add red warning borders for cells above 0.85
    for i in range(n):
        for j in range(n):
            if i != j and corr_values[i][j] > 0.85:
                fig.add_shape(
                    go.layout.Shape(
                        type="rect",
                        x0=j, x1=j + 1,
                        y0=i, y1=i + 1,
                        line=dict(width=3, color=ACCENT_RED),
                        fillcolor="rgba(255,23,68,0.1)",
                    )
                )

    layout = get_plotly_layout(
        height=n * 90 + 80,
        xaxis=dict(
            side="top",
            tickangle=0,
            showspikes=True,
            spikemode="across",
        ),
        yaxis=dict(
            autorange="reversed",
            showspikes=True,
            spikemode="across",
        ),
        showlegend=False,
        margin=dict(l=60, r=20, t=20, b=40),
    )

    fig.update_layout(layout)
    return fig

# ── Build stress test panel ────────────────────────────────────────────────────

def build_stress_test(positions: list[dict]) -> pd.DataFrame:
    """Apply historical drawdowns to current positions. Returns DataFrame with results."""
    results = []

    for scenario, drawdowns in STRESS_SCENARIOS.items():
        scenario_loss = 0
        for pos in positions:
            ticker = pos["ticker"]
            if ticker in drawdowns:
                dd = drawdowns[ticker]
            else:
                # Use SPY drawdown as approximation
                dd = drawdowns.get("SPY", 0)

            position_value = pos["shares"] * pos["current"]
            scenario_loss += position_value * dd

        total_value = sum(pos["shares"] * pos["current"] for pos in positions)
        scenario_pct = scenario_loss / total_value * 100 if total_value > 0 else 0

        # Severity color
        if abs(scenario_pct) < 10:
            severity_color = ACCENT_GREEN
        elif abs(scenario_pct) < 20:
            severity_color = ACCENT_AMBER
        else:
            severity_color = ACCENT_RED

        # Severity bar width (0-100%, capped at 100%)
        bar_width = min(abs(scenario_pct) / 40 * 100, 100)

        results.append({
            "scenario": scenario,
            "loss_dollar": scenario_loss,
            "loss_pct": scenario_pct,
            "severity_color": severity_color,
            "bar_width": bar_width,
        })

    return pd.DataFrame(results)

# ── Build watchlist ────────────────────────────────────────────────────────────

def build_watchlist(tickers: list[str]) -> pd.DataFrame:
    """Download prices, run regime detection, return DataFrame sorted by confidence desc."""
    if not tickers:
        return pd.DataFrame()

    prices = download_current_prices(tickers)
    rows = []
    for ticker in tickers:
        price = prices.get(ticker, 0)
        regime_info = run_regime(ticker, days=365)
        rows.append({
            "ticker": ticker,
            "price": price,
            "regime": regime_info["regime"] if regime_info else "N/A",
            "confidence": regime_info["confidence"] if regime_info else 0,
            "days_in_regime": regime_info["days_in_regime"] if regime_info else 0,
            "stability": regime_info["stability"] if regime_info else "N/A",
            "current_label": regime_info["current_label"] if regime_info else -1,
            "regime_names": regime_info["regime_names"] if regime_info else [],
        })

    df = pd.DataFrame(rows)
    df = df.sort_values("confidence", ascending=False).reset_index(drop=True)
    return df

# ── Sidebar ────────────────────────────────────────────────────────────────────

def render_sidebar():
    """Sidebar with position editor, CSV upload, Alpaca keys, watchlist input."""
    st.sidebar.subheader("Positions")

    # Alpaca API keys
    with st.sidebar.expander("Alpaca API (optional)"):
        st.session_state.setdefault("alpaca_api_key", "")
        st.session_state.setdefault("alpaca_api_secret", "")
        st.text_input("API Key", value=st.session_state.alpaca_api_key,
                      type="password", key="alpaca_api_key",
                      help="Alpaca API key for live positions")
        st.text_input("Secret Key", value=st.session_state.alpaca_api_secret,
                      type="password", key="alpaca_api_secret",
                      help="Alpaca secret key")
        if st.session_state.alpaca_api_key and st.session_state.alpaca_api_secret:
            if st.button("Connect", use_container_width=True):
                with st.spinner("Connecting to Alpaca..."):
                    positions = get_alpaca_positions()
                    if positions:
                        st.session_state.positions = positions
                        st.rerun()
                    else:
                        st.error("Failed to fetch positions")

    # Current positions
    positions = get_positions()

    # Edit positions
    st.sidebar.subheader("Edit Positions")
    new_positions = []
    for i, pos in enumerate(positions):
        with st.sidebar.expander(f"{pos['ticker']}", expanded=False):
            col1, col2, col3 = st.columns([3, 2, 2])
            with col1:
                ticker = st.text_input("Ticker", value=pos["ticker"], key=f"ticker_{i}")
            with col2:
                shares = st.number_input("Shares", value=int(pos["shares"]), step=1,
                                         key=f"shares_{i}")
            with col3:
                entry = st.number_input("Entry Price", value=float(pos["entry"]),
                                        format="%.2f", step=0.01, key=f"entry_{i}")

            if st.button("Remove", key=f"remove_{i}", use_container_width=True):
                new_positions.append(pos)  # keep temporarily, remove after loop
                break

            if ticker != pos["ticker"] or shares != pos["shares"] or entry != pos["entry"]:
                new_pos = {"ticker": ticker, "shares": shares, "entry": entry}
                new_positions.append(new_pos)

    # Remove the position we just removed
    positions = [p for p in positions if p not in new_positions or positions.count(p) > new_positions.count(p)]

    # Add new position
    with st.sidebar.expander("Add Position"):
        col1, col2, col3 = st.columns([3, 2, 2])
        with col1:
            new_ticker = st.text_input("Ticker", key="new_ticker")
        with col2:
            new_shares = st.number_input("Shares", value=1, step=1, key="new_shares")
        with col3:
            new_entry = st.number_input("Entry Price", value=100.0,
                                        format="%.2f", step=0.01, key="new_entry")
        if new_ticker.strip() and st.button("Add", key="add_position",
                                            use_container_width=True):
            positions.append({"ticker": new_ticker.strip().upper(),
                              "shares": new_shares, "entry": new_entry})
            st.rerun()

    # CSV upload
    with st.sidebar.expander("Upload CSV"):
        uploaded = st.file_uploader("Upload positions CSV",
                                    type=["csv"],
                                    help="Columns: ticker, shares, entry")
        if uploaded is not None:
            try:
                df = pd.read_csv(uploaded)
                if "ticker" in df.columns and "shares" in df.columns and "entry" in df.columns:
                    positions = []
                    for _, row in df.iterrows():
                        positions.append({
                            "ticker": str(row["ticker"]).strip().upper(),
                            "shares": int(row["shares"]),
                            "entry": float(row["entry"]),
                        })
                    st.success("CSV loaded!")
                    st.rerun()
            except Exception as e:
                st.error(f"Error parsing CSV: {e}")

    # Watchlist
    st.sidebar.subheader("Watchlist")
    st.session_state.setdefault("watchlist", "")
    watchlist_input = st.text_input("Add ticker", value=st.session_state.watchlist,
                                    key="watchlist_input",
                                    help="Add ticker to watchlist")
    if watchlist_input.strip() and st.button("Add to Watchlist",
                                             key="add_watchlist",
                                             use_container_width=True):
        if "watchlist" not in st.session_state:
            st.session_state.watchlist = []
        t = watchlist_input.strip().upper()
        if t and t not in st.session_state.watchlist:
            st.session_state.watchlist.append(t)
            st.rerun()

    # Show current watchlist
    if "watchlist" in st.session_state and st.session_state.watchlist:
        st.sidebar.write(", ".join(st.session_state.watchlist))

    # Refresh button
    st.markdown(
        f'<hr style="border:1px solid {BORDER};border-radius:2px;" '
        'noshade="noshade" style="width:100%;margin:1rem 0;">',
        unsafe_allow_html=True,
    )
    if st.sidebar.button("Refresh", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    st.set_page_config(
        page_title="Portfolio Risk",
        page_icon="📊",
        layout="wide",
    )

    # ── Determine market status ─────────────────────────────────────────────
    import datetime
    now = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=-4)))  # EST
    market_open = (now.weekday() < 5 and now.hour >= 9 and now.hour < 16)

    # ── Sidebar ─────────────────────────────────────────────────────────────
    render_sidebar()

    # ── Get positions ───────────────────────────────────────────────────────
    positions = get_positions()
    tickers = [p["ticker"] for p in positions]

    # ── Download current prices ─────────────────────────────────────────────
    with st.spinner("Downloading prices..."):
        prices = download_current_prices(tickers)

    # Update positions with current prices
    for pos in positions:
        if pos["ticker"] in prices:
            pos["current"] = prices[pos["ticker"]]

    # ── Run regime detection for each position ──────────────────────────────
    with st.spinner("Running regime detection..."):
        positions_regimes = {}
        for ticker in tickers:
            regime_info = run_regime(ticker, days=365)
            if regime_info:
                positions_regimes[ticker] = regime_info

    # ── Top bar ─────────────────────────────────────────────────────────────
    build_top_bar(positions, positions_regimes, market_open)

    # ── Two-column layout ───────────────────────────────────────────────────
    left_col, right_col = st.columns([0.6, 0.4])

    # ── Left column: Positions ──────────────────────────────────────────────
    with left_col:
        st.markdown(section_header("Positions"), unsafe_allow_html=True)
        for pos in positions:
            regime_info = positions_regimes.get(pos["ticker"])
            components.html(
                build_position_card(pos, regime_info),
                height=180,
                scrolling=False,
            )

    # ── Right column: stacked panels ────────────────────────────────────────
    with right_col:
        # Panel 1: Correlation Heatmap
        st.markdown(section_header("Correlation Risk"), unsafe_allow_html=True)
        if len(tickers) >= 2:
            corr_fig = build_correlation_heatmap(tickers)
            st.plotly_chart(corr_fig, use_container_width=True,
                            config={"displayModeBar": False})
        else:
            st.markdown(
                f'<div style="font-size:12px;color:{TEXT_MUTED};text-align:center;padding:2rem;">'
                "Need at least 2 positions for correlation analysis</div>",
                unsafe_allow_html=True,
            )

        # Panel 2: Stress Test
        st.markdown(section_header("Stress Test"), unsafe_allow_html=True)
        stress_df = build_stress_test(positions)
        for _, row in stress_df.iterrows():
            scenario = row["scenario"]
            loss_dollar = row["loss_dollar"]
            loss_pct = row["loss_pct"]
            severity_color = row["severity_color"]
            bar_width = row["bar_width"]

            # Severity bar
            bar_html = f"""
            <span style="margin-top:0.5rem;position:relative;height:6px;
                        background:{BG_PRIMARY};border-radius:3px;overflow:visible;">
                <span style="position:absolute;left:0;top:0;height:100%;
                            width:{bar_width}%;background:{severity_color};
                            border-radius:3px;opacity:0.8;"></span>
            </span>
            """

            components.html(
                f"""
                <span style="display:block;background:{BG_CARD};border:1px solid {BORDER};
                            border-radius:12px;padding:1rem;margin-bottom:0.75rem;">
                    <span style="font-family:DM Sans,sans-serif;font-size:14px;
                                font-weight:600;color:{TEXT_PRIMARY};margin-bottom:0.5rem;">
                        {scenario}
                    </span>
                    <span style="display:flex;align-items:baseline;gap:1rem;">
                        <span style="font-family:JetBrains Mono,monospace;font-size:20px;
                                     font-weight:700;color:{severity_color};">
                            {"+" if loss_dollar >= 0 else ""}${abs(loss_dollar):,.0f}
                        </span>
                        <span style="font-family:JetBrains Mono,monospace;font-size:14px;
                                     color:{severity_color};">
                            ({loss_pct:+.1f}%)
                        </span>
                    </span>
                    {bar_html}
                </span>
                """,
                height=120,
                scrolling=False,
            )

        # Panel 3: Watchlist
        st.markdown(section_header("Watchlist"), unsafe_allow_html=True)
        watchlist = st.session_state.get("watchlist", [])
        if watchlist:
            watchlist_df = build_watchlist(watchlist)
            for _, row in watchlist_df.iterrows():
                ticker = row["ticker"]
                price = row["price"]
                regime = row["regime"]
                confidence = row["confidence"]
                days_in_regime = row["days_in_regime"]
                stability = row["stability"]

                # Confidence bar
                conf_width = confidence * 100

                # Regime color
                regime_color = REGIME_COLORS.get(regime, TEXT_MUTED)

                # Days in regime bar (visual)
                max_days = 200  # cap at ~200 days for visual scaling
                days_width = min(days_in_regime / max_days * 100, 100)

                components.html(
                    f"""
                    <span style="display:block;background:{BG_CARD};border:1px solid {BORDER};
                                border-radius:12px;padding:1rem;margin-bottom:0.5rem;">
                        <span style="display:flex;align-items:center;justify-content:space-between;">
                            <span style="display:flex;align-items:center;gap:0.75rem;">
                                <span style="font-family:DM Sans,sans-serif;font-size:14px;
                                             font-weight:700;color:{TEXT_PRIMARY};">{ticker}</span>
                                {regime_badge(regime, confidence=confidence * 100)}
                            </span>
                            <span style="font-family:JetBrains Mono,monospace;font-size:14px;
                                         font-weight:700;color:{TEXT_PRIMARY};">
                                ${price:,.0f}
                            </span>
                        </span>
                        <span style="margin-top:0.5rem;display:flex;align-items:center;gap:1rem;">
                            <span style="display:flex;align-items:center;gap:0.5rem;">
                                <span style="font-size:10px;color:{TEXT_MUTED};">Confidence</span>
                                <span style="width:60px;height:4px;background:{BG_PRIMARY};
                                            border-radius:2px;overflow:visible;">
                                    <span style="width:{conf_width}%;height:100%;
                                                background:{regime_color};border-radius:2px;"></span>
                                </span>
                                <span style="font-family:JetBrains Mono,monospace;font-size:10px;color:{TEXT_MUTED};">
                                    {confidence:.0%}
                                </span>
                            </span>
                            <span style="display:flex;align-items:center;gap:0.5rem;">
                                <span style="font-size:10px;color:{TEXT_MUTED};">Days</span>
                                <span style="width:40px;height:4px;background:{BG_PRIMARY};
                                            border-radius:2px;overflow:visible;">
                                    <span style="width:{days_width}%;height:100%;
                                                background:{regime_color};border-radius:2px;"></span>
                                </span>
                                <span style="font-family:JetBrains Mono,monospace;font-size:10px;color:{TEXT_MUTED};">
                                    {days_in_regime}d
                                </span>
                            </span>
                            <span style="font-size:10px;color:{TEXT_MUTED};">{stability}</span>
                        </span>
                    </span>
                    """,
                    height=140,
                    scrolling=False,
                )
        else:
            st.markdown(
                f'<div style="font-size:12px;color:{TEXT_MUTED};text-align:center;padding:2rem;">'
                "Add tickers to watchlist from sidebar</div>",
                unsafe_allow_html=True,
            )

if __name__ == "__main__":
    main()
