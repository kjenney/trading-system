"""Regime-Based Market Screener — scan tickers, classify regimes, filter setups."""

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

DEFAULT_LARGE_CAP = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "JPM", "V", "UNH"]
DEFAULT_ETFS = ["SPY", "QQQ", "IWM", "DIA", "XLF", "XLE", "XLK", "GLD", "TLT", "HYG"]
DEFAULT_CRYPTO = ["BTC-USD", "ETH-USD", "SOL-USD"]
DEFAULT_TICKERS = DEFAULT_LARGE_CAP + DEFAULT_ETFS + DEFAULT_CRYPTO

CATEGORY_COLORS = {
    "Large Cap": ACCENT_CYAN,
    "ETF": ACCENT_VIOLET,
    "Crypto": ACCENT_AMBER,
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

# ── Regime labeling ──────────────────────────────────────────────────────────────

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
            labels.append("Medium Vol")
    return labels

# ── HMM scan for a single ticker ───────────────────────────────────────────────

def scan_ticker(ticker: str, start_date: pd.Timestamp, end_date: pd.Timestamp) -> dict | None:
    """Download data, train HMM, return regime analysis for one ticker.

    Returns None on failure.
    """
    try:
        raw = yf.download(ticker, start=start_date, end=end_date, progress=False)
        if raw.empty:
            return None
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)

        df = build_features(raw)
        if len(df) < 60:
            return None

        feature_cols = ["log_ret", "realized_vol", "vol_ratio", "hl_range"]
        X = df[feature_cols].values

        best_bic = np.inf
        best_n = 3
        best_model = None

        for n_comp in range(3, 6):
            try:
                model = hmm.GaussianHMM(
                    n_components=n_comp,
                    covariance_type="diag",
                    n_iter=200,
                    random_state=42,
                )
                model.fit(X)
                bic = compute_bic(model, X)
                if bic < best_bic:
                    best_bic = bic
                    best_n = n_comp
                    best_model = model
            except Exception:
                continue

        if best_model is None:
            return None

        labels, confidences = forward_filter_labels(best_model, X)
        labels = apply_stability_filter(labels)

        means = best_model.means_
        regime_names = label_regimes(means, best_n)

        # Current regime
        current_label = labels[-1]
        current_conf = confidences[-1]

        if current_label == 999:
            regime_display = "Uncertain"
        elif current_label < best_n:
            regime_display = regime_names[current_label]
        else:
            regime_display = "Unknown"

        # Days in current regime — count consecutive from end
        days_in_regime = 1
        for i in range(len(labels) - 2, -1, -1):
            if labels[i] == current_label:
                days_in_regime += 1
            else:
                break

        # 50-day SMA position
        sma_50 = df["Close"].rolling(50).mean().iloc[-1]
        current_price = df["Close"].iloc[-1]
        sma_position = "Above" if current_price > sma_50 else "Below"

        # 20-day volume trend (compare last 10 days avg to prior 10 days avg)
        recent_vol = df["Volume"].iloc[-10:].mean()
        prior_vol = df["Volume"].iloc[-20:-10].mean()
        volume_trend = "Increasing" if recent_vol > prior_vol else "Decreasing"

        # Regime history over last year (percentage in each regime)
        one_year_ago = df.index[-252:] if len(df) >= 252 else df.index
        one_year_labels = labels[-252:] if len(labels) >= 252 else labels
        regime_history = {}
        for i in range(best_n):
            regime_history[regime_names[i]] = (one_year_labels == i).sum() / len(one_year_labels)
        if 999 in one_year_labels:
            regime_history["Uncertain"] = (one_year_labels == 999).sum() / len(one_year_labels)

        # Full chart data for click-to-chart
        chart_data = {
            "dates": df.index.tolist(),
            "prices": df["Close"].values.tolist(),
            "labels": labels.tolist(),
            "confidences": confidences.tolist(),
            "regime_names": regime_names,
            "sma_50": sma_50,
        }

        return {
            "ticker": ticker,
            "regime": regime_display,
            "confidence": current_conf * 100,
            "days_in_regime": days_in_regime,
            "sma_position": sma_position,
            "volume_trend": volume_trend,
            "price": current_price,
            "regime_history": regime_history,
            "chart_data": chart_data,
            "category": get_category(ticker),
            "n_regimes": best_n,
        }
    except Exception:
        return None

# ── Category classification ────────────────────────────────────────────────────

def get_category(ticker: str) -> str:
    """Classify ticker into Large Cap / ETF / Crypto."""
    if ticker in DEFAULT_CRYPTO:
        return "Crypto"
    if ticker in DEFAULT_ETFS:
        return "ETF"
    if ticker in DEFAULT_LARGE_CAP:
        return "Large Cap"
    # Heuristic: crypto tickers end in -USD
    if "-USD" in ticker:
        return "Crypto"
    # ETFs are usually 1-5 letter tickers
    if len(ticker) <= 4:
        return "ETF"
    return "Large Cap"

# ── Build price chart with regime bands ────────────────────────────────────────

def build_screener_chart(chart_data: dict, ticker: str) -> go.Figure:
    """Small chart with regime-colored background bands."""
    dates = chart_data["dates"]
    prices = chart_data["prices"]
    labels = chart_data["labels"]
    confidences = chart_data["confidences"]
    regime_names = chart_data["regime_names"]
    n = len(dates)

    traces = []
    band_start = 0
    prev_regime = labels[0]

    for i in range(1, n):
        curr = labels[i]
        if curr != prev_regime:
            x0, x1 = dates[band_start], dates[min(i, n - 1)]
            y0, y1 = prices[band_start], prices[min(i, n - 1)]

            if prev_regime == 999:
                band_color = REGIME_COLORS["Uncertain"]
                band_label = "Uncertain"
            elif prev_regime < len(regime_names):
                band_color = REGIME_COLORS.get(regime_names[prev_regime], TEXT_MUTED)
                band_label = regime_names[prev_regime]
            else:
                band_color = TEXT_MUTED
                band_label = "Regime"

            traces.append(
                go.Scatter(
                    x=[x0, x1],
                    y=[y0, y1],
                    mode="lines",
                    line=dict(width=0, color=band_color),
                    opacity=0.13,
                    name=band_label,
                    hoverinfo="skip",
                    showlegend=False,
                    fill="tozeroy",
                    yaxis="y2",
                )
            )
            band_start = i
            prev_regime = curr

    # Last band
    if band_start < n:
        x0, x1 = dates[band_start], dates[n - 1]
        y0, y1 = prices[band_start], prices[n - 1]

        if prev_regime == 999:
            band_color = REGIME_COLORS["Uncertain"]
            band_label = "Uncertain"
        elif prev_regime < len(regime_names):
            band_color = REGIME_COLORS.get(regime_names[prev_regime], TEXT_MUTED)
            band_label = regime_names[prev_regime]
        else:
            band_color = TEXT_MUTED
            band_label = "Regime"

        traces.append(
            go.Scatter(
                x=[x0, x1],
                y=[y0, y1],
                mode="lines",
                line=dict(width=0, color=band_color),
                opacity=0.13,
                name=band_label,
                hoverinfo="skip",
                showlegend=False,
                fill="tozeroy",
                yaxis="y2",
            )
        )

    # Price line
    traces.append(
        go.Scatter(
            x=dates,
            y=prices,
            mode="lines",
            name="Price",
            line=dict(color="white", width=1.5),
        )
    )

    # SMA 50
    sma_50 = chart_data["sma_50"]
    sma_50_series = []
    for i, date in enumerate(dates):
        if i < 49:
            sma_50_series.append(None)
        else:
            sma_50_series.append(prices[i - 49])

    traces.append(
        go.Scatter(
            x=dates,
            y=sma_50_series,
            mode="lines",
            name="SMA 50",
            line=dict(color=ACCENT_AMBER, width=1, dash="dot"),
            opacity=0.6,
        )
    )

    layout = get_plotly_layout(
        height=350,
        xaxis=dict(
            type="date",
            showspikes=True,
            spikemode="across",
        ),
        yaxis=dict(
            title="Price ($)",
            showspikes=True,
            spikemode="across",
        ),
        yaxis2=dict(
            overlaying="y",
            side="right",
            showgrid=False,
            showticklabels=False,
            showline=False,
        ),
        legend=dict(
            x=0.01,
            y=0.99,
            bgcolor="rgba(0,0,0,0)",
            font=dict(color=TEXT_SECONDARY),
        ),
    )

    fig = go.Figure(data=traces, layout=layout)
    fig.update_layout(bargap=0)
    return fig

# ── Build regime history bar ───────────────────────────────────────────────────

def build_regime_history_chart(regime_history: dict) -> go.Figure:
    """Horizontal bar chart: % time in each regime over last year."""
    fig = go.Figure()

    regimes = list(regime_history.keys())
    pct_values = [v * 100 for v in regime_history.values()]
    colors = [REGIME_COLORS.get(r, TEXT_MUTED) for r in regimes]

    fig.add_trace(go.Bar(
        x=pct_values,
        y=regimes,
        orientation="h",
        marker_color=colors,
        marker_line_color=BG_PRIMARY,
        marker_line_width=1,
        hovertemplate="%{y}: %{x:.1f}%<br>",
    ))

    layout = get_plotly_layout(
        height=120,
        xaxis=dict(
            title="% of Time",
            tickformat=".0f%%",
            showgrid=True,
            gridcolor=GRID_LINE,
        ),
        yaxis=dict(
            title="Regime",
            showgrid=False,
            showline=False,
        ),
        showlegend=False,
    )

    fig = go.Figure(layout=layout)
    return fig

# ── Build regime distribution bar ──────────────────────────────────────────────

def build_distribution_bar(results: list[dict]) -> go.Figure:
    """Horizontal stacked bar: what % of tickers in each regime."""
    regime_counts = {}
    for r in results:
        regime = r["regime"]
        regime_counts[regime] = regime_counts.get(regime, 0) + 1

    total = len(results)
    if total == 0:
        return None

    regimes = ["Low Vol", "Medium Vol", "High Vol", "Uncertain"]
    values = [regime_counts.get(r, 0) / total * 100 for r in regimes]
    colors = [REGIME_COLORS.get(r, TEXT_MUTED) for r in regimes]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=values,
        y=["Ticker Distribution"],
        orientation="h",
        marker_color=colors,
        marker_line_color=BG_PRIMARY,
        marker_line_width=2,
        hovertemplate="%{y}<br>%{x:.1f}%<br>",
    ))

    layout = get_plotly_layout(
        height=120,
        xaxis=dict(
            title="Percentage of Tickers",
            tickformat=".0f%%",
            showgrid=True,
            gridcolor=GRID_LINE,
        ),
        yaxis=dict(
            title="",
            showgrid=False,
            showline=False,
            showticklabels=False,
        ),
        showlegend=False,
    )

    fig = go.Figure(layout=layout)
    return fig

# ── Sidebar ────────────────────────────────────────────────────────────────────

def render_sidebar():
    """Sidebar with ticker universe, filters, sort, scan button."""
    if "screener_tickers" not in st.session_state:
        st.session_state.screener_tickers = list(DEFAULT_TICKERS)

    st.sidebar.subheader("Ticker Universe")

    # Large Cap
    st.sidebar.write("**Large Cap**")
    for t in DEFAULT_LARGE_CAP:
        if t in st.session_state.screener_tickers:
            if st.sidebar.button(f"− {t}", key=f"rem_lc_{t}", use_container_width=True, type="secondary"):
                st.session_state.screener_tickers.remove(t)
                st.rerun()

    # ETFs
    st.sidebar.write("**ETFs**")
    for t in DEFAULT_ETFS:
        if t in st.session_state.screener_tickers:
            if st.sidebar.button(f"− {t}", key=f"rem_etf_{t}", use_container_width=True, type="secondary"):
                st.session_state.screener_tickers.remove(t)
                st.rerun()

    # Crypto
    st.sidebar.write("**Crypto**")
    for t in DEFAULT_CRYPTO:
        if t in st.session_state.screener_tickers:
            if st.sidebar.button(f"− {t}", key=f"rem_cry_{t}", use_container_width=True, type="secondary"):
                st.session_state.screener_tickers.remove(t)
                st.rerun()

    # Custom tickers
    st.sidebar.subheader("Custom Tickers")
    custom = [t for t in st.session_state.screener_tickers if t not in DEFAULT_LARGE_CAP and t not in DEFAULT_ETFS and t not in DEFAULT_CRYPTO]
    for t in custom:
        if st.sidebar.button(f"− {t}", key=f"rem_custom_{t}", use_container_width=True, type="secondary"):
            st.session_state.screener_tickers.remove(t)
            st.rerun()

    with st.sidebar.expander("Add Ticker"):
        new_ticker = st.text_input("Ticker", key="new_ticker_input")
        if new_ticker and st.button("Add", key="add_ticker_btn", use_container_width=True):
            t = new_ticker.strip().upper()
            if t and t not in st.session_state.screener_tickers:
                st.session_state.screener_tickers.append(t)
            st.rerun()

    st.sidebar.subheader("Filters")

    # Regime filter
    regime_filter = st.sidebar.multiselect(
        "Regime",
        options=["Low Vol", "Medium Vol", "High Vol", "Uncertain"],
        default=["Low Vol", "Medium Vol", "High Vol", "Uncertain"],
        key="regime_filter",
    )

    # Confidence filter
    min_conf = st.sidebar.slider(
        "Min Confidence",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05,
        key="min_confidence",
    )

    # SMA filter
    sma_filter = st.sidebar.selectbox(
        "50 SMA",
        options=["Both", "Above", "Below"],
        index=0,
        key="sma_filter",
    )

    # Volume filter
    vol_filter = st.sidebar.selectbox(
        "Volume Trend",
        options=["Both", "Increasing", "Decreasing"],
        index=0,
        key="vol_filter",
    )

    st.sidebar.subheader("Sort By")
    sort_by = st.sidebar.selectbox(
        "Sort",
        options=["Confidence", "Days in Regime", "Ticker"],
        index=0,
        key="sort_by",
    )

    # Date range
    st.sidebar.subheader("Date Range")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        start_date = st.date_input("From", value=pd.Timestamp.now() - pd.DateOffset(years=2))
    with col2:
        end_date = st.date_input("To", value=pd.Timestamp.now())

    # Scan button
    st.sidebar.subheader("")
    if st.sidebar.button("🔍 Scan Market", type="primary", use_container_width=True):
        st.session_state._scan_clicked = True
        st.rerun()

    return start_date, end_date, sort_by

# ── Render summary cards ───────────────────────────────────────────────────────

def render_summary_cards(results: list[dict]):
    """Top metric cards: regime counts, avg confidence, strongest regime."""
    if not results:
        return

    regime_counts = {}
    total_conf = 0.0
    for r in results:
        regime = r["regime"]
        regime_counts[regime] = regime_counts.get(regime, 0) + 1
        total_conf += r["confidence"]

    avg_conf = total_conf / len(results)
    strongest = max(regime_counts, key=regime_counts.get)

    html_parts = []
    for regime in ["Low Vol", "Medium Vol", "High Vol", "Uncertain"]:
        count = regime_counts.get(regime, 0)
        color = REGIME_COLORS.get(regime, TEXT_MUTED)
        html_parts.append(metric_card(f"{count} in {regime}", str(count), color))

    html_parts.append(metric_card("Avg Confidence", f"{avg_conf:.0f}%", ACCENT_CYAN))
    html_parts.append(metric_card("Strongest Regime", strongest, REGIME_COLORS.get(strongest, TEXT_MUTED)))

    components.html(
        '<div style="display:flex;flex-wrap:wrap;gap:1.5rem;align-items:flex-start;">'
        + "".join(html_parts) + "</div>",
        height=0,
    )

# ── Render category analysis ───────────────────────────────────────────────────

def render_category_analysis(results: list[dict]):
    """Show which category has the most favorable regimes."""
    if not results:
        return

    category_regime = {}
    for r in results:
        cat = r["category"]
        regime = r["regime"]
        if cat not in category_regime:
            category_regime[cat] = {"Low Vol": 0, "Medium Vol": 0, "High Vol": 0, "Uncertain": 0}
        category_regime[cat][regime] = category_regime[cat].get(regime, 0) + 1

    # "Favorable" = Low Vol count (more = better)
    favorable = {cat: cr.get("Low Vol", 0) for cat, cr in category_regime.items()}
    best_cat = max(favorable, key=favorable.get)

    st.markdown(
        f'<div style="margin-bottom:1rem;font-family:DM Sans,sans-serif;font-size:13px;color:{TEXT_SECONDARY};">'
        f'Category with most Low Vol setups: <b style="color:{REGIME_COLORS.get(best_cat, ACCENT_CYAN)}">{best_cat}</b>'
        f' ({favorable[best_cat]} tickers)</div>',
        unsafe_allow_html=True,
    )

    # Show breakdown
    html_parts = []
    for cat in sorted(favorable.keys()):
        color = CATEGORY_COLORS.get(cat, TEXT_MUTED)
        cr = category_regime[cat]
        total = sum(cr.values())
        parts = ", ".join(f"{k}: {v}" for k, v in cr.items() if v > 0)
        html_parts.append(
            f'<div style="display:inline-block;background:{BG_CARD};border:1px solid {BORDER};'
            f'border-radius:8px;padding:8px 14px;margin-right:8px;margin-bottom:8px;">'
            f'<span style="color:{color};font-family:DM Sans,sans-serif;font-size:12px;font-weight:500;">{cat}</span>'
            f'<span style="color:{TEXT_MUTED};font-size:11px;margin-left:6px;">({total})</span>'
            f'<span style="color:{TEXT_SECONDARY};font-size:11px;margin-left:8px;">{parts}</span>'
            f'</div>'
        )

    st.markdown(
        '<div style="display:flex;flex-wrap:wrap;">' + "".join(html_parts) + "</div>",
        unsafe_allow_html=True,
    )

# ── Render main table ──────────────────────────────────────────────────────────

def render_table(results: list[dict], sort_by: str):
    """Dark styled table with all screener columns."""
    if not results:
        return

    # Sort
    if sort_by == "Confidence":
        results = sorted(results, key=lambda r: r["confidence"], reverse=True)
    elif sort_by == "Days in Regime":
        results = sorted(results, key=lambda r: r["days_in_regime"], reverse=True)
    else:
        results = sorted(results, key=lambda r: r["ticker"])

    rows = []
    for r in results:
        regime_color = REGIME_COLORS.get(r["regime"], TEXT_MUTED)
        conf_color = regime_color
        sma_arrow = "↑" if r["sma_position"] == "Above" else "↓"
        sma_color = ACCENT_GREEN if r["sma_position"] == "Above" else ACCENT_RED
        vol_arrow = "↑" if r["volume_trend"] == "Increasing" else "↓"

        # "Clean setup" = low vol + above SMA + increasing volume
        is_clean = r["regime"] == "Low Vol" and r["sma_position"] == "Above" and r["volume_trend"] == "Increasing"

        row = {
            "Ticker": r["ticker"],
            "Price": f"${r['price']:.2f}",
            "Regime": regime_badge(r["regime"], confidence=r["confidence"]),
            "RegimeName": r["regime"],
            "Confidence": r["confidence"],
            "Days in Regime": r["days_in_regime"],
            "SMA": f'<span style="color:{sma_color};font-family:JetBrains Mono,monospace;font-size:12px;">{sma_arrow} {r["sma_position"]}</span>',
            "Volume": f'<span style="color:{ACCENT_GREEN if r["volume_trend"] == "Increasing" else ACCENT_RED};font-family:JetBrains Mono,monospace;font-size:12px;">{vol_arrow} {r["volume_trend"]}</span>',
            "Category": r["category"],
            "is_clean": is_clean,
        }
        rows.append(row)

    # Confidence progress bar HTML per row — use regime name, not category
    conf_html = []
    for row in rows:
        conf_val = row["Confidence"]
        bar_width = f"{conf_val:.0f}%"
        regime_color = REGIME_COLORS.get(row["RegimeName"], TEXT_MUTED)
        conf_html.append(
            f'<div style="display:flex;align-items:center;gap:6px;">'
            f'<div style="width:60px;height:6px;background:{BG_PRIMARY};border-radius:3px;overflow:hidden;">'
            f'<div style="width:{bar_width};height:100%;background:{regime_color};border-radius:3px;"></div>'
            f'</div>'
            f'<span style="font-family:JetBrains Mono,monospace;font-size:11px;color:{TEXT_SECONDARY};">{conf_val:.0f}%</span>'
            f'</div>'
        )

    # Build styled table with custom HTML cells
    styled_table = build_styled_table(rows, conf_html)

    st.markdown(styled_table, unsafe_allow_html=True)

def build_styled_table(rows: list[dict], conf_html: list[str]) -> str:
    """Build dark styled HTML table for the screener."""
    if not rows:
        return ""

    header = '<table style="width:100%;border-collapse:collapse;font-size:12px;">'
    header += '<thead><tr>'

    # Column headers
    headers = [
        ("Ticker", "width:80px"),
        ("Price", "width:90px"),
        ("Regime", "width:140px"),
        ("Confidence", "width:120px"),
        ("Days", "width:60px"),
        ("SMA 50", "width:70px"),
        ("Volume", "width:80px"),
        ("Category", "width:80px"),
    ]

    for col_name, width in headers:
        header += f'<th style="background:{BG_PRIMARY};color:{TEXT_PRIMARY};font-family:DM Sans,sans-serif;font-size:11px;text-transform:uppercase;letter-spacing:1px;padding:10px 12px;text-align:left;border-bottom:1px solid {BORDER};{width}">{col_name}</th>'

    header += '</tr></thead><tbody>'

    for i, row in enumerate(rows):
        is_clean = row["is_clean"]
        row_bg = BG_CARD if i % 2 == 0 else BG_PRIMARY
        border_left = f'border-left:3px solid {ACCENT_CYAN}' if is_clean else f'border-left:3px solid transparent'

        clean_color = ACCENT_CYAN if is_clean else "transparent"
        header += (
            '<tr style="background:{};border-left:3px solid {};transition:background 0.15s;" '
            'onmouseover="this.style.background=\'{}\'" '
            'onmouseout="this.style.background=\'{}\';this.style.borderLeft=\'3px solid {}\'">'
            .format(row_bg, ACCENT_CYAN if is_clean else "transparent",
                    BG_CARD_HOVER, row_bg, clean_color)
        )

        # Clickable ticker name
        ticker_cell = (
            f'<span data-ticker="{row["Ticker"]}" style="font-family:JetBrains Mono,monospace;font-size:12px;color:{TEXT_PRIMARY};font-weight:500;'
            f'cursor:pointer;text-decoration:underline;text-underline-offset:3px;" '
            f'onmouseover="this.style.opacity=\'0.7\'" onmouseout="this.style.opacity=\'1\'">'
            f'{row["Ticker"]}</span>'
        )

        cells = [
            ticker_cell,
            ('<span style="font-family:JetBrains Mono,monospace;font-size:12px;color:{};">{}</span>'.format(TEXT_SECONDARY, row["Price"])),
            (row["Regime"]),
            (conf_html[i]),
            ('<span style="font-family:JetBrains Mono,monospace;font-size:12px;color:{};">{}</span>'.format(TEXT_SECONDARY, row["Days in Regime"])),
            (row["SMA"]),
            (row["Volume"]),
            ('<span style="font-family:JetBrains Mono,monospace;font-size:11px;color:{};">{}</span>'.format(CATEGORY_COLORS.get(row["Category"], TEXT_MUTED), row["Category"])),
        ]

        for cell in cells:
            header += f'<td style="padding:8px 12px;border-bottom:1px solid {BORDER};font-size:12px;">{cell}</td>'

        header += '</tr>'

    header += '</tbody></table>'
    return header

# ── Click-to-chart handler ─────────────────────────────────────────────────────

def render_chart_modal(ticker: str, chart_data: dict, regime_history: dict):
    """Show modal-like chart for a clicked ticker."""
    components.html(
        f'<div id="chart-modal" style="display:flex;position:fixed;top:0;left:0;right:0;bottom:0;background:rgba(0,0,0,0.8);z-index:9999;'
        f'align-items:center;justify-content:center;" onclick="if(event.target===this)this.style.display=\'none\'">',
        height=0,
    )

    fig = build_screener_chart(chart_data, ticker)
    components.html(
        f'<div style="background:{BG_CARD};border:1px solid {BORDER};border-radius:16px;padding:1.5rem;width:90%;max-width:800px;position:relative;">'
        f'<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:1rem;">'
        f'<div style="font-family:DM Sans,sans-serif;font-size:24px;font-weight:700;color:{TEXT_PRIMARY}">{ticker}</div>'
        f'<button onclick="document.getElementById(\'chart-modal\').style.display=\'none\'" style="background:none;border:none;color:{TEXT_SECONDARY};font-size:24px;cursor:pointer;">&times;</button>'
        f'</div>',
        height=0,
    )

    st.plotly_chart(fig, key=f"modal_chart_{ticker}", use_container_width=True, config={"displayModeBar": False})

    # Regime history
    components.html(section_header("Regime History (Last Year)"), height=0)
    hist_fig = build_regime_history_chart(regime_history)
    st.plotly_chart(hist_fig, key=f"modal_hist_{ticker}", use_container_width=True, config={"displayModeBar": False})

    components.html('</div>', height=0)

# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    st.session_state.setdefault("screener_tickers", list(DEFAULT_TICKERS))
    st.session_state.setdefault("_scan_clicked", False)
    st.session_state.setdefault("_scan_results", {})
    st.session_state.setdefault("_scan_date", None)

    # Check cache validity
    cache_valid = (
        st.session_state.get("_scan_date") is not None
        and (pd.Timestamp.now() - st.session_state["_scan_date"]).hours < 1  # 1-hour cache
    )

    # Render sidebar
    start_date, end_date, sort_by = render_sidebar()

    # Determine if we need to scan
    needs_scan = (
        not cache_valid
        or st.session_state.get("_scan_clicked", False)
        or st.session_state.get("_scan_results") == {}
    )

    if needs_scan and not st.session_state.screener_tickers:
        components.html(
            '<div style="display:flex;align-items:center;justify-content:center;'
            'height:60vh;font-family:DM Sans,sans-serif;color:'
            + TEXT_MUTED + ';font-size:16px;">'
            "No tickers selected. Add tickers in the sidebar."
            "</div>",
            height=200,
        )
        return

    # Scan
    if needs_scan:
        st.session_state._scan_clicked = False
        st.session_state._scan_results = {}
        st.session_state._scan_date = pd.Timestamp.now()

        progress_bar = st.progress(0, text="Scanning market...")
        total = len(st.session_state.screener_tickers)

        for idx, ticker in enumerate(st.session_state.screener_tickers):
            progress_bar.progress((idx + 1) / total, text=f"Scanning {ticker}...")
            result = scan_ticker(ticker, start_date, end_date)
            if result is not None:
                st.session_state._scan_results[ticker] = result

        progress_bar.empty()

    # Apply filters
    results = list(st.session_state._scan_results.values())

    # Regime filter
    regime_filter = st.session_state.get("regime_filter", ["Low Vol", "Medium Vol", "High Vol", "Uncertain"])
    results = [r for r in results if r["regime"] in regime_filter]

    # Confidence filter
    min_conf = st.session_state.get("min_confidence", 0.5)
    results = [r for r in results if r["confidence"] >= min_conf * 100]

    # SMA filter
    sma_filter = st.session_state.get("sma_filter", "Both")
    if sma_filter == "Above":
        results = [r for r in results if r["sma_position"] == "Above"]
    elif sma_filter == "Below":
        results = [r for r in results if r["sma_position"] == "Below"]

    # Volume filter
    vol_filter = st.session_state.get("vol_filter", "Both")
    if vol_filter == "Increasing":
        results = [r for r in results if r["volume_trend"] == "Increasing"]
    elif vol_filter == "Decreasing":
        results = [r for r in results if r["volume_trend"] == "Decreasing"]

    # Sort
    if sort_by == "Confidence":
        results = sorted(results, key=lambda r: r["confidence"], reverse=True)
    elif sort_by == "Days in Regime":
        results = sorted(results, key=lambda r: r["days_in_regime"], reverse=True)
    else:
        results = sorted(results, key=lambda r: r["ticker"])

    # ── Title ────────────────────────────────────────────────────────────────
    st.markdown(
        f'<div style="font-family:DM Sans,sans-serif;font-size:32px;font-weight:700;color:{TEXT_PRIMARY};margin-bottom:0.25rem;">Regime Screener</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div style="font-family:DM Sans,sans-serif;font-size:13px;color:{TEXT_SECONDARY};margin-bottom:1.5rem;">'
        f'{len(results)} tickers · Scanned {len(st.session_state._scan_results)} of {len(st.session_state.screener_tickers)} tickers</div>',
        unsafe_allow_html=True,
    )

    # ── Summary bar ──────────────────────────────────────────────────────────
    render_summary_cards(results)

    # ── Category analysis ────────────────────────────────────────────────────
    render_category_analysis(results)

    # ── Main table ───────────────────────────────────────────────────────────
    if results:
        render_table(results, sort_by)

    # ── Click-to-chart modal ─────────────────────────────────────────────────
    clicked_ticker = st.session_state.get("_clicked_ticker", None)
    if clicked_ticker and clicked_ticker in st.session_state._scan_results:
        r = st.session_state._scan_results[clicked_ticker]
        render_chart_modal(clicked_ticker, r["chart_data"], r["regime_history"])

    # Add JS for click-to-chart — set session state on ticker click
    components.html(
        f'<script>'
        f'document.querySelectorAll("[data-ticker]").forEach(function(el) {{'
        f'    el.style.cursor = "pointer";'
        f'    el.addEventListener("click", function() {{'
        f'        var ticker = this.getAttribute("data-ticker");'
        f'        st.sessionState._clicked_ticker = ticker;'
        f'        setTimeout(function() {{ window.location.reload(); }}, 100);'
        f'    }});'
        f'}});'
        f'</script>',
        height=0,
    )

    # ── Regime distribution ──────────────────────────────────────────────────
    components.html(section_header("Regime Distribution"), height=0)
    dist_fig = build_distribution_bar(results)
    if dist_fig is not None:
        st.plotly_chart(dist_fig, key="regime_distribution", use_container_width=True, config={"displayModeBar": False})
    else:
        st.markdown(
            f'<div style="font-family:DM Sans,sans-serif;font-size:13px;color:{TEXT_MUTED};text-align:center;padding:2rem;">'
            "No tickers match current filters.</div>",
            unsafe_allow_html=True,
        )

# ── Unused helpers (removed) ───────────────────────────────────────────────────
# The original build_styled_table was replaced with inline HTML table rendering
# in the render_table function for better performance with large datasets.
# Progress bar now shows per-ticker scanning status.

if __name__ == "__main__":
    main()
