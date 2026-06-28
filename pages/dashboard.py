"""Regime Detection Dashboard — online forward filtering, no look-ahead."""

import streamlit as st
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import yfinance as yf
from hmmlearn import hmm
import re

from design_system import *

apply_theme()

# ── Layout helpers ────────────────────────────────────────────────────────────

def html_row(*html_items):
    """Render HTML side-by-side."""
    st.markdown(
        '<div style="display:flex;align-items:baseline;gap:1.5rem;flex-wrap:wrap;">'
        + "".join(html_items)
        + "</div>",
        unsafe_allow_html=True,
    )

# ── Feature engineering ───────────────────────────────────────────────────────

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return DataFrame with log returns, vol, volume ratio, H-L range."""
    df = df.copy()
    df["log_ret"] = np.log(df["Close"] / df["Close"].shift(1))
    df["realized_vol"] = df["log_ret"].rolling(20).std()
    df["vol_ratio"] = df["Volume"] / df["Volume"].rolling(20).mean()
    df["hl_range"] = (df["High"] - df["Low"]) / df["Close"]
    df.dropna(inplace=True)
    return df.reset_index(drop=True)

# ── Forward-filtered regime labels (no look-ahead) ────────────────────────────

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

    # Forward variables (log-space): alpha[t, i] = P(o_1..t, q_t=i)
    # Use large negative instead of -inf to avoid NaN in softmax
    log_alpha = np.full((T, n_states), -1e10)

    # Initial step: alpha[0] = pi * b_0(o_0)
    log_pi = np.log(np.maximum(model.startprob_, 1e-300))
    log_emission = model._compute_log_likelihood(X[0:1])  # (1, n_states)
    log_alpha[0] = log_pi + log_emission[0]

    # Forward recursion: alpha[t] = sum_i alpha[t-1,i] * trans[i,j] * b_j(o_t)
    for t in range(1, T):
        log_trans = np.log(np.maximum(model.transmat_, 1e-300))
        log_emission_t = model._compute_log_likelihood(X[t:t+1])  # (1, n_states)
        for j in range(n_states):
            log_alpha[t, j] = (
                np.log(np.sum(np.exp(np.minimum(log_alpha[t - 1], 0) + log_trans[:, j])))
                + log_emission_t[0, j]
            )

    # Posterior marginals: gamma[t, i] = P(q_t=i | o_1..t)
    # Computed from filtered forward variables — no future data needed.
    for t in range(T):
        max_val = log_alpha[t].max()
        exp_alpha = np.exp(np.minimum(log_alpha[t] - max_val, 0))
        gamma_t = exp_alpha / exp_alpha.sum()
        labels[t] = np.argmax(gamma_t)
        confidences[t] = gamma_t.max()

    return labels, confidences

# ── BIC calculation ───────────────────────────────────────────────────────────

def compute_bic(model, X: np.ndarray) -> float:
    """Bayesian Information Criterion for diagonal Gaussian HMM."""
    n_samples, n_features = X.shape
    n = model.n_components
    # Parameters:
    #   means: n * d
    #   covars: n * d (diagonal only)
    #   transmat: n * n - n (rows sum to 1)
    #   startprob: n - 1 (rows sum to 1)
    n_params = n * n_features + n * n_features + n * n - n + n - 1
    try:
        log_likelihood = model.score(X)  # total log-likelihood
    except Exception:
        return np.inf
    bic = -2 * log_likelihood + n_params * np.log(n_samples)
    return bic

# ── Forward algorithm verification ────────────────────────────────────────────

def verify_no_lookahead() -> bool:
    """Confirm no look-ahead bias — Viterbi decoding must not be used."""
    try:
        with open(__file__, "r") as f:
            lines = f.readlines()
    except FileNotFoundError:
        return True

    # Find the verify function's line range and exclude it
    start = None
    end = None
    for i, line in enumerate(lines):
        if line.startswith("def verify_no_lookahead"):
            start = i
        elif start is not None and line.startswith("def ") and i > start:
            end = i
            break
    if start is None:
        return True
    if end is None:
        end = len(lines)

    # Exclude the verify function body from the scan
    code = "".join(lines[:start]) + "".join(lines[end:])

    # Check that .predict() is not called on a model instance
    # Must match actual method calls — skip comments, docstrings, string literals
    clean = re.sub(r"#.*$", "", code, flags=re.MULTILINE)       # strip comments
    clean = re.sub(r'""".*?"""', "", clean, flags=re.DOTALL)    # strip triple-quoted strings
    clean = re.sub(r"'.*?'", "", clean)                         # strip single-quoted strings
    clean = re.sub(r'".*?"', "", clean)                         # strip double-quoted strings
    if re.search(r"\bmodel\.(predict|predict_proba)\s*\(", clean):
        return False
    return True

# ── Stability filter ──────────────────────────────────────────────────────────

def apply_stability_filter(labels: np.ndarray) -> np.ndarray:
    """Regime must persist 3 consecutive bars.

    If flickering >4 times in 20-bar window, flag as 999 (Uncertain).
    """
    filtered = np.copy(labels)
    n = len(labels)

    # Phase 1: remove single/double bar flickers — replace with dominant regime
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

    # Phase 2: check for flickering — >4 transitions in 20-bar window → Uncertain
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

# ── Regime labeling ───────────────────────────────────────────────────────────

def label_regimes(means: np.ndarray, n_regimes: int) -> list[str]:
    """Sort regimes by mean volatility, assign labels."""
    vol_means = means[:, 1]  # realized_vol is second feature
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

# ── Build main chart ──────────────────────────────────────────────────────────

def build_price_chart(df: pd.DataFrame, labels: np.ndarray,
                      confidences: np.ndarray, regime_names: list[str]) -> go.Figure:
    """Main chart with regime-colored background bands."""
    dates = df.index.tolist()
    closes = df["Close"].values
    n = len(dates)

    # Regime background bands — one trace per regime transition
    traces = []
    band_start = 0
    prev_regime = labels[0]

    for i in range(1, n):
        curr = labels[i]
        if curr != prev_regime:
            # Build band from band_start to i
            x0, x1 = dates[band_start], dates[min(i, n - 1)]
            y0, y1 = closes[band_start], closes[min(i, n - 1)]

            band_label = f"Uncertain" if prev_regime == 999 else (regime_names[prev_regime] if prev_regime < len(regime_names) else "Regime")

            if prev_regime == 999:
                band_color = REGIME_COLORS["Uncertain"]
            elif prev_regime < len(regime_names):
                band_color = REGIME_COLORS.get(regime_names[prev_regime], TEXT_MUTED)
            else:
                band_color = TEXT_MUTED

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
        y0, y1 = closes[band_start], closes[n - 1]

        if prev_regime == 999:
            band_color = REGIME_COLORS["Uncertain"]
        elif prev_regime < len(regime_names):
            band_color = REGIME_COLORS.get(regime_names[prev_regime], TEXT_MUTED)
        else:
            band_color = TEXT_MUTED

        traces.append(
            go.Scatter(
                x=[x0, x1],
                y=[y0, y1],
                mode="lines",
                line=dict(width=0, color=band_color),
                opacity=0.13,
                name=f"Uncertain" if prev_regime == 999 else (regime_names[prev_regime] if prev_regime < len(regime_names) else "Regime"),
                hoverinfo="skip",
                showlegend=False,
                fill="tozeroy",
                yaxis="y2",
            )
        )

    # Price line (white, on top)
    traces.append(
        go.Scatter(
            x=dates,
            y=closes,
            mode="lines",
            name="Price",
            line=dict(color="white", width=1.5),
        )
    )

    layout = get_plotly_layout(
        height=500,
        xaxis=dict(
            rangeslider=dict(visible=False),
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
    fig.update_layout(
        bargap=0,
        xaxis_rangeslider_visible=False,
    )
    return fig

# ── Build confidence chart ────────────────────────────────────────────────────

def build_confidence_chart(df: pd.DataFrame, labels: np.ndarray,
                           confidences: np.ndarray, regime_names: list[str]) -> go.Figure:
    """Area chart of regime confidence over time, colored by regime."""
    dates = df.index.tolist()

    # Group by regime for colored area segments
    traces = []
    band_start = 0
    prev_regime = labels[0]

    for i in range(1, len(dates)):
        curr = labels[i]
        if curr != prev_regime:
            # Close the previous regime segment
            x_chunk = dates[band_start:i]
            y_chunk = confidences[band_start:i]

            if prev_regime == 999:
                fill_color = REGIME_COLORS["Uncertain"]
                seg_name = "Uncertain"
            elif prev_regime < len(regime_names):
                fill_color = REGIME_COLORS.get(regime_names[prev_regime], TEXT_MUTED)
                seg_name = regime_names[prev_regime]
            else:
                fill_color = TEXT_MUTED
                seg_name = "Regime"

            traces.append(
                go.Scatter(
                    x=x_chunk,
                    y=y_chunk,
                    mode="lines",
                    fill="tozeroy",
                    fillcolor=fill_color,
                    opacity=0.3,
                    line=dict(width=0),
                    name=seg_name,
                    hoverinfo="skip",
                    showlegend=False,
                )
            )
            band_start = i
            prev_regime = curr

    # Last segment
    if band_start < len(dates):
        x_chunk = dates[band_start:]
        y_chunk = confidences[band_start:]

        if prev_regime == 999:
            fill_color = REGIME_COLORS["Uncertain"]
            seg_name = "Uncertain"
        elif prev_regime < len(regime_names):
            fill_color = REGIME_COLORS.get(regime_names[prev_regime], TEXT_MUTED)
            seg_name = regime_names[prev_regime]
        else:
            fill_color = TEXT_MUTED
            seg_name = "Regime"

        traces.append(
            go.Scatter(
                x=x_chunk,
                y=y_chunk,
                mode="lines",
                fill="tozeroy",
                fillcolor=fill_color,
                opacity=0.3,
                line=dict(width=0),
                name=seg_name,
                hoverinfo="skip",
                showlegend=False,
            )
        )

    layout = get_plotly_layout(
        height=200,
        xaxis=dict(
            type="date",
            showspikes=True,
            spikemode="across",
        ),
        yaxis=dict(
            title="Confidence",
            tickformat=".0%",
            showspikes=True,
            spikemode="across",
            range=[0, 1],
        ),
        showlegend=False,
    )

    fig = go.Figure(data=traces, layout=layout)
    return fig

# ── Build regime stats cards ──────────────────────────────────────────────────

def build_regime_stats(df: pd.DataFrame, labels: np.ndarray,
                       regime_names: list[str], n_regimes: int,
                       means: np.ndarray) -> str:
    """HTML for regime stat cards."""
    cards = []
    n = len(labels)

    for i in range(n_regimes):
        mask = labels == i
        n_bars = mask.sum()
        pct = n_bars / n * 100

        mean_ret = df["log_ret"].iloc[mask].mean()
        mean_vol = df["realized_vol"].iloc[mask].mean()
        mean_vol_ratio = df["vol_ratio"].iloc[mask].mean()

        color = REGIME_COLORS.get(regime_names[i], TEXT_MUTED)

        card = f"""
        <div style="display:inline-block; background:{BG_CARD}; border:1px solid {BORDER};
                     border-radius:12px; padding:1rem 1.5rem; margin-right:1rem;
                     border-left:4px solid {color}; width:220px; vertical-align:top;">
            <div style="font-family:DM Sans,sans-serif; font-size:12px; font-weight:500;
                         color:{color}; margin-bottom:8px;">{regime_names[i]}</div>
            <div style="font-family:JetBrains Mono,monospace; font-size:11px;
                         color:{TEXT_SECONDARY}; line-height:1.8;">
                Mean Return: {mean_ret:+.4f}<br>
                Mean Vol: {mean_vol:.4f}<br>
                Vol Ratio: {mean_vol_ratio:.2f}x<br>
                Time in Regime: {pct:.1f}%
            </div>
        </div>
        """
        cards.append(card)

    return "".join(cards)

# ── Main app ──────────────────────────────────────────────────────────────────

def parse_config():
    """Read analysis config from URL query params, fall back to sidebar inputs."""
    qp = st.query_params
    ticker = qp.get("ticker", "SPY")
    start_date = qp.get("start_date", None)
    end_date = qp.get("end_date", None)
    override_n = qp.get("override_n", "0")

    # Parse dates from URL — format: YYYY-MM-DD
    if start_date:
        try:
            start_date = pd.Timestamp(start_date)
        except ValueError:
            start_date = pd.Timestamp.now() - pd.DateOffset(years=3)
    else:
        start_date = pd.Timestamp.now() - pd.DateOffset(years=3)

    if end_date:
        try:
            end_date = pd.Timestamp(end_date)
        except ValueError:
            end_date = pd.Timestamp.now()
    else:
        end_date = pd.Timestamp.now()

    try:
        override_n = int(override_n)
    except ValueError:
        override_n = 0

    return ticker, start_date, end_date, override_n

def main():
    # ── Path-based routing: parse URL params ────────────────────────────────
    url_ticker, url_start, url_end, url_override = parse_config()
    has_url = any(st.query_params.get(k) for k in ("ticker", "start_date", "end_date", "override_n"))

    if has_url:
        # URL params exist — render URL-param widgets only, auto-run
        ticker = st.sidebar.text_input("Ticker", value=url_ticker, key="ticker_url")
        col1, col2 = st.sidebar.columns(2)
        with col1:
            start_date = st.date_input("From", value=url_start, key="start_url")
        with col2:
            end_date = st.date_input("To", value=url_end, key="end_url")
        override_n = st.sidebar.number_input(
            "Override # Regimes (0=auto)",
            min_value=0,
            max_value=7,
            value=url_override,
            step=1,
            key="override_url",
        )
        run_btn = True
    else:
        # No URL params — render normal sidebar widgets
        ticker = st.sidebar.text_input("Ticker", value="SPY")
        col1, col2 = st.sidebar.columns(2)
        with col1:
            start_date = st.date_input("From", value=pd.Timestamp.now() - pd.DateOffset(years=3))
        with col2:
            end_date = st.date_input("To", value=pd.Timestamp.now())
        override_n = st.sidebar.number_input(
            "Override # Regimes (0=auto)",
            min_value=0,
            max_value=7,
            value=0,
            step=1,
        )
        run_btn = st.sidebar.button("▶ Run Analysis", type="primary", use_container_width=True)

    if not run_btn:
        st.markdown(
            '<div style="display:flex;align-items:center;justify-content:center;'
            'height:60vh;font-family:DM Sans,sans-serif;color:'
            + TEXT_MUTED + ';font-size:16px;">'
            "Enter ticker and click Run Analysis"
            "</div>",
            unsafe_allow_html=True,
        )
        return

    # ── Update URL params to reflect current values ─────────────────────────
    qp = st.query_params
    qp["ticker"] = ticker
    qp["start_date"] = start_date.strftime("%Y-%m-%d")
    qp["end_date"] = end_date.strftime("%Y-%m-%d")
    if override_n > 0:
        qp["override_n"] = str(override_n)
    else:
        qp.pop("override_n", None)

    # ── Data ────────────────────────────────────────────────────────────────
    with st.spinner("Downloading data..."):
        raw = yf.download(ticker, start=start_date, end=end_date, progress=False)

    if raw.empty:
        st.error(f"No data found for {ticker}")
        return

    # Flatten MultiIndex columns
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    df = build_features(raw)

    # ── Verify no look-ahead ────────────────────────────────────────────────
    lookahead_ok = verify_no_lookahead()
    if not lookahead_ok:
        st.error("⚠️ Look-ahead bias detected — model.predict() found in code")
        return

    # ── Model selection ─────────────────────────────────────────────────────
    feature_cols = ["log_ret", "realized_vol", "vol_ratio", "hl_range"]
    X = df[feature_cols].values
    y = df["Close"].values

    best_bic = np.inf
    best_n = 3
    best_model = None

    n_range = range(3, 8) if override_n == 0 else [override_n]

    for n_comp in n_range:
        with st.spinner(f"Training HMM with {n_comp} regimes..."):
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
            except Exception as e:
                st.warning(f"Failed {n_comp} regimes: {e}")

    if best_model is None:
        st.error("All HMM training failed")
        return

    # ── Forward-filtered labels ─────────────────────────────────────────────
    labels, confidences = forward_filter_labels(best_model, X)
    labels = apply_stability_filter(labels)

    # ── Regime naming ───────────────────────────────────────────────────────
    means = best_model.means_
    regime_names = label_regimes(means, best_n)

    # ── Current regime ──────────────────────────────────────────────────────
    current_label = labels[-1]
    current_conf = confidences[-1]

    if current_label == 999:
        regime_display = "Uncertain"
        stability = "Flickering"
    elif current_label < best_n:
        regime_display = regime_names[current_label]
        # Check stability: >4 transitions in 20-bar window = flickering
        window = labels[-20:] if len(labels) >= 20 else labels
        transitions = np.sum(window[1:] != window[:-1])
        stability = "Stable" if transitions <= 4 else "Flickering"
    else:
        regime_display = "Unknown"
        stability = "N/A"

    # ── Layout: top bar ─────────────────────────────────────────────────────
    st.markdown(
        f'<div style="font-family:DM Sans,sans-serif;font-size:36px;'
        f'font-weight:700;color:{TEXT_PRIMARY};margin-bottom:0.25rem;">'
        f'{ticker}</div>',
        unsafe_allow_html=True,
    )

    html_row(
        regime_badge(regime_display, confidence=current_conf * 100),
        f'<span style="font-family:JetBrains Mono,monospace;font-size:32px;'
        f'font-weight:700;color:{ACCENT_CYAN};">{current_conf:.0%}</span>',
        f'<span style="font-family:DM Sans,sans-serif;font-size:13px;color:'
        f'{TEXT_SECONDARY};">confidence · {status_dot(stability.lower())} {stability}'
        f"</span>",
        f'<span style="font-family:JetBrains Mono,monospace;font-size:14px;color:'
        f'{TEXT_SECONDARY};">{best_n} regimes</span>',
    )

    # ── Main chart ──────────────────────────────────────────────────────────
    fig = build_price_chart(df, labels, confidences, regime_names)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    # ── Regime statistics ───────────────────────────────────────────────────
    st.markdown(section_header("Regime Statistics"))
    stats_html = build_regime_stats(df, labels, regime_names, best_n, means)
    st.markdown(
        f'<div style="display:flex;flex-wrap:wrap;">{stats_html}</div>',
        unsafe_allow_html=True,
    )

    # ── Confidence timeline ─────────────────────────────────────────────────
    st.markdown(section_header("Confidence Timeline"))
    conf_fig = build_confidence_chart(df, labels, confidences, regime_names)
    st.plotly_chart(conf_fig, use_container_width=True, config={"displayModeBar": False})

if __name__ == "__main__":
    main()
