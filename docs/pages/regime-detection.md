# Regime Detection

HMM-based market regime classification with online forward filtering. No look-ahead bias.

## Overview

The regime detection page downloads price data, engineers features, trains a Hidden Markov Model, and produces regime-colored price charts with confidence timelines.

## Features

1. **Data download** via `yfinance`
2. **Feature engineering**: log returns, 20-day realized vol, volume ratio, H-L range
3. **Model selection**: trains `hmmlearn.GaussianHMM` (diagonal covariance) with 3–7 regimes, selects via BIC
4. **Forward-filtered labels**: uses forward algorithm (alpha) to compute posterior marginals `gamma[t] = P(q_t | o_1..t)` — NO `model.predict()` used
5. **Stability filter**: regimes must persist 3 consecutive bars; >4 transitions in 20-bar window flags as "Uncertain"
6. **Regime labeling**: sorts by mean volatility — Low Vol / Medium Vol / High Vol

## No Look-Ahead Guarantee

The page verifies no look-ahead bias on every run:

```python
def verify_no_lookahead() -> bool:
    """Confirm no look-ahead bias — Viterbi decoding must not be used."""
    # Scans source code for model.predict() or model.predict_proba() calls
    # Returns True if clean, False if found
```

If `model.predict()` is found in the code, the page displays an error and aborts.

## Forward Filtering

Instead of calling `model.predict()` (which uses Viterbi and requires all observations), the page computes forward-filtered labels using the forward algorithm:

```
alpha[t, j] = sum_i(alpha[t-1, i] * trans[i,j]) * b_j(o_t)
gamma[t, i] = alpha[t, i] / sum_j(alpha[t, j])   # posterior marginals
label[t] = argmax_i(gamma[t, i])
confidence[t] = max(gamma[t, i])
```

This means label at time T uses only data 1..T — zero look-ahead bias.

## Usage

1. Enter ticker (default: SPY)
2. Set date range (default: last 3 years)
3. Optionally override #regimes (0 = auto-select via BIC)
4. Click "Run Analysis"

## Visualizations

- **Price chart** with regime-colored background bands (opacity 0.13)
- **Regime statistics cards** — mean return, vol, vol ratio, time in regime
- **Confidence timeline** — area chart colored by regime, showing posterior probability

## URL Routing

Supports shareable links:

- `?ticker=QQQ&start_date=2022-01-01&end_date=2024-01-01` — auto-runs with QQQ data
- `?override_n=5` — forces 5 regimes instead of auto-selecting
