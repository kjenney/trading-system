# Regime Screener

Multi-ticker HMM regime classification with filtering, sorting, and click-to-chart analysis.

## Overview

The regime screener downloads price data for a configurable universe of tickers, trains a Hidden Markov Model on each one, and displays the results in a sortable table with filters. Click any ticker to view its regime-colored price chart and regime history.

## How It Works

### Feature Engineering

For each ticker, the page computes four features:

```python
df["log_ret"] = np.log(df["Close"] / df["Close"].shift(1))  # log returns
df["realized_vol"] = df["log_ret"].rolling(20).std()         # 20-day realized vol
df["vol_ratio"] = df["Volume"] / df["Volume"].rolling(20).mean()  # volume ratio
df["hl_range"] = (df["High"] - df["Low"]) / df["Close"]     # H-L range
```

### Model Selection

Trains `hmmlearn.GaussianHMM` with diagonal covariance for 3, 4, and 5 regimes. Selects the best model by BIC:

```python
bic = -2 * log_likelihood + n_params * np.log(n_samples)
```

### Forward-Filtered Regime Labels

Uses the forward algorithm to compute posterior marginals `gamma[t] = P(q_t | o_1..t)` — no look-ahead bias. Never calls `model.predict()`.

```
alpha[t, j] = sum_i(alpha[t-1, i] * trans[i,j]) * b_j(o_t)
gamma[t, i] = alpha[t, i] / sum_j(alpha[t, j])
label[t] = argmax_i(gamma[t, i])
confidence[t] = max(gamma[t, i])
```

### Stability Filter

Regimes must persist 3 consecutive bars. If more than 4 transitions occur in a 20-bar window, the window is flagged as "Uncertain" (label 999).

### Regime Labeling

Sorts regimes by mean volatility and assigns names: Low Vol, Medium Vol, High Vol.

## Ticker Universe

Default 23 tickers across three categories:

| Category | Tickers |
|----------|---------|
| Large Cap | AAPL, MSFT, GOOGL, AMZN, NVDA, META, TSLA, JPM, V, UNH |
| ETF | SPY, QQQ, IWM, DIA, XLF, XLE, XLK, GLD, TLT, HYG |
| Crypto | BTC-USD, ETH-USD, SOL-USD |

Custom tickers can be added via the sidebar. Categories are auto-detected by ticker name (crypto ends in `-USD`, ETFs are 1-5 letters).

## Sidebar Controls

- **Remove tickers** — click the minus button next to any default ticker
- **Add ticker** — enter a custom ticker and click Add
- **Regime filter** — filter by regime (Low Vol, Medium Vol, High Vol, Uncertain)
- **Min Confidence** — minimum HMM confidence threshold (0.0–1.0, default 0.5)
- **50 SMA** — filter by price position relative to 50-day SMA (Above, Below, Both)
- **Volume Trend** — filter by volume trend (Increasing, Decreasing, Both)
- **Sort By** — Confidence, Days in Regime, or Ticker
- **Date Range** — From / To date pickers (default: 2 years)
- **Scan Market** — Run the scan

## Table Columns

| Column | Description |
|--------|-------------|
| Ticker | Clickable — opens chart modal |
| Price | Current price |
| Regime | Colored badge with confidence |
| Confidence | HMM confidence percentage with progress bar |
| Days | Consecutive days in current regime |
| SMA 50 | Arrow showing Above/Below position |
| Volume | Arrow showing Increasing/Decreasing trend |
| Category | Large Cap, ETF, or Crypto |

"Clean setup" rows have a cyan left border — regime is Low Vol, price above SMA 50, and volume increasing.

## Visualizations

- **Summary cards** — count in each regime, average confidence, strongest regime
- **Category analysis** — which category has the most favorable regimes
- **Regime distribution** — stacked bar showing % of tickers in each regime
- **Click-to-chart modal** — price chart with regime-colored background bands and regime history bar

## URL Routing

No URL routing support — the screener uses session state for all configuration.
