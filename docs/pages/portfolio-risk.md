# Portfolio Risk

Regime overlay, correlation analysis, and stress testing for a multi-asset portfolio.

## Overview

Dashboard that overlays HMM market regimes onto portfolio positions, computes inter-asset correlations, and runs historical stress tests. Supports manual position entry, CSV upload, and optional Alpaca API integration for live positions.

## Architecture

```
┌─────────────────────────────────────────┐
│  Top Bar: total value, P&L, regime health, market status  │
├──────────────────┬──────────────────────┤
│  Positions       │  Correlation Heatmap │
│  (with regime    │  Risk                │
│   badges)        │                      │
│                  ├──────────────────────┤
│                  │  Stress Tests        │
│                  │  (2008, 2020, 2022)  │
│                  ├──────────────────────┤
│                  │  Watchlist           │
│                  │  (regime + price)    │
└──────────────────┴──────────────────────┘
```

## Inputs

### Manual Position Entry

Add positions in the sidebar with ticker, shares, and entry price. Edit or remove existing positions via expandable sections.

### CSV Upload

Upload a CSV with columns: `ticker`, `shares`, `entry`. Parses and replaces current positions.

### Alpaca Integration (optional)

Enter API key and secret key in the sidebar. Click "Connect" to fetch live positions from Alpaca.

### Watchlist

Add tickers to watch in the sidebar. Watchlist items display current price and HMM regime status.

## Regime Detection

Each position and watchlist ticker runs the same HMM regime detection as the Regime Detection page:

1. Downloads 365 days of data via `yfinance`
2. Engineers features: log returns, 20-day realized vol, volume ratio, H-L range
3. Trains `hmmlearn.GaussianHMM` with 3 regimes (diagonal covariance)
4. Forward-filters labels — no look-ahead bias
5. Applies stability filter (3-bar persistence, 20-bar flicker threshold)

Regime health metric counts positions in favorable regimes ("Low Vol", "Bull").

## Correlation Risk

60-day rolling correlation matrix between all portfolio positions, rendered as a dark-themed heatmap:

- **Color scale**: deep navy (low) → cyan (moderate) → white (high)
- **Red warning borders**: cells above 0.85 correlation flagged as over-concentration risk

## Stress Tests

Applies historical drawdowns to current positions:

| Scenario | SPY | QQQ | AAPL | GLD | TLT |
|----------|-----|-----|------|-----|-----|
| 2008 Crisis | -56% | -54% | -61% | +21% | +33% |
| 2020 Covid | -34% | -28% | -31% | -3% | +21% |
| 2022 Rate Hikes | -25% | -33% | -30% | -4% | -31% |

Severity bars colored green (<10%), amber (<20%), red (≥20% loss).

## Visualizations

### Position Cards

Dark cards showing ticker, P&L ($ and %), entry price, current price, shares, and regime badge with confidence. P&L bars centered at 0, extending left (loss) or right (gain).

### Correlation Heatmap

Square Plotly heatmap with cyan scale and red borders on high-correlation pairs.

### Stress Test Bars

Horizontal bars showing drawdown severity per scenario, color-coded by magnitude.

### Watchlist Cards

Compact cards with ticker, price, regime badge, confidence bar, days-in-regime bar, and stability indicator.

## Usage

1. Enter positions (manual, CSV, or Alpaca)
2. Add tickers to watchlist
3. Dashboard auto-runs — downloads prices, runs regime detection, computes correlations
4. Click "Refresh" to clear cache and re-fetch all data
