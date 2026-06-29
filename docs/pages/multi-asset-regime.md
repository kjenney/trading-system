# Multi-Asset Regime Backtest

Walk-forward HMM-based regime allocation across multiple asset classes with stress testing.

## Overview

Backtests a regime-aware allocation strategy across multiple assets, comparing it against buy-and-hold and 200-day SMA trend-following benchmarks. Uses walk-forward validation: train HMM on a fixed window, test on a rolling period, advance forward.

## Strategy

Regime-based allocation scales exposure based on market volatility regime:

| Regime | Allocation |
|--------|-----------|
| Low Vol | 95% |
| Medium Vol | 77.5% (linear interpolation) |
| High Vol | 60% |
| Uncertain | 50% |

## Walk-Forward Backtest

```
for each window:
    train_days = train_years * 252
    test_days = test_months * 30
    train data → train HMM
    test data → forward-filter labels → compute allocation
    strategy return = allocation * daily_return
    advance window by train_days
```

Default: 1-year train window, 6-month test window.

## Benchmarks

Three equity curves compared per asset:

1. **Regime Strategy** — allocation-weighted returns based on HMM regime
2. **Buy & Hold** — raw buy-and-hold returns
3. **200-Day SMA** — trend-following: only long when price > 200-day SMA

## Metrics

- **Annualized return** — total return annualized via `(equity_end / equity_start)^(252/days) - 1`
- **Sharpe ratio** — annualized daily Sharpe
- **Maximum drawdown** — peak-to-trough maximum
- **Sharpe improvement** — strategy Sharpe minus buy-and-hold Sharpe

## Stress Tests

Maximum drawdown during three crisis windows:

| Crisis | Period |
|--------|--------|
| 2008 Crisis | 2008-09-01 to 2009-03-31 |
| 2020 Covid | 2020-02-01 to 2020-04-30 |
| 2022 Rate Hikes | 2022-01-01 to 2022-10-31 |

## Visualizations

### Equity Curves

Per-asset line chart: regime strategy (colored by asset), buy-and-hold (muted dashed), 200-day SMA (faint dotted).

### Regime Timeline

All assets stacked vertically as horizontal colored bars. Each bar divided into regime segments (Low Vol / Medium Vol / High Vol). Hover shows regime name and date range.

### Asset Comparison Table

Dark-styled table with all assets side by side. Sharpe improvement column colored green/red. Best-performing asset highlighted with cyan left border.

### Stress Test Bar Charts

Grouped bar charts per crisis: strategy drawdown vs buy-and-hold drawdown per asset.

## Usage

1. Add assets via sidebar (default: SPY, BTC-USD, GLD, TLT)
2. Set date range (default: last 5 years)
3. Set walk-forward parameters (train period in years, test period in months)
4. Click "Run Backtest"

## URL Routing

Supports shareable links:

- `?selected=SPY&start_date=2019-01-01&end_date=2024-01-01&train_years=2&test_months=6` — auto-runs with SPY, custom dates and windows
- Clicking an asset tab updates the `selected` param for sharing
