# Monte Carlo Simulation

Equity curve probability cloud from trade-return shuffling.

## Overview

Shuffles historical trade returns, adds noise, and runs many simulations to estimate probability distributions of outcomes. Tests for overfitting by comparing original backtest result against shuffled permutations.

## Data Input

- **Upload CSV** with columns: `date`, `trade_return`
- Or use built-in **sample data** — 200 trades with slight positive edge (+0.15% per trade) and loss clustering

## Simulation Engine

```
for each simulation i:
    shuffled = permute(trade_returns) + noise(±0.3%)
    equity[t+1] = equity[t] * (1 + shuffled[t])
```

Default: 1,000 simulations, starting capital $100,000.

## Statistics

### Probability Metrics

- **Probability of loss** — % of simulations where final value < starting capital
- **Probability of 20%+ drawdown** — % of simulations with max drawdown ≥ 20%
- **Probability of 30%+ drawdown** — % of simulations with max drawdown ≥ 30%

### Percentile Distribution

- 5th, 25th, 50th, 75th, 95th percentiles of max drawdown distribution

### Overfitting Check

Compares original backtest final value against shuffled outcomes:

```
original_pctile = mean(final_values < original_final_value)
```

- **>90th percentile** — HIGH risk: original result looks too good, possible overfitting
- **>80th percentile** — MEDIUM risk
- **≤80th percentile** — LOW risk

## Visualizations

### Fan Chart

Cinematic probability cloud showing:

- **Individual curves** — low opacity lines (0.015) in cyan
- **5th–95th band** — wide probability range
- **25th–75th band** — interquartile range
- **Median line** — 50th percentile
- **Original backtest** — solid white line

### Histograms

- **Final portfolio values** — distribution of ending values with original backtest line
- **Max drawdown distribution** — distribution of worst drawdowns per simulation

### Drawdown Distribution Detail

5th, 25th, 50th, 75th, 95th percentile values displayed as cards.

### Interpretation

Auto-generated text summarizing probability of loss, drawdowns, median outcome, and original backtest percentile.

## Usage

1. Upload CSV or use sample data
2. Set starting capital and number of simulations
3. Click "Run Simulation"
