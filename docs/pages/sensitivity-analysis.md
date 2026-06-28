# Sensitivity Analysis

Parameter robustness scoring via single-parameter sweeps on a MA-crossover strategy.

## Overview

Tests how stable a strategy's performance is when individual parameters change. Built-in demo uses a simple moving average crossover on SPY.

## Strategy

- **Buy**: fast MA crosses above slow MA
- **Sell**: fast MA crosses below slow MA, or stop-loss/take-profit hit

## Parameters

| Parameter | Base | Range | Step |
|-----------|------|-------|------|
| `fast_ma` | 10 | 5–30 | 1 |
| `slow_ma` | 50 | 20–100 | 5 |
| `stop_loss_pct` | 2.0 | 0.5–5.0 | 0.5 |
| `take_profit_pct` | 4.0 | 1.0–10.0 | 0.5 |

## Robustness Scoring

For each parameter, holds all others constant and sweeps the target across its range. Scores 0–100 based on coefficient of variation (CV) of performance:

```
cv = std(values) / abs(mean(values))
score = max(0, min(100, 100 * (1 - cv)))
```

- CV=0 → score=100 (perfectly stable)
- CV=1 → score=50
- CV=2 → score=0

When base value has no trades (all returns are 0), scores all points equally — no degradation possible.

### Classification

- **≥70 Robust** (green) — performance stays stable across range
- **40–70 Moderate** (amber) — some sensitivity, consider tightening range
- **<40 Fragile** (red) — small changes cause large swings, possible overfitting

## Visualizations

### Overall Robustness Gauge

Circular arc meter showing the average of all parameter scores. Colored green/amber/red by classification.

### Degradation Heatmap

Parameters on Y-axis, metrics on X-axis. Cells colored by degradation from base performance:

- **Deep green** (≤10% degradation)
- **Cyan** (≤20%)
- **Amber** (≤40%)
- **Red** (>40%)

### Per-Parameter Detail Charts

Line charts showing total return vs parameter value. Base value marked with dashed cyan line. Stable zone (within 20% of base) shown as a subtle green-tinted background band.

## Custom Ranges

Toggle "Use custom ranges" in the sidebar to override parameter ranges.

## URL Routing

Supports shareable links:

- `?ticker=QQQ&start_date=2021-01-01&end_date=2024-01-01` — auto-runs with custom ticker/date range
