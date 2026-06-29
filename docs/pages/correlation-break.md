# Correlation Break Detector

Rolling pairwise correlations with z-score alerting and historical context.

## Overview

Detects when the correlation between two assets breaks from its historical range, surfacing potential regime shifts in asset relationships.

## Default Pairs

| Pair | Rationale |
|------|-----------|
| SPY / QQQ | Large-cap equity |
| GLD / TLT | Safe-haven assets |
| SPY / IWM | Large-cap vs small-cap |
| BTC-USD / ETH-USD | Crypto majors |
| SPY / EEM | US vs emerging markets |

Custom pairs can be added via the sidebar.

## How It Works

### Rolling Correlation

For each pair, daily returns are calculated and the rolling correlation is computed at the selected window (20-day, 60-day, or 120-day).

### Z-Score Break Detection

The historical mean and standard deviation of the rolling correlation are calculated over the selected date range. The current z-score is:

```
z = (current_correlation - historical_mean) / historical_std
```

### Severity Classification

| Severity | Z-Score Range | Visual |
|----------|--------------|--------|
| Normal | > -1.5 | Dark gray, muted |
| Notable | -1.5 to -2.0 | Amber glow |
| Significant | -2.0 to -2.5 | Orange glow + border |
| Extreme | < -2.5 | Red pulsing glow |

### Historical Context

When a pair enters Notable or worse, the detector finds all prior periods where the z-score was within ±0.5 of the current value. For each historical instance, it computes:

- **Asset 1 returns**: 5-day, 10-day, 20-day forward returns from the break date
- **Asset 2 returns**: 5-day, 10-day, 20-day forward returns from the break date
- **Averages**: mean returns at each horizon across all historical instances

> ⚠ Past correlation breaks do not guarantee similar outcomes. Correlations can drift without signaling directional moves.

## Visualizations

### Status Cards

Horizontal row of cards, one per pair. Normal cards are visually quiet; break-status cards visually "pop" with CSS glow animations. Clicking a card navigates to its analysis.

### Main Correlation Chart

- 60-day correlation line (cyan)
- 20-day correlation line (violet dashed)
- Historical mean (cyan dashed)
- -2σ threshold (red dashed)
- Green area fill when above threshold, red when below
- Red shading on historical break periods
- Current correlation marker

### 20-Day vs 60-Day Comparison

Single chart showing both rolling windows together — helps distinguish short-term blips (20-day breaks but 60-day stable) from sustained shifts (both breaking).

## Inputs

### Sidebar Controls

- **Pair management** — edit/remove pairs, add new ones via text inputs
- **Correlation window** — 20-day, 60-day, or 120-day
- **Z-score threshold** — slider from -4.0 to -1.0 (default: -2.0). Alerts fire when z-score drops below this value.
- **Date range** — start/end date pickers (default: last 3 years)
- **Check Correlations** — runs analysis for all pairs
- **Refresh** — clears cache and re-fetches all data

## Alert Logging

Alerts are saved to `data/correlation_alerts.json`:

```json
[
  {
    "timestamp": "2025-01-15T14:32:00",
    "pair": "SPY / IWM",
    "z_score": -2.35,
    "severity": "Significant"
  }
]
```

The alert log section at the bottom of the page shows the most recent 50 alerts with timestamp, pair, z-score, and severity badge.

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│  Status Cards: Normal cards subdued, breaks glow           │
├──────────────────────────────────────────────────────────────┤
│  Main Chart: 60d corr line, mean, -2σ threshold,            │
│               green/red area fill, historical break shading │
├──────────────────────────────────────────────────────────────┤
│  20d vs 60d Comparison                                      │
├──────────────────────────────────────────────────────────────┤
│  Current Values: 60d corr, 20d corr, z-score, severity     │
├──────────────────────────────────────────────────────────────┤
│  Historical Context (only when z < -1.5):                   │
│    Forward returns table with AVERAGE row                   │
├──────────────────────────────────────────────────────────────┤
│  Alert Log (recent 50 from JSON file)                       │
└──────────────────────────────────────────────────────────────┘
```
