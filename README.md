# Claude for Trading — Dashboard

Dark trading-terminal Streamlit app with three analytical pages: regime detection, sensitivity analysis, and Monte Carlo simulation.

## Quick Start

```bash
cd claude-for-trading
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run home.py
```

## Pages

| Page | Icon | Description |
|------|------|-------------|
| **Dashboard** | 🚀 | HMM regime detection — online forward filtering, no look-ahead bias |
| **Sensitivity** | 🔬 | Parameter robustness scoring via single-parameter sweeps |
| **Monte Carlo** | 🎲 | Equity curve probability cloud from trade-return shuffling |
| **Demo** | — | Design system component showcase |

## Architecture

```
claude-for-trading/
├── home.py                  # Landing page
├── design_system.py         # Shared theme, colors, Plotly layouts, HTML helpers
├── demo/
│   └── demo.py              # Design system walkthrough
├── pages/
│   ├── dashboard.py         # HMM regime detection
│   ├── sensitivity_analysis.py  # Parameter sensitivity sweep
│   └── monte_carlo.py       # Monte Carlo equity simulation
└── requirements.txt         # Python dependencies
```

### Design System

`design_system.py` exports shared building blocks used by every page:

- **Theme** — `apply_theme()` injects dark CSS (fonts: DM Sans + JetBrains Mono, accent cyan `#00d4ff`, card backgrounds `#12121a`)
- **HTML helpers** — `metric_card()`, `regime_badge()`, `section_header()`, `status_dot()`, `pnl_color()`
- **Plotly** — `get_plotly_layout()` returns dark-themed Plotly dict with override merging
- **Data** — `style_dataframe()` applies dark formatting to pandas DataFrames
- **Colors** — constants: `BG_PRIMARY`, `ACCENT_CYAN`, `ACCENT_GREEN`, `ACCENT_RED`, `ACCENT_AMBER`, `REGIME_COLORS`

Pages import selectively:
```python
from design_system import ACCENT_CYAN, ACCENT_GREEN, metric_card, section_header
```

### Regime Detection (`pages/dashboard.py`)

- Downloads price data via `yfinance`
- Builds features: log returns, realized vol (20-day), volume ratio, H-L range
- Trains `hmmlearn.GaussianHMM` with diagonal covariance, selects #regimes via BIC (3–7 states)
- **Forward-filtered** labels only — uses forward algorithm (alpha) to compute posterior marginals `gamma[t] = P(q_t | o_1..t)`. No `model.predict()` — zero look-ahead bias
- Stability filter: regimes must persist 3 consecutive bars; >4 transitions in 20-bar window flags as "Uncertain" (label 999)
- Regimes labeled by mean volatility: Low Vol / Medium Vol / High Vol

### Sensitivity Analysis (`pages/sensitivity_analysis.py`)

- Built-in demo: MA-crossover on SPY (fast MA crosses above slow MA = buy, cross below = sell)
- Parameter sweeps: hold all but one parameter constant, sweep across range, run backtest for each value
- Robustness scoring: 0–100 based on coefficient of variation of performance across the sweep range
  - CV=0 → score=100 (perfectly stable)
  - CV=1 → score=50, CV=2 → score=0
- Classification: ≥70 Robust (green), 40–70 Moderate (amber), <40 Fragile (red)
- Visualizations: circular gauge, degradation heatmap, per-parameter line charts with stable zone bands

### Monte Carlo (`pages/monte_carlo.py`)

- Upload CSV (`date`, `trade_return`) or use built-in sample data (slightly positive edge + loss clustering)
- Shuffles trade returns + adds noise (±0.3%), runs N simulations (default 1000)
- Computes percentile bands (5th–95th, 25th–75th), median line
- Overfitting check: original backtest result percentile vs shuffled outcomes — >90th percentile = HIGH risk
- Visualizations: cinematic fan chart, final-value histogram, drawdown distribution histogram

## Dependencies

```
streamlit>=1.30.0
plotly>=5.18.0
numpy>=1.24.0
pandas>=2.0.0
yfinance>=0.2.31
hmmlearn>=0.3.0
```

## URL Routing

Dashboard and Sensitivity pages support path-based routing via URL query params:

- `?ticker=SPY&start_date=2021-01-01&end_date=2024-01-01` — auto-runs with those params
- URL updates as you change inputs — shareable links
