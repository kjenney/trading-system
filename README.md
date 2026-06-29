# Claude for Trading — Dashboard

Dark trading-terminal Streamlit app with seven analytical pages: regime detection, sensitivity analysis, Monte Carlo simulation, portfolio risk, multi-asset regime backtesting, sentiment analysis, and correlation break detection.

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
| **Portfolio Risk** | 📊 | Regime overlay, correlation analysis, stress testing |
| **Multi-Asset Regime** | 📈 | Walk-forward HMM regime allocation across assets |
| **Regime Screener** | 🔍 | Multi-ticker HMM regime scan with filtering, sorting, click-to-chart |
| **Correlation Break** | 📉 | Rolling pairwise correlations, z-score alerts, historical context |
| **Demo** | — | Design system component showcase |

## Architecture

```
claude-for-trading/
├── home.py                  # Landing page
├── design_system.py         # Shared theme, colors, Plotly layouts, HTML helpers
├── demo/
│   └── demo.py              # Design system walkthrough
├── pages/
│   ├── dashboard.py              # HMM regime detection
│   ├── sensitivity_analysis.py   # Parameter sensitivity sweep
│   ├── monte_carlo.py            # Monte Carlo equity simulation
│   ├── portfolio_risk.py         # Regime overlay, correlation, stress testing
│   └── multi_asset_regime.py     # Walk-forward HMM regime allocation
│   └── regime_screener.py        # Multi-ticker HMM regime screener
│   └── correlation_break.py      # Rolling correlation z-score break detector
├── scripts/
│   └── detectwords.py            # Amazon Rekognition text detection (troubleshooting)
└── requirements.txt              # Python dependencies
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

### Portfolio Risk (`pages/portfolio_risk.py`)

- Input positions manually, via CSV upload, or via Alpaca API integration
- Runs HMM regime detection on each position and watchlist ticker — overlays regime badges on position cards
- Computes 60-day rolling correlation matrix between all positions — heatmap with red borders on pairs >0.85
- Stress tests: applies historical drawdowns (2008 Crisis, 2020 Covid, 2022 Rate Hikes) to portfolio
- Regime health metric: counts positions in favorable regimes ("Low Vol", "Bull")

### Multi-Asset Regime (`pages/multi_asset_regime.py`)

- Walk-forward backtest: train HMM on fixed window, test on rolling period, advance forward
- Regime-based allocation: 95% in Low Vol, 60% in High Vol, linear interpolation for Medium Vol
- Benchmarks: buy-and-hold, 200-day SMA trend-following
- Metrics: annualized return, Sharpe, max drawdown, Sharpe improvement
- Stress tests: drawdowns during 2008, 2020, 2022 crisis windows
- Visualizations: equity curves (per-asset colored), regime timeline strips (all assets stacked), comparison table, stress bar charts
- URL routing: `?selected=SPY&train_years=2&test_months=6` for shareable links

### Regime Screener (`pages/regime_screener.py`)

- Scans configurable universe of tickers (default: 23 across Large Cap, ETF, Crypto)
- Downloads price data via `yfinance`, engineers features: log returns, realized vol, volume ratio, H-L range
- Trains `hmmlearn.GaussianHMM` with diagonal covariance, selects #regimes via BIC (3–5 states)
- Forward-filtered regime labels — no look-ahead bias, stability filter flags flickering as "Uncertain"
- Regime labeling by mean volatility: Low Vol / Medium Vol / High Vol
- Sidebar: remove/add tickers, filter by regime, confidence, SMA 50 position, volume trend, sort order, date range
- Table: colored regime badges, confidence progress bars, clean setup indicator (cyan border)
- Click-to-chart modal: regime-colored price chart + regime history bar for individual tickers

### Correlation Break Detector (`pages/correlation_break.py`)

- Monitors configurable asset pairs (default: SPY/QQQ, GLD/TLT, SPY/IWM, BTC-USD/ETH-USD, SPY/EEM)
- Downloads daily data via `yfinance`, computes rolling correlations at 20-day and 60-day windows
- Calculates historical mean and standard deviation of each pair's 60-day rolling correlation
- Z-score alerts: classifies breaks as Normal (>-1.5), Notable (-1.5 to -2.0), Significant (-2.0 to -2.5), Extreme (<-2.5)
- Status cards with visual hierarchy — Normal cards are subdued, break-status cards "pop" with glow animations
- Main correlation chart with mean line, -2σ threshold, green/red area fill, historical break shading
- 20-day/60-day comparison chart for quick assessment
- Historical context: when a break is detected, finds prior similar z-score events and computes 5/10/20-day forward returns
- Alert logging: alerts saved to `data/correlation_alerts.json` with timestamp, pair, z-score, severity
- Sidebar: pair management (add/remove/edit), correlation window selector (20/60/120-day), z-score threshold slider, date range

### Detect Words (`scripts/detectwords.py`)

- Command-line tool for detecting text in images via Amazon Rekognition `detect_text`
- Usage: `python detectwords.py --profile <aws_profile> <image_path>`
- Outputs each detected word with confidence percentage
- Troubleshooting utility — not part of the Streamlit app

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

Dashboard, Sensitivity, and Multi-Asset Regime pages support path-based routing via URL query params:

- `?ticker=SPY&start_date=2021-01-01&end_date=2024-01-01` — Dashboard auto-runs with those params
- `?ticker=QQQ&start_date=2021-01-01&end_date=2024-01-01` — Sensitivity auto-runs with those params
- `?selected=SPY&start_date=2019-01-01&end_date=2024-01-01&train_years=2&test_months=6` — Multi-Asset Regime auto-runs
- URL updates as you change inputs — shareable links
