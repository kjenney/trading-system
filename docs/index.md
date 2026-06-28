# Claude for Trading

Dark trading-terminal Streamlit dashboard with three analytical pages: regime detection, sensitivity analysis, and Monte Carlo simulation.

## Pages

| Page | Description |
|------|-------------|
| [Regime Detection](pages/regime-detection.md) | HMM-based market regime classification with online forward filtering |
| [Sensitivity Analysis](pages/sensitivity-analysis.md) | Parameter robustness scoring via single-parameter sweeps |
| [Monte Carlo](pages/monte-carlo.md) | Equity curve probability cloud from trade-return shuffling |

## Architecture

```
claude-for-trading/
├── home.py                  # Landing page
├── design_system.py         # Shared theme, colors, Plotly layouts, HTML helpers
├── demo/demo.py             # Design system walkthrough
├── pages/
│   ├── dashboard.py         # HMM regime detection
│   ├── sensitivity_analysis.py  # Parameter sensitivity sweep
│   └── monte_carlo.py       # Monte Carlo equity simulation
└── requirements.txt         # Python dependencies
```

## Design System

`design_system.py` provides shared building blocks:

- **Theme** — `apply_theme()` injects dark CSS
- **HTML helpers** — `metric_card()`, `regime_badge()`, `section_header()`, `status_dot()`, `pnl_color()`
- **Plotly** — `get_plotly_layout()` returns dark-themed Plotly dict
- **Data** — `style_dataframe()` applies dark formatting to pandas DataFrames
- **Colors** — constants: `BG_PRIMARY`, `ACCENT_CYAN`, `ACCENT_GREEN`, `ACCENT_RED`, `ACCENT_AMBER`, `REGIME_COLORS`

See [Design System](design-system.md) for the full reference.

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run home.py
```

## URL Routing

Dashboard and Sensitivity pages support path-based routing via URL query params:

- `?ticker=SPY&start_date=2021-01-01&end_date=2024-01-01` — auto-runs with those params
- URL updates as you change inputs — shareable links
