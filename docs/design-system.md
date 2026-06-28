# Design System

Shared design system used by all dashboard pages. Import via:

```python
from design_system import *

apply_theme()
```

## Theme

`apply_theme()` injects dark trading-terminal CSS:

- **Fonts**: DM Sans (UI text), JetBrains Mono (code/numbers)
- **Background**: `#0a0a0f` primary, `#12121a` cards
- **Accent**: `#00d4ff` cyan (primary accent)
- **Top bar**: 2px cyan line via `::before` pseudo-element
- **Chrome removed**: main menu, sidebar nav, footer hidden

## Colors

```python
BG_PRIMARY = "#0a0a0f"          # Main background
BG_CARD = "#12121a"             # Card background
BG_CARD_HOVER = "#1a1a24"       # Card hover state
BORDER = "rgba(255,255,255,0.06)"

ACCENT_CYAN = "#00d4ff"         # Primary accent
ACCENT_GREEN = "#00e676"        # Positive / low vol
ACCENT_RED = "#ff1744"          # Negative / high vol
ACCENT_AMBER = "#ffc107"        # Neutral / medium
ACCENT_VIOLET = "#7c4dff"       # Uncertain

TEXT_PRIMARY = "#ffffff"        # Headings
TEXT_SECONDARY = "#8a8a9a"      # Body text
TEXT_MUTED = "#5a5a6a"          # Subtle labels
GRID_LINE = "rgba(255,255,255,0.04)"

REGIME_COLORS = {
    "Low Vol": ACCENT_GREEN,
    "Bull": ACCENT_GREEN,
    "Medium Vol": ACCENT_AMBER,
    "Neutral": ACCENT_AMBER,
    "High Vol": ACCENT_RED,
    "Bear": ACCENT_RED,
    "Uncertain": ACCENT_VIOLET,
}
```

## HTML Helpers

### `metric_card(label, value, color)`

Renders a metric display with uppercase label and large value:

```python
st.markdown(metric_card("PnL", "+$12,345", ACCENT_GREEN), unsafe_allow_html=True)
```

### `regime_badge(regime_name, confidence=None)`

Renders a colored pill/badge with optional confidence percentage:

```python
st.markdown(regime_badge("Low Vol", confidence=87.3), unsafe_allow_html=True)
```

### `section_header(text)`

Renders an uppercase section header with divider line:

```python
st.markdown(section_header("Performance"), unsafe_allow_html=True)
```

### `status_dot(status)`

Renders a colored dot indicator (green for connected/active, red for disconnected/error, amber for warning):

```python
st.markdown(status_dot("connected") + " Connected", unsafe_allow_html=True)
```

### `pnl_color(value)`

Returns `ACCENT_GREEN` if value ≥ 0, `ACCENT_RED` if value < 0:

```python
color = pnl_color(-500)  # Returns ACCENT_RED
```

## Plotly

### `get_plotly_layout(**overrides)`

Returns a base Plotly layout dict with dark trading-terminal styling. Pass keyword overrides to merge:

```python
layout = get_plotly_layout(
    xaxis=dict(title="Date", type="date"),
    yaxis=dict(title="Price ($)", tickformat="$,.0f"),
    height=450,
)
fig = go.Figure(layout=layout)
```

## DataFrames

### `style_dataframe(df)`

Returns a pandas Styler with dark formatting:

```python
styled = style_dataframe(df)
styled = styled.format({
    "PnL": lambda v: f"${v:,.0f}",
    "Return": "{:.1%}",
})
st.dataframe(styled, width="stretch", hide_index=True)
```
