"""Shared design system for all trading dashboards.

Usage in any Streamlit app:

    from design_system import *

    apply_theme()
    st.metric_card("PnL", "+$12,345", ACCENT_GREEN)
    st.regime_badge("Low Vol", confidence=87.3)
    st.section_header("Performance")
    ...
"""

import streamlit as st
import plotly.graph_objects as go
import pandas as pd

# ── Color Constants ──────────────────────────────────────────────────────────

BG_PRIMARY = "#0a0a0f"
BG_CARD = "#12121a"
BG_CARD_HOVER = "#1a1a24"
BORDER = "rgba(255,255,255,0.06)"

ACCENT_CYAN = "#00d4ff"
ACCENT_GREEN = "#00e676"
ACCENT_RED = "#ff1744"
ACCENT_AMBER = "#ffc107"
ACCENT_VIOLET = "#7c4dff"

TEXT_PRIMARY = "#ffffff"
TEXT_SECONDARY = "#8a8a9a"
TEXT_MUTED = "#5a5a6a"
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

# ── Theme Injection ──────────────────────────────────────────────────────────

def apply_theme() -> None:
    """Inject dark trading-terminal CSS into the Streamlit app."""
    css = f"""
        <link href="https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&family=JetBrains+Mono:wght@400;500;700&display=swap" rel="stylesheet">
        <style>
        /* Reset */
        html, body, .stApp {{
            background-color: {BG_PRIMARY} !important;
            padding: 0 !important;
            margin: 0 !important;
        }}
        /* Top accent line */
        .stApp::before {{
            content: "";
            display: block;
            height: 2px;
            background: {ACCENT_CYAN};
        }}
        /* Hide Streamlit chrome */
        #MainMenu {{ visibility: hidden; }}
        #stSidebarNav {{ visibility: hidden; }}
        footer {{ visibility: hidden; }}
        .st-emotion-cache-1e58t6n {{ visibility: hidden; }}
        .st-emotion-cache-1k6o7q5 {{ visibility: hidden; }}
        /* Metrics */
        .stMetric {{
            background: transparent !important;
            padding: 0.5rem 0 !important;
        }}
        .stMetric .stMetricValue {{
            font-family: 'JetBrains Mono', monospace !important;
            font-size: 28px !important;
            font-weight: 700 !important;
            color: {ACCENT_CYAN} !important;
        }}
        .stMetric .stMetricLabel {{
            font-family: 'DM Sans', sans-serif !important;
            font-size: 11px !important;
            text-transform: uppercase !important;
            letter-spacing: 3px !important;
            color: {TEXT_MUTED} !important;
        }}
        /* Cards / containers */
        .st-emotion-cache-1kyxreq {{
            background: {BG_CARD} !important;
            border: 1px solid {BORDER} !important;
            border-radius: 12px !important;
            padding: 1rem !important;
        }}
        .st-emotion-cache-1kyxreq:hover {{
            background: {BG_CARD_HOVER} !important;
        }}
        /* Sidebar */
        [data-testid="stSidebar"] {{
            background: {BG_CARD} !important;
            border-right: 1px solid {BORDER} !important;
            padding: 1rem !important;
        }}
        /* Text */
        .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {{
            font-family: 'DM Sans', sans-serif !important;
            color: {TEXT_PRIMARY} !important;
        }}
        .stMarkdown p, .stMarkdown div {{
            font-family: 'DM Sans', sans-serif !important;
            color: {TEXT_SECONDARY} !important;
        }}
        /* Buttons */
        .stButton > button {{
            font-family: 'DM Sans', sans-serif !important;
            font-size: 13px !important;
            text-transform: uppercase !important;
            letter-spacing: 1px !important;
            color: {TEXT_PRIMARY} !important;
            background: {BG_CARD} !important;
            border: 1px solid {BORDER} !important;
            border-radius: 8px !important;
        }}
        .stButton > button:hover {{
            background: {BG_CARD_HOVER} !important;
            border-color: {ACCENT_CYAN} !important;
        }}
        /* Selectbox / Dropdowns */
        .stSelectbox > div[data-baseweb="select"] {{
            background: {BG_CARD} !important;
            border: 1px solid {BORDER} !important;
            border-radius: 8px !important;
            color: {TEXT_PRIMARY} !important;
        }}
        /* Scrollbar */
        ::-webkit-scrollbar {{ width: 6px; }}
        ::-webkit-scrollbar-track {{ background: {BG_PRIMARY}; }}
        ::-webkit-scrollbar-thumb {{ background: {BORDER}; border-radius: 3px; }}
        /* Expander */
        .stExpander {{
            background: {BG_CARD} !important;
            border: 1px solid {BORDER} !important;
            border-radius: 12px !important;
        }}
        </style>
        """
    st.html(css)

# ── Helper Functions ─────────────────────────────────────────────────────────

def metric_card(label: str, value: str, color: str = ACCENT_CYAN) -> str:
    """Return styled HTML for a metric display."""
    return (
        f'<div style="display:inline-block; text-align:center; '
        f'margin-right:2rem; margin-bottom:1rem;">'
        f'<div style="font-family:DM Sans,sans-serif; font-size:11px; '
        f'text-transform:uppercase; letter-spacing:3px; '
        f'color:{TEXT_MUTED}; margin-bottom:4px;">{label}</div>'
        f'<div style="font-family:JetBrains Mono,monospace; font-size:28px; '
        f'font-weight:700; color:{color};">{value}</div>'
        f'</div>'
    )


def regime_badge(regime_name: str, confidence: float | None = None) -> str:
    """Return styled HTML pill/badge for a market regime."""
    color = REGIME_COLORS.get(regime_name, TEXT_MUTED)
    glow = color + "40"  # 25% opacity hex
    bg = color + "33"    # 20% opacity hex
    confidence_str = f"<br><span style=\"font-size:11px;color:{TEXT_MUTED};\">{confidence:.1f}%</span>" if confidence is not None else ""
    return (
        f'<span style="display:inline-block; '
        f'background:{bg}; '
        f'color:{color}; '
        f'font-family:DM Sans,sans-serif; '
        f'font-size:12px; '
        f'font-weight:500; '
        f'padding:4px 14px; '
        f'border-radius:20px; '
        f'border:1px solid {color}40; '
        f'box-shadow:0 0 8px {glow};">'
        f'{regime_name}{confidence_str}'
        f'</span>'
    )


def section_header(text: str) -> str:
    """Return styled HTML for a section header."""
    return (
        f'<div style="margin-bottom:1rem; margin-top:1.5rem;">'
        f'<div style="font-family:DM Sans,sans-serif; font-size:11px; '
        f'text-transform:uppercase; letter-spacing:3px; '
        f'color:{TEXT_MUTED};">{' ' * 0}{text}</div>'
        f'<div style="height:1px; background:{BORDER}; margin-top:6px;"></div>'
        f'</div>'
    )


def status_dot(status: str) -> str:
    """Return a small colored dot for connection/status indicators."""
    color_map = {
        "connected": ACCENT_GREEN,
        "active": ACCENT_GREEN,
        "disconnected": ACCENT_RED,
        "error": ACCENT_RED,
        "warning": ACCENT_AMBER,
    }
    color = color_map.get(status.lower(), TEXT_MUTED)
    return (
        f'<span style="display:inline-block; width:8px; height:8px; '
        f'background:{color}; border-radius:50%; '
        f'box-shadow:0 0 4px {color}60;"></span>'
    )


def pnl_color(value: float) -> str:
    """Return accent color based on PnL sign."""
    return ACCENT_GREEN if value >= 0 else ACCENT_RED

# ── Plotly Layout ────────────────────────────────────────────────────────────

def get_plotly_layout(**overrides) -> dict:
    """Return a base Plotly layout dict with dark trading-terminal styling.

    Pass keyword overrides to merge into the returned dict.

    Example:
        layout = get_plotly_layout(title="Equity Curve")
    """
    base = {
        "paper_bgcolor": BG_PRIMARY,
        "plot_bgcolor": BG_PRIMARY,
        "font": {
            "family": "JetBrains Mono, monospace",
            "size": 12,
            "color": TEXT_SECONDARY,
        },
        "xaxis": {
            "gridcolor": GRID_LINE,
            "zerolinecolor": GRID_LINE,
            "tickfont": {"color": TEXT_MUTED},
            "linecolor": BORDER,
            "zeroline": False,
        },
        "yaxis": {
            "gridcolor": GRID_LINE,
            "zerolinecolor": GRID_LINE,
            "tickfont": {"color": TEXT_MUTED},
            "linecolor": BORDER,
            "zeroline": False,
        },
        "hoverlabel": {
            "bgcolor": BG_CARD,
            "bordercolor": ACCENT_CYAN,
            "font": {
                "family": "JetBrains Mono, monospace",
                "size": 12,
                "color": TEXT_PRIMARY,
            },
        },
        "margin": {"l": 40, "r": 20, "t": 40, "b": 40},
        "showlegend": True,
        "legend": {
            "font": {
                "family": "DM Sans, sans-serif",
                "size": 12,
                "color": TEXT_SECONDARY,
            },
            "bgcolor": "rgba(0,0,0,0)",
        },
    }
    base.update(overrides)
    return base

# ── DataFrame Styling ────────────────────────────────────────────────────────

def style_dataframe(df: pd.DataFrame):
    """Return a styled pandas Styler with dark trading-terminal formatting."""
    styler = df.style.set_properties(**{
        "background-color": BG_CARD,
        "color": TEXT_SECONDARY,
        "border-color": BORDER,
        "font-family": "JetBrains Mono, monospace",
        "font-size": "12px",
    })
    styler = styler.set_table_styles([
        {"selector": "th", "props": [
            ("background-color", BG_PRIMARY),
            ("color", TEXT_PRIMARY),
            ("font-family", "DM Sans, sans-serif"),
            ("font-weight", "500"),
            ("font-size", "11px"),
            ("text-transform", "uppercase"),
            ("letter-spacing", "1px"),
            ("border-color", BORDER),
            ("padding", "8px 12px"),
        ]},
        {"selector": "tbody tr", "props": [
            ("border-bottom", f"1px solid {BORDER}"),
        ]},
        {"selector": "tbody tr:hover", "props": [
            ("background-color", BG_CARD_HOVER),
        ]},
    ])
    return styler
