"""Sentiment Analysis Dashboard — news-driven sentiment gauges and key drivers."""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import datetime
import requests
import xml.etree.ElementTree as ET
import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import plotly.graph_objects as go
import re

from nltk.sentiment.vader import SentimentIntensityAnalyzer

from design_system import (
    ACCENT_CYAN, ACCENT_GREEN, ACCENT_RED, ACCENT_AMBER,
    BG_CARD, BG_CARD_HOVER, BG_PRIMARY, BORDER, TEXT_MUTED, TEXT_SECONDARY, TEXT_PRIMARY,
    apply_theme, metric_card, section_header, get_plotly_layout,
)

apply_theme()

# ── Constants ──────────────────────────────────────────────────────────────────

DEFAULT_TICKERS = ["SPY", "AAPL", "NVDA", "TSLA", "BTC-USD"]
GOOGLE_NEWS_RSS_URL = "https://news.google.com/rss/search?q={ticker}+stock&hl=en-US&gl=US&ceid=US:en"
LOOKBACK_DAYS_DEFAULT = 3

# ── Utilities ──────────────────────────────────────────────────────────────────

def sentiment_color(score: float) -> str:
    """Return color based on sentiment score."""
    if score > 0.05:
        return ACCENT_GREEN
    elif score < -0.05:
        return ACCENT_RED
    return TEXT_MUTED

def sentiment_direction(score: float) -> tuple[str, str]:
    """Return (emoji, text) for sentiment direction."""
    if score > 0.05:
        return ("up", "Bullish")
    elif score < -0.05:
        return ("down", "Bearish")
    return ("flat", "Neutral")

# ── News Fetching ──────────────────────────────────────────────────────────────

@st.cache_data(ttl=300)
def fetch_google_news_rss(ticker: str, lookback_days: int = LOOKBACK_DAYS_DEFAULT) -> list[dict]:
    """Fetch articles from Google News RSS for a ticker."""
    articles = []
    try:
        url = GOOGLE_NEWS_RSS_URL.format(ticker=ticker)
        resp = requests.get(url, timeout=15, headers={
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
        })
        resp.raise_for_status()
        root = ET.fromstring(resp.content)
        cutoff = datetime.datetime.now() - datetime.timedelta(days=lookback_days)
        for item in root.findall(".//item"):
            title = item.findtext("title", "")
            source = item.findtext("source", "")
            published = item.findtext("pubDate", "")
            snippet = item.findtext("description", "")
            snippet = re.sub(r"<[^>]+>", "", snippet).strip()[:200]
            pub_date = None
            try:
                pub_date = datetime.datetime.strptime(published, "%a, %d %b %Y %H:%M:%S %Z")
            except (ValueError, TypeError):
                pass
            if pub_date and pub_date < cutoff:
                continue
            if title:
                articles.append({
                    "title": title,
                    "source": source,
                    "pub_date": pub_date,
                    "snippet": snippet,
                    "published_str": pub_date.strftime("%b %d, %Y %H:%M") if pub_date else published,
                })
    except Exception as e:
        st.warning(f"Google News fetch failed for {ticker}: {e}")
    return articles

@st.cache_data(ttl=300)
def fetch_newsapi_articles(ticker: str, api_key: str, lookback_days: int = LOOKBACK_DAYS_DEFAULT) -> list[dict]:
    """Fetch articles from NewsAPI for a ticker."""
    articles = []
    try:
        url = "https://newsapi.org/v2/everything"
        params = {
            "q": ticker,
            "apiKey": api_key,
            "sortBy": "publishedAt",
            "language": "en",
        }
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        cutoff = datetime.datetime.now() - datetime.timedelta(days=lookback_days)
        for item in data.get("articles", []):
            title = item.get("title", "")
            source = item.get("source", {}).get("name", "")
            pub_date = None
            published_str = ""
            if item.get("publishedAt"):
                try:
                    pub_date = datetime.datetime.strptime(
                        item["publishedAt"], "%Y-%m-%dT%H:%M:%SZ"
                    )
                    published_str = pub_date.strftime("%b %d, %Y %H:%M")
                except ValueError:
                    published_str = item["publishedAt"]
            if pub_date and pub_date < cutoff:
                continue
            snippet = item.get("description", "") or ""
            if title:
                articles.append({
                    "title": title,
                    "source": source,
                    "pub_date": pub_date,
                    "snippet": snippet,
                    "published_str": published_str,
                })
    except Exception as e:
        st.warning(f"NewsAPI fetch failed for {ticker}: {e}")
    return articles

def fetch_articles(ticker: str, api_key: str, lookback_days: int) -> list[dict]:
    """Fetch articles, trying NewsAPI first, then Google News RSS."""
    if api_key:
        articles = fetch_newsapi_articles(ticker, api_key, lookback_days)
    else:
        articles = []
    if not articles:
        articles = fetch_google_news_rss(ticker, lookback_days)
    return articles

# ── Sentiment Scoring ─────────────────────────────────────────────────────────

_sia = None

def get_sia() -> SentimentIntensityAnalyzer:
    """Lazy-init VADER analyzer."""
    global _sia
    if _sia is None:
        try:
            _sia = SentimentIntensityAnalyzer()
        except LookupError:
            import nltk
            nltk.download("vader_lexicon", quiet=True)
            _sia = SentimentIntensityAnalyzer()
    return _sia

def score_article(title: str) -> float:
    """Score article title using VADER compound. Returns score in [-1, 1]."""
    sia = get_sia()
    return sia.polarity_scores(title)["compound"]

def compute_ticker_sentiment(articles: list[dict]) -> dict:
    """Compute aggregated sentiment for a ticker."""
    if not articles:
        return {
            "score": 0.0,
            "articles": [],
            "momentum": 0.0,
            "momentum_direction": "flat",
            "article_count": 0,
        }

    now = datetime.datetime.now()
    scored_articles = []
    weights = []

    for article in articles:
        score = score_article(article["title"])
        # Weight: recent articles (within 3 days) get 2x weight
        weight = 2.0 if (article["pub_date"] and (now - article["pub_date"]).days <= 3) else 1.0
        scored_articles.append({**article, "sentiment": score, "weight": weight})
        weights.append(weight)

    # Weighted average
    total_weight = sum(weights)
    weighted_score = sum(a["sentiment"] * a["weight"] for a in scored_articles) / total_weight if total_weight else 0.0

    # Momentum: recent vs older
    recent = [a for a in scored_articles if a["pub_date"] and (now - a["pub_date"]).days <= 3]
    older = [a for a in scored_articles if a["pub_date"] and (now - a["pub_date"]).days > 3]

    momentum = 0.0
    if recent and older:
        recent_weight = sum(a["weight"] for a in recent)
        older_weight = sum(a["weight"] for a in older)
        recent_score = sum(a["sentiment"] * a["weight"] for a in recent) / recent_weight if recent_weight else 0.0
        older_score = sum(a["sentiment"] * a["weight"] for a in older) / older_weight if older_weight else 0.0
        momentum = recent_score - older_score

    momentum_direction = "up" if momentum > 0.05 else ("down" if momentum < -0.05 else "flat")

    # Top 5 by absolute sentiment
    top_articles = sorted(scored_articles, key=lambda a: abs(a["sentiment"]), reverse=True)[:5]

    return {
        "score": weighted_score,
        "articles": top_articles,
        "momentum": momentum,
        "momentum_direction": momentum_direction,
        "article_count": len(articles),
    }

# ── Gauge SVG ──────────────────────────────────────────────────────────────────

def render_gauge_card(ticker: str, sentiment: dict) -> str:
    """Render a single ticker gauge card — instrument panel style."""
    score = sentiment["score"]
    article_count = sentiment["article_count"]
    momentum_dir = sentiment["momentum_direction"]

    color = sentiment_color(score)
    glow = color + "80"
    emoji, label = sentiment_direction(score)

    # Needle angle: map score [-1, 1] to [-135, -45] degrees
    angle = -135 + (score + 1) * 90
    needle_rad = np.radians(angle)
    needle_x = 100 + 70 * np.cos(needle_rad)
    needle_y = 100 + 70 * np.sin(needle_rad)

    # Score label position (outside the arc)
    label_rad = np.radians(angle)
    label_x = 100 + 85 * np.cos(label_rad)
    label_y = 100 + 85 * np.sin(label_rad)

    score_display = f"{score:+.2f}"

    # Momentum
    arrow_color = ACCENT_GREEN if momentum_dir == "up" else (ACCENT_RED if momentum_dir == "down" else TEXT_MUTED)
    arrow_symbol = "↑" if momentum_dir == "up" else ("↓" if momentum_dir == "down" else "→")
    arrow_text = "Improving" if momentum_dir == "up" else ("Declining" if momentum_dir == "down" else "Stable")

    # Semicircle arc paths (center 100,100 radius 80)
    # Red: -1 to 0 (left half)
    # Green: 0 to +1 (right half)
    # We draw the arc from -135° to -45° for the full semicircle, but split colors
    # Actually let's do it simpler: one arc, colored by segments

    # Full arc from -135° to -45°
    # Red segment: -135° to -90° (scores -1 to 0)
    # Green segment: -90° to -45° (scores 0 to +1)
    # But wait: angle=-135 is left side, angle=-45 is right side.
    # So red goes left-to-center, green goes center-to-right.

    # SVG arc: A rx ry x-axis-rotation large-arc-flag sweep-flag x y
    # From left end (-135°): x=20, y=100
    # To right end (-45°): x=180, y=100
    # Sweep 0 = counter-clockwise (top arc)
    red_arc = 'M 20 100 A 80 80 0 0 0 100 20'  # left to top
    green_arc = 'M 100 20 A 80 80 0 0 0 180 100'  # top to right

    # Tick marks (radius 65-72, labels at 80)
    tick_data = [
        (-1, -135), (-0.5, -112.5), (0, -90), (0.5, -67.5), (1, -45),
    ]
    tick_lines = ""
    tick_labels = ""
    for val, deg in tick_data:
        rad = np.radians(deg)
        x1 = 100 + 65 * np.cos(rad)
        y1 = 100 + 65 * np.sin(rad)
        x2 = 100 + 72 * np.cos(rad)
        y2 = 100 + 72 * np.sin(rad)
        lx = 100 + 80 * np.cos(rad)
        ly = 100 + 80 * np.sin(rad)
        tick_lines += f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{TEXT_MUTED}" stroke-width="1" opacity="0.4"/>'
        tick_labels += f'<text x="{lx:.0f}" y="{ly:.0f}" text-anchor="middle" dominant-baseline="middle" font-family="JetBrains Mono,monospace" font-size="9" fill="{TEXT_MUTED}">{val}</text>'

    # Needle glow circle at tip
    needle_glow = f'<circle cx="{needle_x:.1f}" cy="{needle_y:.1f}" r="6" fill="{color}" opacity="0.15" style="filter: drop-shadow(0 0 6px {glow});"/>'
    needle_tip = f'<circle cx="{needle_x:.1f}" cy="{needle_y:.1f}" r="2.5" fill="{color}" style="filter: drop-shadow(0 0 4px {glow});"/>'

    return f"""
    <span style="display:inline-block; vertical-align:top; width:18.5%; margin-right:0.5%; min-width:180px; max-width:280px;">
        <div style="background:{BG_CARD};border:1px solid {BORDER};border-radius:12px;padding:1.25rem;text-align:center;box-shadow:0 2px 12px rgba(0,0,0,0.3);">
            <div style="font-family:DM Sans,sans-serif;font-size:22px;font-weight:700;color:{TEXT_PRIMARY};margin-bottom:0.75rem;letter-spacing:1px;">
                {ticker}
            </div>

            <svg width="180" height="160" viewBox="0 0 200 180">
                <!-- Red arc: -1 to 0 -->
                <path d="{red_arc}" fill="none" stroke="{ACCENT_RED}" stroke-width="10" stroke-linecap="round" opacity="0.15"/>
                <!-- Green arc: 0 to +1 -->
                <path d="{green_arc}" fill="none" stroke="{ACCENT_GREEN}" stroke-width="10" stroke-linecap="round" opacity="0.15"/>

                <!-- Tick marks -->
                {tick_lines}

                <!-- Tick labels -->
                {tick_labels}

                <!-- Needle shadow -->
                <line x1="100" y1="100" x2="{needle_x + 1:.1f}" y2="{needle_y + 1:.1f}"
                      stroke="rgba(0,0,0,0.5)" stroke-width="2.5" stroke-linecap="round"/>

                <!-- Needle -->
                <line x1="100" y1="100" x2="{needle_x:.1f}" y2="{needle_y:.1f}"
                      stroke="{color}" stroke-width="2" stroke-linecap="round"
                      style="filter: drop-shadow(0 0 6px {glow});"/>

                <!-- Glow dot at tip -->
                {needle_glow}
                {needle_tip}

                <!-- Center pivot -->
                <circle cx="100" cy="100" r="4" fill="{BG_CARD}" stroke="{color}" stroke-width="1.5"/>
                <circle cx="100" cy="100" r="2" fill="{color}"/>

                <!-- Score label -->
                <text x="{label_x:.0f}" y="{label_y:.0f}" text-anchor="middle" dominant-baseline="middle"
                      font-family="JetBrains Mono,monospace" font-size="13" font-weight="700"
                      fill="{color}"
                      style="filter: drop-shadow(0 0 6px {glow});">
                    {score_display}
                </text>
            </svg>

            <!-- Momentum -->
            <div style="margin-top:0.25rem;">
                <span style="font-family:DM Sans,sans-serif;font-size:14px;font-weight:600;color:{arrow_color};">
                    {arrow_symbol} {arrow_text}
                </span>
            </div>

            <!-- Article count -->
            <div style="margin-top:0.75rem;font-family:JetBrains Mono,monospace;font-size:10px;color:{TEXT_MUTED};">
                {article_count} articles analyzed
            </div>
        </div>
    </span>
    """

# ── Article Card ───────────────────────────────────────────────────────────────

def render_article_card(article: dict) -> str:
    """Render a single dark card for a key driver article."""
    score = article["sentiment"]
    card_color = ACCENT_GREEN if score > 0.05 else (ACCENT_RED if score < -0.05 else TEXT_MUTED)
    dot_color = sentiment_color(score)

    return f"""
    <span style="display:block;background:{BG_CARD};border:1px solid {BORDER};
                border-radius:8px;padding:0.75rem 1rem;margin-bottom:0.5rem;
                border-left:3px solid {card_color};">
        <span style="display:block;font-size:10px;color:{TEXT_MUTED};
                     font-family:DM Sans,sans-serif;margin-bottom:4px;">
            {article.get('source', 'Unknown')} · {article.get('published_str', 'N/A')}
        </span>
        <span style="display:block;font-family:DM Sans,sans-serif;font-size:13px;
                     font-weight:600;color:{TEXT_PRIMARY};margin-bottom:4px;
                     line-height:1.3;">
            {article.get('title', '')}
        </span>
        <span style="display:inline-block;width:8px;height:8px;
                     background:{dot_color};border-radius:50%;
                     box-shadow:0 0 4px {dot_color}60;vertical-align:middle;">
        </span>
    </span>
    """

# ── Aggregate Bar (Plotly) ─────────────────────────────────────────────────────

def render_aggregate_bar(ticker_sentiments: dict) -> go.Figure:
    """Render horizontal stacked bar showing overall market sentiment via Plotly."""
    segments = []
    for ticker, sentiment in ticker_sentiments.items():
        score = sentiment["score"]
        if abs(score) < 0.01:
            continue
        segments.append({
            "ticker": ticker,
            "score": score,
            "color": sentiment_color(score),
        })

    if not segments:
        return go.Figure()

    # Sort: bearish (negative) first, then bullish (positive)
    bearish = [s for s in segments if s["score"] < 0]
    bullish = [s for s in segments if s["score"] > 0]

    # Y positions
    y_positions = list(range(len(segments)))
    y_tickers = [s["ticker"] for s in bearish + bullish]

    # X values: negative for bearish, positive for bullish
    x_values = [s["score"] for s in bearish + bullish]
    colors = [s["color"] for s in bearish + bullish]

    fig = go.Figure()

    # Main bar
    fig.add_trace(go.Bar(
        x=x_values,
        y=y_positions,
        orientation="h",
        marker_color=colors,
        marker_line_width=0,
        opacity=0.75,
        text=y_tickers,
        textposition="inside",
        textfont=dict(
            family="JetBrains Mono, monospace",
            size=12,
            color="#ffffff",
            weight=700,
        ),
        hovertemplate="<b>%{text}</b><br>Sentiment: %{x:.2f}<extra></extra>",
        width=0.6,
    ))

    # Center zero line
    fig.add_shape(
        type="line",
        x0=0, y0=-0.5, x1=0, y1=len(segments) - 0.5,
        line=dict(color=TEXT_MUTED, width=1.5, dash="dot"),
    )

    # Zero label
    fig.add_annotation(
        x=0, y=len(segments) - 0.5,
        text="0",
        showarrow=False,
        font=dict(
            family="JetBrains Mono, monospace",
            size=9,
            color=TEXT_MUTED,
        ),
    )

    # X-axis labels
    x_range = max(abs(min(x_values, default=-1)), abs(max(x_values, default=1)))
    x_range = max(x_range, 1.0)

    fig.update_layout(
        **get_plotly_layout(
            height=len(segments) * 36 + 60,
            barmode="relative",
            xaxis=dict(
                range=[-x_range - 0.1, x_range + 0.1],
                title=dict(text="Sentiment", font=dict(size=11)),
                dtick=0.5,
            ),
            yaxis=dict(
                autorange="reversed",
                showgrid=False,
                showticklabels=False,
                showline=False,
            ),
            margin=dict(l=60, r=20, t=10, b=30),
            showlegend=False,
        )
    )

    return fig

# ── Sidebar ────────────────────────────────────────────────────────────────────

def render_sidebar():
    """Sidebar with ticker management, API key, lookback period, refresh."""
    st.sidebar.subheader("Tickers")

    st.session_state.setdefault("tickers", DEFAULT_TICKERS.copy())
    tickers_input = st.session_state.tickers

    with st.sidebar.expander("Manage Tickers", expanded=True):
        ticker_list = st.text_area(
            "Tickers (one per line)",
            value="\n".join(tickers_input),
            key="ticker_input",
            help="One ticker per line, e.g. SPY, AAPL, TSLA",
        )
        if st.button("Apply", use_container_width=True):
            new_tickers = [t.strip().upper() for t in ticker_list.split("\n") if t.strip()]
            if new_tickers:
                st.session_state.tickers = new_tickers
                st.rerun()

    st.session_state.setdefault("newsapi_key", "")
    st.text_input(
        "NewsAPI Key (optional)",
        value=st.session_state.newsapi_key,
        type="password",
        key="newsapi_key",
        help="Get a free key at newsapi.org. Falls back to Google News RSS if not provided.",
    )

    st.session_state.setdefault("lookback_days", LOOKBACK_DAYS_DEFAULT)
    lookback = st.number_input(
        "Lookback Period",
        min_value=1,
        max_value=30,
        value=st.session_state.lookback_days,
        key="lookback_days",
        help="Number of days to look back for articles",
    )
    if lookback != st.session_state.lookback_days:
        st.session_state.lookback_days = lookback

    st.markdown(
        f'<hr style="border:1px solid {BORDER};border-radius:2px;" '
        'noshade="noshade" style="width:100%;margin:1rem 0;">',
        unsafe_allow_html=True,
    )
    if st.sidebar.button("Refresh", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

# ── Top Bar ────────────────────────────────────────────────────────────────────

def build_top_bar(ticker_sentiments: dict, total_articles: int) -> str:
    """Render the newsroom/financial terminal header."""
    now = datetime.datetime.now()
    date_str = now.strftime("%A, %B %d, %Y")
    time_str = now.strftime("%I:%M %p")

    return f"""
    <div style="background:{BG_CARD};border-bottom:1px solid {BORDER};
                padding:1.25rem 2rem;margin-bottom:1.5rem;">
        <div style="display:flex;align-items:center;justify-content:space-between;">
            <div style="display:flex;align-items:center;gap:1rem;">
                <span style="font-family:DM Sans,sans-serif;font-size:28px;
                             font-weight:700;color:{TEXT_PRIMARY};
                             text-shadow:0 0 10px {ACCENT_CYAN}40;">
                    MARKET SENTIMENT BRIEFING
                </span>
                <span style="display:inline-block;width:6px;height:6px;
                             background:{ACCENT_GREEN};border-radius:50%;
                             box-shadow:0 0 4px {ACCENT_GREEN}80;"></span>
            </div>
            <div style="display:flex;align-items:center;gap:2rem;">
                <span style="font-family:DM Sans,sans-serif;font-size:13px;color:{TEXT_SECONDARY};">
                    {date_str} · {time_str}
                </span>
                <span style="font-family:JetBrains Mono,monospace;font-size:13px;color:{ACCENT_CYAN};">
                    {total_articles} articles analyzed
                </span>
            </div>
        </div>
    </div>
    """

# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    st.set_page_config(
        page_title="Sentiment Analysis",
        page_icon="📰",
        layout="wide",
    )

    # ── Sidebar ─────────────────────────────────────────────────────────────
    render_sidebar()

    # ── Get tickers ─────────────────────────────────────────────────────────
    tickers = st.session_state.tickers
    api_key = st.session_state.newsapi_key
    lookback = st.session_state.lookback_days

    # ── Fetch articles and compute sentiment ────────────────────────────────
    ticker_sentiments = {}
    total_articles = 0

    with st.spinner("Fetching news and analyzing sentiment..."):
        for ticker in tickers:
            articles = fetch_articles(ticker, api_key, lookback)
            sentiment = compute_ticker_sentiment(articles)
            ticker_sentiments[ticker] = sentiment
            total_articles += sentiment["article_count"]

    # ── Top bar ─────────────────────────────────────────────────────────────
    st.markdown(build_top_bar(ticker_sentiments, total_articles), unsafe_allow_html=True)

    # ── Ticker gauges row ───────────────────────────────────────────────────
    gauge_html = ""
    for ticker in tickers:
        sentiment = ticker_sentiments[ticker]
        gauge_html += render_gauge_card(ticker, sentiment)

    components.html(gauge_html, height=320, scrolling=False)

    # ── Ticker detail selection ─────────────────────────────────────────────
    st.session_state.setdefault("selected_ticker", None)

    # Selection buttons
    for ticker in tickers:
        s = ticker_sentiments[ticker]
        color = sentiment_color(s["score"])
        is_selected = st.session_state.selected_ticker == ticker
        bg = BG_CARD_HOVER if is_selected else BG_CARD
        border_color = color if is_selected else BORDER

        if st.button(f"__select__{ticker}", key=f"sel_{ticker}",
                      use_container_width=False,
                      help=f"View details for {ticker}"):
            st.session_state.selected_ticker = ticker
            st.rerun()

    # Show detail panel
    if st.session_state.selected_ticker:
        selected = st.session_state.selected_ticker
        sentiment = ticker_sentiments[selected]
        st.markdown(section_header(f"Key Drivers — {selected}"), unsafe_allow_html=True)

        articles = sentiment["articles"]
        if articles:
            for article in articles:
                components.html(
                    render_article_card(article),
                    height=90,
                    scrolling=False,
                )
        else:
            st.markdown(
                f'<div style="font-size:12px;color:{TEXT_MUTED};text-align:center;padding:2rem;">'
                "No articles found for this ticker</div>",
                unsafe_allow_html=True,
            )

    # ── Aggregate bar ───────────────────────────────────────────────────────
    st.markdown(section_header("Market Sentiment Overview"), unsafe_allow_html=True)
    agg_fig = render_aggregate_bar(ticker_sentiments)
    if agg_fig.data:
        st.plotly_chart(agg_fig, use_container_width=True, config={"displayModeBar": False})

    # ── Disclaimer ──────────────────────────────────────────────────────────
    st.markdown(
        f'<div style="text-align:center;padding:1.5rem 0;margin-top:1.5rem;'
        f'font-family:DM Sans,sans-serif;font-size:11px;color:{TEXT_MUTED};'
        f'border-top:1px solid {BORDER};" '
        'unsafe_allow_html=True>'
        "Sentiment scores are based on automated text analysis and may misinterpret context. "
        "Use as one input among many."
        f'</div>',
        unsafe_allow_html=True,
    )

if __name__ == "__main__":
    main()
