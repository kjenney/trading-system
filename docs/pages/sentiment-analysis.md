# Sentiment Analysis

News-driven sentiment gauges and key drivers — a professional morning briefing for market sentiment.

## Overview

Dashboard that fetches news articles for configurable tickers, scores sentiment using VADER, and displays results as instrument-panel-style gauges with momentum indicators and key driver articles.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  Top Bar: "MARKET SENTIMENT BRIEFING", date/time, article count │
├─────────────────────────────────────────────────────────────────┤
│  Gauge Row (equal-width cards):                                 │
│  ┌──────┐  ┌──────┐  ┌──────┐  ┌──────┐  ┌──────┐             │
│  │ SPY  │  │ AAPL │  │ NVDA │  │ TSLA │  │ BTC  │             │
│  │ Gauge│  │ Gauge│  │ Gauge│  │ Gauge│  │ Gauge│             │
│  └──────┘  └──────┘  └──────┘  └──────┘  └──────┘             │
├─────────────────────────────────────────────────────────────────┤
│  Selected Ticker Detail Panel:                                  │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │ Key Drivers — {TICKER}                                      ││
│  │ ┌─────────────────────────────────────────────────────────┐ ││
│  │ │ Source · Date                                           │ ││
│  │ │ Headline                                                │ ││
│  │ │ [sentiment dot]                                         │ ││
│  │ └─────────────────────────────────────────────────────────┘ ││
│  └─────────────────────────────────────────────────────────────┘│
├─────────────────────────────────────────────────────────────────┤
│  Market Sentiment Overview: horizontal stacked bar (-1 to +1)   │
└─────────────────────────────────────────────────────────────────┘
```

## Data Sources

### Google News RSS (default)

No API key required. Fetches RSS feeds from Google News for each ticker:

```
https://news.google.com/rss/search?q={ticker}+stock&hl=en-US&gl=US&ceid=US:en
```

Parses article title, source, publication date, and snippet from the XML feed. Filters articles by the configured lookback period.

### NewsAPI (optional)

Enter a NewsAPI key in the sidebar to use NewsAPI instead of Google News RSS. Get a free API key at [newsapi.org](https://newsapi.org).

If both sources are available, NewsAPI is tried first; if it returns no results, falls back to Google News RSS.

## Sentiment Scoring

Each article title is scored using VADER (Valence Aware Dictionary and sEntiment Analyzer) from `nltk.sentiment.vader`:

1. **VADER compound score** — single value from -1 (bearish) to +1 (bullish) computed from the article headline
2. **Weighted average** — more recent articles (within the lookback period, default 3 days) receive 2x weight compared to older articles
3. **Momentum** — compares recent weighted sentiment to older weighted sentiment; direction shown as "Improving", "Declining", or "Stable" with arrow indicators

## Gauges

Hero visual: semicircular instrument-panel gauges for each ticker:

- **Arc**: red left half (negative sentiment), green right half (positive sentiment), gray center
- **Needle**: positioned at current weighted sentiment score, with glow effect in the corresponding color
- **Score display**: numeric value at the needle tip with colored glow
- **Tick marks**: labeled at -1, -0.5, 0, 0.5, +1
- **Momentum**: arrow (↑/↓/→) with "Improving/Declining/Stable" text below the gauge
- **Article count**: small muted text at bottom

## Key Drivers

When a ticker card is selected, the detail panel shows the top 5 articles ranked by absolute sentiment score:

- **Source and date** — small muted text above the headline
- **Headline** — bold white text
- **Sentiment indicator** — colored dot (green for bullish, red for bearish, gray for neutral)
- **Colored left border** — green for bullish articles, red for bearish, gray for neutral

## Market Sentiment Overview

Horizontal stacked bar showing overall market sentiment:

- Bearish tickers (negative sentiment) extend left from zero center line
- Bullish tickers (positive sentiment) extend right from center line
- Each segment colored by sentiment direction, labeled with ticker symbol
- Scale from -1.0 (left) to +1.0 (right), zero marker at center

## Usage

1. Add tickers in the sidebar (default: SPY, AAPL, NVDA, TSLA, BTC-USD)
2. Optionally enter a NewsAPI key for NewsAPI data source
3. Set the lookback period for article filtering (default: 3 days)
4. Dashboard auto-runs — fetches articles and scores sentiment
5. Click "Refresh" to clear cache and re-fetch all data
6. Click a ticker card to view its key driver articles
