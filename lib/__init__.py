"""lib – reusable building blocks for the S&P 500 analyser notebook.

Modules
-------
data         : ticker universe, OHLCV fetching, top-N by market cap
indicators   : technical indicator calculations (RSI, MACD, Bollinger, …)
scoring      : technical scoring, fundamental blending, signal classification
fundamentals : fundamental data fetching and scoring (P/E, ROE, growth, …)
sentiment    : AI news-sentiment scoring (FinBERT / VADER fallback)
display      : terminal output and matplotlib visualisations
"""

from lib import data, indicators, scoring, fundamentals, sentiment, display

__all__ = ["data", "indicators", "scoring", "fundamentals", "sentiment", "display"]
