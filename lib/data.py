"""
lib.data
========
Stock universe and data-fetching utilities.

Public API
----------
TOP_50_SP500        : list[str]   - tickers ranked by market cap
fetch_ohlcv(ticker) : pd.DataFrame | None
fetch_all(tickers)  : dict[str, pd.DataFrame]
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd

TOP_50_SP500: list[str] = [
    "AAPL",  # Apple
    "MSFT",  # Microsoft
    "NVDA",  # NVIDIA
    "AMZN",  # Amazon
    "GOOGL", # Alphabet (Class A)
    "META",  # Meta Platforms
    "TSLA",  # Tesla
    "AVGO",  # Broadcom
    "BRK-B", # Berkshire Hathaway
    "LLY",   # Eli Lilly
    "JPM",   # JPMorgan Chase
    "V",     # Visa
    "UNH",   # UnitedHealth Group
    "XOM",   # ExxonMobil
    "MA",    # Mastercard
    "COST",  # Costco
    "HD",    # Home Depot
    "PG",    # Procter & Gamble
    "JNJ",   # Johnson & Johnson
    "WMT",   # Walmart
    "NFLX",  # Netflix
    "BAC",   # Bank of America
    "ORCL",  # Oracle
    "KO",    # Coca-Cola
    "CRM",   # Salesforce
    "CVX",   # Chevron
    "AMD",   # Advanced Micro Devices
    "MRK",   # Merck
    "ABBV",  # AbbVie
    "PEP",   # PepsiCo
    "LIN",   # Linde
    "TMO",   # Thermo Fisher Scientific
    "ACN",   # Accenture
    "ADBE",  # Adobe
    "CSCO",  # Cisco
    "MCD",   # McDonald's
    "TXN",   # Texas Instruments
    "NKE",   # Nike
    "WFC",   # Wells Fargo
    "DIS",   # Walt Disney
    "PM",    # Philip Morris
    "AMGN",  # Amgen
    "DHR",   # Danaher
    "IBM",   # IBM
    "QCOM",  # Qualcomm
    "INTC",  # Intel
    "LOW",   # Lowe's
    "CAT",   # Caterpillar
    "NOW",   # ServiceNow
    "GE",    # GE Aerospace
]


def fetch_ohlcv(ticker: str, lookback_days: int = 400, min_rows: int = 50) -> Optional[pd.DataFrame]:
    """Download adjusted OHLCV data for *ticker* covering the last *lookback_days* calendar days."""
    try:
        import yfinance as yf
    except ImportError:
        raise ImportError("yfinance is required: pip install yfinance")

    end   = datetime.today()
    start = end - timedelta(days=lookback_days)
    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)

    if df is None or len(df) < min_rows:
        return None

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    return df[["Open", "High", "Low", "Close", "Volume"]].copy()


def fetch_all(tickers: list[str] | None = None, lookback_days: int = 400,
              delay: float = 0.3, verbose: bool = True) -> dict[str, pd.DataFrame]:
    """Fetch OHLCV data for every ticker and return a {ticker: DataFrame} mapping."""
    if tickers is None:
        tickers = TOP_50_SP500

    results: dict[str, pd.DataFrame] = {}

    for i, ticker in enumerate(tickers, 1):
        if verbose:
            print(f"  [{i:>2}/{len(tickers)}] Fetching {ticker:<7}...", end=" ")

        df = fetch_ohlcv(ticker, lookback_days=lookback_days)

        if df is not None:
            results[ticker] = df
            if verbose:
                print(f"OK  ({len(df)} rows)")
        else:
            if verbose:
                print("skipped")

        time.sleep(delay)

    return results
