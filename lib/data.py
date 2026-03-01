"""
lib.data
========
Stock universe and data-fetching utilities.

Public API
----------
TOP_50_SP500                        : list[str]  — hardcoded fallback, ~current top 50 by mkt cap
get_sp500_tickers()                 -> list[str]  — live S&P 500 constituents from Wikipedia
get_top_n_sp500(n, verbose)         -> list[str]  — top N by live market cap (falls back to hardcoded)
fetch_ohlcv(ticker)                 -> pd.DataFrame | None
fetch_all(tickers, lookback_days)   -> dict[str, pd.DataFrame]
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# HARDCODED FALLBACK UNIVERSE
# Used when dynamic fetching is unavailable (no internet, Wikipedia down, etc.)
# ─────────────────────────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────────────────────────
# DYNAMIC UNIVERSE HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def get_sp500_tickers(verbose: bool = False) -> list[str]:
    """
    Scrape the current S&P 500 constituent list from Wikipedia.

    Returns a list of ticker symbols (BRK.B → BRK-B style normalised for
    yfinance).  Falls back to TOP_50_SP500 on any network or parse error.
    """
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    try:
        if verbose:
            print("  Fetching S&P 500 constituent list from Wikipedia…", end=" ")

        tables = pd.read_html(url, attrs={"id": "constituents"})
        tickers = (
            tables[0]["Symbol"]
            .str.replace(".", "-", regex=False)   # BRK.B → BRK-B
            .str.strip()
            .tolist()
        )

        if verbose:
            print(f"✓  ({len(tickers)} constituents)")

        return tickers

    except Exception as e:
        if verbose:
            print(f"✗  failed: {e}")
        return TOP_50_SP500


def get_top_n_sp500(
    n: int = 50,
    verbose: bool = True,
    _batch_size: int = 100,
) -> list[str]:
    """
    Return the top *n* S&P 500 stocks ranked by **live market cap**.

    Workflow
    --------
    1. Fetch the full constituent list from Wikipedia (≈ 503 tickers).
    2. Query yfinance ``fast_info.market_cap`` in batches of *_batch_size*.
    3. Sort descending and slice to *n*.
    4. On **any** failure, fall back silently to the hardcoded TOP_50_SP500.

    Parameters
    ----------
    n           : Number of top stocks to return (default 50).
    verbose     : Print progress to stdout.
    _batch_size : Tickers per yfinance batch call (tune for speed vs. rate limits).

    Returns
    -------
    list[str]  Ticker symbols sorted by market cap, largest first.

    Notes
    -----
    * Fetching market caps for ~500 tickers takes roughly 60–120 s on a typical
      connection.  Cache the result in a notebook variable if you plan to re-run
      cells frequently::

          TICKERS = get_top_n_sp500(50)   # run once, store in TICKERS

    * The fallback list (TOP_50_SP500) was accurate as of early 2025; use the
      dynamic path for up-to-date rankings.
    """
    try:
        import yfinance as yf
    except ImportError:
        raise ImportError("yfinance is required: pip install yfinance")

    # ── Step 1: constituent list ──────────────────────────────────────────
    all_tickers = get_sp500_tickers(verbose=verbose)

    if verbose:
        print(f"  Fetching live market caps for {len(all_tickers)} tickers "
              f"(batches of {_batch_size})…")

    # ── Step 2: market caps in batches ────────────────────────────────────
    market_caps: dict[str, float] = {}
    total = len(all_tickers)

    for batch_start in range(0, total, _batch_size):
        batch = all_tickers[batch_start : batch_start + _batch_size]

        try:
            tickers_obj = yf.Tickers(" ".join(batch))

            for symbol in batch:
                try:
                    mc = tickers_obj.tickers[symbol].fast_info.market_cap
                    if mc and mc > 0:
                        market_caps[symbol] = float(mc)
                except Exception:
                    pass  # skip tickers with missing data

        except Exception as e:
            if verbose:
                print(f"\n  [!] Batch error at index {batch_start}: {e}")

        if verbose:
            done = min(batch_start + _batch_size, total)
            pct  = done / total * 100
            bar  = "█" * round(pct / 5) + "░" * (20 - round(pct / 5))
            print(f"  [{bar}] {done:>3}/{total}  ({len(market_caps)} caps retrieved)\r",
                  end="", flush=True)

        time.sleep(0.2)   # light rate-limit guard between batches

    if verbose:
        print(f"\n  ✓ Market caps retrieved for {len(market_caps)} / {total} stocks.")

    # ── Step 3: rank and slice ────────────────────────────────────────────
    if not market_caps:
        if verbose:
            print("  [!] No market caps retrieved — using hardcoded fallback.")
        return TOP_50_SP500[:n]

    ranked = sorted(market_caps, key=lambda t: market_caps[t], reverse=True)
    top_n  = ranked[:n]

    if verbose:
        print(f"  ✓ Top {n} by market cap:")
        for rank, sym in enumerate(top_n[:5], 1):
            mc_b = market_caps[sym] / 1e12
            print(f"    #{rank:>2}  {sym:<8}  ${mc_b:.2f} T")
        if n > 5:
            print(f"    … and {n - 5} more.")

    return top_n


# ─────────────────────────────────────────────────────────────────────────────
# OHLCV DATA FETCHING
# ─────────────────────────────────────────────────────────────────────────────

def fetch_ohlcv(
    ticker: str,
    lookback_days: int = 400,
    min_rows: int = 50,
) -> Optional[pd.DataFrame]:
    """
    Download adjusted OHLCV data for *ticker* covering the last
    *lookback_days* calendar days.

    Returns a DataFrame with columns [Open, High, Low, Close, Volume]
    or None if data is unavailable / insufficient.
    """
    try:
        import yfinance as yf
    except ImportError:
        raise ImportError("yfinance is required: pip install yfinance")

    end   = datetime.today()
    start = end - timedelta(days=lookback_days)

    df = yf.download(
        ticker,
        start=start,
        end=end,
        progress=False,
        auto_adjust=True,
    )

    if df is None or len(df) < min_rows:
        return None

    # Flatten MultiIndex columns if present (single-ticker download)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    return df[["Open", "High", "Low", "Close", "Volume"]].copy()


def fetch_all(
    tickers: list[str] | None = None,
    lookback_days: int = 400,
    delay: float = 0.3,
    verbose: bool = True,
) -> dict[str, pd.DataFrame]:
    """
    Fetch OHLCV data for every ticker in *tickers* and return a
    ``{ticker: DataFrame}`` mapping.  Failed downloads are silently skipped.

    Parameters
    ----------
    tickers       : list of ticker strings.
                    Pass ``get_top_n_sp500()`` for a live-ranked list, or
                    leave as ``None`` to use the hardcoded TOP_50_SP500.
    lookback_days : calendar days of history to request (default 400).
    delay         : seconds to wait between requests (rate-limit guard).
    verbose       : print progress to stdout.

    Example
    -------
    >>> from lib import data
    >>> tickers = data.get_top_n_sp500(50)   # live top 50 by market cap
    >>> ohlcv   = data.fetch_all(tickers)
    """
    if tickers is None:
        tickers = TOP_50_SP500

    results: dict[str, pd.DataFrame] = {}

    for i, ticker in enumerate(tickers, 1):
        if verbose:
            print(f"  [{i:>2}/{len(tickers)}] Fetching {ticker:<7}…", end=" ")

        df = fetch_ohlcv(ticker, lookback_days=lookback_days)

        if df is not None:
            results[ticker] = df
            if verbose:
                print(f"✓  ({len(df)} rows)")
        else:
            if verbose:
                print("✗  skipped")

        time.sleep(delay)

    return results
