"""lib/fundamentals.py
--------------------
Fetches and scores fundamental financial data for S&P 500 stocks via yfinance.

Metric categories
-----------------
  Valuation  : trailing P/E, price-to-book, PEG ratio
  Growth     : revenue growth (YoY), earnings growth (YoY)
  Quality    : return on equity, net profit margin, free-cash-flow yield
  Risk       : debt-to-equity ratio
  Sentiment  : analyst consensus recommendation

Usage
-----
>>> from lib import fundamentals
>>> fund_data = fundamentals.fetch_fundamentals(TICKERS, verbose=True)
>>> for ticker, rec in fund_data.items():
...     scores = fundamentals.score_fundamentals(rec)
...     comp   = fundamentals.fundamental_composite(scores)
...     print(f"{ticker}: {comp:.3f}")
"""

import time
import yfinance as yf

# ── Fields pulled from yfinance .info ─────────────────────────────────────────
FUNDAMENTAL_FIELDS = [
    "trailingPE",
    "priceToBook",
    "pegRatio",
    "revenueGrowth",
    "earningsGrowth",
    "returnOnEquity",
    "profitMargins",
    "debtToEquity",
    "currentRatio",
    "freeCashflow",
    "marketCap",
    "recommendationMean",
    "forwardPE",
    "trailingEps",
    "earningsQuarterlyGrowth",
]


# ── Data fetching ──────────────────────────────────────────────────────────────
def fetch_fundamentals(
    tickers: list[str],
    verbose: bool = True,
    delay: float = 0.15,
) -> dict[str, dict]:
    """Fetch fundamental metrics for each ticker via yfinance .info.

    Parameters
    ----------
    tickers  : list of ticker symbols
    verbose  : print per-ticker progress
    delay    : seconds to sleep between requests (rate-limit guard)

    Returns
    -------
    dict  ticker -> dict of raw fundamental values (None where unavailable)
    """
    results: dict[str, dict] = {}

    for i, ticker in enumerate(tickers, 1):
        if verbose:
            print(f"  [{i:>2}/{len(tickers)}] {ticker:<7}", end="  ")
        try:
            info = yf.Ticker(ticker).info
            rec: dict = {}
            for field in FUNDAMENTAL_FIELDS:
                val = info.get(field)
                rec[field] = float(val) if val is not None else None
            results[ticker] = rec

            if verbose:
                pe  = rec.get("trailingPE")
                roe = rec.get("returnOnEquity")
                rg  = rec.get("revenueGrowth")
                parts = []
                if pe  is not None: parts.append(f"PE={pe:.1f}")
                if roe is not None: parts.append(f"ROE={100*roe:.0f}%")
                if rg  is not None: parts.append(f"RevGrowth={100*rg:.0f}%")
                print("✓  " + "  ".join(parts))

        except Exception as exc:
            if verbose:
                print(f"✗  ({exc})")
            results[ticker] = {f: None for f in FUNDAMENTAL_FIELDS}

        time.sleep(delay)

    return results


# ── Individual scorers (each returns a float in [0.0, 1.0]) ───────────────────
def _clamp(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def score_pe(pe) -> float:
    """Trailing P/E – lower is better (value signal).
    Negative or missing → neutral 0.5 (can't penalise loss-makers on P/E alone).
    """
    if pe is None or pe <= 0:
        return 0.5
    if pe <= 10:
        return 1.0
    if pe >= 50:
        return 0.0
    return _clamp(1.0 - (pe - 10) / 40.0)


def score_pb(pb) -> float:
    """Price-to-book – lower is better."""
    if pb is None or pb <= 0:
        return 0.5
    if pb <= 1.0:
        return 1.0
    if pb >= 10.0:
        return 0.0
    return _clamp(1.0 - (pb - 1.0) / 9.0)


def score_peg(peg) -> float:
    """PEG ratio – < 1 indicates undervalued relative to growth."""
    if peg is None or peg <= 0:
        return 0.5
    if peg <= 0.5:
        return 1.0
    if peg >= 3.0:
        return 0.0
    return _clamp(1.0 - (peg - 0.5) / 2.5)


def score_revenue_growth(growth) -> float:
    """Year-over-year revenue growth – higher is better.
    Score linearly from -10% (→ 0) to +30% (→ 1).
    """
    if growth is None:
        return 0.5
    if growth <= -0.10:
        return 0.0
    if growth >= 0.30:
        return 1.0
    return _clamp((growth + 0.10) / 0.40)


def score_earnings_growth(growth) -> float:
    """Year-over-year earnings growth – higher is better.
    Score linearly from -20% (→ 0) to +50% (→ 1).
    """
    if growth is None:
        return 0.5
    if growth <= -0.20:
        return 0.0
    if growth >= 0.50:
        return 1.0
    return _clamp((growth + 0.20) / 0.70)


def score_roe(roe) -> float:
    """Return on equity – higher is better.
    Score linearly from 0% (→ 0) to 30%+ (→ 1).
    """
    if roe is None:
        return 0.5
    if roe <= 0.0:
        return 0.0
    if roe >= 0.30:
        return 1.0
    return _clamp(roe / 0.30)


def score_profit_margin(margin) -> float:
    """Net profit margin – higher is better.
    Score linearly from 0% (→ 0) to 25%+ (→ 1).
    """
    if margin is None:
        return 0.5
    if margin <= 0.0:
        return 0.0
    if margin >= 0.25:
        return 1.0
    return _clamp(margin / 0.25)


def score_debt_equity(de) -> float:
    """Debt-to-equity – lower is better.
    Score linearly from 2.0+ (→ 0) to 0 (→ 1).
    """
    if de is None:
        return 0.5
    if de <= 0.0:
        return 1.0   # no debt
    if de >= 2.0:
        return 0.0
    return _clamp(1.0 - de / 2.0)


def score_fcf_yield(fcf, market_cap) -> float:
    """Free-cash-flow yield = FCF / market cap – higher is better.
    Score linearly from 0% (→ 0) to 5%+ (→ 1).
    """
    if fcf is None or market_cap is None or market_cap <= 0:
        return 0.5
    yield_ = fcf / market_cap
    if yield_ <= 0.0:
        return 0.0
    if yield_ >= 0.05:
        return 1.0
    return _clamp(yield_ / 0.05)


def score_analyst(rating) -> float:
    """yfinance recommendationMean: 1 = Strong Buy … 5 = Strong Sell.
    Inverted and normalised to [0, 1].
    """
    if rating is None:
        return 0.5
    return _clamp((5.0 - rating) / 4.0)


# ── Weights for the fundamental composite ─────────────────────────────────────
FUNDAMENTAL_WEIGHTS: dict[str, float] = {
    "pe":              0.12,
    "pb":              0.05,
    "peg":             0.13,
    "revenue_growth":  0.15,
    "earnings_growth": 0.15,
    "roe":             0.15,
    "profit_margin":   0.10,
    "debt_equity":     0.08,
    "fcf_yield":       0.05,
    "analyst":         0.02,
}
# Weights sum to 1.00


# ── Aggregate scoring ─────────────────────────────────────────────────────────
def score_fundamentals(rec: dict) -> dict[str, float]:
    """Score a single stock's fundamental record.

    Parameters
    ----------
    rec : dict of raw fundamental values (as returned by fetch_fundamentals)

    Returns
    -------
    dict  metric_name -> score in [0, 1]
    """
    return {
        "pe":              score_pe(rec.get("trailingPE")),
        "pb":              score_pb(rec.get("priceToBook")),
        "peg":             score_peg(rec.get("pegRatio")),
        "revenue_growth":  score_revenue_growth(rec.get("revenueGrowth")),
        "earnings_growth": score_earnings_growth(rec.get("earningsGrowth")),
        "roe":             score_roe(rec.get("returnOnEquity")),
        "profit_margin":   score_profit_margin(rec.get("profitMargins")),
        "debt_equity":     score_debt_equity(rec.get("debtToEquity")),
        "fcf_yield":       score_fcf_yield(rec.get("freeCashflow"), rec.get("marketCap")),
        "analyst":         score_analyst(rec.get("recommendationMean")),
    }


def fundamental_composite(component_scores: dict[str, float]) -> float:
    """Weighted average of fundamental component scores.

    Weights are read from FUNDAMENTAL_WEIGHTS; any metric not present in
    component_scores is silently skipped (its weight is excluded from the
    denominator so the result is still normalised).
    """
    total_w = sum(FUNDAMENTAL_WEIGHTS[k] for k in component_scores if k in FUNDAMENTAL_WEIGHTS)
    if total_w == 0:
        return 0.5
    weighted = sum(
        FUNDAMENTAL_WEIGHTS[k] * component_scores[k]
        for k in component_scores
        if k in FUNDAMENTAL_WEIGHTS
    )
    return weighted / total_w


def analyze_fundamentals(
    tickers: list[str],
    verbose: bool = True,
    delay: float = 0.15,
) -> dict[str, dict]:
    """Fetch + score fundamentals for all tickers in one call.

    Returns
    -------
    dict  ticker -> {
        "raw":       raw fundamental values,
        "scores":    component scores [0,1],
        "composite": weighted fundamental composite [0,1],
    }
    """
    raw_data = fetch_fundamentals(tickers, verbose=verbose, delay=delay)
    results: dict[str, dict] = {}
    for ticker, rec in raw_data.items():
        scores    = score_fundamentals(rec)
        composite = fundamental_composite(scores)
        results[ticker] = {
            "raw":       rec,
            "scores":    scores,
            "composite": composite,
        }
    return results
