"""lib/scoring.py
----------------
Weighted scoring for technical indicators, plus a combined
technical + fundamental composite score.

Technical composite
-------------------
  Seven indicators are each scored 0–1 and blended using WEIGHTS.

Fundamental composite
---------------------
  Imported from lib.fundamentals and blended with the technical
  score using TECH_WEIGHT / FUND_WEIGHT (configurable at call time).

Combined composite
------------------
  combined = TECH_WEIGHT * tech_composite + FUND_WEIGHT * fund_composite
  Default split: 60 % technical, 40 % fundamental.

Usage
-----
>>> result   = scoring.analyze(ticker, df, indicators)
>>> combined = scoring.combined_composite(result["composite"],
...                                       fund_composite,
...                                       tech_weight=0.60)
>>> label    = scoring.classify(combined)
"""

from __future__ import annotations

import pandas as pd

from lib import data, indicators as ind

# ── Technical indicator weights (must sum to 1.0) ─────────────────────────────
WEIGHTS: dict[str, float] = {
    "rsi":             0.20,
    "macd":            0.20,
    "bollinger":       0.15,
    "moving_averages": 0.15,
    "stochastic":      0.10,
    "momentum":        0.10,
    "volume":          0.10,
}

# ── Technical / fundamental blend defaults ────────────────────────────────────
TECH_WEIGHT: float = 0.60   # weight applied to technical composite
FUND_WEIGHT: float = 0.40   # weight applied to fundamental composite

# ── Signal classification thresholds ─────────────────────────────────────────
SIGNAL_THRESHOLDS = {
    "Strong Buy":  0.75,
    "Buy":         0.60,
    "Neutral":     0.40,
    "Sell":        0.25,
    "Strong Sell": 0.00,
}


# ── Individual technical scorers (each returns 0.0–1.0) ──────────────────────
def score_rsi(indicators: dict) -> float:
    rsi = indicators.get("rsi")
    if rsi is None:
        return 0.5
    if rsi < 30:
        return 1.0   # oversold → bullish
    if rsi > 70:
        return 0.0   # overbought → bearish
    if rsi <= 50:
        return 0.5 + (50 - rsi) / 40
    return 0.5 - (rsi - 50) / 40


def score_macd(indicators: dict) -> float:
    macd_val  = indicators.get("macd")
    macd_sig  = indicators.get("macd_signal")
    macd_hist = indicators.get("macd_hist")
    if macd_val is None or macd_sig is None:
        return 0.5
    score = 0.5
    if macd_val > macd_sig:
        score += 0.3
    else:
        score -= 0.3
    if macd_hist is not None:
        if macd_hist > 0:
            score += 0.2
        else:
            score -= 0.2
    return max(0.0, min(1.0, score))


def score_bollinger(indicators: dict) -> float:
    pct_b = indicators.get("bb_pct_b")
    if pct_b is None:
        return 0.5
    if pct_b < 0:
        return 1.0
    if pct_b > 1:
        return 0.0
    return 1.0 - pct_b


def score_moving_averages(indicators: dict) -> float:
    score = 0.5
    price       = indicators.get("price")
    sma_20      = indicators.get("sma_20")
    sma_50      = indicators.get("sma_50")
    sma_200     = indicators.get("sma_200")
    ema_12      = indicators.get("ema_12")
    ema_26      = indicators.get("ema_26")
    if price and sma_20:
        score += 0.15 if price > sma_20 else -0.15
    if price and sma_50:
        score += 0.15 if price > sma_50 else -0.15
    if price and sma_200:
        score += 0.15 if price > sma_200 else -0.15
    if ema_12 and ema_26:
        score += 0.05 if ema_12 > ema_26 else -0.05
    return max(0.0, min(1.0, score))


def score_stochastic(indicators: dict) -> float:
    k = indicators.get("stoch_k")
    d = indicators.get("stoch_d")
    if k is None:
        return 0.5
    score = 0.5
    if k < 20:
        score += 0.3
    elif k > 80:
        score -= 0.3
    if d is not None:
        if k > d:
            score += 0.2
        else:
            score -= 0.2
    return max(0.0, min(1.0, score))


def score_momentum(indicators: dict) -> float:
    mom = indicators.get("momentum")
    if mom is None:
        return 0.5
    if mom > 0.20:
        return 1.0
    if mom < -0.20:
        return 0.0
    return (mom + 0.20) / 0.40


def score_volume(indicators: dict) -> float:
    vol_ratio = indicators.get("volume_ratio")
    if vol_ratio is None:
        return 0.5
    score = 0.5
    mom = indicators.get("momentum")
    if mom is not None and mom > 0 and vol_ratio > 1:
        score += min(0.5, (vol_ratio - 1) * 0.25)
    elif mom is not None and mom < 0 and vol_ratio > 1:
        score -= min(0.5, (vol_ratio - 1) * 0.25)
    return max(0.0, min(1.0, score))


# ── Technical aggregate ───────────────────────────────────────────────────────
def score_all(indicators: dict) -> dict[str, float]:
    """Return a dict of component technical scores."""
    return {
        "rsi":             score_rsi(indicators),
        "macd":            score_macd(indicators),
        "bollinger":       score_bollinger(indicators),
        "moving_averages": score_moving_averages(indicators),
        "stochastic":      score_stochastic(indicators),
        "momentum":        score_momentum(indicators),
        "volume":          score_volume(indicators),
    }


def composite(component_scores: dict[str, float]) -> float:
    """Weighted average of technical component scores."""
    total_w = sum(WEIGHTS.get(k, 0) for k in component_scores)
    if total_w == 0:
        return 0.5
    return sum(WEIGHTS.get(k, 0) * v for k, v in component_scores.items()) / total_w


# ── Combined technical + fundamental composite ────────────────────────────────
def combined_composite(
    tech_score: float,
    fund_score: float,
    tech_weight: float = TECH_WEIGHT,
    fund_weight: float = FUND_WEIGHT,
) -> float:
    """Blend a technical composite and a fundamental composite.

    Parameters
    ----------
    tech_score   : technical composite [0, 1]
    fund_score   : fundamental composite [0, 1]
    tech_weight  : relative weight for technical (default 0.60)
    fund_weight  : relative weight for fundamental (default 0.40)

    Returns
    -------
    float in [0, 1]
    """
    total = tech_weight + fund_weight
    if total == 0:
        return 0.5
    return (tech_weight * tech_score + fund_weight * fund_score) / total


# ── Signal classification ─────────────────────────────────────────────────────
def classify(score: float) -> str:
    """Map a composite score to a signal label."""
    if score >= SIGNAL_THRESHOLDS["Strong Buy"]:
        return "Strong Buy"
    if score >= SIGNAL_THRESHOLDS["Buy"]:
        return "Buy"
    if score >= SIGNAL_THRESHOLDS["Neutral"]:
        return "Neutral"
    if score >= SIGNAL_THRESHOLDS["Sell"]:
        return "Sell"
    return "Strong Sell"


# ── Single-ticker analysis ────────────────────────────────────────────────────
def analyze(ticker: str, df: pd.DataFrame, indicator_dict: dict) -> dict:
    """Score one ticker from its OHLCV DataFrame and pre-computed indicators.

    Parameters
    ----------
    ticker         : symbol string
    df             : OHLCV DataFrame
    indicator_dict : dict from indicators.compute_all(df)

    Returns
    -------
    dict with keys: ticker, price, scores, composite, signal
    """
    scores        = score_all(indicator_dict)
    tech_comp     = composite(scores)
    signal        = classify(tech_comp)
    return {
        "ticker":    ticker,
        "price":     indicator_dict.get("price"),
        "scores":    scores,
        "composite": tech_comp,
        "signal":    signal,
    }


# ── Universe analysis (technical only) ────────────────────────────────────────
def analyze_universe(
    tickers: list[str],
    lookback_days: int = 400,
    verbose: bool = True,
    delay: float = 0.1,
) -> dict[str, dict]:
    """Fetch OHLCV data and compute technical scores for every ticker.

    To add fundamental analysis, call lib.fundamentals.analyze_fundamentals()
    separately and then use combined_composite() to blend the two scores.

    Parameters
    ----------
    tickers       : list of ticker symbols
    lookback_days : calendar days of history to request (default 400)
    verbose       : print per-ticker progress
    delay         : seconds to wait between requests (rate-limit guard)

    Returns
    -------
    dict  ticker -> {ticker, price, scores, composite, signal}
    """
    results: dict[str, dict] = {}

    for i, ticker in enumerate(tickers, 1):
        if verbose:
            print(f"  [{i:>2}/{len(tickers)}] Fetching {ticker:<7}…", end=" ")
        df = data.fetch_ohlcv(ticker, lookback_days=lookback_days)
        if df is None:
            if verbose:
                print("✗ skipped")
            continue

        indicator_dict = ind.compute_all(df)
        result         = analyze(ticker, df, indicator_dict)
        results[ticker] = result

        if verbose:
            print(f"✓  {result['signal']:<11}  score={result['composite']:.3f}")

        import time
        time.sleep(delay)

    return results


# ── Combined universe analysis (technical + fundamental) ──────────────────────
def merge_fundamental(
    tech_results: dict[str, dict],
    fund_results: dict[str, dict],
    tech_weight: float = TECH_WEIGHT,
    fund_weight: float = FUND_WEIGHT,
) -> dict[str, dict]:
    """Merge pre-computed technical and fundamental results into one record.

    Parameters
    ----------
    tech_results : output of analyze_universe()
    fund_results : output of fundamentals.analyze_fundamentals()
    tech_weight  : weight for technical composite (default 0.60)
    fund_weight  : weight for fundamental composite (default 0.40)

    Returns
    -------
    dict  ticker -> {
        ticker, price,
        tech_scores, tech_composite,
        fund_scores, fund_composite,
        composite,   signal,
    }
    """
    merged: dict[str, dict] = {}

    all_tickers = set(tech_results) | set(fund_results)
    for ticker in all_tickers:
        t = tech_results.get(ticker, {})
        f = fund_results.get(ticker, {})

        tech_comp = t.get("composite", 0.5)
        fund_comp = f.get("composite", 0.5)
        comp      = combined_composite(tech_comp, fund_comp, tech_weight, fund_weight)
        signal    = classify(comp)

        merged[ticker] = {
            "ticker":         ticker,
            "price":          t.get("price"),
            "tech_scores":    t.get("scores", {}),
            "tech_composite": tech_comp,
            "fund_scores":    f.get("scores", {}),
            "fund_raw":       f.get("raw", {}),
            "fund_composite": fund_comp,
            "composite":      comp,
            "signal":         signal,
        }

    return merged
