"""
lib.scoring
===========
Converts raw indicator values into normalised 0–1 scores and combines
them into a single composite probability score.

Public API
----------
WEIGHTS                          : dict[str, float]  — indicator weights (sum = 1)
score_rsi(rsi)                   -> float
score_macd(macd, signal, hist)   -> float
score_bollinger(pct_b)           -> float
score_moving_averages(price, sma20, sma50, sma200) -> float
score_stochastic(pct_k, pct_d)   -> float
score_momentum(mom5, mom10, mom20) -> float
score_volume(vol_ratio)          -> float
score_all(indicators_dict)       -> dict[str, float]
composite(scores_dict)           -> float
classify(score)                  -> str
analyze(ticker, ohlcv_df)        -> dict
"""

from __future__ import annotations

import numpy as np

from . import indicators as ind

# ─────────────────────────────────────────────
# WEIGHTS  (must sum to 1.0)
# ─────────────────────────────────────────────

WEIGHTS: dict[str, float] = {
    "rsi":        0.18,
    "macd":       0.18,
    "bollinger":  0.14,
    "ma":         0.20,
    "stochastic": 0.12,
    "momentum":   0.12,
    "volume":     0.06,
}

assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-9, "WEIGHTS must sum to 1.0"

# Signal thresholds
SIGNAL_THRESHOLDS = {
    "STRONG BUY":  0.72,
    "BUY":         0.58,
    "NEUTRAL":     0.45,
    "SELL":        0.32,
    # Below 0.32 → STRONG SELL
}


# ─────────────────────────────────────────────
# INDIVIDUAL SCORING FUNCTIONS
# ─────────────────────────────────────────────

def score_rsi(rsi: float) -> float:
    """
    RSI → 0–1 score.
    Oversold (low RSI) is bullish; overbought (high RSI) is bearish.
    """
    if   rsi <= 20: return 0.90
    elif rsi <= 30: return 0.80
    elif rsi <= 40: return 0.65
    elif rsi <= 55: return 0.55
    elif rsi <= 65: return 0.45
    elif rsi <= 70: return 0.35
    elif rsi <= 80: return 0.20
    else:           return 0.10


def score_macd(macd: float, signal: float, histogram: float) -> float:
    """
    MACD → 0–1 score based on:
      • position of MACD vs signal line  (±0.15)
      • histogram sign                    (±0.10)
      • MACD vs zero line                 (±0.05)
    """
    score = 0.5
    score += 0.15 if macd > signal    else -0.15
    score += 0.10 if histogram > 0    else -0.10
    score += 0.05 if macd > 0         else -0.05
    return float(np.clip(score, 0.0, 1.0))


def score_bollinger(pct_b: float) -> float:
    """
    Bollinger %B → 0–1 score (mean-reversion framing).
    Price near/below lower band → bullish; near/above upper band → bearish.
    """
    if np.isnan(pct_b): return 0.5
    if   pct_b <  0.0:  return 0.85   # below lower band
    elif pct_b <  0.2:  return 0.70
    elif pct_b <  0.4:  return 0.60
    elif pct_b <  0.6:  return 0.50   # mid-band neutral
    elif pct_b <  0.8:  return 0.40
    elif pct_b <  1.0:  return 0.30
    else:               return 0.20   # above upper band


def score_moving_averages(
    price: float,
    sma20: float,
    sma50: float,
    sma200: float,
) -> float:
    """
    Moving-average trend score.
    Awards +0.08 for each MA the price is above, and +0.08 for a
    golden cross (SMA50 > SMA200); deducts symmetrically otherwise.
    """
    score = 0.5
    for s in [sma20, sma50, sma200]:
        if s > 0:
            score += 0.08 if price > s else -0.08
    if sma50 > 0 and sma200 > 0:
        score += 0.08 if sma50 > sma200 else -0.08  # golden / death cross
    return float(np.clip(score, 0.0, 1.0))


def score_stochastic(pct_k: float, pct_d: float) -> float:
    """
    Stochastic %K/%D → 0–1 score.
    Oversold region (<20) is bullish; overbought (>80) is bearish.
    %K > %D (bullish crossover) adds a small bonus.
    """
    score = 0.5
    if   pct_k <  20: score += 0.20
    elif pct_k <  40: score += 0.10
    elif pct_k >  80: score -= 0.20
    elif pct_k >  60: score -= 0.10
    score += 0.05 if pct_k > pct_d else -0.05
    return float(np.clip(score, 0.0, 1.0))


def score_momentum(mom5: float, mom10: float, mom20: float) -> float:
    """
    Multi-period momentum → 0–1 score.
    Each return is capped at ±10% before being weighted
    (5-day: 0.15, 10-day: 0.10, 20-day: 0.05).
    """
    score = 0.5
    for mom, w in [(mom5, 0.15), (mom10, 0.10), (mom20, 0.05)]:
        score += w * float(np.clip(mom, -10, 10)) / 10
    return float(np.clip(score, 0.0, 1.0))


def score_volume(vol_ratio: float) -> float:
    """
    Volume trend ratio → 0–1 score.
    Rising volume (> 1.0) signals stronger conviction behind price moves.
    """
    if   vol_ratio >= 2.0: return 0.75
    elif vol_ratio >= 1.5: return 0.70
    elif vol_ratio >= 1.1: return 0.60
    elif vol_ratio >= 0.9: return 0.50
    elif vol_ratio >= 0.7: return 0.40
    else:                  return 0.30


# ─────────────────────────────────────────────
# COMPOSITE HELPERS
# ─────────────────────────────────────────────

def score_all(indic: dict) -> dict[str, float]:
    """
    Convert a raw indicators dict (from ``indicators.compute_all``)
    into a component-scores dict keyed by WEIGHTS keys.
    """
    return {
        "rsi":        score_rsi(indic["rsi"]),
        "macd":       score_macd(indic["macd"], indic["macd_signal"], indic["macd_hist"]),
        "bollinger":  score_bollinger(indic["bb_pct_b"]),
        "ma":         score_moving_averages(
                          indic["price"], indic["sma20"], indic["sma50"], indic["sma200"]
                      ),
        "stochastic": score_stochastic(indic["stoch_k"], indic["stoch_d"]),
        "momentum":   score_momentum(indic["mom5"], indic["mom10"], indic["mom20"]),
        "volume":     score_volume(indic["vol_trend"]),
    }


def composite(scores: dict[str, float]) -> float:
    """
    Weighted average of component scores → single probability in [0, 1].
    """
    total = sum(WEIGHTS[k] * scores[k] for k in WEIGHTS)
    return float(np.clip(total, 0.0, 1.0))


def classify(score: float) -> str:
    """Map a composite score to a human-readable signal label."""
    if   score >= SIGNAL_THRESHOLDS["STRONG BUY"]: return "STRONG BUY"
    elif score >= SIGNAL_THRESHOLDS["BUY"]:        return "BUY"
    elif score >= SIGNAL_THRESHOLDS["NEUTRAL"]:    return "NEUTRAL"
    elif score >= SIGNAL_THRESHOLDS["SELL"]:       return "SELL"
    else:                                          return "STRONG SELL"


# ─────────────────────────────────────────────
# TOP-LEVEL CONVENIENCE
# ─────────────────────────────────────────────

def analyze(ticker: str, df) -> dict | None:
    """
    Full pipeline for a single stock: indicators → scores → composite.

    Parameters
    ----------
    ticker : str          — stock ticker symbol
    df     : pd.DataFrame — OHLCV data (from lib.data.fetch_ohlcv)

    Returns
    -------
    Flat result dict ready for display or DataFrame construction,
    or None if the OHLCV data is insufficient.
    """
    if df is None or len(df) < 50:
        return None

    try:
        raw    = ind.compute_all(df)
        scores = score_all(raw)
        prob   = composite(scores)
        signal = classify(prob)

        return {
            # Identity
            "ticker":       ticker,
            "price":        round(raw["price"], 2),
            # Composite
            "score":        round(prob, 4),
            "signal":       signal,
            # Raw indicator values
            "rsi":          round(raw["rsi"],        1),
            "macd":         round(raw["macd"],       4),
            "macd_signal":  round(raw["macd_signal"],4),
            "macd_hist":    round(raw["macd_hist"],  4),
            "bb_pct_b":     round(raw["bb_pct_b"],   3),
            "sma20":        round(raw["sma20"],       2),
            "sma50":        round(raw["sma50"],       2),
            "sma200":       round(raw["sma200"],      2),
            "stoch_k":      round(raw["stoch_k"],     1),
            "stoch_d":      round(raw["stoch_d"],     1),
            "mom5":         round(raw["mom5"],        2),
            "mom10":        round(raw["mom10"],       2),
            "mom20":        round(raw["mom20"],       2),
            "vol_trend":    round(raw["vol_trend"],   2),
            "atr":          round(raw["atr"],         2),
            # Derived flags
            "above_sma20":  raw["price"] > raw["sma20"],
            "above_sma50":  raw["price"] > raw["sma50"],
            "above_sma200": raw["price"] > raw["sma200"],
            "golden_cross": raw["sma50"] > raw["sma200"],
            # Component scores
            "score_rsi":    round(scores["rsi"],        3),
            "score_macd":   round(scores["macd"],       3),
            "score_boll":   round(scores["bollinger"],  3),
            "score_ma":     round(scores["ma"],         3),
            "score_stoch":  round(scores["stochastic"], 3),
            "score_mom":    round(scores["momentum"],   3),
            "score_vol":    round(scores["volume"],     3),
        }

    except Exception as e:
        print(f"  [!] Error analysing {ticker}: {e}")
        return None


def analyze_universe(
    ohlcv_data: dict,
    verbose: bool = True,
) -> list[dict]:
    """
    Run ``analyze`` over an entire ``{ticker: DataFrame}`` mapping and
    return a list of result dicts sorted by score descending.
    """
    results = []
    tickers = list(ohlcv_data.keys())

    for i, (ticker, df) in enumerate(ohlcv_data.items(), 1):
        if verbose:
            print(f"  [{i:>2}/{len(tickers)}] Scoring {ticker:<7}...", end=" ")

        result = analyze(ticker, df)

        if result:
            results.append(result)
            if verbose:
                print(f"score={result['score']:.4f}  [{result['signal']}]")
        else:
            if verbose:
                print("skipped")

    return sorted(results, key=lambda r: r["score"], reverse=True)
