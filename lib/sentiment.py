"""lib/sentiment.py
-----------------
AI-powered news-sentiment scoring for S&P 500 stocks.

Data source : yfinance Ticker.news (headline titles, up to N per ticker)
Scoring     : FinBERT (ProsusAI/finbert) via Transformers  [preferred]
              VADER (nltk)                                   [lightweight fallback]

Composite score : 0.0 = very negative  ·  1.0 = very positive

Usage
-----
    from lib import sentiment
    results = sentiment.analyze_sentiment(["AAPL", "MSFT"], model="auto")
    print(results["AAPL"]["composite"])   # e.g. 0.63
    print(results["AAPL"]["signal"])      # e.g. "Positive"
"""
from __future__ import annotations

import time
import datetime
import numpy as np
from typing import Optional

# ── lazy model singletons ─────────────────────────────────────────────────────
_FINBERT_PIPE: Optional[object] = None
_VADER_SIA:   Optional[object] = None

# ── backend loading ───────────────────────────────────────────────────────────

def _load_finbert():
    """Load FinBERT pipeline (cached after first call)."""
    global _FINBERT_PIPE
    if _FINBERT_PIPE is not None:
        return _FINBERT_PIPE
    try:
        from transformers import pipeline as hf_pipeline
    except ImportError:
        raise ImportError(
            "transformers is required for FinBERT.\n"
            "Install:  pip install transformers torch"
        )
    print("  Loading FinBERT (ProsusAI/finbert)...")
    print("  First run downloads ~400 MB and then is permanently cached.")
    _FINBERT_PIPE = hf_pipeline(
        "text-classification",
        model="ProsusAI/finbert",
        top_k=None,        # return scores for all 3 labels: positive/negative/neutral
        truncation=True,
        max_length=512,
    )
    print("  FinBERT loaded.")
    return _FINBERT_PIPE


def _load_vader():
    """Load VADER SentimentIntensityAnalyzer (cached after first call)."""
    global _VADER_SIA
    if _VADER_SIA is not None:
        return _VADER_SIA
    try:
        import nltk
    except ImportError:
        raise ImportError(
            "nltk is required for VADER.\n"
            "Install:  pip install nltk"
        )
    try:
        nltk.data.find("sentiment/vader_lexicon.zip")
    except LookupError:
        nltk.download("vader_lexicon", quiet=True)
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    _VADER_SIA = SentimentIntensityAnalyzer()
    return _VADER_SIA


def load_model(model: str = "auto") -> tuple[str, object]:
    """
    Load and return ``(method_name, model_object)``.

    Parameters
    ----------
    model : str
        ``"finbert"``  – FinBERT (best quality; ~400 MB download on first use)
        ``"vader"``    – VADER  (lightweight, rule-based, no download needed)
        ``"auto"``     – try FinBERT first, fall back to VADER
    """
    if model == "finbert":
        return "finbert", _load_finbert()
    if model == "vader":
        return "vader", _load_vader()
    # auto: prefer FinBERT, fall back gracefully
    try:
        return "finbert", _load_finbert()
    except ImportError:
        pass
    return "vader", _load_vader()


# ── per-text scoring ──────────────────────────────────────────────────────────

def _finbert_result_to_score(result) -> float:
    """
    Convert one FinBERT output item to a 0-1 score.

    Handles both:
      - list of dicts  [{'label': 'positive', 'score': 0.97}, ...]  (top_k=None)
      - single dict    {'label': 'positive', 'score': 0.97}         (top_k=1)
    """
    if isinstance(result, dict):
        label, score = result["label"], result["score"]
        if label == "positive":
            return 0.5 + score * 0.5
        elif label == "negative":
            return 0.5 - score * 0.5
        return 0.5  # neutral
    # list of dicts (top_k=None)
    d = {r["label"]: r["score"] for r in result}
    pos = d.get("positive", 0.0)
    neg = d.get("negative", 0.0)
    return float(np.clip(0.5 + (pos - neg) / 2.0, 0.0, 1.0))


def score_texts_finbert(
    texts: list[str], pipe, batch_size: int = 16
) -> list[float]:
    """Batch-score *texts* with FinBERT. Returns list of floats in [0, 1]."""
    PLACEHOLDER = "earnings results"
    safe, is_empty = [], []
    for t in texts:
        clean = (t or "").strip()[:512]
        safe.append(clean if clean else PLACEHOLDER)
        is_empty.append(not clean)

    raw = pipe(safe, batch_size=batch_size)
    scores = []
    for i, result in enumerate(raw):
        scores.append(0.5 if is_empty[i] else _finbert_result_to_score(result))
    return scores


def score_texts_vader(texts: list[str], sia) -> list[float]:
    """Score *texts* with VADER. Returns list of floats in [0, 1]."""
    scores = []
    for t in texts:
        clean = (t or "").strip()
        if not clean:
            scores.append(0.5)
        else:
            compound = sia.polarity_scores(clean)["compound"]
            scores.append(float((compound + 1.0) / 2.0))
    return scores


# ── news fetching ─────────────────────────────────────────────────────────────

def fetch_news(ticker: str, max_articles: int = 15) -> list[dict]:
    """
    Fetch recent news headlines for *ticker* via ``yfinance.Ticker.news``.

    Returns
    -------
    list of dicts with keys: ``title``, ``publisher``, ``age_days``
    """
    import yfinance as yf
    now = datetime.datetime.utcnow().timestamp()
    try:
        raw = yf.Ticker(ticker).news or []
    except Exception:
        return []
    articles = []
    for item in raw[:max_articles]:
        title = (item.get("title") or "").strip()
        if not title:
            continue
        pub_ts = item.get("providerPublishTime", now)
        articles.append({
            "title"    : title,
            "publisher": item.get("publisher", ""),
            "age_days" : max(0.0, (now - pub_ts) / 86_400),
        })
    return articles


# ── score aggregation ─────────────────────────────────────────────────────────

_DECAY = 0.12   # exponential recency decay constant (per day)


def _recency_weight(age_days: float) -> float:
    return float(np.exp(-_DECAY * age_days))


def aggregate(articles: list[dict], scores: list[float]) -> float:
    """
    Recency-weighted average of headline scores.
    More recent articles receive higher weight.
    Returns 0.5 (neutral) when no articles are available.
    """
    if not articles or not scores:
        return 0.5
    weights = [_recency_weight(a["age_days"]) for a in articles]
    total_w = sum(weights)
    if total_w < 1e-9:
        return float(np.mean(scores))
    return float(np.clip(
        sum(w * s for w, s in zip(weights, scores)) / total_w,
        0.0, 1.0
    ))


# ── signal labels ─────────────────────────────────────────────────────────────

_SIGNAL_THRESHOLDS = [
    (0.70, "Very Positive"),
    (0.55, "Positive"),
    (0.45, "Neutral"),
    (0.30, "Negative"),
    (0.00, "Very Negative"),
]


def sentiment_signal(score: float) -> str:
    """Map a 0-1 composite score to a human-readable signal label."""
    for threshold, label in _SIGNAL_THRESHOLDS:
        if score >= threshold:
            return label
    return "Very Negative"


# ── main public function ──────────────────────────────────────────────────────

def analyze_sentiment(
    tickers: list[str],
    model: str = "auto",
    max_articles: int = 15,
    verbose: bool = True,
    delay: float = 0.05,
) -> dict[str, dict]:
    """
    Fetch news and score sentiment for each ticker.

    Parameters
    ----------
    tickers      : list of ticker symbols
    model        : ``"auto"`` | ``"finbert"`` | ``"vader"``
    max_articles : max headlines per ticker (default 15)
    verbose      : print progress
    delay        : seconds to sleep between news fetches (rate-limit courtesy)

    Returns
    -------
    dict  ticker -> {
        "headlines"  : list[str],
        "publishers" : list[str],
        "raw_scores" : list[float],   # per-headline 0-1 scores
        "composite"  : float,         # recency-weighted aggregate 0-1
        "signal"     : str,           # "Very Positive" … "Very Negative"
        "n_articles" : int,
        "method"     : str,           # "finbert" or "vader"
    }
    """
    method, model_obj = load_model(model)

    if verbose:
        print(f"  Backend  : {method.upper()}")
        print(f"  Tickers  : {len(tickers)}")
        print(f"  Articles : up to {max_articles} per stock\n")

    # ── Step 1: fetch all news ────────────────────────────────────────────────
    if verbose:
        print("  Fetching headlines...")
    ticker_news: dict[str, list[dict]] = {}
    for ticker in tickers:
        ticker_news[ticker] = fetch_news(ticker, max_articles=max_articles)
        time.sleep(delay)

    # ── Step 2: batch-score all headlines at once ─────────────────────────────
    # Collect (ticker, local_idx, text) tuples
    all_items: list[tuple[str, int]] = []
    all_texts: list[str] = []
    for ticker, articles in ticker_news.items():
        for i, a in enumerate(articles):
            all_items.append((ticker, i))
            all_texts.append(a["title"])

    total = len(all_texts)
    if verbose:
        print(f"  Scoring {total} headlines with {method.upper()}...")

    if all_texts:
        if method == "finbert":
            all_scores = score_texts_finbert(all_texts, model_obj)
        else:
            all_scores = score_texts_vader(all_texts, model_obj)
    else:
        all_scores = []

    # ── Step 3: map scores back to tickers ───────────────────────────────────
    per_ticker_scores: dict[str, list[float]] = {t: [] for t in tickers}
    for (ticker, _), score in zip(all_items, all_scores):
        per_ticker_scores[ticker].append(score)

    # ── Step 4: aggregate & build result dict ─────────────────────────────────
    results: dict[str, dict] = {}
    for ticker in tickers:
        articles = ticker_news[ticker]
        scores   = per_ticker_scores[ticker]
        comp     = aggregate(articles, scores)
        results[ticker] = {
            "headlines"  : [a["title"]     for a in articles],
            "publishers" : [a["publisher"] for a in articles],
            "raw_scores" : scores,
            "composite"  : comp,
            "signal"     : sentiment_signal(comp),
            "n_articles" : len(articles),
            "method"     : method,
        }

    if verbose:
        print()
        print(f"  {'Ticker':<8} {'Score':>6}  {'Signal':<15}  {'N':>3}  Bar")
        print("  " + "-" * 56)
        for ticker, r in results.items():
            bar = "\u2588" * round(r["composite"] * 20)
            print(
                f"  {ticker:<8} {r['composite']:>6.3f}  "
                f"{r['signal']:<15}  {r['n_articles']:>3}  {bar}"
            )

    return results
