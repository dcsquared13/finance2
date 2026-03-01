#!/usr/bin/env python3
"""
S&P 500 Top 50 Stock Probability Analyzer
==========================================
Analyzes the top 50 S&P 500 stocks using:
  - Technical indicators  (RSI, MACD, Bollinger Bands, MA, Stochastic, Momentum, Volume)
  - Fundamental metrics   (P/E, P/B, PEG, ROE, growth, margins, D/E, FCF yield, analyst)
  - AI Sentiment analysis (FinBERT primary / NLTK VADER fallback)

Scores all three dimensions 0→1 (higher = stronger buy signal) and prints
a side-by-side comparison table + optionally saves charts to PNG.

Usage:
    pip install yfinance pandas numpy matplotlib requests lxml html5lib
    pip install transformers torch          # for FinBERT (recommended)
    pip install nltk                        # for VADER (lightweight fallback)
    python sp500_analyzer.py
"""

import sys
import os
import warnings
import time
from datetime import datetime
from collections import Counter

warnings.filterwarnings("ignore")

# ── Ensure lib/ is importable ─────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── Core deps ─────────────────────────────────────────────────────────────────
try:
    import numpy as np
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")          # non-interactive backend; charts saved to PNG
    import matplotlib.pyplot as plt
except ImportError:
    print("Installing core dependencies …")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install",
                           "numpy", "pandas", "matplotlib", "--quiet"])
    import numpy as np
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

# ── lib package ───────────────────────────────────────────────────────────────
try:
    from lib import data, scoring, fundamentals, sentiment
except ImportError as e:
    print(f"\n[ERROR] Cannot import lib package: {e}")
    print("  Make sure lib/ lives in the same directory as this script.")
    sys.exit(1)

# ═════════════════════════════════════════════════════════════════════════════
# CONFIGURATION — edit these to taste
# ═════════════════════════════════════════════════════════════════════════════

# Stock universe — choose ONE option:
#   "live"   → fetch current top-N S&P 500 by market cap (needs network)
#   "static" → use the hardcoded TOP_50_SP500 list in lib/data.py
#   "custom" → use CUSTOM_TICKERS list below
UNIVERSE_MODE  = "live"
TOP_N          = 50
CUSTOM_TICKERS = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL"]

# Tech / Fundamental blend (must sum to 1.0)
TECH_WEIGHT = scoring.TECH_WEIGHT    # default 0.60
FUND_WEIGHT = scoring.FUND_WEIGHT    # default 0.40

# Sentiment model
#   "auto"    → tries FinBERT, falls back to VADER if torch not installed
#   "finbert" → ProsusAI/finbert (~400 MB download on first run, then cached)
#   "vader"   → NLTK VADER (lightweight rule-based, no large download)
SENTIMENT_MODEL = "auto"
MAX_ARTICLES    = 15     # headlines per stock

# History
LOOKBACK_DAYS = 400      # calendar days of OHLCV data

# Output
SAVE_CHARTS  = True      # save PNG charts alongside the script
CHART_DIR    = "."       # directory for PNG files
SAVE_CSV     = True
CSV_FILENAME = "sp500_analysis_results.csv"

# ═════════════════════════════════════════════════════════════════════════════
# TERMINAL COLOURS
# ═════════════════════════════════════════════════════════════════════════════

TF_COLOR = {
    "Strong Buy":   "\033[92m",
    "Buy":          "\033[32m",
    "Neutral":      "\033[33m",
    "Sell":         "\033[31m",
    "Strong Sell":  "\033[91m",
    # legacy uppercase variants from lib/scoring
    "STRONG BUY":   "\033[92m",
    "BUY":          "\033[32m",
    "NEUTRAL":      "\033[33m",
    "SELL":         "\033[31m",
    "STRONG SELL":  "\033[91m",
}
SENT_COLOR = {
    "Very Positive": "\033[94m",
    "Positive":      "\033[34m",
    "Neutral":       "\033[33m",
    "Negative":      "\033[31m",
    "Very Negative": "\033[91m",
}
RESET = "\033[0m"
SEP   = "-" * 100

# Chart colours (matplotlib)
SIGNAL_COLORS = {
    "Strong Buy":   "#2e7d32",
    "Buy":          "#66bb6a",
    "Neutral":      "#ffa726",
    "Sell":         "#ef5350",
    "Strong Sell":  "#b71c1c",
    "STRONG BUY":   "#2e7d32",
    "BUY":          "#66bb6a",
    "NEUTRAL":      "#ffa726",
    "SELL":         "#ef5350",
    "STRONG SELL":  "#b71c1c",
}
SENT_COLORS = {
    "Very Positive": "#1a5276",
    "Positive":      "#2980b9",
    "Neutral":       "#95a5a6",
    "Negative":      "#e74c3c",
    "Very Negative": "#7b241c",
    "N/A":           "#bdc3c7",
}


def score_bar(score: float, width: int = 20) -> str:
    filled = round(score * width)
    return "█" * filled + "░" * (width - filled)


def _save_or_show(fig, filename: str) -> None:
    if SAVE_CHARTS:
        path = os.path.join(CHART_DIR, filename)
        fig.savefig(path, dpi=120, bbox_inches="tight")
        print(f"  [chart] Saved → {path}")
    plt.close(fig)


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 1 — Print header
# ═════════════════════════════════════════════════════════════════════════════

def print_header(tickers):
    print("\n" + "=" * 70)
    print("  S&P 500 TOP 50 — STOCK PROBABILITY ANALYZER")
    print(f"  Run date : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Universe : {len(tickers)} stocks")
    print(f"  Blend    : {TECH_WEIGHT:.0%} Technical + {FUND_WEIGHT:.0%} Fundamental")
    print(f"  Sentiment: {SENTIMENT_MODEL.upper()} model | {MAX_ARTICLES} headlines/stock")
    print("=" * 70)
    print(f"\n  Indicators: RSI · MACD · Bollinger · MA · Stochastic · Momentum · Volume")
    print(f"  Fundamentals: P/E · P/B · PEG · ROE · Growth · Margins · D/E · FCF · Analyst")
    print(f"  Sentiment: FinBERT (ProsusAI) / VADER fallback\n")


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 2 — Ranked summary table (terminal)
# ═════════════════════════════════════════════════════════════════════════════

def print_summary_table(results: dict, sent_results: dict) -> None:
    print("\n" + SEP)
    print(
        f"  {'#':>2}  {'TICKER':<7} {'PRICE':>8}  "
        f"{'TECH':>6}  {'FUND':>6}  {'TECH+FUND':>9}  {'TF SIGNAL':<12}  "
        f"{'SENTIMENT':>9}  {'SENT SIGNAL':<14}"
    )
    print(SEP)
    for rank, (ticker, r) in enumerate(results.items(), 1):
        s   = sent_results.get(ticker, {})
        sig = r.get("signal", "N/A")
        ssg = s.get("signal", "N/A")
        tc  = TF_COLOR.get(sig, "")
        sc  = SENT_COLOR.get(ssg, "")
        print(
            f"  {rank:>2}  {ticker:<7} ${r.get('price', 0):>7.2f}  "
            f"{r.get('tech_composite', 0):>6.3f}  {r.get('fund_composite', 0):>6.3f}  "
            f"{r.get('composite', 0):>9.3f}  {tc}{sig:<12}{RESET}  "
            f"{s.get('composite', 0.5):>9.3f}  {sc}{ssg:<14}{RESET}"
        )
    print(SEP)


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 3 — Detailed breakdown for top-N stocks
# ═════════════════════════════════════════════════════════════════════════════

def print_detailed(results: dict, sent_results: dict, top_n: int = 10) -> None:
    print("\n" + "=" * 70)
    print(f"  DETAILED BREAKDOWN — TOP {top_n} STOCKS")
    print("=" * 70)

    fund_names = {
        "pe": "P/E ratio",           "pb": "Price/Book",
        "peg": "PEG ratio",          "revenue_growth": "Revenue growth",
        "earnings_growth": "Earnings growth", "roe": "Return on equity",
        "profit_margin": "Profit margin",    "debt_equity": "Debt/Equity",
        "fcf_yield": "FCF yield",    "analyst": "Analyst rating",
    }

    for rank, (ticker, r) in enumerate(list(results.items())[:top_n], 1):
        s   = sent_results.get(ticker, {})
        sig = r.get("signal", "N/A")
        color = TF_COLOR.get(sig, "")

        print(f"\n  {rank}. {ticker}  ${r.get('price', 0):.2f}"
              f"  →  {color}Tech+Fund={r.get('composite', 0):.4f}  [{sig}]{RESET}"
              f"  |  Sentiment={s.get('composite', 0.5):.4f}  [{s.get('signal', 'N/A')}]")
        print(f"     {'─'*64}")

        # Technical components
        print("     Technical:")
        for k, v in r.get("tech_scores", {}).items():
            print(f"       {k:<22} {v:.3f}  {score_bar(v, 12)}")

        # Fundamental components
        print("     Fundamental:")
        for k, v in r.get("fund_scores", {}).items():
            label = fund_names.get(k, k)
            print(f"       {label:<22} {v:.3f}  {score_bar(v, 12)}")

        # Sentiment headlines
        method = s.get("method", "").upper()
        print(f"     Sentiment ({method}) — {s.get('n_articles', 0)} articles:")
        for headline, score in zip(s.get("headlines", [])[:5], s.get("raw_scores", [])[:5]):
            marker = "▲" if score >= 0.55 else ("▼" if score <= 0.45 else "■")
            sc_color = "\033[32m" if score >= 0.55 else ("\033[31m" if score <= 0.45 else "\033[33m")
            print(f"       {sc_color}{marker} [{score:.2f}]{RESET} {headline[:75]}")


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 4 — Sentiment explorer (most positive / most negative)
# ═════════════════════════════════════════════════════════════════════════════

def print_sentiment_explorer(sent_results: dict, top: int = 5, bottom: int = 3) -> None:
    sent_sorted = sorted(sent_results.items(), key=lambda x: x[1]["composite"], reverse=True)

    print("\n★ Most Positive Sentiment")
    print("=" * 70)
    for ticker, r in sent_sorted[:top]:
        print(f"\n  {ticker}  score={r['composite']:.3f}  signal={r['signal']}"
              f"  n={r['n_articles']}  backend={r.get('method', 'N/A').upper()}")
        for headline, score in zip(r["headlines"][:4], r["raw_scores"][:4]):
            m = "▲" if score >= 0.55 else ("▼" if score <= 0.45 else "■")
            print(f"    {m} [{score:.2f}] {headline[:78]}")

    print("\n★ Most Negative Sentiment")
    print("=" * 70)
    for ticker, r in sent_sorted[-bottom:]:
        print(f"\n  {ticker}  score={r['composite']:.3f}  signal={r['signal']}")
        for headline, score in zip(r["headlines"][:4], r["raw_scores"][:4]):
            m = "▲" if score >= 0.55 else ("▼" if score <= 0.45 else "■")
            print(f"    {m} [{score:.2f}] {headline[:78]}")


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 5 — Signal distribution summary
# ═════════════════════════════════════════════════════════════════════════════

def print_signal_distribution(results: dict, sent_results: dict) -> None:
    tf_dist   = Counter(r.get("signal", "N/A") for r in results.values())
    sent_dist = Counter(v.get("signal", "N/A") for v in sent_results.values())

    scores    = [r.get("composite", 0) for r in results.values()]
    sscores   = [v.get("composite", 0.5) for v in sent_results.values()]

    print("\n" + "─" * 50)
    print(f"  Stocks analysed  : {len(results)}")
    print(f"  Avg Tech+Fund    : {np.mean(scores):.4f}")
    print(f"  Avg Sentiment    : {np.mean(sscores):.4f}")
    print("─" * 50)

    print("\n  Tech+Fund Signal distribution:")
    for sig in ["Strong Buy","STRONG BUY","Buy","BUY","Neutral","NEUTRAL","Sell","SELL","Strong Sell","STRONG SELL"]:
        count = tf_dist.get(sig, 0)
        if count:
            color = TF_COLOR.get(sig, "")
            print(f"    {color}{sig:<14}{RESET}  {'|' * count} ({count})")

    print("\n  Sentiment Signal distribution:")
    for sig in ["Very Positive","Positive","Neutral","Negative","Very Negative","N/A"]:
        count = sent_dist.get(sig, 0)
        if count:
            color = SENT_COLOR.get(sig, "")
            print(f"    {color}{sig:<16}{RESET}  {'|' * count} ({count})")
    print("─" * 50)


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 6 — Charts
# ═════════════════════════════════════════════════════════════════════════════

def chart_side_by_side(results: dict, sent_results: dict) -> None:
    """Two-panel horizontal bar: Tech+Fund (left) | Sentiment (right)."""
    tickers    = list(results.keys())
    tf_sigs    = [results[t].get("signal", "Neutral") for t in tickers]
    sent_sigs  = [sent_results.get(t, {}).get("signal", "Neutral") for t in tickers]
    comb       = [results[t].get("composite", 0) for t in tickers]
    sent_vals  = [sent_results.get(t, {}).get("composite", 0.5) for t in tickers]

    fig, axes = plt.subplots(1, 2, figsize=(20, max(10, len(tickers) * 0.38)))

    # Left: Tech+Fund
    ax = axes[0]
    tf_cols = [SIGNAL_COLORS.get(s, "#ffa726") for s in tf_sigs]
    bars = ax.barh(tickers[::-1], comb[::-1], color=tf_cols[::-1], alpha=0.88)
    ax.axvline(x=0.75, color="#2e7d32", linestyle="--", alpha=0.5, label="Strong Buy (0.75)")
    ax.axvline(x=0.60, color="#66bb6a", linestyle="--", alpha=0.5, label="Buy (0.60)")
    ax.axvline(x=0.40, color="#ffa726", linestyle="--", alpha=0.5, label="Neutral (0.40)")
    for bar, score in zip(bars, comb[::-1]):
        ax.text(min(score + 0.01, 0.97), bar.get_y() + bar.get_height() / 2,
                f"{score:.3f}", va="center", fontsize=7)
    ax.set_xlabel("Tech+Fund Composite Score", fontsize=11)
    ax.set_title(f"Tech+Fund Score ({TECH_WEIGHT:.0%} Tech + {FUND_WEIGHT:.0%} Fund)",
                 fontsize=13, fontweight="bold")
    ax.set_xlim(0, 1)
    ax.legend(loc="lower right", fontsize=8)

    # Right: Sentiment
    ax2 = axes[1]
    s_cols = [SENT_COLORS.get(s, "#95a5a6") for s in sent_sigs]
    bars2 = ax2.barh(tickers[::-1], sent_vals[::-1], color=s_cols[::-1], alpha=0.88)
    ax2.axvline(x=0.70, color="#1a5276", linestyle="--", alpha=0.5, label="Very Positive (0.70)")
    ax2.axvline(x=0.55, color="#2980b9", linestyle="--", alpha=0.5, label="Positive (0.55)")
    ax2.axvline(x=0.45, color="#95a5a6", linestyle="--", alpha=0.5, label="Neutral (0.45)")
    ax2.axvline(x=0.30, color="#e74c3c", linestyle="--", alpha=0.5, label="Negative (0.30)")
    for bar, score in zip(bars2, sent_vals[::-1]):
        ax2.text(min(score + 0.01, 0.97), bar.get_y() + bar.get_height() / 2,
                 f"{score:.3f}", va="center", fontsize=7)
    ax2.set_xlabel("Sentiment Score (AI)", fontsize=11)
    ax2.set_title("News Sentiment Score (FinBERT / VADER)", fontsize=13, fontweight="bold")
    ax2.set_xlim(0, 1)
    ax2.set_yticklabels([])
    ax2.legend(loc="lower right", fontsize=8)

    plt.suptitle("Tech+Fund vs Sentiment — Side-by-Side", fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()
    _save_or_show(fig, "chart_side_by_side.png")


def chart_three_way(results: dict, sent_results: dict) -> None:
    """Grouped bar (Technical / Fundamental / Sentiment) + signal distribution."""
    tickers    = list(results.keys())
    tech_sc    = [results[t].get("tech_composite", 0) for t in tickers]
    fund_sc    = [results[t].get("fund_composite", 0) for t in tickers]
    sent_sc    = [sent_results.get(t, {}).get("composite", 0.5) for t in tickers]
    tf_sigs    = [results[t].get("signal", "Neutral") for t in tickers]
    sent_sigs  = [sent_results.get(t, {}).get("signal", "Neutral") for t in tickers]

    fig, axes = plt.subplots(2, 1, figsize=(18, 12))

    # Top: grouped bar
    ax = axes[0]
    x = np.arange(len(tickers))
    w = 0.28
    ax.bar(x - w, tech_sc, w, label="Technical",   color="#1565c0", alpha=0.85)
    ax.bar(x,     fund_sc, w, label="Fundamental", color="#e65100", alpha=0.85)
    ax.bar(x + w, sent_sc, w, label="Sentiment",   color="#6a1b9a", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(tickers, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Score (0–1)")
    ax.set_title("Technical vs Fundamental vs Sentiment Scores", fontsize=13, fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.axhline(0.60, color="gray", linestyle="--", alpha=0.4)
    ax.legend(loc="upper right")

    # Bottom: signal distribution
    ax2 = axes[1]
    tf_dist   = Counter(tf_sigs)
    sent_dist = Counter(sent_sigs)
    tf_order   = ["Strong Buy", "STRONG BUY", "Buy", "BUY", "Neutral", "NEUTRAL",
                  "Sell", "SELL", "Strong Sell", "STRONG SELL"]
    sent_order = ["Very Positive", "Positive", "Neutral", "Negative", "Very Negative"]
    labels     = ["Very Pos /\nStrong Buy", "Pos /\nBuy", "Neutral",
                  "Neg /\nSell", "Very Neg /\nStrong Sell"]
    tf_counts   = [sum(tf_dist.get(s, 0) for s in tf_order[i*2:(i+1)*2]) for i in range(5)]
    sent_counts = [sent_dist.get(s, 0) for s in sent_order]
    tf_chart_colors   = ["#2e7d32", "#66bb6a", "#ffa726", "#ef5350", "#b71c1c"]
    sent_chart_colors = [SENT_COLORS[s] for s in sent_order]
    x2 = np.arange(5)
    ax2.bar(x2 - 0.2, tf_counts,   0.38, label="Tech+Fund", color=tf_chart_colors,   alpha=0.9)
    ax2.bar(x2 + 0.2, sent_counts, 0.38, label="Sentiment", color=sent_chart_colors, alpha=0.75)
    ax2.set_xticks(x2)
    ax2.set_xticklabels(labels, fontsize=9)
    ax2.set_ylabel("Number of Stocks")
    ax2.set_title("Signal Distribution — Tech+Fund vs Sentiment", fontsize=13, fontweight="bold")
    ax2.legend()

    plt.tight_layout()
    _save_or_show(fig, "chart_three_way.png")


def chart_tech_heatmap(results: dict) -> None:
    """Heatmap of 7 technical component scores."""
    tickers      = list(results.keys())
    components   = ["rsi", "macd", "bollinger", "moving_averages", "stochastic", "momentum", "volume"]
    rows_data    = [[results[t].get("tech_scores", {}).get(c, 0.5) for c in components]
                    for t in tickers]
    matrix = np.array(rows_data)

    fig, ax = plt.subplots(figsize=(12, max(8, len(tickers) * 0.35)))
    im = ax.imshow(matrix, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(components)))
    ax.set_xticklabels([c.replace("_", "\n") for c in components], fontsize=9)
    ax.set_yticks(range(len(tickers)))
    ax.set_yticklabels(tickers, fontsize=8)
    for i in range(len(tickers)):
        for j in range(len(components)):
            val = matrix[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7,
                    color="black" if 0.25 < val < 0.75 else "white")
    plt.colorbar(im, ax=ax, shrink=0.6, label="Score (0=Bearish, 1=Bullish)")
    ax.set_title("Technical Indicator Heatmap", fontsize=13, fontweight="bold", pad=12)
    plt.tight_layout()
    _save_or_show(fig, "chart_tech_heatmap.png")


def chart_fund_heatmap(results: dict) -> None:
    """Heatmap of 10 fundamental component scores."""
    tickers    = list(results.keys())
    components = ["pe", "pb", "peg", "revenue_growth", "earnings_growth",
                  "roe", "profit_margin", "debt_equity", "fcf_yield", "analyst"]
    labels     = ["P/E", "P/B", "PEG", "Rev\nGrowth", "EPS\nGrowth",
                  "ROE", "Profit\nMargin", "D/E", "FCF\nYield", "Analyst"]
    rows_data  = [[results[t].get("fund_scores", {}).get(c, 0.5) for c in components]
                  for t in tickers]
    matrix = np.array(rows_data)

    fig, ax = plt.subplots(figsize=(14, max(8, len(tickers) * 0.35)))
    im = ax.imshow(matrix, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_yticks(range(len(tickers)))
    ax.set_yticklabels(tickers, fontsize=8)
    for i in range(len(tickers)):
        for j in range(len(components)):
            val = matrix[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7,
                    color="black" if 0.25 < val < 0.75 else "white")
    plt.colorbar(im, ax=ax, shrink=0.6, label="Score (0=Weak, 1=Strong)")
    ax.set_title("Fundamental Indicator Heatmap", fontsize=13, fontweight="bold", pad=12)
    plt.tight_layout()
    _save_or_show(fig, "chart_fund_heatmap.png")


def chart_sentiment_ranking(sent_results: dict) -> None:
    """Sentiment-ranked horizontal bar chart."""
    sent_sorted = sorted(sent_results.items(), key=lambda x: x[1]["composite"], reverse=True)
    tickers     = [t for t, _ in sent_sorted]
    vals        = [r["composite"] for _, r in sent_sorted]
    sigs        = [r["signal"]    for _, r in sent_sorted]

    fig, ax = plt.subplots(figsize=(12, max(8, len(tickers) * 0.35)))
    s_cols = [SENT_COLORS.get(s, "#95a5a6") for s in sigs]
    bars   = ax.barh(tickers[::-1], vals[::-1], color=s_cols[::-1], alpha=0.88)
    for bar, val in zip(bars, vals[::-1]):
        ax.text(min(val + 0.01, 0.97), bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=7)
    ax.axvline(x=0.70, color="#1a5276", linestyle="--", alpha=0.5, label="Very Positive (0.70)")
    ax.axvline(x=0.55, color="#2980b9", linestyle="--", alpha=0.5, label="Positive (0.55)")
    ax.axvline(x=0.45, color="#95a5a6", linestyle="--", alpha=0.5, label="Neutral (0.45)")
    ax.axvline(x=0.30, color="#e74c3c", linestyle="--", alpha=0.5, label="Negative (0.30)")
    ax.set_xlabel("Sentiment Score (AI-scored headlines)")
    ax.set_title("News Sentiment Ranking", fontsize=13, fontweight="bold")
    ax.set_xlim(0, 1)
    ax.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    _save_or_show(fig, "chart_sentiment_ranking.png")


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 7 — CSV export
# ═════════════════════════════════════════════════════════════════════════════

def save_csv(results: dict, sent_results: dict, filename: str) -> None:
    rows = []
    for rank, (ticker, r) in enumerate(results.items(), 1):
        s = sent_results.get(ticker, {})
        row = {
            "Rank":              rank,
            "Ticker":            ticker,
            "Price":             r.get("price"),
            "Tech_Composite":    round(r.get("tech_composite", 0), 4),
            "Fund_Composite":    round(r.get("fund_composite", 0), 4),
            "TechFund_Score":    round(r.get("composite", 0), 4),
            "TechFund_Signal":   r.get("signal", "N/A"),
            "Sentiment_Score":   round(s.get("composite", 0.5), 4),
            "Sentiment_Signal":  s.get("signal", "N/A"),
            "Sent_Articles":     s.get("n_articles", 0),
            "Sent_Backend":      s.get("method", "N/A"),
        }
        for k, v in r.get("tech_scores", {}).items():
            row[f"tech_{k}"] = round(v, 4)
        for k, v in r.get("fund_scores", {}).items():
            row[f"fund_{k}"] = round(v, 4)
        raw = r.get("fund_raw", {})
        for field in ["trailingPE", "priceToBook", "pegRatio", "revenueGrowth",
                      "earningsGrowth", "returnOnEquity", "profitMargins",
                      "debtToEquity", "recommendationMean"]:
            row[field] = raw.get(field)
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(filename, index=False)
    print(f"\n  Results saved → {filename}  ({len(df)} rows, {len(df.columns)} columns)")


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main():
    t0 = time.time()

    # ── 1. Stock universe ─────────────────────────────────────────────────────
    if UNIVERSE_MODE == "live":
        print("\n  Fetching live S&P 500 top-50 by market cap …")
        tickers = data.get_top_n_sp500(n=TOP_N, verbose=True)
    elif UNIVERSE_MODE == "static":
        tickers = data.TOP_50_SP500
    else:
        tickers = CUSTOM_TICKERS

    print_header(tickers)

    # ── 2. Technical analysis ─────────────────────────────────────────────────
    print("  ┌─ Step 1/3: Technical analysis ─────────────────────────────────┐")
    tech_results = scoring.analyze_universe(
        tickers,
        lookback_days=LOOKBACK_DAYS,
        verbose=True,
    )
    print(f"  └─ ✓ Technical analysis complete for {len(tech_results)} stocks.\n")

    # ── 3. Fundamental analysis ───────────────────────────────────────────────
    print("  ┌─ Step 2/3: Fundamental analysis ───────────────────────────────┐")
    fund_results = fundamentals.analyze_fundamentals(
        list(tech_results.keys()),
        verbose=True,
    )
    print(f"  └─ ✓ Fundamental analysis complete for {len(fund_results)} stocks.\n")

    # ── 4. Merge tech + fund ──────────────────────────────────────────────────
    results = scoring.merge_fundamental(
        tech_results,
        fund_results,
        tech_weight=TECH_WEIGHT,
        fund_weight=FUND_WEIGHT,
    )
    # Sort by combined composite (descending)
    results = dict(sorted(results.items(), key=lambda x: x[1]["composite"], reverse=True))

    # ── 5. AI Sentiment analysis ──────────────────────────────────────────────
    print("  ┌─ Step 3/3: AI Sentiment analysis ──────────────────────────────┐")
    sent_results = sentiment.analyze_sentiment(
        list(results.keys()),
        model=SENTIMENT_MODEL,
        max_articles=MAX_ARTICLES,
        verbose=True,
    )
    print(f"  └─ ✓ Sentiment analysis complete for {len(sent_results)} stocks.\n")

    # ── 6. Terminal output ────────────────────────────────────────────────────
    print_signal_distribution(results, sent_results)
    print_summary_table(results, sent_results)
    print_detailed(results, sent_results, top_n=10)
    print_sentiment_explorer(sent_results, top=5, bottom=3)

    # ── 7. Charts ─────────────────────────────────────────────────────────────
    if SAVE_CHARTS:
        print("\n  Generating charts …")
        chart_side_by_side(results, sent_results)
        chart_three_way(results, sent_results)
        chart_tech_heatmap(results)
        chart_fund_heatmap(results)
        chart_sentiment_ranking(sent_results)

    # ── 8. CSV export ─────────────────────────────────────────────────────────
    if SAVE_CSV:
        save_csv(results, sent_results, CSV_FILENAME)

    elapsed = time.time() - t0
    print(f"\n  Total runtime: {elapsed/60:.1f} min\n")
    print("=" * 70)
    print("  DISCLAIMER: For educational/informational purposes only.")
    print("  NOT financial advice. Consult a licensed financial advisor.")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
