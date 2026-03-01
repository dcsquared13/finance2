"""
lib.display
===========
Formatting, terminal output, and matplotlib/pandas-based visualisation.

Public API
----------
Terminal output
    print_summary_table(results)
    print_detailed(results, top_n=10)

Matplotlib charts (return Figure objects for notebook embedding)
    plot_scores(results, top_n=50)          -> Figure
    plot_signal_distribution(results)       -> Figure
    plot_component_heatmap(results, top_n)  -> Figure
    plot_score_gauge(result)                -> Figure

Utilities
    results_to_dataframe(results)           -> pd.DataFrame
    save_csv(results, path)
"""

from __future__ import annotations

from collections import Counter
from typing import Sequence

import pandas as pd

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────

SIGNAL_COLORS_ANSI = {
    "STRONG BUY":  "\033[92m",
    "BUY":         "\033[32m",
    "NEUTRAL":     "\033[33m",
    "SELL":        "\033[31m",
    "STRONG SELL": "\033[91m",
}
RESET = "\033[0m"

# Matplotlib colour map for signals
SIGNAL_COLORS_MPL = {
    "STRONG BUY":  "#27ae60",
    "BUY":         "#2ecc71",
    "NEUTRAL":     "#f39c12",
    "SELL":        "#e74c3c",
    "STRONG SELL": "#c0392b",
}

SIGNAL_ORDER = ["STRONG BUY", "BUY", "NEUTRAL", "SELL", "STRONG SELL"]

COMPONENT_LABELS = {
    "score_rsi":   "RSI",
    "score_macd":  "MACD",
    "score_boll":  "Bollinger",
    "score_ma":    "Moving Avgs",
    "score_stoch": "Stochastic",
    "score_mom":   "Momentum",
    "score_vol":   "Volume",
}


# ─────────────────────────────────────────────
# TERMINAL OUTPUT
# ─────────────────────────────────────────────

def _bar(score: float, width: int = 20) -> str:
    filled = round(score * width)
    return "█" * filled + "░" * (width - filled)


def print_summary_table(results: list[dict]) -> None:
    """Print a colour-coded ranked table to stdout."""
    sep = "─" * 98
    print(f"\n{sep}")
    print(
        f" {'#':>2}  {'TICKER':<7} {'PRICE':>8}  {'SCORE':>6}  "
        f"{'BAR':^22}  {'SIGNAL':<12}  {'RSI':>5}  "
        f"{'MOM5%':>6}  {'MOM20%':>7}  {'VOL×':>5}"
    )
    print(sep)

    for i, r in enumerate(results, 1):
        sig   = r["signal"]
        color = SIGNAL_COLORS_ANSI.get(sig, "")
        flag  = " ★" if r.get("golden_cross") else "  "
        print(
            f" {i:>2}  {r['ticker']:<7} ${r['price']:>7.2f}  "
            f"{r['score']:.4f}  [{_bar(r['score'])}]  "
            f"{color}{sig:<12}{RESET}  "
            f"{r['rsi']:>5.1f}  "
            f"{r['mom5']:>+6.1f}%  "
            f"{r['mom20']:>+6.1f}%  "
            f"{r['vol_trend']:>5.2f}×{flag}"
        )

    print(sep)
    print(" ★ = Golden Cross (SMA50 > SMA200)\n")


def print_detailed(results: list[dict], top_n: int = 10) -> None:
    """Print a per-indicator breakdown for the top N stocks."""
    print(f"\n{'═' * 62}")
    print(f"  DETAILED BREAKDOWN — TOP {top_n} STOCKS")
    print(f"{'═' * 62}")

    for r in results[:top_n]:
        sig   = r["signal"]
        color = SIGNAL_COLORS_ANSI.get(sig, "")
        print(
            f"\n  {r['ticker']}  ${r['price']:.2f}  →  "
            f"{color}Score: {r['score']:.4f}  [{sig}]{RESET}"
        )
        print(f"  {'─' * 42}")
        components = [
            ("RSI",          r["score_rsi"],   f"RSI={r['rsi']}"),
            ("MACD",         r["score_macd"],  ""),
            ("Bollinger %B", r["score_boll"],  f"%B={r['bb_pct_b']:.2f}"),
            ("Moving Avgs",  r["score_ma"],
             f"↑20={'✓' if r['above_sma20'] else '✗'} "
             f"↑50={'✓' if r['above_sma50'] else '✗'} "
             f"↑200={'✓' if r['above_sma200'] else '✗'}"),
            ("Stochastic",   r["score_stoch"], ""),
            ("Momentum",     r["score_mom"],   f"5d={r['mom5']:+.1f}%  20d={r['mom20']:+.1f}%"),
            ("Volume",       r["score_vol"],   f"×{r['vol_trend']:.2f}"),
        ]
        for name, score, note in components:
            bar      = _bar(score, width=14)
            note_str = f"  ({note})" if note else ""
            print(f"  {name:<22} {score:.3f}  [{bar}]{note_str}")


def print_signal_distribution(results: list[dict]) -> None:
    """Print a simple text summary of signal counts."""
    dist = Counter(r["signal"] for r in results)
    print("\n  Signal distribution:")
    for sig in SIGNAL_ORDER:
        count = dist.get(sig, 0)
        bar   = "■" * count
        color = SIGNAL_COLORS_ANSI.get(sig, "")
        print(f"    {color}{sig:<12}{RESET}  {bar}  ({count})")
    print()


# ─────────────────────────────────────────────
# MATPLOTLIB CHARTS
# ─────────────────────────────────────────────

def _mpl():
    """Lazy-import matplotlib so the module works without it."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        return plt, mpatches
    except ImportError:
        raise ImportError("matplotlib is required for charting: pip install matplotlib")


def plot_scores(results: list[dict], top_n: int = 50):
    """
    Horizontal bar chart of probability scores for all stocks.

    Returns a matplotlib Figure.
    """
    plt, mpatches = _mpl()

    data     = results[:top_n]
    tickers  = [r["ticker"] for r in data]
    scores   = [r["score"]  for r in data]
    signals  = [r["signal"] for r in data]
    colors   = [SIGNAL_COLORS_MPL.get(s, "#95a5a6") for s in signals]

    fig, ax = plt.subplots(figsize=(10, max(6, len(data) * 0.35)))
    bars = ax.barh(tickers[::-1], scores[::-1], color=colors[::-1], edgecolor="none")

    ax.set_xlabel("Probability Score (0 = Bearish → 1 = Bullish)", fontsize=11)
    ax.set_title("S&P 500 Top 50 — Upward Price Probability", fontsize=13, fontweight="bold")
    ax.axvline(0.5, color="#7f8c8d", linewidth=1, linestyle="--", alpha=0.7)
    ax.set_xlim(0, 1)

    # Score labels
    for bar, score in zip(bars, scores[::-1]):
        ax.text(
            min(score + 0.01, 0.98), bar.get_y() + bar.get_height() / 2,
            f"{score:.3f}", va="center", ha="left", fontsize=8, color="#2c3e50",
        )

    # Legend
    legend_patches = [
        mpatches.Patch(color=SIGNAL_COLORS_MPL[s], label=s)
        for s in SIGNAL_ORDER
    ]
    ax.legend(handles=legend_patches, loc="lower right", fontsize=9)

    fig.tight_layout()
    return fig


def plot_signal_distribution(results: list[dict]):
    """
    Pie chart of signal distribution.

    Returns a matplotlib Figure.
    """
    plt, mpatches = _mpl()

    dist   = Counter(r["signal"] for r in results)
    labels = [s for s in SIGNAL_ORDER if s in dist]
    sizes  = [dist[s] for s in labels]
    colors = [SIGNAL_COLORS_MPL[s] for s in labels]

    fig, ax = plt.subplots(figsize=(6, 6))
    wedges, texts, autotexts = ax.pie(
        sizes,
        labels=labels,
        colors=colors,
        autopct="%1.0f%%",
        startangle=140,
        pctdistance=0.80,
        wedgeprops={"linewidth": 1.5, "edgecolor": "white"},
    )
    for t in autotexts:
        t.set_fontsize(10)
        t.set_color("white")
        t.set_fontweight("bold")

    ax.set_title("Signal Distribution", fontsize=13, fontweight="bold")
    fig.tight_layout()
    return fig


def plot_component_heatmap(results: list[dict], top_n: int = 20):
    """
    Heatmap of per-indicator scores for the top N stocks.

    Returns a matplotlib Figure.
    """
    plt, _ = _mpl()
    try:
        import matplotlib.colors as mcolors
    except ImportError:
        raise ImportError("matplotlib is required: pip install matplotlib")

    data     = results[:top_n]
    tickers  = [r["ticker"] for r in data]
    comp_keys = list(COMPONENT_LABELS.keys())
    comp_names = list(COMPONENT_LABELS.values())

    matrix = [[r[k] for k in comp_keys] for r in data]

    fig, ax = plt.subplots(figsize=(11, max(5, len(data) * 0.45)))
    cmap = plt.cm.RdYlGn
    im   = ax.imshow(matrix, cmap=cmap, aspect="auto", vmin=0, vmax=1)

    ax.set_xticks(range(len(comp_names)))
    ax.set_xticklabels(comp_names, rotation=30, ha="right", fontsize=10)
    ax.set_yticks(range(len(tickers)))
    ax.set_yticklabels(tickers, fontsize=10)

    for i, row in enumerate(matrix):
        for j, val in enumerate(row):
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=8, color="black" if 0.3 < val < 0.8 else "white")

    fig.colorbar(im, ax=ax, label="Component Score")
    ax.set_title(f"Indicator Heatmap — Top {top_n} Stocks", fontsize=13, fontweight="bold")
    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────
# DATA UTILITIES
# ─────────────────────────────────────────────

def results_to_dataframe(results: list[dict]) -> pd.DataFrame:
    """Convert a results list to a tidy pandas DataFrame."""
    return pd.DataFrame(results)


def save_csv(results: list[dict], path: str = "sp500_analysis.csv") -> None:
    """Save results to a CSV file."""
    df = results_to_dataframe(results)
    df.to_csv(path, index=False)
    print(f"  ✓ Results saved → {path}")
