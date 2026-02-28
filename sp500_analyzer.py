#!/usr/bin/env python3
"""
S&P 500 Top 50 Stock Probability Analyzer
==========================================
Analyzes the top 50 S&P 500 stocks using technical indicators and returns
a probability score (0-1) indicating the likelihood of price appreciation.

Indicators used:
  - RSI (Relative Strength Index)
  - MACD (Moving Average Convergence Divergence)
  - Bollinger Bands
  - Moving Averages (SMA20, SMA50, SMA200)
  - Volume Trend
  - Price Momentum (5d, 10d, 20d)
  - Stochastic Oscillator
  - ATR-based volatility context

Usage:
  pip install yfinance pandas numpy
  python sp500_analyzer.py
"""

import sys
import warnings
warnings.filterwarnings("ignore")

try:
    import yfinance as yf
except ImportError:
    print("Installing yfinance...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "yfinance", "--quiet"])
    import yfinance as yf

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time

TOP_50_SP500 = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AVGO",
    "BRK-B", "LLY", "JPM", "V", "UNH", "XOM", "MA", "COST", "HD",
    "PG", "JNJ", "WMT", "NFLX", "BAC", "ORCL", "KO", "CRM", "CVX",
    "AMD", "MRK", "ABBV", "PEP", "LIN", "TMO", "ACN", "ADBE", "CSCO",
    "MCD", "TXN", "NKE", "WFC", "DIS", "PM", "AMGN", "DHR", "IBM",
    "QCOM", "INTC", "LOW", "CAT", "NOW", "GE",
]

def calculate_rsi(prices, period=14):
    delta = prices.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period-1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period-1, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return float(rsi.iloc[-1]) if not rsi.empty else 50.0

def calculate_macd(prices):
    ema12 = prices.ewm(span=12, adjust=False).mean()
    ema26 = prices.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    histogram = macd_line - signal_line
    return float(macd_line.iloc[-1]), float(signal_line.iloc[-1]), float(histogram.iloc[-1])

def calculate_bollinger_bands(prices, period=20):
    sma = prices.rolling(window=period).mean()
    std = prices.rolling(window=period).std()
    upper = sma + 2 * std
    lower = sma - 2 * std
    pct_b = (prices - lower) / (upper - lower).replace(0, np.nan)
    return float(upper.iloc[-1]), float(sma.iloc[-1]), float(lower.iloc[-1]), float(pct_b.iloc[-1]) if not pct_b.empty else 0.5

def calculate_sma(prices, period):
    sma = prices.rolling(window=period).mean()
    return float(sma.iloc[-1]) if len(prices) >= period else float(prices.mean())

def calculate_stochastic(high, low, close, k_period=14, d_period=3):
    lowest_low = low.rolling(window=k_period).min()
    highest_high = high.rolling(window=k_period).max()
    range_ = (highest_high - lowest_low).replace(0, np.nan)
    pct_k = 100 * (close - lowest_low) / range_
    pct_d = pct_k.rolling(window=d_period).mean()
    return (float(pct_k.iloc[-1]) if not pct_k.empty else 50.0,
            float(pct_d.iloc[-1]) if not pct_d.empty else 50.0)

def calculate_volume_trend(volume, period=20):
    if len(volume) < period:
        return 1.0
    recent_vol = float(volume.iloc[-5:].mean())
    avg_vol = float(volume.iloc[-period:].mean())
    return recent_vol / avg_vol if avg_vol > 0 else 1.0

def calculate_momentum(prices, period):
    if len(prices) < period + 1:
        return 0.0
    return float((prices.iloc[-1] / prices.iloc[-period] - 1) * 100)

def calculate_atr(high, low, close, period=14):
    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    return float(atr.iloc[-1]) if not atr.empty else 0.0

def score_rsi(rsi):
    if rsi <= 20: return 0.90
    elif rsi <= 30: return 0.80
    elif rsi <= 40: return 0.65
    elif rsi <= 55: return 0.55
    elif rsi <= 65: return 0.45
    elif rsi <= 70: return 0.35
    elif rsi <= 80: return 0.20
    else: return 0.10

def score_macd(macd, signal, histogram):
    score = 0.5
    score += 0.15 if macd > signal else -0.15
    score += 0.10 if histogram > 0 else -0.10
    score += 0.05 if macd > 0 else -0.05
    return float(np.clip(score, 0.0, 1.0))

def score_bollinger(pct_b):
    if np.isnan(pct_b): return 0.5
    if pct_b < 0: return 0.85
    elif pct_b < 0.2: return 0.70
    elif pct_b < 0.4: return 0.60
    elif pct_b < 0.6: return 0.50
    elif pct_b < 0.8: return 0.40
    elif pct_b < 1.0: return 0.30
    else: return 0.20

def score_moving_averages(price, sma20, sma50, sma200):
    score = 0.5
    for sma in [sma20, sma50, sma200]:
        if sma > 0:
            score += 0.08 if price > sma else -0.08
    if sma50 > 0 and sma200 > 0:
        score += 0.08 if sma50 > sma200 else -0.08
    return float(np.clip(score, 0.0, 1.0))

def score_stochastic(pct_k, pct_d):
    score = 0.5
    if pct_k < 20: score += 0.20
    elif pct_k < 40: score += 0.10
    elif pct_k > 80: score -= 0.20
    elif pct_k > 60: score -= 0.10
    score += 0.05 if pct_k > pct_d else -0.05
    return float(np.clip(score, 0.0, 1.0))

def score_momentum(mom5, mom10, mom20):
    score = 0.5
    for mom, weight in [(mom5, 0.15), (mom10, 0.10), (mom20, 0.05)]:
        score += weight * np.clip(mom, -10, 10) / 10
    return float(np.clip(score, 0.0, 1.0))

def score_volume_trend(vol_ratio):
    if vol_ratio >= 2.0: return 0.75
    elif vol_ratio >= 1.5: return 0.70
    elif vol_ratio >= 1.1: return 0.60
    elif vol_ratio >= 0.9: return 0.50
    elif vol_ratio >= 0.7: return 0.40
    else: return 0.30

WEIGHTS = {"rsi": 0.18, "macd": 0.18, "bollinger": 0.14, "ma": 0.20,
           "stochastic": 0.12, "momentum": 0.12, "volume": 0.06}

def compute_composite_score(scores):
    return float(np.clip(sum(WEIGHTS[k] * scores[k] for k in WEIGHTS), 0.0, 1.0))

def classify_signal(score):
    if score >= 0.72: return "STRONG BUY"
    elif score >= 0.58: return "BUY"
    elif score >= 0.45: return "NEUTRAL"
    elif score >= 0.32: return "SELL"
    else: return "STRONG SELL"

def analyze_stock(ticker):
    try:
        end = datetime.today()
        start = end - timedelta(days=400)
        data = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
        if data is None or len(data) < 50:
            return None
        close  = data["Close"].squeeze()
        high   = data["High"].squeeze()
        low    = data["Low"].squeeze()
        volume = data["Volume"].squeeze()
        current_price = float(close.iloc[-1])
        rsi = calculate_rsi(close)
        macd, sig, hist = calculate_macd(close)
        _, _, _, pct_b = calculate_bollinger_bands(close)
        sma20  = calculate_sma(close, 20)
        sma50  = calculate_sma(close, 50)
        sma200 = calculate_sma(close, 200)
        pct_k, pct_d = calculate_stochastic(high, low, close)
        mom5   = calculate_momentum(close, 5)
        mom10  = calculate_momentum(close, 10)
        mom20  = calculate_momentum(close, 20)
        vol_ratio = calculate_volume_trend(volume)
        atr = calculate_atr(high, low, close)
        scores = {
            "rsi":        score_rsi(rsi),
            "macd":       score_macd(macd, sig, hist),
            "bollinger":  score_bollinger(pct_b),
            "ma":         score_moving_averages(current_price, sma20, sma50, sma200),
            "stochastic": score_stochastic(pct_k, pct_d),
            "momentum":   score_momentum(mom5, mom10, mom20),
            "volume":     score_volume_trend(vol_ratio),
        }
        composite = compute_composite_score(scores)
        return {
            "ticker": ticker, "price": current_price, "score": composite,
            "signal": classify_signal(composite),
            "rsi_value": round(rsi, 1), "rsi_score": round(scores["rsi"], 3),
            "macd_score": round(scores["macd"], 3),
            "bollinger_%b": round(pct_b, 3), "boll_score": round(scores["bollinger"], 3),
            "ma_score": round(scores["ma"], 3), "stoch_score": round(scores["stochastic"], 3),
            "momentum_5d": round(mom5, 2), "momentum_10d": round(mom10, 2),
            "momentum_20d": round(mom20, 2), "mom_score": round(scores["momentum"], 3),
            "vol_ratio": round(vol_ratio, 2), "vol_score": round(scores["volume"], 3),
            "atr": round(atr, 2), "sma20": round(sma20, 2),
            "sma50": round(sma50, 2), "sma200": round(sma200, 2),
            "above_sma20": current_price > sma20, "above_sma50": current_price > sma50,
            "above_sma200": current_price > sma200, "golden_cross": sma50 > sma200,
        }
    except Exception as e:
        print(f"  [!] Error analyzing {ticker}: {e}")
        return None

SIGNAL_COLOR = {
    "STRONG BUY": "\033[92m", "BUY": "\033[32m", "NEUTRAL": "\033[33m",
    "SELL": "\033[31m", "STRONG SELL": "\033[91m",
}
RESET = "\033[0m"

def score_bar(score, width=20):
    filled = round(score * width)
    return "#" * filled + "." * (width - filled)

def print_results(results):
    sorted_results = sorted(results, key=lambda x: x["score"], reverse=True)
    print("\n" + "-"*95)
    print(f" {'#':>2}  {'TICKER':<7} {'PRICE':>8}  {'SCORE':>6}  {'BAR':^22}  {'SIGNAL':<12}  {'RSI':>5}  {'MOM5%':>6}  {'MOM20%':>7}  {'VOL':>5}")
    print("-"*95)
    for i, r in enumerate(sorted_results, 1):
        sig = r["signal"]
        color = SIGNAL_COLOR.get(sig, "")
        bar = score_bar(r["score"])
        flag = " *" if r.get("golden_cross") else "  "
        print(f" {i:>2}  {r['ticker']:<7} ${r['price']:>7.2f}  {r['score']:.4f}  [{bar}]  {color}{sig:<12}{RESET}  {r['rsi_value']:>5.1f}  {r['momentum_5d']:>+6.1f}%  {r['momentum_20d']:>+6.1f}%  {r['vol_ratio']:>5.2f}x{flag}")
    print("-"*95)
    print(" * = Golden Cross (SMA50 > SMA200)")

def print_detailed(results, top_n=10):
    sorted_results = sorted(results, key=lambda x: x["score"], reverse=True)[:top_n]
    print("\n" + "="*60)
    print(f"  DETAILED BREAKDOWN - TOP {top_n} STOCKS")
    print("="*60)
    for r in sorted_results:
        sig = r["signal"]
        color = SIGNAL_COLOR.get(sig, "")
        print(f"\n  {r['ticker']}  ${r['price']:.2f}  ->  {color}Score: {r['score']:.4f}  [{sig}]{RESET}")
        print(f"  {'-'*40}")
        components = [
            ("RSI",          r["rsi_score"],   f"RSI={r['rsi_value']}"),
            ("MACD",         r["macd_score"],  ""),
            ("Bollinger %B", r["boll_score"],  f"%B={r['bollinger_%b']:.2f}"),
            ("Moving Avgs",  r["ma_score"],    f"20={'Y' if r['above_sma20'] else 'N'} 50={'Y' if r['above_sma50'] else 'N'} 200={'Y' if r['above_sma200'] else 'N'}"),
            ("Stochastic",   r["stoch_score"], ""),
            ("Momentum",     r["mom_score"],   f"5d={r['momentum_5d']:+.1f}% 20d={r['momentum_20d']:+.1f}%"),
            ("Volume",       r["vol_score"],   f"x{r['vol_ratio']:.2f}"),
        ]
        for name, score, note in components:
            bar = score_bar(score, width=12)
            note_str = f"  ({note})" if note else ""
            print(f"  {name:<22} {score:.3f}  [{bar}]{note_str}")

def save_csv(results, filename="sp500_analysis.csv"):
    df = pd.DataFrame(results).sort_values("score", ascending=False)
    df.to_csv(filename, index=False)
    print(f"\n  Results saved to: {filename}")

def main():
    print("\n" + "="*60)
    print("  S&P 500 TOP 50 - STOCK PROBABILITY ANALYZER")
    print(f"  Run date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("  Indicators: RSI / MACD / Bollinger / MA / Stochastic / Momentum / Volume")
    print("="*60)
    print(f"\n  Fetching data for {len(TOP_50_SP500)} stocks...\n")
    results = []
    failed  = []
    for idx, ticker in enumerate(TOP_50_SP500, 1):
        sys.stdout.write(f"  [{idx:>2}/{len(TOP_50_SP500)}] Analyzing {ticker:<7}...  ")
        sys.stdout.flush()
        result = analyze_stock(ticker)
        if result:
            results.append(result)
            sig = result["signal"]
            color = SIGNAL_COLOR.get(sig, "")
            print(f"score={result['score']:.4f}  {color}{sig}{RESET}")
        else:
            failed.append(ticker)
            print("FAILED")
        time.sleep(0.3)
    if not results:
        print("\n  No results - check your internet connection.")
        return
    scores = [r["score"] for r in results]
    print(f"\n  {'-'*40}")
    print(f"  Analyzed:   {len(results)} / {len(TOP_50_SP500)} stocks")
    print(f"  Failed:     {len(failed)} ({', '.join(failed) if failed else 'none'})")
    print(f"  Avg score:  {np.mean(scores):.4f}")
    print(f"  Max score:  {max(scores):.4f}  ({results[scores.index(max(scores))]['ticker']})")
    print(f"  Min score:  {min(scores):.4f}  ({results[scores.index(min(scores))]['ticker']})")
    print(f"  {'-'*40}")
    from collections import Counter
    dist = Counter(r["signal"] for r in results)
    print(f"\n  Signal distribution:")
    for sig in ["STRONG BUY", "BUY", "NEUTRAL", "SELL", "STRONG SELL"]:
        count = dist.get(sig, 0)
        color = SIGNAL_COLOR.get(sig, "")
        print(f"    {color}{sig:<12}{RESET}  {'|' * count}  ({count})")
    print_results(results)
    print_detailed(results, top_n=10)
    save_csv(results, filename="sp500_analysis.csv")
    print("\n" + "="*60)
    print("  DISCLAIMER: For educational/informational purposes only.")
    print("  NOT financial advice. Consult a licensed financial advisor.")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
