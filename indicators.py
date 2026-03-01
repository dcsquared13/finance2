"""
lib.indicators
==============
Pure functions that compute technical indicators from OHLCV price series.
All functions are stateless and return the most recent value as a float.

Public API
----------
rsi(close, period=14)                       -> float
macd(close, fast=12, slow=26, signal=9)     -> tuple[float, float, float]
bollinger_bands(close, period=20, n_std=2)  -> tuple[float, float, float, float]
sma(close, period)                          -> float
ema(close, period)                          -> float
stochastic(high, low, close, k=14, d=3)     -> tuple[float, float]
volume_trend(volume, recent=5, baseline=20) -> float
momentum(close, period)                     -> float
atr(high, low, close, period=14)            -> float
compute_all(df)                             -> dict
"""

from __future__ import annotations
import numpy as np
import pandas as pd


def rsi(close: pd.Series, period: int = 14) -> float:
    """RSI (0-100). Below 30 = oversold, above 70 = overbought."""
    delta    = close.diff()
    gain     = delta.clip(lower=0)
    loss     = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs       = avg_gain / avg_loss.replace(0, np.nan)
    rsi_s    = 100 - (100 / (1 + rs))
    return float(rsi_s.iloc[-1]) if not rsi_s.empty else 50.0


def macd(close: pd.Series, fast: int = 12, slow: int = 26,
         signal_period: int = 9) -> tuple[float, float, float]:
    """MACD. Returns (macd_line, signal_line, histogram)."""
    ema_fast    = close.ewm(span=fast, adjust=False).mean()
    ema_slow    = close.ewm(span=slow, adjust=False).mean()
    macd_line   = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    histogram   = macd_line - signal_line
    return float(macd_line.iloc[-1]), float(signal_line.iloc[-1]), float(histogram.iloc[-1])


def sma(close: pd.Series, period: int) -> float:
    """Simple Moving Average - latest value."""
    s = close.rolling(window=period).mean()
    return float(s.iloc[-1]) if len(close) >= period else float(close.mean())


def ema(close: pd.Series, period: int) -> float:
    """Exponential Moving Average - latest value."""
    return float(close.ewm(span=period, adjust=False).mean().iloc[-1])


def momentum(close: pd.Series, period: int) -> float:
    """Rate-of-change momentum (%). Returns 100*(close_t/close_{t-period} - 1)."""
    if len(close) < period + 1:
        return 0.0
    return float((close.iloc[-1] / close.iloc[-period] - 1) * 100)


def bollinger_bands(close: pd.Series, period: int = 20,
                    n_std: float = 2.0) -> tuple[float, float, float, float]:
    """Bollinger Bands. Returns (upper, middle, lower, %B).
    %B < 0 = below lower band (oversold); %B > 1 = above upper (overbought)."""
    mid   = close.rolling(window=period).mean()
    std   = close.rolling(window=period).std()
    upper = mid + n_std * std
    lower = mid - n_std * std
    pct_b = (close - lower) / (upper - lower).replace(0, np.nan)
    return (float(upper.iloc[-1]), float(mid.iloc[-1]),
            float(lower.iloc[-1]), float(pct_b.iloc[-1]) if not pct_b.empty else 0.5)


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> float:
    """Average True Range - measures market volatility."""
    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(),
                    (low - prev_close).abs()], axis=1).max(axis=1)
    atr_s = tr.rolling(window=period).mean()
    return float(atr_s.iloc[-1]) if not atr_s.empty else 0.0


def stochastic(high: pd.Series, low: pd.Series, close: pd.Series,
               k_period: int = 14, d_period: int = 3) -> tuple[float, float]:
    """Stochastic Oscillator. Returns (%K, %D).
    Below 20 = oversold, above 80 = overbought. %K > %D = buy signal."""
    lowest_low   = low.rolling(window=k_period).min()
    highest_high = high.rolling(window=k_period).max()
    denom        = (highest_high - lowest_low).replace(0, np.nan)
    pct_k        = 100 * (close - lowest_low) / denom
    pct_d        = pct_k.rolling(window=d_period).mean()
    return (float(pct_k.iloc[-1]) if not pct_k.empty else 50.0,
            float(pct_d.iloc[-1]) if not pct_d.empty else 50.0)


def volume_trend(volume: pd.Series, recent_bars: int = 5,
                 baseline_bars: int = 20) -> float:
    """Volume trend ratio: mean(recent)/mean(baseline). >1.0 = rising volume."""
    if len(volume) < baseline_bars:
        return 1.0
    recent_avg   = float(volume.iloc[-recent_bars:].mean())
    baseline_avg = float(volume.iloc[-baseline_bars:].mean())
    return recent_avg / baseline_avg if baseline_avg > 0 else 1.0


def compute_all(df: pd.DataFrame) -> dict:
    """Compute all indicators for a single OHLCV DataFrame. Returns a flat dict."""
    close  = df["Close"].squeeze()
    high   = df["High"].squeeze()
    low    = df["Low"].squeeze()
    volume = df["Volume"].squeeze()

    macd_v, macd_sig, macd_hist      = macd(close)
    bb_upper, bb_mid, bb_lower, bb_b = bollinger_bands(close)
    stoch_k, stoch_d                 = stochastic(high, low, close)

    return {
        "rsi":         rsi(close),
        "macd":        macd_v,
        "macd_signal": macd_sig,
        "macd_hist":   macd_hist,
        "bb_upper":    bb_upper,
        "bb_mid":      bb_mid,
        "bb_lower":    bb_lower,
        "bb_pct_b":    bb_b,
        "sma20":       sma(close, 20),
        "sma50":       sma(close, 50),
        "sma200":      sma(close, 200),
        "stoch_k":     stoch_k,
        "stoch_d":     stoch_d,
        "mom5":        momentum(close, 5),
        "mom10":       momentum(close, 10),
        "mom20":       momentum(close, 20),
        "vol_trend":   volume_trend(volume),
        "atr":         atr(high, low, close),
        "price":       float(close.iloc[-1]),
    }
