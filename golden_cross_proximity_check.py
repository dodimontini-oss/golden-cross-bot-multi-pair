"""
Live 50/200 SMA Golden Cross proximity check - NOT a new signal source, just
a "how close is each currently-traded pair to a crossover right now" snapshot
for discretionary awareness. Mirrors multi_pair_bot/bot.py's indicator math
exactly (SMA, not EMA - the deployed strategy is a Simple Moving Average
crossover) so the numbers here match what the live bot is actually watching.

Checks the same H4 timeframe and 9-pair shortlist the live bot trades, using
the most recent candles only (not a full-history backtest).

"Closeness" is measured as the SMA50-SMA200 gap divided by ATR(14), since raw
price gaps aren't comparable across pairs with different volatility/pip
scales. A pair is flagged as "near crossover" if that ratio is below
NEAR_THRESHOLD_ATR.

Environment variables required:
    OANDA_API_KEY
"""

import os

import pandas as pd
import requests

OANDA_API_KEY = os.environ["OANDA_API_KEY"]
OANDA_BASE_URL = "https://api-fxpractice.oanda.com"
HEADERS = {"Authorization": f"Bearer {OANDA_API_KEY}"}

GRANULARITY = "H4"  # matches the live multi-pair bot's timeframe
CANDLE_COUNT = 260   # SLOW_LEN(200) + ATR_LEN(14) + trend lookback + warm-up buffer

FAST_LEN = 50
SLOW_LEN = 200
ATR_LEN = 14
NEAR_THRESHOLD_ATR = 0.25  # gap below this fraction of ATR(14) is flagged as "near crossover"
LOOKBACK_FOR_TREND = 3     # candles used to say whether the gap is narrowing or widening

PAIRS = [
    "GBP_CHF", "GBP_CAD", "GBP_USD", "GBP_JPY", "GBP_AUD",
    "USD_CHF", "EUR_USD", "EUR_AUD", "AUD_NZD",
]


def pip_size(instrument: str) -> float:
    return 0.01 if "JPY" in instrument else 0.0001


def get_recent_candles(instrument: str, count: int) -> pd.DataFrame:
    url = f"{OANDA_BASE_URL}/v3/instruments/{instrument}/candles"
    params = {"count": count, "granularity": GRANULARITY, "price": "M"}
    resp = requests.get(url, headers=HEADERS, params=params, timeout=15)
    resp.raise_for_status()
    rows = [
        {"time": c["time"], "open": float(c["mid"]["o"]), "high": float(c["mid"]["h"]),
         "low": float(c["mid"]["l"]), "close": float(c["mid"]["c"])}
        for c in resp.json()["candles"] if c["complete"]
    ]
    return pd.DataFrame(rows)


def sma(series: pd.Series, length: int) -> pd.Series:
    return series.rolling(length).mean()


def wilder_rma(series: pd.Series, length: int) -> pd.Series:
    rma = pd.Series(index=series.index, dtype=float)
    if len(series) < length:
        return rma
    rma.iloc[length - 1] = series.iloc[:length].mean()
    for i in range(length, len(series)):
        rma.iloc[i] = (rma.iloc[i - 1] * (length - 1) + series.iloc[i]) / length
    return rma


def atr_wilder(df: pd.DataFrame, length: int) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    return wilder_rma(tr, length)


def check_pair(instrument: str):
    df = get_recent_candles(instrument, CANDLE_COUNT)
    if len(df) < SLOW_LEN + ATR_LEN + LOOKBACK_FOR_TREND:
        return {"symbol": instrument, "error": "not enough candle history"}

    df["fast_ma"] = sma(df["close"], FAST_LEN)
    df["slow_ma"] = sma(df["close"], SLOW_LEN)
    df["atr"] = atr_wilder(df, ATR_LEN)
    df["gap"] = df["fast_ma"] - df["slow_ma"]

    latest = df.iloc[-1]
    prior = df.iloc[-1 - LOOKBACK_FOR_TREND]

    if pd.isna(latest["slow_ma"]) or pd.isna(prior["slow_ma"]):
        return {"symbol": instrument, "error": "SMA200 not warmed up yet"}

    gap_atr = abs(latest["gap"]) / latest["atr"] if latest["atr"] else float("inf")
    gap_pips = abs(latest["gap"]) / pip_size(instrument)
    narrowing = abs(latest["gap"]) < abs(prior["gap"])
    side = "fast>slow (bullish stance)" if latest["gap"] > 0 else "fast<slow (bearish stance)"

    return {
        "symbol": instrument,
        "close": latest["close"],
        "fast_sma": latest["fast_ma"],
        "slow_sma": latest["slow_ma"],
        "gap_pips": gap_pips,
        "gap_atr": gap_atr,
        "side": side,
        "narrowing": narrowing,
        "near": gap_atr < NEAR_THRESHOLD_ATR,
    }


def run():
    results = []
    for instrument in PAIRS:
        try:
            results.append(check_pair(instrument))
        except Exception as e:
            results.append({"symbol": instrument, "error": str(e)})

    ok = [r for r in results if "error" not in r]
    errors = [r for r in results if "error" in r]
    ok.sort(key=lambda r: r["gap_atr"])

    print(f"\nSMA {FAST_LEN}/{SLOW_LEN} Golden Cross proximity check - {GRANULARITY} timeframe, {len(PAIRS)} pairs")
    print(f"(gap/ATR below {NEAR_THRESHOLD_ATR} = flagged as near a possible crossover)\n")
    print(f"{'Pair':<10} {'Close':>10} {'SMA50':>10} {'SMA200':>10} {'Gap(pips)':>10} {'Gap/ATR':>9}  {'Trend':<10} Side")
    for r in ok:
        flag = " <-- NEAR" if r["near"] else ""
        trend = "narrowing" if r["narrowing"] else "widening"
        print(f"{r['symbol']:<10} {r['close']:>10.5f} {r['fast_sma']:>10.5f} {r['slow_sma']:>10.5f} "
              f"{r['gap_pips']:>10.1f} {r['gap_atr']:>9.3f}  {trend:<10} {r['side']}{flag}")

    for r in errors:
        print(f"{r['symbol']:<10} error - {r['error']}")

    print()


if __name__ == "__main__":
    run()
