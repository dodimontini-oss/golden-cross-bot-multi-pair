"""
Live "what might trade soon" snapshot for the 9 deployed forex pairs - NOT a
new signal source, mirrors multi_pair_bot/bot.py's actual live logic exactly
(SMA 50/200, ATR, H4, plus the GAPCONFIRM entry filter deployed 2026-08-25)
so this reports what the bot itself is watching.

Two states are checked, most-actionable first:

1. CONFIRMING - a crossover already fired within the last CONFIRM_MAX_BARS
   H4 candles and is still valid (no reversal since). These are the pairs
   closest to an actual trade: bot.py will place the order as soon as the
   fast/slow gap widens past CONFIRM_MIN_GAP_ATR (or already has, in which
   case it should trade on its very next scheduled check). Reports current
   gap progress toward that threshold.
2. WATCHING - no recent crossover, but the fast/slow gap (relative to ATR)
   is close enough to be worth keeping an eye on. This is a much earlier/
   weaker signal than CONFIRMING - most WATCHING pairs never actually cross.

Uses the most recent candles only (not a backtest).

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
CANDLE_COUNT = 260   # SLOW_LEN(200) + ATR_LEN(14) + confirm/trend lookback + warm-up buffer

FAST_LEN = 50
SLOW_LEN = 200
ATR_LEN = 14

# Must match multi_pair_bot/bot.py exactly - this reports against the bot's
# actual live thresholds, not independently-chosen ones.
CONFIRM_MIN_GAP_ATR = 0.10
CONFIRM_MAX_BARS = 5

NEAR_THRESHOLD_ATR = 0.25  # WATCHING-state threshold, independent of CONFIRM_MIN_GAP_ATR
LOOKBACK_FOR_TREND = 3

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


def find_crossovers(df: pd.DataFrame):
    for i in range(1, len(df)):
        prev, curr = df.iloc[i - 1], df.iloc[i]
        if pd.isna(prev["slow_ma"]) or pd.isna(curr["slow_ma"]):
            continue
        if prev["fast_ma"] <= prev["slow_ma"] and curr["fast_ma"] > curr["slow_ma"]:
            yield i, "LONG"
        elif prev["fast_ma"] >= prev["slow_ma"] and curr["fast_ma"] < curr["slow_ma"]:
            yield i, "SHORT"


def check_pair(instrument: str):
    df = get_recent_candles(instrument, CANDLE_COUNT)
    if len(df) < SLOW_LEN + ATR_LEN + LOOKBACK_FOR_TREND:
        return {"symbol": instrument, "error": "not enough candle history"}

    df["fast_ma"] = sma(df["close"], FAST_LEN)
    df["slow_ma"] = sma(df["close"], SLOW_LEN)
    df["atr"] = atr_wilder(df, ATR_LEN)
    df["gap"] = df["fast_ma"] - df["slow_ma"]

    latest = df.iloc[-1]
    if pd.isna(latest["slow_ma"]) or not latest["atr"]:
        return {"symbol": instrument, "error": "SMA200 not warmed up yet"}

    gap_atr = abs(latest["gap"]) / latest["atr"]

    # --- state 1: recent crossover, still valid, awaiting/already past gap confirmation ---
    signals = list(find_crossovers(df))
    if signals:
        i, direction = signals[-1]
        bars_since = (len(df) - 1) - i
        if bars_since <= CONFIRM_MAX_BARS:
            return {
                "symbol": instrument, "state": "CONFIRMING", "direction": direction,
                "bars_since": bars_since, "bars_left": CONFIRM_MAX_BARS - bars_since,
                "gap_atr": gap_atr, "confirmed_now": gap_atr >= CONFIRM_MIN_GAP_ATR,
                "close": latest["close"],
            }

    # --- state 2: no recent crossover - plain distance-to-crossover watch ---
    prior = df.iloc[-1 - LOOKBACK_FOR_TREND]
    if pd.isna(prior["slow_ma"]):
        return {"symbol": instrument, "error": "not enough history for trend comparison"}
    narrowing = abs(latest["gap"]) < abs(prior["gap"])
    side = "fast>slow (bullish stance)" if latest["gap"] > 0 else "fast<slow (bearish stance)"
    return {
        "symbol": instrument, "state": "WATCHING", "close": latest["close"],
        "fast_sma": latest["fast_ma"], "slow_sma": latest["slow_ma"],
        "gap_pips": abs(latest["gap"]) / pip_size(instrument), "gap_atr": gap_atr,
        "side": side, "narrowing": narrowing, "near": gap_atr < NEAR_THRESHOLD_ATR,
    }


def run():
    results = []
    for instrument in PAIRS:
        try:
            results.append(check_pair(instrument))
        except Exception as e:
            results.append({"symbol": instrument, "error": str(e)})

    confirming = [r for r in results if r.get("state") == "CONFIRMING"]
    watching = [r for r in results if r.get("state") == "WATCHING"]
    errors = [r for r in results if "error" in r]

    print(f"\nGolden Cross 'what might trade soon' check - {GRANULARITY}, {len(PAIRS)} pairs "
          f"(matches multi_pair_bot/bot.py's live GAPCONFIRM logic exactly)\n")

    if confirming:
        confirming.sort(key=lambda r: (not r["confirmed_now"], -r["gap_atr"]))
        print(f"CONFIRMING - crossed recently, closest to an actual trade:")
        print(f"{'Pair':<10} {'Direction':<7} {'Close':>10} {'BarsSince':>10} {'BarsLeft':>9} "
              f"{'Gap/ATR':>9} {'Status'}")
        for r in confirming:
            status = "READY - should trade next check" if r["confirmed_now"] else \
                     f"waiting (need {CONFIRM_MIN_GAP_ATR}, have {r['gap_atr']:.3f})"
            print(f"{r['symbol']:<10} {r['direction']:<7} {r['close']:>10.5f} {r['bars_since']:>10} "
                  f"{r['bars_left']:>9} {r['gap_atr']:>9.3f} {status}")
        print()
    else:
        print("CONFIRMING - none. No pair has crossed within the last "
              f"{CONFIRM_MAX_BARS} bars.\n")

    if watching:
        watching.sort(key=lambda r: r["gap_atr"])
        print(f"WATCHING - no recent crossover, ranked by distance (gap/ATR below {NEAR_THRESHOLD_ATR} flagged):")
        print(f"{'Pair':<10} {'Close':>10} {'SMA50':>10} {'SMA200':>10} {'Gap(pips)':>10} {'Gap/ATR':>9}  {'Trend':<10} Side")
        for r in watching:
            flag = " <-- NEAR" if r["near"] else ""
            trend = "narrowing" if r["narrowing"] else "widening"
            print(f"{r['symbol']:<10} {r['close']:>10.5f} {r['fast_sma']:>10.5f} {r['slow_sma']:>10.5f} "
                  f"{r['gap_pips']:>10.1f} {r['gap_atr']:>9.3f}  {trend:<10} {r['side']}{flag}")
        print()

    for r in errors:
        print(f"{r['symbol']:<10} error - {r['error']}")

    print()


if __name__ == "__main__":
    run()
