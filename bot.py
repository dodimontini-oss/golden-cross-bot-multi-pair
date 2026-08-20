"""
Multi-pair Golden Cross bot - checks a validated shortlist of forex pairs
every run and places a trade on whichever ones show a fresh crossover
signal. Talks directly to OANDA, no TradingView, no separate hosting.

The 9 pairs below were chosen from a 23-pair screen over the full 2015-2026
H4 history (see forex_screener/screener.py) - each has profit factor > 1.15
and controlled drawdown. Backtesting is not duplicated here; use the
screener for that. This script is for live execution only.

Modes:
    python bot.py once   - checks all pairs once, then exits (for scheduled
                            execution, e.g. GitHub Actions cron).
    python bot.py test   - places one small symbolic order to confirm the
                            OANDA connection works end-to-end.
    python bot.py        - runs forever, checking all pairs every 5 minutes.

Environment variables required:
    OANDA_API_KEY     - Personal Access Token from OANDA account settings
    OANDA_ACCOUNT_ID  - your practice account ID
"""

import logging
import os
import sys
import time

import pandas as pd
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("golden-cross-multi")

OANDA_API_KEY = os.environ["OANDA_API_KEY"]
OANDA_ACCOUNT_ID = os.environ["OANDA_ACCOUNT_ID"]
OANDA_BASE_URL = "https://api-fxpractice.oanda.com"  # practice only - never change without a deliberate decision

GRANULARITY = "H4"
FAST_LEN = 50
SLOW_LEN = 200
ATR_LEN = 14
ATR_STOP_MULT = 2.0
RR_RATIO = 2.0
RISK_PER_TRADE_PCT = 1.0

# Validated shortlist from the 23-pair screen - profit factor > 1.15 and
# controlled drawdown over the full 2015-2026 backtest.
PAIRS = [
    "GBP_CHF", "GBP_CAD", "GBP_USD", "GBP_JPY", "GBP_AUD",
    "USD_CHF", "EUR_USD", "EUR_AUD", "AUD_NZD",
]

HEADERS = {
    "Authorization": f"Bearer {OANDA_API_KEY}",
    "Content-Type": "application/json",
}

# ---------------- OANDA helpers - all parameterized by instrument ----------------

def get_recent_candles(instrument: str, count: int) -> pd.DataFrame:
    url = f"{OANDA_BASE_URL}/v3/instruments/{instrument}/candles"
    params = {"count": count, "granularity": GRANULARITY, "price": "M"}
    resp = requests.get(url, headers=HEADERS, params=params, timeout=15)
    resp.raise_for_status()
    return _candles_to_df(resp.json()["candles"])


def _candles_to_df(candles: list) -> pd.DataFrame:
    rows = [
        {"time": c["time"], "open": float(c["mid"]["o"]), "high": float(c["mid"]["h"]),
         "low": float(c["mid"]["l"]), "close": float(c["mid"]["c"]), "complete": c["complete"]}
        for c in candles
    ]
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df[df["complete"]].reset_index(drop=True)


def get_account_balance() -> float:
    url = f"{OANDA_BASE_URL}/v3/accounts/{OANDA_ACCOUNT_ID}/summary"
    resp = requests.get(url, headers=HEADERS, timeout=10)
    resp.raise_for_status()
    return float(resp.json()["account"]["balance"])


def get_open_position_units(instrument: str) -> float:
    """Signed units currently held for this instrument (0 if flat)."""
    url = f"{OANDA_BASE_URL}/v3/accounts/{OANDA_ACCOUNT_ID}/positions/{instrument}"
    resp = requests.get(url, headers=HEADERS, timeout=10)
    if resp.status_code == 404:
        return 0.0
    resp.raise_for_status()
    pos = resp.json()["position"]
    return float(pos["long"]["units"]) + float(pos["short"]["units"])


def place_order(instrument: str, units: int, stop: float, target: float) -> dict:
    url = f"{OANDA_BASE_URL}/v3/accounts/{OANDA_ACCOUNT_ID}/orders"
    body = {
        "order": {
            "type": "MARKET",
            "instrument": instrument,
            "units": str(units),
            "timeInForce": "FOK",
            "positionFill": "DEFAULT",
            "stopLossOnFill": {"price": f"{stop:.5f}"},
            "takeProfitOnFill": {"price": f"{target:.5f}"},
        }
    }
    resp = requests.post(url, headers=HEADERS, json=body, timeout=15)
    resp.raise_for_status()
    return resp.json()


# ---------------- Indicators - matches Pine's ta.sma()/ta.atr() exactly ----------------

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


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["fast_ma"] = sma(df["close"], FAST_LEN)
    df["slow_ma"] = sma(df["close"], SLOW_LEN)
    df["atr"] = atr_wilder(df, ATR_LEN)
    return df


def find_crossovers(df: pd.DataFrame):
    for i in range(1, len(df)):
        prev, curr = df.iloc[i - 1], df.iloc[i]
        if pd.isna(prev["slow_ma"]) or pd.isna(curr["slow_ma"]):
            continue
        if prev["fast_ma"] <= prev["slow_ma"] and curr["fast_ma"] > curr["slow_ma"]:
            yield i, "LONG"
        elif prev["fast_ma"] >= prev["slow_ma"] and curr["fast_ma"] < curr["slow_ma"]:
            yield i, "SHORT"


# ---------------- Core check-and-trade, per instrument ----------------

def check_and_trade(instrument: str, df: pd.DataFrame):
    df = add_indicators(df)
    signals = list(find_crossovers(df))
    if not signals:
        log.info("[%s] No fresh crossover. No action.", instrument)
        return

    i, direction = signals[-1]
    if i != len(df) - 1:
        log.info("[%s] Most recent signal isn't on the latest candle - stale. No action.", instrument)
        return

    row = df.iloc[i]
    position = get_open_position_units(instrument)
    if position != 0:
        log.info("[%s] Already in a position (%s units). No action.", instrument, position)
        return

    balance = get_account_balance()
    risk_amount = balance * (RISK_PER_TRADE_PCT / 100)
    stop_distance = row["atr"] * ATR_STOP_MULT
    units = int(risk_amount / stop_distance)

    if direction == "LONG":
        stop, target = row["close"] - stop_distance, row["close"] + stop_distance * RR_RATIO
    else:
        stop, target = row["close"] + stop_distance, row["close"] - stop_distance * RR_RATIO
        units = -units

    log.info("[%s] %s signal at %s (close=%.5f) - placing order: units=%d stop=%.5f target=%.5f",
              instrument, direction, row["time"], row["close"], units, stop, target)
    result = place_order(instrument, units, stop, target)
    log.info("[%s] OANDA response: %s", instrument, result)


def check_all_pairs():
    for instrument in PAIRS:
        try:
            df = get_recent_candles(instrument, count=SLOW_LEN + ATR_LEN + 10)
            if len(df) < SLOW_LEN + 2:
                log.warning("[%s] Not enough candle history yet (%d bars).", instrument, len(df))
                continue
            check_and_trade(instrument, df)
        except Exception:
            log.exception("[%s] Error during check - skipping this pair this cycle.", instrument)


# ---------------- Modes ----------------

def run_once():
    log.info("Golden Cross multi-pair bot - single check across %d pairs: %s", len(PAIRS), ", ".join(PAIRS))
    check_all_pairs()


def run_live(poll_interval_seconds: int = 300):
    log.info("Golden Cross multi-pair bot starting (live) - %d pairs, polling every %ds.",
              len(PAIRS), poll_interval_seconds)
    while True:
        try:
            check_all_pairs()
        except Exception:
            log.exception("Unexpected error in main loop - will retry next poll.")
        time.sleep(poll_interval_seconds)


def run_test():
    """Places one small symbolic order on the first pair in the list, just
    to confirm the OANDA connection works end-to-end."""
    instrument = PAIRS[0]
    log.info("Connectivity test - placing one small order on %s.", instrument)
    balance = get_account_balance()
    log.info("Balance fetched OK: %.2f", balance)
    candles = get_recent_candles(instrument, count=5)
    last_close = candles.iloc[-1]["close"]
    stop, target = last_close - 0.0050, last_close + 0.0050
    units = 100
    log.info("Placing test order: %s units=%d close=%.5f stop=%.5f target=%.5f",
              instrument, units, last_close, stop, target)
    result = place_order(instrument, units, stop, target)
    log.info("Test order placed OK: %s", result)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "live"
    if mode == "once":
        run_once()
    elif mode == "test":
        run_test()
    else:
        run_live()
