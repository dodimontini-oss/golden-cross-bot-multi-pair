"""
Multi-pair Golden Cross bot - checks a validated shortlist of forex pairs
every run and places a trade on whichever ones show a fresh crossover
signal. Talks directly to OANDA, no TradingView, no separate hosting.

The 9 pairs below were chosen from a 23-pair screen over the full 2015-2026
H4 history (see forex_screener/screener.py) - each has profit factor > 1.15
and controlled drawdown. Backtesting is not duplicated here; use the
screener for that. This script is for live execution only.

On top of the base 50/200 SMA Golden Cross, two filters validated in
forex_screener/golden_cross_portfolio_lab.py (full 2015-2026 history, walk-
forward checked on an early/late split - both beat baseline on profit
factor, win rate, and net return in both halves) are applied:

  - GAPCONFIRM: don't enter the instant a crossover fires - wait up to
    CONFIRM_MAX_BARS H4 candles for the fast/slow MA gap to widen past
    CONFIRM_MIN_GAP_ATR (relative to ATR), skipping narrow/marginal
    crossovers that are more likely whipsaws. Recomputed fresh from OANDA
    candle history every run rather than stored as bot-side state, since
    GitHub Actions runs are stateless containers - this just looks back at
    the last few candles each cycle instead of remembering "still waiting".
  - CURCAP: blocks a new trade if it would push total open risk-dollars
    sharing a currency (5 of the 9 pairs are GBP crosses) above
    MAX_CURRENCY_RISK_PCT of equity. Each open trade's risk is reconstructed
    from its live OANDA entry price and attached stop-loss price (same
    "recompute from broker state, don't persist" pattern crypto_bot/
    live_bot.py already uses for its own stop reconstruction).

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

# Validated in forex_screener/golden_cross_portfolio_lab.py - same values used
# there, don't change these without re-running that backtest.
CONFIRM_MIN_GAP_ATR = 0.10
CONFIRM_MAX_BARS = 5
MAX_CURRENCY_RISK_PCT = 2.5

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


def get_account_info() -> tuple:
    """Returns (balance, currency) - the account's own currency is needed to
    correctly convert quote-currency price distances into account-currency
    risk amounts (see get_conversion_rate)."""
    url = f"{OANDA_BASE_URL}/v3/accounts/{OANDA_ACCOUNT_ID}/summary"
    resp = requests.get(url, headers=HEADERS, timeout=10)
    resp.raise_for_status()
    account = resp.json()["account"]
    return float(account["balance"]), account["currency"]


def get_conversion_rate(from_currency: str, to_currency: str) -> float:
    """How many units of to_currency 1 unit of from_currency is worth, using
    the latest OANDA price. ATR/stop/target are computed in the QUOTE
    currency of the traded pair (e.g. JPY for GBP_JPY), but risk sizing and
    the currency cap both need amounts in the ACCOUNT's currency - for any
    pair whose quote currency isn't the account currency (7 of the 9 live
    pairs; only GBP_USD/EUR_USD already quote in USD), skipping this
    conversion silently mis-sizes the position by the exchange rate.
    Confirmed live 2026-08-26: GBP_JPY was undersized ~160x (JPY/USD rate),
    risking ~$0.61 instead of the intended ~1% of equity."""
    if from_currency == to_currency:
        return 1.0
    try:
        df = get_recent_candles(f"{from_currency}_{to_currency}", count=2)
        if not df.empty:
            return df.iloc[-1]["close"]
    except Exception:
        pass
    df = get_recent_candles(f"{to_currency}_{from_currency}", count=2)
    return 1.0 / df.iloc[-1]["close"]


def get_open_position_units(instrument: str) -> float:
    """Signed units currently held for this instrument (0 if flat)."""
    url = f"{OANDA_BASE_URL}/v3/accounts/{OANDA_ACCOUNT_ID}/positions/{instrument}"
    resp = requests.get(url, headers=HEADERS, timeout=10)
    if resp.status_code == 404:
        return 0.0
    resp.raise_for_status()
    pos = resp.json()["position"]
    return float(pos["long"]["units"]) + float(pos["short"]["units"])


def get_open_trades_risk_by_currency(account_ccy: str) -> dict:
    """Risk, in account-currency terms, currently committed per currency,
    across ALL open trades on the account (not just one instrument) -
    reconstructed from each trade's live entry price and attached stop-loss
    price rather than any bot-stored value, so this stays correct even
    across GitHub Actions' stateless runs. Returns {} (no cap enforced this
    cycle) if OANDA's response doesn't have the shape we expect, rather than
    risking a schema surprise silently blocking every future trade forever.
    """
    risk_by_ccy = {}
    try:
        url = f"{OANDA_BASE_URL}/v3/accounts/{OANDA_ACCOUNT_ID}/openTrades"
        resp = requests.get(url, headers=HEADERS, timeout=10)
        resp.raise_for_status()
        trades = resp.json()["trades"]
    except Exception:
        log.exception("Could not fetch open trades for currency-risk check - skipping CURCAP this cycle.")
        return {}

    for trade in trades:
        try:
            instrument = trade["instrument"]
            entry_price = float(trade["price"])
            units = abs(float(trade["currentUnits"]))
            stop_price = float(trade["stopLossOrder"]["price"])
            _, quote = currency_legs(instrument)
            quote_to_account = get_conversion_rate(quote, account_ccy)
            risk_amount = units * abs(entry_price - stop_price) * quote_to_account
            for ccy in currency_legs(instrument):
                risk_by_ccy[ccy] = risk_by_ccy.get(ccy, 0.0) + risk_amount
        except (KeyError, ValueError, TypeError):
            log.warning("Open trade %s missing expected fields for risk calc - "
                        "not counted toward the currency cap this cycle.", trade.get("instrument", "?"))
    return risk_by_ccy


def price_decimals(instrument: str) -> int:
    """OANDA requires JPY-quoted pairs priced to 3 decimals, everything else
    to 5 - sending the wrong precision gets the order rejected with 400."""
    return 3 if "JPY" in instrument else 5


def place_order(instrument: str, units: int, stop: float, target: float) -> dict:
    url = f"{OANDA_BASE_URL}/v3/accounts/{OANDA_ACCOUNT_ID}/orders"
    decimals = price_decimals(instrument)
    body = {
        "order": {
            "type": "MARKET",
            "instrument": instrument,
            "units": str(units),
            "timeInForce": "FOK",
            "positionFill": "DEFAULT",
            "stopLossOnFill": {"price": f"{stop:.{decimals}f}"},
            "takeProfitOnFill": {"price": f"{target:.{decimals}f}"},
        }
    }
    resp = requests.post(url, headers=HEADERS, json=body, timeout=15)
    resp.raise_for_status()
    return resp.json()


def currency_legs(pair: str) -> tuple:
    base, quote = pair.split("_")
    return base, quote


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


def find_confirmed_signal(df: pd.DataFrame):
    """Most recent crossover, gap-confirmed, within CONFIRM_MAX_BARS of
    forming. Mirrors golden_cross_portfolio_lab.py's GAPCONFIRM logic, just
    recomputed from the recent candle window each call instead of carried as
    state between runs - if a reversal happened after the crossover, that
    reversal is itself the most recent crossover (find_crossovers naturally
    picks it up as signals[-1]), so no separate "still valid" check is needed.

    Returns (i, direction, bars_since) or None.
    """
    signals = list(find_crossovers(df))
    if not signals:
        return None
    i, direction = signals[-1]
    bars_since = (len(df) - 1) - i
    if bars_since > CONFIRM_MAX_BARS:
        return None  # crossover is real but timed out waiting for confirmation

    row = df.iloc[-1]
    if pd.isna(row["atr"]) or row["atr"] <= 0:
        return None
    gap_atr = abs(row["fast_ma"] - row["slow_ma"]) / row["atr"]
    if gap_atr < CONFIRM_MIN_GAP_ATR:
        return None  # crossover is real but hasn't widened enough yet

    return i, direction, bars_since


# ---------------- Core check-and-trade, per instrument ----------------

def check_and_trade(instrument: str, df: pd.DataFrame, balance: float, account_ccy: str, risk_by_ccy: dict):
    df = add_indicators(df)
    signal = find_confirmed_signal(df)
    if signal is None:
        log.info("[%s] No gap-confirmed crossover. No action.", instrument)
        return

    i, direction, bars_since = signal
    row = df.iloc[i]  # signal candle - use ITS atr for stop sizing, not the latest bar's

    position = get_open_position_units(instrument)
    if position != 0:
        log.info("[%s] Already in a position (%s units). No action.", instrument, position)
        return

    base, quote = currency_legs(instrument)
    risk_amount = balance * (RISK_PER_TRADE_PCT / 100)
    cap_dollars = balance * (MAX_CURRENCY_RISK_PCT / 100)
    base_risk = risk_by_ccy.get(base, 0.0)
    quote_risk = risk_by_ccy.get(quote, 0.0)
    if base_risk + risk_amount > cap_dollars or quote_risk + risk_amount > cap_dollars:
        log.info("[%s] %s signal confirmed (%d bars ago) but blocked by currency exposure cap "
                  "(%s open risk=%.2f, %s open risk=%.2f, cap=%.2f). No action.",
                  instrument, direction, bars_since, base, base_risk, quote, quote_risk, cap_dollars)
        return

    latest_close = df.iloc[-1]["close"]
    stop_distance = row["atr"] * ATR_STOP_MULT
    quote_to_account = get_conversion_rate(quote, account_ccy)
    units = int(risk_amount / (stop_distance * quote_to_account))

    if direction == "LONG":
        stop, target = latest_close - stop_distance, latest_close + stop_distance * RR_RATIO
    else:
        stop, target = latest_close + stop_distance, latest_close - stop_distance * RR_RATIO
        units = -units

    log.info("[%s] %s signal confirmed (%d bars ago, gap-confirmed) at close=%.5f - "
              "placing order: units=%d stop=%.5f target=%.5f",
              instrument, direction, bars_since, latest_close, units, stop, target)
    result = place_order(instrument, units, stop, target)
    log.info("[%s] OANDA response: %s", instrument, result)

    # update in-memory so a second pair sharing this currency, checked later
    # in the same cycle, sees this trade's risk too - not just what OANDA
    # already knew about at the start of this run.
    for ccy in (base, quote):
        risk_by_ccy[ccy] = risk_by_ccy.get(ccy, 0.0) + risk_amount


def check_all_pairs():
    balance, account_ccy = get_account_info()
    risk_by_ccy = get_open_trades_risk_by_currency(account_ccy)
    for instrument in PAIRS:
        try:
            df = get_recent_candles(instrument, count=SLOW_LEN + ATR_LEN + 10)
            if len(df) < SLOW_LEN + 2:
                log.warning("[%s] Not enough candle history yet (%d bars).", instrument, len(df))
                continue
            check_and_trade(instrument, df, balance, account_ccy, risk_by_ccy)
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
    balance, account_ccy = get_account_info()
    log.info("Balance fetched OK: %.2f %s", balance, account_ccy)
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
