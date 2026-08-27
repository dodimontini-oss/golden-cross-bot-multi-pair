"""
Stop/target distance sweep for the DEPLOYED forex config (GAPCONFIRM +
CURCAP, i.e. the "WINNERS" combo from golden_cross_portfolio_lab.py, which
is what multi_pair_bot/bot.py actually runs live) - user's question: what if
the ATR stop-loss multiple (and therefore the target too, since R:R stays
fixed at 2:1) were pushed farther out, with units sized down proportionally
so dollar risk stays at exactly 1% of equity either way? Does more room to
breathe (fewer stop-outs on noise) beat the current ATR_STOP_MULT=2.0?

Same shared-equity portfolio engine, same 9-pair/H4/2015-2026 history, same
GAPCONFIRM+CURCAP logic as golden_cross_portfolio_lab.py's WINNERS config -
only ATR_STOP_MULT varies. RR_RATIO stays fixed at 2.0 throughout (target
always 2x the stop distance), matching what the user described: push TP and
SL out together, not independently. No fee model here (forex trading cost
is the spread, already embedded in OANDA's own bid/ask - this project has
never modeled an explicit forex commission, unlike the crypto fee findings,
so this stays consistent with golden_cross_portfolio_lab.py's convention).

Run: python stop_distance_sweep_lab.py

Environment variables required:
    OANDA_API_KEY
"""

import logging
import os
import time

import pandas as pd
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("forex-stop-distance-sweep")

OANDA_API_KEY = os.environ["OANDA_API_KEY"]
OANDA_BASE_URL = "https://api-fxpractice.oanda.com"
HEADERS = {"Authorization": f"Bearer {OANDA_API_KEY}"}

GRANULARITY = "H4"
FROM_TIME = "2015-01-01T00:00:00Z"

FAST_LEN = 50
SLOW_LEN = 200
ATR_LEN = 14
RR_RATIO = 2.0  # fixed throughout - target always 2x the stop distance
RISK_PER_TRADE_PCT = 1.0
STARTING_EQUITY = 10000.0

# Must match multi_pair_bot/bot.py's actual deployed values exactly.
CONFIRM_MIN_GAP_ATR = 0.10
CONFIRM_MAX_BARS = 5
MAX_CURRENCY_RISK_PCT = 2.5

ATR_STOP_MULTS_TO_TEST = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0]  # 2.0 is the current deployed value

PAIRS = [
    "GBP_CHF", "GBP_CAD", "GBP_USD", "GBP_JPY", "GBP_AUD",
    "USD_CHF", "EUR_USD", "EUR_AUD", "AUD_NZD",
]


def get_candles_range(instrument: str, from_time: str) -> pd.DataFrame:
    all_candles = []
    current_from = from_time
    while True:
        url = f"{OANDA_BASE_URL}/v3/instruments/{instrument}/candles"
        params = {"granularity": GRANULARITY, "price": "M", "from": current_from, "count": 5000}
        resp = requests.get(url, headers=HEADERS, params=params, timeout=30)
        resp.raise_for_status()
        batch = resp.json()["candles"]
        if not batch:
            break
        all_candles.extend(batch)
        if len(batch) < 5000:
            break
        current_from = batch[-1]["time"]
    rows = [
        {"time": c["time"], "open": float(c["mid"]["o"]), "high": float(c["mid"]["h"]),
         "low": float(c["mid"]["l"]), "close": float(c["mid"]["c"]), "complete": c["complete"]}
        for c in all_candles
    ]
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df[df["complete"]].drop_duplicates(subset="time").reset_index(drop=True)
    df["time"] = pd.to_datetime(df["time"])
    return df


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


def currency_legs(pair: str) -> tuple:
    base, quote = pair.split("_")
    return base, quote


def load_pair_data(instrument: str) -> pd.DataFrame:
    df = get_candles_range(instrument, FROM_TIME)
    df["fast_ma"] = sma(df["close"], FAST_LEN)
    df["slow_ma"] = sma(df["close"], SLOW_LEN)
    df["atr"] = atr_wilder(df, ATR_LEN)
    return df


class Position:
    def __init__(self, pair, direction, entry_price, stop, target, units, risk_dollars, entry_time):
        self.pair = pair
        self.direction = direction
        self.entry_price = entry_price
        self.stop = stop
        self.target = target
        self.units = units
        self.risk_dollars = risk_dollars
        self.entry_time = entry_time


class PendingEntry:
    def __init__(self, direction, bars_waited=0):
        self.direction = direction
        self.bars_waited = bars_waited


def run_portfolio_backtest(pair_data: dict, atr_stop_mult: float) -> dict:
    """GAPCONFIRM + CURCAP always on (matches deployed bot.py exactly) -
    only atr_stop_mult varies."""
    all_times = sorted(set().union(*[set(df["time"]) for df in pair_data.values()]))
    time_index = {pair: {t: i for i, t in enumerate(df["time"])} for pair, df in pair_data.items()}

    equity = STARTING_EQUITY
    peak_equity = equity
    max_drawdown_pct = 0.0
    trades = []

    positions = {}
    pending = {}

    def open_risk_for_currency(ccy):
        return sum(pos.risk_dollars for p, pos in positions.items() if ccy in currency_legs(p))

    for t in all_times:
        for pair in PAIRS:
            idx = time_index[pair].get(t)
            if idx is None:
                continue
            row = pair_data[pair].iloc[idx]
            if pd.isna(row["slow_ma"]) or pd.isna(row["atr"]):
                continue

            pos = positions.get(pair)
            if pos is not None:
                if pos.direction == "LONG":
                    hit_stop = row["low"] <= pos.stop
                    hit_target = row["high"] >= pos.target
                else:
                    hit_stop = row["high"] >= pos.stop
                    hit_target = row["low"] <= pos.target

                if hit_stop or hit_target:
                    exit_price = pos.stop if hit_stop else pos.target
                    if pos.direction == "LONG":
                        pnl = (exit_price - pos.entry_price) * pos.units
                    else:
                        pnl = (pos.entry_price - exit_price) * pos.units
                    equity += pnl
                    peak_equity = max(peak_equity, equity)
                    dd = ((peak_equity - equity) / peak_equity * 100) if peak_equity > 0 else 0
                    max_drawdown_pct = max(max_drawdown_pct, dd)
                    trades.append({"pair": pair, "pnl": pnl})
                    del positions[pair]
                continue

            pend = pending.get(pair)
            if pend is not None and pair not in positions:
                direction_ok = (
                    row["fast_ma"] > row["slow_ma"] if pend.direction == "LONG"
                    else row["fast_ma"] < row["slow_ma"]
                )
                do_enter = False
                if not direction_ok:
                    del pending[pair]
                else:
                    gap_atr = abs(row["fast_ma"] - row["slow_ma"]) / row["atr"] if row["atr"] else 0
                    if gap_atr >= CONFIRM_MIN_GAP_ATR:
                        do_enter = True
                    else:
                        pend.bars_waited += 1
                        if pend.bars_waited >= CONFIRM_MAX_BARS:
                            del pending[pair]

                if do_enter:
                    entry_price = row["open"]
                    stop_distance = row["atr"] * atr_stop_mult
                    if stop_distance <= 0:
                        del pending[pair]
                        continue
                    risk_amount = equity * RISK_PER_TRADE_PCT / 100

                    base, quote = currency_legs(pair)
                    cap_dollars = equity * MAX_CURRENCY_RISK_PCT / 100
                    if (open_risk_for_currency(base) + risk_amount > cap_dollars or
                            open_risk_for_currency(quote) + risk_amount > cap_dollars):
                        del pending[pair]
                        continue

                    units = risk_amount / stop_distance
                    if pend.direction == "LONG":
                        stop = entry_price - stop_distance
                        target = entry_price + stop_distance * RR_RATIO
                    else:
                        stop = entry_price + stop_distance
                        target = entry_price - stop_distance * RR_RATIO

                    positions[pair] = Position(pair, pend.direction, entry_price, stop, target,
                                                units, risk_amount, t)
                    del pending[pair]
                continue

            if pair not in positions and pair not in pending:
                if idx == 0:
                    continue
                prev = pair_data[pair].iloc[idx - 1]
                if pd.isna(prev["slow_ma"]):
                    continue
                fresh_long = prev["fast_ma"] <= prev["slow_ma"] and row["fast_ma"] > row["slow_ma"]
                fresh_short = prev["fast_ma"] >= prev["slow_ma"] and row["fast_ma"] < row["slow_ma"]
                if fresh_long or fresh_short:
                    pending[pair] = PendingEntry("LONG" if fresh_long else "SHORT")

    wins = [t for t in trades if t["pnl"] > 0]
    losses = [t for t in trades if t["pnl"] <= 0]
    gross_profit = sum(t["pnl"] for t in wins)
    gross_loss = abs(sum(t["pnl"] for t in losses))
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float("inf")
    win_rate = (len(wins) / len(trades) * 100) if trades else 0
    net_pnl_pct = (equity - STARTING_EQUITY) / STARTING_EQUITY * 100

    return {
        "trades": len(trades), "win_rate": win_rate, "profit_factor": profit_factor,
        "net_pnl_pct": net_pnl_pct, "max_drawdown_pct": max_drawdown_pct, "final_equity": equity,
    }


def main():
    log.info("Fetching %d H4 candles for %d pairs (full 2015-2026 history) ...", len(PAIRS), len(PAIRS))
    pair_data = {}
    for pair in PAIRS:
        log.info("  %s ...", pair)
        pair_data[pair] = load_pair_data(pair)
        time.sleep(0.3)

    results = []
    for mult in ATR_STOP_MULTS_TO_TEST:
        log.info("Running ATR_STOP_MULT=%.1f (target=%.1fx ATR, R:R fixed at %.1f:1, GAPCONFIRM+CURCAP on)...",
                  mult, mult * RR_RATIO, RR_RATIO)
        r = run_portfolio_backtest(pair_data, atr_stop_mult=mult)
        r["mult"] = mult
        results.append(r)

    print("\n" + "=" * 100)
    print(f"{'Stop=ATRx':>10} {'Target=ATRx':>12} {'Trades':>7} {'Win%':>7} {'ProfitFactor':>13} {'Net P&L%':>10} {'MaxDD%':>8}")
    print("=" * 100)
    for r in results:
        marker = "  <-- current deployed" if r["mult"] == 2.0 else ""
        print(f"{r['mult']:>10.1f} {r['mult']*RR_RATIO:>12.1f} {r['trades']:>7} {r['win_rate']:>6.1f}% "
              f"{r['profit_factor']:>13.3f} {r['net_pnl_pct']:>+9.1f}% {r['max_drawdown_pct']:>7.1f}%{marker}")
    print("=" * 100)
    print(f"\nStarting equity: ${STARTING_EQUITY:,.0f}  |  Shared portfolio equity across all {len(PAIRS)} pairs.")
    print("GAPCONFIRM + CURCAP always on (matches deployed bot.py). R:R fixed at 2:1 throughout - "
          "only the ATR multiple (both stop and target scale together) varies.\n")


if __name__ == "__main__":
    main()
