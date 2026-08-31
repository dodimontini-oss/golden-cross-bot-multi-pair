"""
Cross-market validation of the crypto result from
crypto_bot/multi_strategy_lab.py + donchian_deep_dive.py.

On 17-pair Daily crypto, a Donchian channel breakout that additionally
requires the close to clear the channel by a minimum ATR margin
("breakout strength") comfortably beat the deployed Golden Cross:
PF 1.966 vs 1.406, +120% vs +25% net, 13.6% vs 17.4% max drawdown, and it
passed a walk-forward split in both halves. It was also robust to channel
length (a broad 20-90 plateau, not a spike) and was not carried by any
single coin (12 of 17 pairs profitable, best pair only 16.7% of net profit).

The open question this script answers: is that a REAL structural edge, or
just a crypto artifact? Project methodology says never assume a result
transfers across instruments/timeframes - the GAPCONFIRM finding won on
forex and then explicitly did NOT transfer to crypto, so the reverse
direction has to be tested too, not assumed.

Ported deliberately, with two honest adaptations:
  - No BTC_REGIME analog exists. That filter was the single biggest
    contributor on crypto (BTC leads the whole complex in a way no forex
    pair leads the others). Forex has no equivalent single leader, so the
    port drops it rather than inventing a fake substitute. SELF_TREND
    (price vs its own 200-period SMA) is tested instead as the closest
    honest regime analog.
  - Forex allows shorts (the deployed bot trades both directions and
    crypto's long-only constraint is an Alpaca platform limit, not a
    strategy choice), so Donchian is tested long AND short here - upper
    channel break = long, lower channel break = short.

Costs: unlike the existing forex labs in this folder, which model NO
transaction cost at all, this charges a round-trip spread on every trade
(SPREAD_PIPS, converted per-pair so JPY crosses use the right pip size).
That makes this script's BASELINE row NOT directly comparable to the
previously-recorded forex numbers - compare within this script's own
output only, which is why BASELINE (the deployed Golden Cross logic) is
re-run here rather than quoted from memory.

Run: python donchian_forex_lab.py

Environment variables required:
    OANDA_API_KEY
"""

import logging
import os
import time

import numpy as np
import pandas as pd
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("donchian-forex-lab")

OANDA_API_KEY = os.environ["OANDA_API_KEY"]
OANDA_BASE_URL = "https://api-fxpractice.oanda.com"
HEADERS = {"Authorization": f"Bearer {OANDA_API_KEY}"}

GRANULARITY = "H4"
FROM_TIME = "2015-01-01T00:00:00Z"

FAST_LEN, SLOW_LEN = 50, 200
ATR_LEN = 14
ATR_STOP_MULT = 2.0          # validated best for forex H4 (stop-distance sweep) - reused, not re-tuned
RR_RATIO = 2.0
RISK_PER_TRADE_PCT = 1.0
STARTING_EQUITY = 10000.0
SPREAD_PIPS = 1.5            # round-trip, charged once per trade

PAIRS = [
    "GBP_CHF", "GBP_CAD", "GBP_USD", "GBP_JPY", "GBP_AUD",
    "USD_CHF", "EUR_USD", "EUR_AUD", "AUD_NZD",
]


def pip_size(pair: str) -> float:
    return 0.01 if pair.endswith("_JPY") else 0.0001


# ---------------- data ----------------

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
        {"time": pd.to_datetime(c["time"]), "open": float(c["mid"]["o"]), "high": float(c["mid"]["h"]),
         "low": float(c["mid"]["l"]), "close": float(c["mid"]["c"])}
        for c in all_candles if c["complete"]
    ]
    df = pd.DataFrame(rows).drop_duplicates(subset="time").reset_index(drop=True)
    return df


def wilder_rma(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(alpha=1.0 / n, adjust=False).mean()


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    c = df["close"]
    prev = c.shift(1)
    tr = pd.concat([df["high"] - df["low"], (df["high"] - prev).abs(), (df["low"] - prev).abs()], axis=1).max(axis=1)
    df["atr"] = wilder_rma(tr, ATR_LEN)
    df["fast_ma"] = c.rolling(FAST_LEN).mean()
    df["slow_ma"] = c.rolling(SLOW_LEN).mean()
    df["sma200"] = c.rolling(200).mean()
    for n in (20, 55, 100):
        df[f"dc_hi{n}"] = df["high"].rolling(n).max().shift(1)
        df[f"dc_lo{n}"] = df["low"].rolling(n).min().shift(1)
    return df


# ---------------- signals: return (+1 long, -1 short, 0 flat) per bar ----------------

def sig_golden_cross(d):
    f, s = d["fast_ma"], d["slow_ma"]
    up = (f.shift(1) <= s.shift(1)) & (f > s)
    dn = (f.shift(1) >= s.shift(1)) & (f < s)
    return np.where(up, 1, np.where(dn, -1, 0))


def sig_donchian(n, atr_mult=0.0, long_only=False):
    def _f(d):
        up = d["close"] > (d[f"dc_hi{n}"] + atr_mult * d["atr"])
        dn = d["close"] < (d[f"dc_lo{n}"] - atr_mult * d["atr"])
        if long_only:
            return np.where(up, 1, 0)
        return np.where(up, 1, np.where(dn, -1, 0))
    return _f


def gate_none(d):
    return pd.Series(True, index=d.index)


def gate_self_trend(d):
    """Only take longs above the 200SMA and shorts below it."""
    return pd.Series(True, index=d.index)  # direction-aware handling happens in the engine


# ---------------- engine ----------------

def prep(pair_data, signal_fn, use_self_trend):
    out = {}
    for pair, df in pair_data.items():
        sig = signal_fn(df)
        sig = np.nan_to_num(np.asarray(sig, dtype=float)).astype(int)
        if use_self_trend:
            above = (df["close"] > df["sma200"]).fillna(False).to_numpy()
            sig = np.where((sig == 1) & ~above, 0, sig)
            sig = np.where((sig == -1) & above, 0, sig)
        out[pair] = {
            "time": df["time"].to_numpy(),
            "open": df["open"].to_numpy(float), "high": df["high"].to_numpy(float),
            "low": df["low"].to_numpy(float), "atr": df["atr"].to_numpy(float),
            "sig": sig, "pip": pip_size(pair),
        }
    return out


def run_portfolio(prepped, date_filter=None):
    all_times = sorted(set().union(*[set(a["time"]) for a in prepped.values()]))
    if date_filter is not None:
        lo, hi = date_filter
        all_times = [t for t in all_times if lo <= t < hi]
    idx_of = {p: {t: i for i, t in enumerate(a["time"])} for p, a in prepped.items()}

    equity = peak = STARTING_EQUITY
    max_dd = 0.0
    trades = []
    positions = {}
    pending = {}

    for t in all_times:
        for pair in list(positions):
            a = prepped[pair]
            i = idx_of[pair].get(t)
            if i is None:
                continue
            p = positions[pair]
            if p["dir"] == 1:
                hit_stop, hit_tgt = a["low"][i] <= p["stop"], a["high"][i] >= p["target"]
            else:
                hit_stop, hit_tgt = a["high"][i] >= p["stop"], a["low"][i] <= p["target"]
            if not (hit_stop or hit_tgt):
                continue
            exit_price = p["stop"] if hit_stop else p["target"]   # stop assumed first if both touched
            pnl = (exit_price - p["entry"]) * p["units"] * p["dir"] - p["cost"]
            equity += pnl
            peak = max(peak, equity)
            max_dd = max(max_dd, (peak - equity) / peak * 100 if peak > 0 else 0)
            trades.append({"pair": pair, "pnl": pnl})
            del positions[pair]

        for pair, d in list(pending.items()):
            a = prepped[pair]
            i = idx_of[pair].get(t)
            if i is None or pair in positions:
                continue
            entry = a["open"][i]
            atr = a["atr"][i]
            if not np.isfinite(atr) or atr <= 0:
                continue
            sd = atr * ATR_STOP_MULT
            units = (equity * RISK_PER_TRADE_PCT / 100) / sd
            positions[pair] = {
                "dir": d, "entry": entry, "units": units,
                "stop": entry - sd * d, "target": entry + sd * RR_RATIO * d,
                "cost": SPREAD_PIPS * a["pip"] * units,
            }
        pending.clear()

        for pair, a in prepped.items():
            i = idx_of[pair].get(t)
            if i is None or pair in positions:
                continue
            s = a["sig"][i]
            if s != 0:
                pending[pair] = int(s)

    wins = [x for x in trades if x["pnl"] > 0]
    losses = [x for x in trades if x["pnl"] <= 0]
    gp, gl = sum(x["pnl"] for x in wins), abs(sum(x["pnl"] for x in losses))
    net = (equity - STARTING_EQUITY) / STARTING_EQUITY * 100
    return {
        "trades": len(trades), "win_rate": (len(wins) / len(trades) * 100) if trades else 0.0,
        "profit_factor": (gp / gl) if gl > 0 else 0.0, "net_pnl_pct": net,
        "max_drawdown_pct": max_dd, "return_over_dd": (net / max_dd) if max_dd > 0 else float("nan"),
    }


HDR = (f"{'Config':<50} {'Trades':>7} {'Win%':>7} {'PF':>8} {'Net P&L%':>11} {'MaxDD%':>8} {'Ret/DD':>7}")


def line(label, r, extra=""):
    return (f"{label:<50} {r['trades']:>7} {r['win_rate']:>6.1f}% {r['profit_factor']:>8.3f} "
            f"{r['net_pnl_pct']:>+10.1f}% {r['max_drawdown_pct']:>7.1f}% {r['return_over_dd']:>7.2f}{extra}")


def main():
    log.info("Fetching %s candles for %d forex pairs from %s ...", GRANULARITY, len(PAIRS), FROM_TIME)
    pair_data = {}
    for p in PAIRS:
        log.info("  %s ...", p)
        pair_data[p] = add_indicators(get_candles_range(p, FROM_TIME))
        time.sleep(0.2)
    log.info("Loaded %d pairs.", len(pair_data))

    configs = [
        ("GOLDEN_CROSS 50/200 (deployed logic) - REFERENCE", sig_golden_cross, False),
        ("DONCHIAN_20 (long+short)", sig_donchian(20), False),
        ("DONCHIAN_55 (long+short)", sig_donchian(55), False),
        ("DONCHIAN_100 (long+short)", sig_donchian(100), False),
        ("DONCHIAN_55 + SELF_TREND", sig_donchian(55), True),
        ("DONCHIAN_55 + STRENGTH 0.5xATR", sig_donchian(55, 0.5), False),
        ("DONCHIAN_55 + STRENGTH 1.0xATR", sig_donchian(55, 1.0), False),
        ("DONCHIAN_55 + STRENGTH 1.0 + SELF_TREND", sig_donchian(55, 1.0), True),
        ("DONCHIAN_55 long-only (crypto-comparable)", sig_donchian(55, 0.0, True), False),
        ("DONCHIAN_55 long-only + STRENGTH 1.0", sig_donchian(55, 1.0, True), False),
    ]

    print("\n" + "=" * 120)
    print(f"PART 1: FOREX H4, 9 pairs, {SPREAD_PIPS}-pip round-trip spread charged, ATR{ATR_STOP_MULT} stop / 2:1 target, 1% risk")
    print("=" * 120)
    print(HDR)
    print("-" * 120)
    results = {}
    for label, fn, st in configs:
        r = run_portfolio(prep(pair_data, fn, st))
        results[label] = (r, fn, st)
        print(line(label, r))
    print("=" * 120)

    all_t = sorted(set().union(*[set(d["time"]) for d in pair_data.values()]))
    mid = all_t[len(all_t) // 2]
    lo, hi = all_t[0], all_t[-1] + pd.Timedelta(hours=4)
    print("\n" + "=" * 120)
    print(f"PART 2: WALK-FORWARD  (EARLY {str(lo)[:10]} -> {str(mid)[:10]}  |  LATE {str(mid)[:10]} -> {str(all_t[-1])[:10]})")
    print("=" * 120)
    print(f"{'Config':<50} {'EARLY PF':>9} {'EARLY P&L%':>12} {'LATE PF':>9} {'LATE P&L%':>12}   Verdict")
    print("-" * 120)
    for label, (r, fn, st) in results.items():
        p = prep(pair_data, fn, st)
        e = run_portfolio(p, date_filter=(lo, mid))
        l = run_portfolio(p, date_filter=(mid, hi))
        ok = e["profit_factor"] > 1.0 and l["profit_factor"] > 1.0
        print(f"{label:<50} {e['profit_factor']:>9.3f} {e['net_pnl_pct']:>+11.1f}% "
              f"{l['profit_factor']:>9.3f} {l['net_pnl_pct']:>+11.1f}%   {'PASS' if ok else 'fail'}")
    print("=" * 120)
    print("\nThe question this answers: does the crypto Donchian+breakout-strength win GENERALIZE, or was it")
    print("a crypto-specific artifact? Compare every Donchian row to the GOLDEN_CROSS reference row above.\n")


if __name__ == "__main__":
    main()
