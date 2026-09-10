"""
Diagnostic: WHY does the London/Overlap opening-range breakout work on
GBP_CHF (PF 1.45-1.75 across every variation, frictionless AND realistic
cost, both walk-forward halves) when it fails on the other 8 pairs?

Follow-up to project_forex_london_overlap_breakout_backtest.md's closing
note. Not another strategy test - a decomposition. Runs the single config
of interest (LONDON RELVOL trigger, enter next bar open, 3x-ATR chandelier
trail, the one that passed frictionless walk-forward) once per pair, plus
several structural measurements, and lays every pair's numbers side by
side so GBP_CHF's difference (if any) is visible.

Sections:
  A. SIGNAL ACCURACY   - does the London/Overlap bar's direction predict
                          the day's close-to-close direction better on
                          GBP_CHF? (unconditional, on RELVOL days, on
                          big-move days, on RELVOL-and-big-move days)
  B. POST-SIGNAL DRIFT - after a LONDON RELVOL signal, how far does the
                          rest of the day travel in the signal direction,
                          in ATR units? (continuation vs reversal)
  C. COST BURDEN        - spread vs how far the pair actually moves; the
                          per-trade cost drag in R-multiples
  D. TRADE ANATOMY      - for the actual CH=3.0 trailing trades: win rate,
                          avg win R, avg loss R, expectancy, hold time -
                          frictionless AND realistic-cost, per pair
  E. GBP_CHF BY YEAR    - is its edge persistent or a couple of good years?
  F. RETURN AUTOCORR    - lag-1 autocorrelation of H4 and daily returns
                          (momentum vs mean-reversion character per pair)

Run (needs OANDA_API_KEY):
    python forex_gbpchf_diagnostic.py
"""

import os
import time

import numpy as np
import pandas as pd
import requests

OANDA_API_KEY = os.environ["OANDA_API_KEY"]
OANDA_BASE_URL = "https://api-fxpractice.oanda.com"
HEADERS = {"Authorization": f"Bearer {OANDA_API_KEY}", "Content-Type": "application/json"}

PAIRS = ["GBP_CHF", "GBP_CAD", "GBP_USD", "GBP_JPY", "GBP_AUD", "USD_CHF", "EUR_USD", "EUR_AUD", "AUD_NZD"]
GRANULARITY = "H4"
FROM_TIME = "2015-01-01T00:00:00Z"

ATR_LEN = 14
ATR_STOP_MULT = 2.0
RISK_PCT = 0.01
CHANDELIER_MULT = 3.0
BIG_MOVE_MULT = 2.0
AVG_LEN = 20
RELVOL_MULT = 1.5

# day = 6 H4 bars from 21:00 UTC: [21:00, 01:00, 05:00, 09:00, 13:00, 17:00]
LONDON_SIG, LONDON_ENTRY = 3, 4     # 09:00-13:00 UTC signal bar, enter at 13:00 open
OVERLAP_SIG, OVERLAP_ENTRY = 4, 5   # 13:00-17:00 UTC signal bar, enter at 17:00 open

# same cost model as the main lab
SPREAD_PIPS = {
    "GBP_CHF": 2.5, "GBP_CAD": 2.6, "GBP_USD": 1.3, "GBP_JPY": 2.2, "GBP_AUD": 2.8,
    "USD_CHF": 1.6, "EUR_USD": 0.9, "EUR_AUD": 1.9, "AUD_NZD": 2.4,
}
SLIPPAGE_PIPS_PER_SIDE = 0.5
PIP_SIZE = {p: (0.01 if p.endswith("_JPY") else 0.0001) for p in PAIRS}


def get_candles_range(instrument: str, from_time: str) -> pd.DataFrame:
    all_candles, current_from = [], from_time
    while True:
        url = f"{OANDA_BASE_URL}/v3/instruments/{instrument}/candles"
        params = {"granularity": GRANULARITY, "price": "M", "from": current_from, "count": 5000}
        resp = requests.get(url, headers=HEADERS, params=params, timeout=30)
        resp.raise_for_status()
        batch = resp.json()["candles"]
        if not batch:
            break
        all_candles.extend([c for c in batch if c["complete"]])
        if len(batch) < 5000:
            break
        current_from = batch[-1]["time"]
        time.sleep(0.2)
    rows = [{"time": pd.Timestamp(c["time"]), "open": float(c["mid"]["o"]), "high": float(c["mid"]["h"]),
             "low": float(c["mid"]["l"]), "close": float(c["mid"]["c"]), "volume": int(c.get("volume", 0))}
            for c in all_candles]
    return pd.DataFrame(rows).drop_duplicates(subset="time").sort_values("time").reset_index(drop=True)


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
    tr = pd.concat([df["high"] - df["low"], (df["high"] - prev_close).abs(),
                    (df["low"] - prev_close).abs()], axis=1).max(axis=1)
    return wilder_rma(tr, length)


def build_days(h4: pd.DataFrame, atr: pd.Series) -> pd.DataFrame:
    first_idx = h4.index[h4["time"].dt.hour == 21]
    if len(first_idx) == 0:
        raise ValueError("no 21:00 UTC bar")
    start = first_idx[0]
    d = h4.iloc[start:].reset_index(drop=True)
    a = atr.iloc[start:].reset_index(drop=True)
    n_days = len(d) // 6
    rows = []
    for i in range(n_days):
        base = i * 6
        chunk = d.iloc[base:base + 6]
        day_open, day_close = chunk["open"].iloc[0], chunk["close"].iloc[-1]
        row = {
            "day_start": chunk["time"].iloc[0],
            "year": chunk["time"].iloc[0].year,
            "day_open": day_open,
            "day_close": day_close,
            "day_ret": (day_close / day_open - 1) * 100,
            "day_dir": 1 if day_close > day_open else (-1 if day_close < day_open else 0),
        }
        for name, sig_i, ent_i in (("LONDON", LONDON_SIG, LONDON_ENTRY), ("OVERLAP", OVERLAP_SIG, OVERLAP_ENTRY)):
            bar = chunk.iloc[sig_i]
            row[f"{name}_vol"] = bar["volume"]
            row[f"{name}_close"] = bar["close"]
            row[f"{name}_dir"] = 1 if bar["close"] > bar["open"] else (-1 if bar["close"] < bar["open"] else 0)
            row[f"{name}_atr"] = a.iloc[base + sig_i]          # ATR through the signal bar
            row[f"{name}_atr_prev"] = a.iloc[base + ent_i - 1]  # ATR used to size the trade
            row[f"{name}_entry_global_idx"] = start + base + ent_i
        rows.append(row)
    df = pd.DataFrame(rows)
    df["avg_abs_ret_20"] = df["day_ret"].abs().rolling(AVG_LEN).mean().shift(1)
    df["big_move"] = df["day_ret"].abs() > BIG_MOVE_MULT * df["avg_abs_ret_20"]
    for name in ("LONDON", "OVERLAP"):
        av = df[f"{name}_vol"].rolling(AVG_LEN).mean().shift(1)
        df[f"{name}_relvol"] = df[f"{name}_vol"] >= RELVOL_MULT * av
    return df


def pf_of(rs):
    rs = list(rs)
    if not rs:
        return float("nan")
    gw = sum(x for x in rs if x > 0)
    gl = abs(sum(x for x in rs if x <= 0))
    return gw / gl if gl > 0 else float("inf")


def cost_r(pair, atr_prev):
    rt_price = (SPREAD_PIPS[pair] + 2 * SLIPPAGE_PIPS_PER_SIDE) * PIP_SIZE[pair]
    return rt_price / (ATR_STOP_MULT * atr_prev)


def trailing_trades(pair, h4, days, window):
    """Run the CH=3.0 chandelier-trail config on `window` (LONDON/OVERLAP)
    RELVOL signals. Returns a list of dicts, one per trade, with gross R,
    cost R, bars held, entry year."""
    flag, dcol = f"{window}_relvol", f"{window}_dir"
    hi_arr = h4["high"].to_numpy()
    lo_arr = h4["low"].to_numpy()
    cl_arr = h4["close"].to_numpy()
    op_arr = h4["open"].to_numpy()
    n = len(h4)
    out = []
    blocked_until = -1
    for _, day in days.iterrows():
        if not bool(day.get(flag, False)):
            continue
        direction = day[dcol]
        if direction == 0:
            continue
        ei = int(day[f"{window}_entry_global_idx"])
        if ei <= blocked_until or ei >= n:
            continue
        a_prev = day[f"{window}_atr_prev"]
        if pd.isna(a_prev) or a_prev <= 0:
            continue
        entry = op_arr[ei]
        init_stop_dist = ATR_STOP_MULT * a_prev
        trail_dist = CHANDELIER_MULT * a_prev
        peak = entry
        stop = entry - init_stop_dist if direction == 1 else entry + init_stop_dist
        exit_i = n - 1
        r_gross = None
        for j in range(ei, n):
            if direction == 1:
                if lo_arr[j] <= stop:
                    r_gross = (stop - entry) / init_stop_dist
                    exit_i = j
                    break
                peak = max(peak, hi_arr[j])
                stop = max(stop, peak - trail_dist)
            else:
                if hi_arr[j] >= stop:
                    r_gross = (entry - stop) / init_stop_dist
                    exit_i = j
                    break
                peak = min(peak, lo_arr[j])
                stop = min(stop, peak + trail_dist)
        if r_gross is None:
            move = (cl_arr[-1] - entry) if direction == 1 else (entry - cl_arr[-1])
            r_gross = move / init_stop_dist
        blocked_until = exit_i
        out.append({
            "year": day["year"],
            "r_gross": r_gross,
            "r_cost": r_gross - cost_r(pair, a_prev),
            "bars_held": exit_i - ei + 1,
            "direction": direction,
        })
    return out


def main():
    print(f"Fetching full {FROM_TIME[:10]}-present H4 history for {len(PAIRS)} pairs...")
    data = {}
    for pair in PAIRS:
        h4 = get_candles_range(pair, FROM_TIME)
        atr = atr_wilder(h4, ATR_LEN)
        days = build_days(h4, atr)
        h4r = h4["close"].pct_change()
        data[pair] = {"h4": h4, "atr": atr, "days": days, "h4ret": h4r}
        print(f"  {pair}: {len(h4)} H4 bars, {len(days)} days, "
              f"{int(days['big_move'].sum())} big-move days ({days['big_move'].mean()*100:.1f}%)")
        time.sleep(0.3)

    # ---------------- A. SIGNAL ACCURACY ----------------
    print("\n" + "=" * 120)
    print("A. SIGNAL DIRECTIONAL ACCURACY - P(opening-range bar direction == day's close-to-close direction)")
    print("=" * 120)
    for window in ("LONDON", "OVERLAP"):
        print(f"\n  {window} bar:")
        print(f"  {'pair':<10} {'all days':>12} {'RELVOL days':>14} {'big-move days':>15} {'RELVOL & big':>14}  (n RELVOL&big)")
        for pair in PAIRS:
            d = data[pair]["days"]
            valid = d[f"{window}_dir"] != 0
            def acc(mask):
                m = valid & mask
                if m.sum() == 0:
                    return float("nan"), 0
                return (d.loc[m, f"{window}_dir"] == d.loc[m, "day_dir"]).mean() * 100, int(m.sum())
            a_all, _ = acc(pd.Series(True, index=d.index))
            a_rv, _ = acc(d[f"{window}_relvol"].fillna(False))
            a_bm, _ = acc(d["big_move"].fillna(False))
            a_rvbm, n_rvbm = acc(d[f"{window}_relvol"].fillna(False) & d["big_move"].fillna(False))
            print(f"  {pair:<10} {a_all:>11.1f}% {a_rv:>13.1f}% {a_bm:>14.1f}% {a_rvbm:>13.1f}%  ({n_rvbm})")

    # ---------------- B. POST-SIGNAL DRIFT ----------------
    print("\n" + "=" * 120)
    print("B. POST-SIGNAL DRIFT - after a LONDON RELVOL signal, mean move from signal-bar close to day close,")
    print("   in the signal direction, measured in ATR(signal-bar) units.  >0 = continuation, <=0 = reversal/chop")
    print("=" * 120)
    print(f"  {'pair':<10} {'mean drift (ATR)':>18} {'median drift (ATR)':>20} {'% continued':>14}  (n signals)")
    for pair in PAIRS:
        d = data[pair]["days"]
        sig = d[d["LONDON_relvol"].fillna(False) & (d["LONDON_dir"] != 0)]
        drift = (sig["day_close"] - sig["LONDON_close"]) * sig["LONDON_dir"] / sig["LONDON_atr"]
        cont = (drift > 0).mean() * 100
        print(f"  {pair:<10} {drift.mean():>18.3f} {drift.median():>20.3f} {cont:>13.1f}%  ({len(sig)})")

    # ---------------- C. COST BURDEN ----------------
    print("\n" + "=" * 120)
    print("C. COST BURDEN - spread vs how far the pair actually moves")
    print("=" * 120)
    print(f"  {'pair':<10} {'spread(pip)':>12} {'med |dayRet|(pip)':>19} {'med ATR H4(pip)':>17} "
          f"{'spread/dayMove':>15} {'cost per trade (R)':>19}")
    for pair in PAIRS:
        d = data[pair]["days"]
        h4 = data[pair]["h4"]
        atr = data[pair]["atr"]
        pip = PIP_SIZE[pair]
        med_day_pips = (d["day_ret"].abs() / 100 * d["day_open"]).median() / pip
        med_atr_pips = atr.median() / pip
        spread_pips = SPREAD_PIPS[pair]
        rt_pips = spread_pips + 2 * SLIPPAGE_PIPS_PER_SIDE
        cost_per_trade_r = (rt_pips * pip) / (ATR_STOP_MULT * atr.median())
        print(f"  {pair:<10} {spread_pips:>12.1f} {med_day_pips:>19.1f} {med_atr_pips:>17.1f} "
              f"{spread_pips / med_day_pips * 100:>14.1f}% {cost_per_trade_r:>19.3f}")

    # ---------------- D. TRADE ANATOMY ----------------
    print("\n" + "=" * 120)
    print("D. TRADE ANATOMY - LONDON RELVOL, 3x-ATR chandelier trail (the config that passed frictionless WF)")
    print("=" * 120)
    trades_cache = {p: trailing_trades(p, data[p]["h4"], data[p]["days"], "LONDON") for p in PAIRS}
    for cost_label in ("FRICTIONLESS", "REALISTIC COST"):
        rk = "r_gross" if cost_label == "FRICTIONLESS" else "r_cost"
        print(f"\n  {cost_label}:")
        print(f"  {'pair':<10} {'n':>5} {'win%':>7} {'avgWin R':>10} {'avgLoss R':>11} "
              f"{'expectancy R':>13} {'PF':>7} {'med bars held':>14}")
        for pair in PAIRS:
            tr = trades_cache[pair]
            rs = [t[rk] for t in tr]
            wins = [x for x in rs if x > 0]
            losses = [x for x in rs if x <= 0]
            win_pct = len(wins) / len(rs) * 100 if rs else float("nan")
            aw = np.mean(wins) if wins else float("nan")
            al = np.mean(losses) if losses else float("nan")
            exp = np.mean(rs) if rs else float("nan")
            held = np.median([t["bars_held"] for t in tr]) if tr else float("nan")
            print(f"  {pair:<10} {len(rs):>5} {win_pct:>6.1f}% {aw:>10.3f} {al:>11.3f} "
                  f"{exp:>13.3f} {pf_of(rs):>7.3f} {held:>14.0f}")

    # ---------------- E. GBP_CHF BY YEAR ----------------
    print("\n" + "=" * 120)
    print("E. GBP_CHF YEAR BY YEAR - LONDON RELVOL, 3x-ATR trail, REALISTIC COST")
    print("=" * 120)
    by_year = {}
    for t in trades_cache["GBP_CHF"]:
        by_year.setdefault(t["year"], []).append(t["r_cost"])
    print(f"  {'year':<6} {'n':>5} {'win%':>7} {'expectancy R':>13} {'PF':>7} {'sum R':>9}")
    for yr in sorted(by_year):
        rs = by_year[yr]
        wp = sum(1 for x in rs if x > 0) / len(rs) * 100
        print(f"  {yr:<6} {len(rs):>5} {wp:>6.1f}% {np.mean(rs):>13.3f} {pf_of(rs):>7.3f} {sum(rs):>9.2f}")
    # same for the 3 that "worked" vs a couple that didn't, one line each
    print("\n  (for contrast, same config REALISTIC COST, whole-sample:)")
    for pair in ("GBP_CHF", "EUR_AUD", "AUD_NZD", "GBP_USD", "USD_CHF", "EUR_USD"):
        rs = [t["r_cost"] for t in trades_cache[pair]]
        print(f"    {pair:<10} n={len(rs):>4}  PF={pf_of(rs):.3f}  expR={np.mean(rs):+.3f}  sumR={sum(rs):+.1f}")

    # ---------------- F. RETURN AUTOCORRELATION ----------------
    print("\n" + "=" * 120)
    print("F. RETURN AUTOCORRELATION (lag-1) - momentum (>0) vs mean-reversion (<0) character")
    print("=" * 120)
    print(f"  {'pair':<10} {'H4 ret acf(1)':>15} {'daily ret acf(1)':>18} {'H4 ret |skew|':>15} {'daily ret kurtosis':>20}")
    for pair in PAIRS:
        h4r = data[pair]["h4ret"].dropna()
        dr = data[pair]["days"]["day_ret"].dropna()
        acf1_h4 = h4r.autocorr(1)
        acf1_d = dr.autocorr(1)
        print(f"  {pair:<10} {acf1_h4:>15.4f} {acf1_d:>18.4f} {abs(h4r.skew()):>15.3f} {dr.kurt():>20.3f}")

    print("\n" + "=" * 120)
    print("Read A+B for whether the SIGNAL is genuinely better on GBP_CHF; C for whether it's just")
    print("cheaper to trade relative to its moves; D for whether the edge is win-rate or payoff;")
    print("E for persistence; F for the pair's underlying momentum/reversion character.")
    print("=" * 120 + "\n")


if __name__ == "__main__":
    main()
