"""
Forex analog of qqq_opening_range_signature_lab.py - the same-day
"opening range predicts the day's own big-move-ness and direction" test,
adapted to FX where there is no single RTH open like QQQ has.

DAY DEFINITION: OANDA's H4 bars for these 9 pairs land on fixed UTC
boundaries - 21:00, 01:00, 05:00, 09:00, 13:00, 17:00 - six bars per
trading day, with 21:00 UTC matching the standard forex "day rollover"
(5pm New York) that MT4/MT5 and most retail platforms use for their own
D1 candles. Bars are grouped into days of exactly 6 H4 bars starting at
the first 21:00 UTC bar in each pair's history.

Three candidate "opening range" windows are tested, since FX has no
single obvious equivalent of "the first 5 minutes of RTH":
  - FIRST   the 21:00-01:00 UTC bar (literal day-open, Sydney/early-Asia,
            lowest liquidity of the day - included for completeness)
  - LONDON  the 09:00-13:00 UTC bar (pure London session, the most
            liquid single-session window for these pairs - the closest
            economic analog to QQQ's RTH open)
  - OVERLAP the 13:00-17:00 UTC bar (London/NY overlap - the single
            highest-liquidity window of the FX day)

For each window: does elevated tick-volume or range in that bar predict
the DAY becomes a big-move day (same adaptive definition as every other
lab in this project - |day return| > 2.0x trailing 20-day avg |return|,
per pair, shifted, no lookahead)? And separately - the QQQ study's
headline finding - does that window's OWN direction match the day's
eventual close-to-close direction, and does that match rate improve
specifically on big-move days?

No separate "day gap" candidate: unlike QQQ's real overnight gap
(market literally closed), FX trades continuously through the 21:00 UTC
boundary on weeknights, so there's no real gap there to test - the one
real FX gap (Friday close -> Sunday reopen) is already covered by
WEEKEND_GAP_TOP20 in forex_premove_pattern_lab.py.

Run (needs OANDA_API_KEY - the candles endpoint doesn't need an account ID):
    python forex_opening_range_lab.py
"""

import os

import numpy as np
import pandas as pd
import requests

OANDA_API_KEY = os.environ["OANDA_API_KEY"]
OANDA_BASE_URL = "https://api-fxpractice.oanda.com"
HEADERS = {"Authorization": f"Bearer {OANDA_API_KEY}", "Content-Type": "application/json"}

PAIRS = ["GBP_CHF", "GBP_CAD", "GBP_USD", "GBP_JPY", "GBP_AUD", "USD_CHF", "EUR_USD", "EUR_AUD", "AUD_NZD"]
GRANULARITY = "H4"
CANDLE_COUNT = 5000

BIG_MOVE_MULT = 2.0
AVG_RET_LEN = 20
PCTL_LOOKBACK = 60
PCTL_THRESH = 0.20
RELVOL_MULT = 1.5

# (label, bar-index-within-day) - day = [21:00, 01:00, 05:00, 09:00, 13:00, 17:00] UTC
OR_WINDOWS = [("FIRST", 0), ("LONDON", 3), ("OVERLAP", 4)]


def get_candles(instrument: str) -> pd.DataFrame:
    url = f"{OANDA_BASE_URL}/v3/instruments/{instrument}/candles"
    params = {"count": CANDLE_COUNT, "granularity": GRANULARITY, "price": "M"}
    resp = requests.get(url, headers=HEADERS, params=params, timeout=30)
    resp.raise_for_status()
    rows = [{"time": pd.Timestamp(c["time"]), "open": float(c["mid"]["o"]), "high": float(c["mid"]["h"]),
             "low": float(c["mid"]["l"]), "close": float(c["mid"]["c"]), "volume": int(c.get("volume", 0))}
            for c in resp.json()["candles"] if c["complete"]]
    df = pd.DataFrame(rows).sort_values("time").reset_index(drop=True)
    return df


def build_daily(df: pd.DataFrame, instrument: str) -> pd.DataFrame:
    d = df.copy()
    first_idx = d.index[d["time"].dt.hour == 21]
    if len(first_idx) == 0:
        raise ValueError(f"{instrument}: no 21:00 UTC bar found - can't align day boundaries")
    d = d.iloc[first_idx[0]:].reset_index(drop=True)
    n_days = len(d) // 6
    d = d.iloc[: n_days * 6].reset_index(drop=True)

    rows = []
    for i in range(n_days):
        chunk = d.iloc[i * 6: i * 6 + 6]
        day_open, day_close = chunk["open"].iloc[0], chunk["close"].iloc[-1]
        row = {
            "instrument": instrument,
            "day_start": chunk["time"].iloc[0],
            "day_ret": (day_close / day_open - 1) * 100,
            "day_range_pct": (chunk["high"].max() - chunk["low"].min()) / day_open * 100,
        }
        for label, idx in OR_WINDOWS:
            bar = chunk.iloc[idx]
            row[f"{label}_range_pct"] = (bar["high"] - bar["low"]) / bar["open"] * 100
            row[f"{label}_vol"] = bar["volume"]
            row[f"{label}_dir"] = 1 if bar["close"] > bar["open"] else (-1 if bar["close"] < bar["open"] else 0)
        rows.append(row)
    return pd.DataFrame(rows)


def build_features(dd: pd.DataFrame) -> pd.DataFrame:
    dd = dd.sort_values("day_start").reset_index(drop=True)
    dd["avg_abs_ret_20"] = dd["day_ret"].abs().rolling(AVG_RET_LEN).mean().shift(1)
    dd["big_move"] = dd["day_ret"].abs() > BIG_MOVE_MULT * dd["avg_abs_ret_20"]
    dd["day_dir"] = np.where(dd["day_ret"] > 0, 1, np.where(dd["day_ret"] < 0, -1, 0))

    for label, _ in OR_WINDOWS:
        avg_vol_20 = dd[f"{label}_vol"].rolling(AVG_RET_LEN).mean().shift(1)
        dd[f"{label}_relvol"] = dd[f"{label}_vol"] >= RELVOL_MULT * avg_vol_20
        range_pctl = dd[f"{label}_range_pct"].rolling(PCTL_LOOKBACK).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False)
        dd[f"{label}_range_top20"] = range_pctl >= (1 - PCTL_THRESH)
    return dd


def score(mask: pd.Series, big_move: pd.Series, valid: pd.Series) -> dict:
    m, b = mask[valid], big_move[valid]
    n = len(m)
    if n == 0:
        return {"n": 0, "base_rate": float("nan"), "hit_rate": float("nan"), "lift": float("nan"), "n_big": 0}
    base_rate = m.mean()
    n_big = int(b.sum())
    if n_big == 0 or base_rate == 0:
        return {"n": n, "base_rate": base_rate, "hit_rate": float("nan"), "lift": float("nan"), "n_big": n_big}
    hit_rate = m[b].mean()
    return {"n": n, "base_rate": base_rate, "hit_rate": hit_rate, "lift": hit_rate / base_rate, "n_big": n_big}


def fmt(s: dict) -> str:
    if s["n_big"] == 0 or pd.isna(s["lift"]):
        return "n/a"
    return f"base={s['base_rate']*100:5.1f}% hit={s['hit_rate']*100:5.1f}% lift={s['lift']:5.2f}x (n_big={s['n_big']})"


CANDIDATES = {}
for _label, _ in OR_WINDOWS:
    CANDIDATES[f"{_label}_RELVOL"] = (lambda d, lab=_label: d[f"{lab}_relvol"])
    CANDIDATES[f"{_label}_RANGE_TOP20"] = (lambda d, lab=_label: d[f"{lab}_range_top20"])


def main():
    print("Fetching H4 candles for 9 pairs from OANDA and building daily bars (21:00 UTC rollover)...")
    per_pair = {}
    for pair in PAIRS:
        raw = get_candles(pair)
        daily = build_features(build_daily(raw, pair))
        per_pair[pair] = daily
        print(f"  {pair}: {len(raw)} H4 bars -> {len(daily)} trading days, "
              f"{daily['day_start'].iloc[0]} to {daily['day_start'].iloc[-1]}")

    warmup = max(PCTL_LOOKBACK, AVG_RET_LEN) + 5
    pooled = pd.concat([df.iloc[warmup:] for df in per_pair.values()], ignore_index=True)
    pooled = pooled.sort_values("day_start").reset_index(drop=True)

    times = pooled["day_start"]
    mid = times.iloc[len(times) // 2]
    early_mask, late_mask = times < mid, times >= mid

    n_big_total = int(pooled["big_move"].sum())
    print("\n" + "=" * 115)
    print(f"FOREX OPENING-RANGE SIGNATURE SCREEN (pooled, 9 pairs) - {len(pooled)} trading days, "
          f"{n_big_total} big-move days ({n_big_total/len(pooled)*100:.1f}%)")
    print("'Big move' = |day close-to-close return| > 2.0x trailing 20-day avg |return|, per pair, shifted")
    print("Day = 6 H4 bars starting 21:00 UTC. FIRST=21:00-01:00 (Asia open), "
          "LONDON=09:00-13:00, OVERLAP=13:00-17:00 (London/NY)")
    print("=" * 115)
    print(f"{'Candidate':<24} {'FULL SAMPLE':<40} {'EARLY HALF':<40} {'LATE HALF'}")
    print("-" * 115)

    results = []
    for name, fn in CANDIDATES.items():
        mask = fn(pooled).fillna(False).infer_objects(copy=False)
        full = score(mask, pooled["big_move"], pd.Series(True, index=pooled.index))
        early = score(mask, pooled["big_move"], early_mask)
        late = score(mask, pooled["big_move"], late_mask)
        results.append((name, full, early, late))
        print(f"{name:<24} {fmt(full):<40} {fmt(early):<40} {fmt(late)}")

    print("=" * 115)
    print("VERDICT (lift > 1.0 in BOTH halves, and >=15 big-move days per half to trust it):")
    survivors = []
    for name, full, early, late in results:
        if early["n_big"] < 15 or late["n_big"] < 15:
            print(f"  {name:<24} SKIP - too few big-move days (early={early['n_big']}, late={late['n_big']})")
            continue
        if early["lift"] > 1.05 and late["lift"] > 1.05:
            print(f"  {name:<24} SURVIVES (early {early['lift']:.2f}x, late {late['lift']:.2f}x)")
            survivors.append(name)
        else:
            print(f"  {name:<24} FAILS (early {early['lift']:.2f}x, late {late['lift']:.2f}x)")
    print()
    if survivors:
        print(f"{len(survivors)} candidate(s) survived pooled: {survivors}")
    else:
        print("No candidate survived the pooled walk-forward bar - a clean negative result.")
    print("=" * 115 + "\n")

    if survivors:
        print("=" * 115)
        print("PER-PAIR BREAKDOWN of survivors (does it hold up pair-by-pair, or is it 1-2 pairs driving it?)")
        print("=" * 115)
        for name in survivors:
            print(f"\n{name}:")
            for pair, df in per_pair.items():
                dd = df.iloc[warmup:].reset_index(drop=True)
                mask = CANDIDATES[name](dd).fillna(False).infer_objects(copy=False)
                s = score(mask, dd["big_move"], pd.Series(True, index=dd.index))
                print(f"  {pair:<10} {fmt(s)}")
        print("=" * 115 + "\n")

    # ---------------------------------------------------------------------
    # FOLLOW-UP: the QQQ study's headline check - does the opening-range
    # window's OWN direction match the day's eventual close-to-close
    # direction, and does that improve specifically on big-move days?
    # ---------------------------------------------------------------------
    print("=" * 115)
    print("FOLLOW-UP - does each opening-range window's direction predict the DAY's eventual direction?")
    print("(QQQ found this jumped from ~60% on all days to 72-76% on big-move days specifically)")
    print("=" * 115)
    for label, _ in OR_WINDOWS:
        dir_col = f"{label}_dir"
        valid = pooled[dir_col] != 0
        matches_all = pooled.loc[valid, dir_col] == pooled.loc[valid, "day_dir"]
        all_rate = matches_all.mean() * 100
        big_valid = valid & pooled["big_move"]
        matches_big = pooled.loc[big_valid, dir_col] == pooled.loc[big_valid, "day_dir"]
        big_rate = matches_big.mean() * 100 if big_valid.sum() > 0 else float("nan")
        print(f"{label:<10} ALL DAYS: {all_rate:5.1f}% (n={valid.sum():5d})   "
              f"BIG-MOVE DAYS ONLY: {big_rate:5.1f}% (n={int(big_valid.sum())})")
    print("=" * 115 + "\n")


if __name__ == "__main__":
    main()
