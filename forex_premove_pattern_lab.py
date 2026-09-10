"""
Forex analog of qqq_premove_pattern_lab.py / qqq_opening_range_signature_lab.py -
screens the same style of candidate "precursor" patterns against the 9
pairs golden-cross-bot-multi-pair actually trades, at H4 (the bot's own
native timeframe, so a "big move" here is defined at exactly the
granularity that triggers real trades - not a daily aggregate).

BIG MOVE DEFINITION: identical spirit to the QQQ labs - a bar's
|H4 return| > BIG_MOVE_MULT x the trailing 20-bar average |H4 return|
(shifted, no lookahead). Adapts to each pair's own volatility regime
rather than a fixed-pips threshold, so GBP_JPY (high vol) and AUD_NZD
(low vol) are judged on their own terms.

Pooled across all 9 live pairs for sample size (FX H4 history gives far
fewer bars per pair than QQQ's 5-min equity data), with a per-pair
breakdown printed too so a pooled "survives" isn't secretly just one
pair's idiosyncrasy.

CANDIDATES SCREENED (evaluated on bar D-1, predicting bar D):
  1. NR4 / NR7        - D-1's range is the narrowest of the last 4 / 7 H4 bars
  2. ATR_PCTL20        - D-1's ATR(14) in the bottom 20th %ile of its own
                          trailing 60-bar distribution (a "squeeze")
  3. TICKVOL_DOWN      - D-1's OANDA tick-count ("volume") below its own
                          trailing 20-bar average (FX has no real trade
                          volume - this is OANDA's tick-count proxy,
                          flagged as such, not treated as real volume)
  4. PRIOR_BIG_1       - D-1 was itself a big-move bar
  5. PRIOR_BIG_1-3     - any of D-1..D-3 was a big-move bar (volatility
                          clustering analog)
  6. INSIDE_BAR        - D-1's range sits entirely inside D-2's range
  7. WEEKEND_GAP_TOP20 - the Friday-close-to-Sunday-open gap magnitude in
                          the top 20th %ile of its own trailing 60-week
                          distribution (tested only on the first H4 bar of
                          the week, a genuinely different mechanism from
                          the other candidates - FX has no overnight gap
                          most nights, but a real one every weekend)

Same walk-forward bar as every other finding in this project: only counts
as real if lift > 1.0 in BOTH the early and late half of the pooled
sample.

Run (needs OANDA_API_KEY - the candles endpoint doesn't need an account ID):
    python forex_premove_pattern_lab.py
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
CANDLE_COUNT = 5000  # OANDA's per-request max; ~5000 H4 bars is several years of history

BIG_MOVE_MULT = 2.0
AVG_RET_LEN = 20
ATR_LEN = 14
PCTL_LOOKBACK = 60
PCTL_THRESH = 0.20


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


def wilder_rma(series: pd.Series, length: int) -> pd.Series:
    rma = pd.Series(index=series.index, dtype=float)
    if len(series) < length:
        return rma
    rma.iloc[length - 1] = series.iloc[:length].mean()
    for i in range(length, len(series)):
        rma.iloc[i] = (rma.iloc[i - 1] * (length - 1) + series.iloc[i]) / length
    return rma


def build_features(df: pd.DataFrame, instrument: str) -> pd.DataFrame:
    d = df.copy()
    d["instrument"] = instrument
    d["ret"] = d["close"].pct_change() * 100
    d["range"] = d["high"] - d["low"]

    prev_close = d["close"].shift(1)
    tr = pd.concat([d["high"] - d["low"], (d["high"] - prev_close).abs(),
                    (d["low"] - prev_close).abs()], axis=1).max(axis=1)
    d["atr"] = wilder_rma(tr, ATR_LEN)

    d["avg_abs_ret_20"] = d["ret"].abs().rolling(AVG_RET_LEN).mean().shift(1)
    d["big_move"] = d["ret"].abs() > BIG_MOVE_MULT * d["avg_abs_ret_20"]

    for w in (4, 7):
        d[f"nr{w}"] = d["range"] == d["range"].rolling(w).min()

    atr_pctl = d["atr"].rolling(PCTL_LOOKBACK).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False)
    d["atr_squeeze"] = atr_pctl <= PCTL_THRESH

    avg_vol_20 = d["volume"].rolling(AVG_RET_LEN).mean()
    d["tickvol_down"] = d["volume"] < avg_vol_20

    d["prior_big_1"] = d["big_move"].shift(1).fillna(False)
    pb13 = pd.Series(False, index=d.index)
    for lag in (1, 2, 3):
        pb13 = pb13 | d["big_move"].shift(lag).fillna(False)
    d["prior_big_1_3"] = pb13

    prev_high, prev_low = d["high"].shift(1), d["low"].shift(1)
    d["inside_bar"] = (d["high"] <= prev_high) & (d["low"] >= prev_low)

    # Weekend gap: flag only the FIRST H4 bar after a gap of > 24h since the previous bar
    # (i.e. the Sunday/Monday reopen after Friday's close) - everywhere else this is False.
    # Reopens are sparse (~1 per ~30 H4 bars/week), so the percentile is computed over the
    # trailing N REOPEN occurrences specifically, not the trailing N raw bars - a plain
    # PCTL_LOOKBACK-bar window would contain only ~2 actual reopens and never reach
    # min_periods, which is exactly what happened before this fix (WEEKEND_GAP_TOP20 came
    # back "n/a" - zero valid flags - on the first run).
    hours_since_prev = d["time"].diff().dt.total_seconds() / 3600
    is_reopen = hours_since_prev > 24
    prev_close_before_gap = d["close"].shift(1)
    gap_pct = (d["open"] - prev_close_before_gap).abs() / prev_close_before_gap * 100

    d["weekend_gap_top20"] = False
    reopen_idx = d.index[is_reopen]
    if len(reopen_idx) >= 10:
        reopen_gap_vals = gap_pct.loc[reopen_idx].reset_index(drop=True)
        reopen_pctl = reopen_gap_vals.rolling(20, min_periods=10).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False)
        reopen_flag = (reopen_pctl >= (1 - PCTL_THRESH)).fillna(False)
        d.loc[reopen_idx, "weekend_gap_top20"] = reopen_flag.values

    return d


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


CANDIDATES = {
    "NR4": lambda d: d["nr4"].shift(1),
    "NR7": lambda d: d["nr7"].shift(1),
    "ATR_PCTL20 (squeeze)": lambda d: d["atr_squeeze"].shift(1),
    "TICKVOL_DOWN": lambda d: d["tickvol_down"].shift(1),
    "PRIOR_BIG_1": lambda d: d["prior_big_1"],
    "PRIOR_BIG_1-3": lambda d: d["prior_big_1_3"],
    "INSIDE_BAR": lambda d: d["inside_bar"].shift(1),
    "WEEKEND_GAP_TOP20": lambda d: d["weekend_gap_top20"],
}


def main():
    print("Fetching H4 candles for 9 pairs from OANDA...")
    per_pair = {}
    for pair in PAIRS:
        raw = get_candles(pair)
        per_pair[pair] = build_features(raw, pair)
        print(f"  {pair}: {len(raw)} H4 bars, {raw['time'].iloc[0]} to {raw['time'].iloc[-1]}")

    warmup = max(PCTL_LOOKBACK, AVG_RET_LEN) + 5
    pooled = pd.concat([df.iloc[warmup:] for df in per_pair.values()], ignore_index=True)
    pooled = pooled.sort_values("time").reset_index(drop=True)

    times = pooled["time"]
    mid = times.iloc[len(times) // 2]
    early_mask, late_mask = times < mid, times >= mid

    n_big_total = int(pooled["big_move"].sum())
    print("\n" + "=" * 115)
    print(f"FOREX H4 PRE-MOVE PATTERN SCREEN (pooled, 9 pairs) - {len(pooled)} bars, "
          f"{n_big_total} big-move bars ({n_big_total/len(pooled)*100:.1f}%)")
    print("'Big move' = |H4 return| > 2.0x trailing 20-bar avg |return|, per pair, shifted (no lookahead)")
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
    print("VERDICT (lift > 1.0 in BOTH halves, and >=15 big-move bars per half to trust it):")
    survivors = []
    for name, full, early, late in results:
        if early["n_big"] < 15 or late["n_big"] < 15:
            print(f"  {name:<24} SKIP - too few big-move bars (early={early['n_big']}, late={late['n_big']})")
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

    # ---- per-pair breakdown for whichever candidates survived, so a pooled "survives" ----
    # ---- isn't secretly just one or two pairs driving the whole effect ----
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
    # FOLLOW-UP: directional information, same check as the QQQ study - does
    # PRIOR_BIG_1 (the candidate that actually survived here - NOT
    # PRIOR_BIG_1-3, which failed on magnitude above) predict WHICH WAY,
    # not just THAT a big move is coming?
    # ---------------------------------------------------------------------
    print("=" * 115)
    print("FOLLOW-UP - does PRIOR_BIG_1 (the actual survivor above) carry directional information?")
    print("=" * 115)

    rows = []
    for pair, df in per_pair.items():
        dd = df.iloc[warmup:].reset_index(drop=True)
        for i in range(len(dd)):
            if not dd["big_move"].iloc[i] or not dd["prior_big_1"].iloc[i]:
                continue
            j = i - 1
            if j < 0 or not dd["big_move"].iloc[j]:
                continue  # prior_big_1 true but the actual prior bar isn't available/flagged - skip
            prior_dir = 1 if dd["ret"].iloc[j] > 0 else -1
            this_dir = 1 if dd["ret"].iloc[i] > 0 else -1
            rows.append({"time": dd["time"].iloc[i], "continuation": this_dir == prior_dir})
    clust = pd.DataFrame(rows)
    if len(clust):
        clust = clust.sort_values("time")
        cont_rate = clust["continuation"].mean() * 100
        cmid = clust["time"].iloc[len(clust) // 2]
        early_c = clust[clust["time"] < cmid]["continuation"].mean() * 100
        late_c = clust[clust["time"] >= cmid]["continuation"].mean() * 100
        print(f"n={len(clust)} clustered big-move bars (pooled) | continuation rate: full={cont_rate:.1f}%  "
              f"early={early_c:.1f}%  late={late_c:.1f}%")
    else:
        print("No qualifying clustered big-move bars found.")
    print("(50% = coin flip. This is exactly the check that came back non-directional on QQQ.)")
    print("=" * 115 + "\n")


if __name__ == "__main__":
    main()
