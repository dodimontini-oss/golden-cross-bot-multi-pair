"""
Backtests a NEW candidate strategy - enter in the direction of the London
(09:00-13:00 UTC) or London/NY-Overlap (13:00-17:00 UTC) H4 bar whenever it
shows unusually large volume or range - against the deployed golden-cross
forex bot's own baseline numbers, using the SAME risk-management framework
(ATR*2.0 stop, ATR*4.0 target = 2:1 R:R, 1% risk per trade) as
`multi_pair_bot/bot.py` so the comparison is apples-to-apples on
methodology even though the entry signal itself is completely different.

!!! BUGFIX 2026-09-10 - EVERY BACKTEST NUMBER FROM RUNS #1-#7 IS WRONG !!!
`build_daily` stored each trade's entry-bar index as an index into the
post-slice frame (sliced from the first 21:00 UTC bar), but the simulators
used it to index the FULL h4 frame. Every trade therefore entered ~285 H4
bars (~7 weeks) BEFORE its actual signal - correct direction, wrong price
series - which turned the P&L into near-noise and, e.g., made GBP_CHF look
robust (it isn't) while making GBP_USD look terrible (it's the best pair).
The forex_gbpchf_diagnostic.py run exposed it. Fixed here + the exit loops
rewritten on numpy arrays (10min -> ~1min). project_forex_london_overlap_
breakout_backtest.md's tables from before this fix are RETRACTED.

Direct follow-up to forex_opening_range_lab.py's finding: on a big-move
day, the LONDON and OVERLAP H4 bars' own direction matches the day's
eventual close-to-close direction 82-83% of the time - a real directional
edge, not just a magnitude signal. This tests whether that translates into
an actual tradeable entry rule, not just a same-day descriptive statistic.

METHODOLOGY NOTES (read before trusting the numbers):
  - Full 2015-2026 H4 history, paginated OANDA fetch (NOT the 5000-candle
    single-request cap used in forex_premove_pattern_lab.py and
    forex_opening_range_lab.py) - matches the live bot's own backtest
    period, unlike those two labs which were capped to ~3.2 years.
  - Per-pair INDEPENDENT $10,000 equity (not golden_cross_portfolio_lab.py's
    shared-portfolio, cross-currency-converted simulation) - this matches
    the older screener.py methodology that produced the memory-recorded
    BASELINE PF 1.232 figure, and there's no currency-cap idea being
    tested here anyway, so the simpler per-pair model is the fairer,
    more directly comparable reference point.
  - Outcomes computed in R-multiples (+2.0R on target, -1.0R on stop,
    partial R if still open at the end of history) rather than literal
    P&L - deliberately sidesteps JPY/CHF/AUD cross-currency-to-USD
    conversion, since every trade is sized so a stop-out costs exactly
    1% of that PAIR's OWN equity regardless of quote currency. Per-pair
    equity compounds at 1% risk per trade from these R-multiples.
  - Does NOT model spread/slippage/commission (unlike
    golden_cross_portfolio_lab.py's fee=0.00005 / leverage=20). This is a
    first-pass go/no-go signal test for a brand-new idea, not a
    deployment-ready number - if it looks promising, a full fee-inclusive
    backtest is the natural next step before ever considering deployment.
  - Same-bar dual-hit convention: if a single H4 bar's range would touch
    BOTH the stop and the target, the stop is assumed to hit first
    (conservative), matching this project's usual convention.
  - Entry executes at the OPEN of the bar immediately AFTER the signal bar
    (the signal - elevated volume/range - is only knowable once that bar
    has closed), not at the signal bar's own close - the honest, earliest
    actually-tradeable execution point.
  - One open position per pair at a time (skip new signals while a
    position from an earlier signal is still open), matching the live
    bot's own one-at-a-time-per-pair behavior.
  - Applies the "same-day resolution" fix already required in every other
    backtest lab this project has built: exit-scanning starts AT the entry
    bar itself, not the bar after it.

UPDATE 2026-09-10: the single-signal (RELVOL alone, or RANGE_TOP20 alone)
version of this backtest came back a clean negative (PF 0.93-1.03, worse
than the live bot's own 1.232/1.320, failing walk-forward on 3 of 4
candidates) - see project_forex_london_overlap_breakout_backtest.md. The
diagnosed reason: RELVOL/RANGE alone fire on 7-22% of ALL days, a much
broader and less special population than the ~13% "big-move days"
population the 82-83% directional accuracy was actually measured on.
Added a THIRD candidate type, "relvol_and_range" (require both signals on
the SAME window simultaneously) to test whether that confluence narrows
the trigger down closer to the real big-move-day population, the same way
ORB's RELVOL+GAP confluence beat RELVOL alone on QQQ. UPDATE: that also
came back flat-to-worse (PF unchanged or slightly down, trade count
roughly halved) - RELVOL and RANGE on one H4 bar turn out to be far more
correlated with each other than QQQ's GAP and RELVOL were, so requiring
both barely narrows the population.

UPDATE 2026-09-10 (again): added CROSS-WINDOW confluence via
`simulate_pair_cross()` - require LONDON AND OVERLAP to BOTH fire the
same signal type AND agree on direction, entering once OVERLAP (the
later bar) has closed. This is a genuinely different idea from
relvol_and_range: LONDON and OVERLAP are different bars four hours apart
(not one bar's two correlated stats), so this cross-window agreement
could plausibly narrow down to a cleaner subset even though the
same-window confluence didn't. UPDATE: this ALSO failed - CROSS_RELVOL
(PF 0.827) was the worst result of the whole line of research, and
CROSS_RANGE_TOP20 (PF 0.963) stayed net negative. Three different
selectivity variations (single-signal, same-window AND, cross-window
AND) all landed in the same 0.83-1.03 PF band - strong evidence the
bottleneck is the fixed 2:1 exit structure, not insufficient selectivity.

UPDATE 2026-09-10 (third): added an R:R SWEEP (`RR_SWEEP`, stop fixed at
ATR*2.0, only the target multiple varies) on the four single-window
candidates, since selectivity was now ruled out as the lever to pull.
UPDATE: the pooled PF looked like it improved with a wider target (up to
1.256 at RR=6.0 on LONDON_RELVOL) but walk-forward showed this was
ENTIRELY an early-half (2015-~2020) effect - the late half stayed stuck
at 0.93-1.07 across the whole sweep. Rejected as an overfitting trap, not
a real edge.

UPDATE 2026-09-10 (fourth): added a TRAILING EXIT sweep via
`simulate_pair_trailing()`/`_resolve_trade_trailing()` - a chandelier
stop (no fixed target) trailing peak-since-entry by `chandelier_mult`*ATR,
per the user's "try the trailing exit instead." UPDATE: this is the
first genuine pass in the whole arc - LONDON_RELVOL@CH=3.0 clears both
walk-forward halves (pooled 1.157, early 1.224, late 1.093).

UPDATE 2026-09-10 (fifth): added a PER-PAIR BREAKDOWN of the LONDON
trailing candidates (user: "run the per-pair check next") - 6/9 pairs
pass both halves for LONDON_RELVOL@CH=3.0, and it survives dropping the
strongest pair. Neighbour CH multiples collapse to 2-3/9, so CH=3.0 is a
real sweet spot, not a cherry-pick.

UPDATE 2026-09-10 (sixth): added TRANSACTION COST MODELLING (user: "yes
add cost modeling"). `cost_in_r()` + `COST_SCENARIOS`: per-pair round-trip
= full quoted spread (mid-to-mid) + 0.5pip/side slippage, converted to
R-multiples and subtracted from every trade. Every table now shows
frictionless vs. realistic-cost PF, plus a cost-sensitivity sweep
(frictionless / realistic / pessimistic 1.5x). The R:R sweep is dropped
from this run (fully recorded, clean fail). THIS is the run that decides
whether LONDON_RELVOL@CH=3.0 is real or just a no-cost artefact.

Run (needs OANDA_API_KEY):
    python forex_london_overlap_breakout_lab.py
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
RR_RATIO = 2.0
RISK_PCT = 0.01
STARTING_EQUITY = 10000.0

AVG_LEN = 20
RELVOL_MULT = 1.5
PCTL_LOOKBACK = 60
PCTL_THRESH = 0.20

# (label, signal-bar-index-within-day, entry-bar-index) - day = 6 H4 bars
# starting 21:00 UTC: [21:00, 01:00, 05:00, 09:00, 13:00, 17:00]
OR_WINDOWS = [("LONDON", 3, 4), ("OVERLAP", 4, 5)]
CANDIDATE_TYPES = ["relvol", "range_top20", "relvol_and_range"]
CHANDELIER_SWEEP = [2.0, 3.0, 4.0, 5.0]  # trailing distance (x ATR from peak); 3.0 matches the live-bot-family default

# --- transaction cost model (added 2026-09-10, user: "yes add cost modeling") ---
# Every result before this run was frictionless. Per-pair round-trip cost is
# modeled as: cross the full quoted spread once (mid-to-mid, since the backtest
# uses OANDA mid candles) + a fixed slippage allowance on each side. The spread
# figures are deliberately a touch WIDER than OANDA fxpractice typically shows
# on majors, because a trailing-stop strategy exits on stop orders (which slip
# in fast markets) far more often than on resting limit targets. Cost is
# converted to R-multiples per trade (fraction of the ATR*2.0 initial risk
# distance) and subtracted from every trade's gross outcome.
SPREAD_PIPS = {
    "GBP_CHF": 2.5, "GBP_CAD": 2.6, "GBP_USD": 1.3, "GBP_JPY": 2.2, "GBP_AUD": 2.8,
    "USD_CHF": 1.6, "EUR_USD": 0.9, "EUR_AUD": 1.9, "AUD_NZD": 2.4,
}
SLIPPAGE_PIPS_PER_SIDE = 0.5  # -> +1.0 pip added to the round trip
PIP_SIZE = {p: (0.01 if p.endswith("_JPY") else 0.0001) for p in PAIRS}
# multiplier on the whole round-trip cost, for the sensitivity sweep
COST_SCENARIOS = {"frictionless": 0.0, "realistic": 1.0, "pessimistic": 1.5}


def cost_in_r(pair: str, stop_dist: float, cost_mult: float) -> float:
    """Round-trip transaction cost for one trade, expressed in R (fraction of
    the ATR*2.0 initial risk distance, so it's directly subtractable from the
    R-multiple trade outcomes used throughout this file). cost_mult scales it
    for the scenario sweep (0.0 reproduces the old frictionless numbers)."""
    if cost_mult == 0.0:
        return 0.0
    rt_pips = SPREAD_PIPS[pair] + 2 * SLIPPAGE_PIPS_PER_SIDE
    rt_price = rt_pips * PIP_SIZE[pair]
    return cost_mult * rt_price / stop_dist


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
        all_candles.extend([c for c in batch if c["complete"]])
        if len(batch) < 5000:
            break
        current_from = batch[-1]["time"]
        time.sleep(0.2)
    rows = [{"time": pd.Timestamp(c["time"]), "open": float(c["mid"]["o"]), "high": float(c["mid"]["h"]),
             "low": float(c["mid"]["l"]), "close": float(c["mid"]["c"]), "volume": int(c.get("volume", 0))}
            for c in all_candles]
    df = pd.DataFrame(rows).drop_duplicates(subset="time").sort_values("time").reset_index(drop=True)
    return df


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


def build_daily(df: pd.DataFrame) -> pd.DataFrame:
    first_idx = df.index[df["time"].dt.hour == 21]
    if len(first_idx) == 0:
        raise ValueError("no 21:00 UTC bar found - can't align day boundaries")
    # BUGFIX 2026-09-10: `offset` is the position of the first 21:00 UTC bar in
    # the FULL df. Every `*_entry_bar_idx` stored below MUST be an index into
    # that full df, because simulate_pair*/_resolve_trade* index the full `h4`
    # frame with it. The pre-fix code stored `base + entry_idx` (an index into
    # the post-slice frame `d`), so every trade entered `offset` bars (~7 weeks
    # of H4 data) before its actual signal - correct direction, wrong price
    # series. This silently scrambled every backtest P&L number in this file
    # and its memory doc; the forex_gbpchf_diagnostic.py run exposed it.
    offset = first_idx[0]
    d = df.iloc[offset:].reset_index(drop=True)
    n_days = len(d) // 6
    rows = []
    for i in range(n_days):
        base = i * 6
        chunk = d.iloc[base: base + 6]
        row = {"day_start": chunk["time"].iloc[0]}
        for label, sig_idx, entry_idx in OR_WINDOWS:
            bar = chunk.iloc[sig_idx]
            row[f"{label}_range_pct"] = (bar["high"] - bar["low"]) / bar["open"] * 100
            row[f"{label}_vol"] = bar["volume"]
            row[f"{label}_dir"] = 1 if bar["close"] > bar["open"] else (-1 if bar["close"] < bar["open"] else 0)
            row[f"{label}_entry_bar_idx"] = offset + base + entry_idx
        rows.append(row)
    return pd.DataFrame(rows)


def add_day_flags(daily: pd.DataFrame) -> pd.DataFrame:
    daily = daily.sort_values("day_start").reset_index(drop=True)
    for label, _, _ in OR_WINDOWS:
        avg_vol = daily[f"{label}_vol"].rolling(AVG_LEN).mean().shift(1)
        daily[f"{label}_relvol"] = daily[f"{label}_vol"] >= RELVOL_MULT * avg_vol
        range_pctl = daily[f"{label}_range_pct"].rolling(PCTL_LOOKBACK).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False)
        daily[f"{label}_range_top20"] = range_pctl >= (1 - PCTL_THRESH)
        # CONFLUENCE: require both signals at once - a narrower, more selective
        # trigger than either alone, testing whether it filters down to a
        # subset closer to the actual big-move-day population (the population
        # the 82-83% directional accuracy in forex_opening_range_lab.py was
        # actually measured on), rather than the much broader "RELVOL OR
        # RANGE elevated on an otherwise ordinary day" population that the
        # single-signal version in this script traded unconditionally.
        daily[f"{label}_relvol_and_range"] = daily[f"{label}_relvol"] & daily[f"{label}_range_top20"]
    return daily


class Trade:
    __slots__ = ("pair", "label", "candidate", "direction", "entry_time", "exit_time", "r_multiple")

    def __init__(self, pair, label, candidate, direction, entry_time, exit_time, r_multiple):
        self.pair, self.label, self.candidate = pair, label, candidate
        self.direction, self.entry_time, self.exit_time, self.r_multiple = direction, entry_time, exit_time, r_multiple


class _Arrays:
    """numpy views of an h4 frame + its ATR, built once per pair so the
    forward-scan exit loops don't pay pandas .iloc overhead per bar (this
    is what took the pre-numpy version ~10min per full run)."""
    __slots__ = ("op", "hi", "lo", "cl", "tm", "atr", "n")

    def __init__(self, h4: pd.DataFrame):
        self.op = h4["open"].to_numpy()
        self.hi = h4["high"].to_numpy()
        self.lo = h4["low"].to_numpy()
        self.cl = h4["close"].to_numpy()
        self.tm = h4["time"].to_numpy()
        self.atr = atr_wilder(h4, ATR_LEN).to_numpy()
        self.n = len(h4)


def _resolve_trade(A: "_Arrays", entry_idx: int, direction: int, rr_ratio: float = RR_RATIO):
    """Fixed-target exit scan. Enter at A.op[entry_idx], ATR from the
    just-closed prior bar, scan forward from the entry bar itself
    (same-day-resolution fix). Stop is always ATR_STOP_MULT*ATR; rr_ratio
    sets the target at that many multiples of the stop distance.
    Returns (r_outcome, entry_price, exit_idx, stop_dist) or None."""
    if entry_idx <= 0:
        return None
    a = A.atr[entry_idx - 1]
    if not (a > 0):  # also catches NaN
        return None
    entry_price = A.op[entry_idx]
    stop_dist = ATR_STOP_MULT * a
    if direction == 1:
        stop_price, target_price = entry_price - stop_dist, entry_price + rr_ratio * stop_dist
    else:
        stop_price, target_price = entry_price + stop_dist, entry_price - rr_ratio * stop_dist

    for j in range(entry_idx, A.n):
        if direction == 1:
            if A.lo[j] <= stop_price:
                return -1.0, entry_price, j, stop_dist
            if A.hi[j] >= target_price:
                return rr_ratio, entry_price, j, stop_dist
        else:
            if A.hi[j] >= stop_price:
                return -1.0, entry_price, j, stop_dist
            if A.lo[j] <= target_price:
                return rr_ratio, entry_price, j, stop_dist

    move = (A.cl[-1] - entry_price) if direction == 1 else (entry_price - A.cl[-1])
    return move / stop_dist, entry_price, A.n - 1, stop_dist


def _resolve_trade_trailing(A: "_Arrays", entry_idx: int, direction: int, chandelier_mult: float):
    """Chandelier trailing exit: no fixed target. Stop starts at the same
    ATR_STOP_MULT*ATR initial risk distance, then ratchets to
    (peak-since-entry -/+ chandelier_mult*ATR) once that is more
    protective. Within a bar the OLD stop is checked FIRST, then the peak
    (and trailing stop) update from that same bar's extreme for the next
    bar - no optimistic intrabar ordering.
    Returns (r_outcome, entry_price, exit_idx, stop_dist) or None."""
    if entry_idx <= 0:
        return None
    a = A.atr[entry_idx - 1]
    if not (a > 0):
        return None
    entry_price = A.op[entry_idx]
    init_stop_dist = ATR_STOP_MULT * a
    trail_dist = chandelier_mult * a
    peak = entry_price
    stop_price = entry_price - init_stop_dist if direction == 1 else entry_price + init_stop_dist

    for j in range(entry_idx, A.n):
        if direction == 1:
            if A.lo[j] <= stop_price:
                return (stop_price - entry_price) / init_stop_dist, entry_price, j, init_stop_dist
            if A.hi[j] > peak:
                peak = A.hi[j]
            lvl = peak - trail_dist
            if lvl > stop_price:
                stop_price = lvl
        else:
            if A.hi[j] >= stop_price:
                return (entry_price - stop_price) / init_stop_dist, entry_price, j, init_stop_dist
            if A.lo[j] < peak:
                peak = A.lo[j]
            lvl = peak + trail_dist
            if lvl < stop_price:
                stop_price = lvl

    move = (A.cl[-1] - entry_price) if direction == 1 else (entry_price - A.cl[-1])
    return move / init_stop_dist, entry_price, A.n - 1, init_stop_dist


def _run(pair, label, candidate, A, daily, entry_idx_col, day_direction, resolver, cost_mult):
    """Shared trade loop. `day_direction(day)` returns +1/-1 if the day
    fires (0/None if not); `resolver(A, entry_idx, direction)` returns the
    (r, entry_price, exit_idx, stop_dist) tuple or None."""
    trades = []
    equity_curve = [STARTING_EQUITY]
    equity = STARTING_EQUITY
    blocked_until_idx = -1
    for _, day in daily.iterrows():
        direction = day_direction(day)
        if not direction:
            continue
        entry_idx = int(day[entry_idx_col])
        if entry_idx <= blocked_until_idx or entry_idx >= A.n:
            continue
        res = resolver(A, entry_idx, direction)
        if res is None:
            continue
        r_outcome, _entry, exit_idx, stop_dist = res
        r_outcome -= cost_in_r(pair, stop_dist, cost_mult)
        equity *= (1 + RISK_PCT * r_outcome)
        equity_curve.append(equity)
        blocked_until_idx = exit_idx
        trades.append(Trade(pair, label, candidate, direction, A.tm[entry_idx], A.tm[exit_idx], r_outcome))
    return trades, equity_curve


def _arrays(h4):
    return h4 if isinstance(h4, _Arrays) else _Arrays(h4)


def _single_window_dir(flag_col, dir_col):
    def f(day):
        return int(day[dir_col]) if bool(day.get(flag_col, False)) else 0
    return f


def _cross_window_dir(candidate):
    fa, fb = f"LONDON_{candidate}", f"OVERLAP_{candidate}"
    def f(day):
        if not (bool(day.get(fa, False)) and bool(day.get(fb, False))):
            return 0
        ld, od = int(day["LONDON_dir"]), int(day["OVERLAP_dir"])
        return od if (ld != 0 and ld == od) else 0
    return f


def simulate_pair(pair, h4, daily, label, candidate, rr_ratio=RR_RATIO, cost_mult=0.0):
    return _run(pair, label, candidate, _arrays(h4), daily, f"{label}_entry_bar_idx",
                _single_window_dir(f"{label}_{candidate}", f"{label}_dir"),
                lambda A, ei, d: _resolve_trade(A, ei, d, rr_ratio), cost_mult)


def simulate_pair_trailing(pair, h4, daily, label, candidate, chandelier_mult, cost_mult=0.0):
    return _run(pair, label, candidate, _arrays(h4), daily, f"{label}_entry_bar_idx",
                _single_window_dir(f"{label}_{candidate}", f"{label}_dir"),
                lambda A, ei, d: _resolve_trade_trailing(A, ei, d, chandelier_mult), cost_mult)


def simulate_pair_cross(pair, h4, daily, candidate, cost_mult=0.0):
    """Cross-window confluence: LONDON AND OVERLAP both fire the same signal
    AND agree on direction; enter once OVERLAP (the later bar) has closed."""
    return _run(pair, "CROSS", candidate, _arrays(h4), daily, "OVERLAP_entry_bar_idx",
                _cross_window_dir(candidate),
                lambda A, ei, d: _resolve_trade(A, ei, d), cost_mult)


def pf_stats(trades: list) -> dict:
    if not trades:
        return {"trades": 0, "win_rate": float("nan"), "pf": float("nan")}
    r = [t.r_multiple for t in trades]
    gross_win = sum(x for x in r if x > 0)
    gross_loss = abs(sum(x for x in r if x <= 0))
    pf = gross_win / gross_loss if gross_loss > 0 else float("inf")
    win_rate = sum(1 for x in r if x > 0) / len(r) * 100
    return {"trades": len(r), "win_rate": win_rate, "pf": pf}


def max_dd_pct(curve: list) -> float:
    peak, worst = curve[0], 0.0
    for e in curve:
        peak = max(peak, e)
        worst = max(worst, (peak - e) / peak * 100)
    return worst


def main():
    print(f"Fetching full {FROM_TIME[:10]}-present H4 history for {len(PAIRS)} pairs (paginated)...")
    per_pair_h4, per_pair_daily = {}, {}
    for pair in PAIRS:
        h4 = get_candles_range(pair, FROM_TIME)
        daily = add_day_flags(build_daily(h4))
        # build the numpy views ONCE per pair (ATR's wilder_rma is a slow
        # Python loop; the sweeps call the simulators ~100x per pair)
        per_pair_h4[pair], per_pair_daily[pair] = _Arrays(h4), daily
        print(f"  {pair}: {len(h4)} H4 bars -> {len(daily)} days, {h4['time'].iloc[0]} to {h4['time'].iloc[-1]}")
        time.sleep(0.3)

    realistic = COST_SCENARIOS["realistic"]

    print("\n" + "=" * 110)
    print("LONDON / OVERLAP BREAKOUT BACKTEST vs. deployed golden-cross-bot-multi-pair")
    print("Same risk framework: ATR*2.0 stop, ATR*4.0 target (2:1 R:R), 1% risk/trade, per-pair independent equity")
    print("PF shown BOTH frictionless (old numbers) and with the realistic per-pair spread+slippage cost model")
    print("Reference (memory, same per-pair-independent methodology): BASELINE PF 1.232, WINNERS (GAPCONFIRM+CURCAP) PF 1.320")
    print("=" * 110)
    print(f"{'Candidate':<24} {'Trades':>7} {'Win%':>7} {'PF(free)':>9} {'PF(cost)':>9} {'WorstPairDD%(cost)':>18}")
    print("-" * 110)

    all_results = {}          # realistic-cost trade lists, consumed by walk-forward + per-pair below
    for label, _, _ in OR_WINDOWS:
        for candidate in CANDIDATE_TYPES:
            free_trades, cost_trades, dds = [], [], []
            for pair in PAIRS:
                ft, _ = simulate_pair(pair, per_pair_h4[pair], per_pair_daily[pair], label, candidate, cost_mult=0.0)
                ct, curve = simulate_pair(pair, per_pair_h4[pair], per_pair_daily[pair], label, candidate, cost_mult=realistic)
                free_trades.extend(ft)
                cost_trades.extend(ct)
                if len(curve) > 1:
                    dds.append(max_dd_pct(curve))
            fs, cs = pf_stats(free_trades), pf_stats(cost_trades)
            worst_dd = max(dds) if dds else float("nan")
            name = f"{label}_{candidate.upper()}"
            all_results[name] = (cost_trades, cs, worst_dd)
            print(f"{name:<24} {cs['trades']:>7} {cs['win_rate']:>6.1f}% {fs['pf']:>9.3f} {cs['pf']:>9.3f} {worst_dd:>17.2f}%")

    # CROSS-WINDOW confluence: LONDON and OVERLAP both fire AND agree on direction.
    for candidate in ("relvol", "range_top20"):
        free_trades, cost_trades, dds = [], [], []
        for pair in PAIRS:
            ft, _ = simulate_pair_cross(pair, per_pair_h4[pair], per_pair_daily[pair], candidate, cost_mult=0.0)
            ct, curve = simulate_pair_cross(pair, per_pair_h4[pair], per_pair_daily[pair], candidate, cost_mult=realistic)
            free_trades.extend(ft)
            cost_trades.extend(ct)
            if len(curve) > 1:
                dds.append(max_dd_pct(curve))
        fs = pf_stats(free_trades)
        stats = pf_stats(cost_trades)
        worst_dd = max(dds) if dds else float("nan")
        name = f"CROSS_{candidate.upper()}"
        all_results[name] = (cost_trades, stats, worst_dd)
        print(f"{name:<24} {stats['trades']:>7} {stats['win_rate']:>6.1f}% {fs['pf']:>9.3f} {stats['pf']:>9.3f} {worst_dd:>17.2f}%")
    print("=" * 110)

    print("\n" + "=" * 110)
    print("WALK-FORWARD CHECK - REALISTIC COST (split at pooled-trade time median per candidate)")
    print("=" * 110)
    print(f"{'Candidate':<24} {'EarlyTrades':>11} {'EarlyPF':>9} {'LateTrades':>11} {'LatePF':>9}")
    print("-" * 110)
    for name, (trades, stats, worst_dd) in all_results.items():
        if len(trades) < 10:
            print(f"{name:<24} SKIP - too few trades ({len(trades)}) to split")
            continue
        times = sorted(t.entry_time for t in trades)
        mid = times[len(times) // 2]
        early = [t for t in trades if t.entry_time < mid]
        late = [t for t in trades if t.entry_time >= mid]
        es, ls = pf_stats(early), pf_stats(late)
        print(f"{name:<24} {es['trades']:>11} {es['pf']:>9.3f} {ls['trades']:>11} {ls['pf']:>9.3f}")
    print("=" * 110)

    print("\n" + "=" * 110)
    print("PER-PAIR BREAKDOWN - REALISTIC COST (does it hold up pair-by-pair with spread/slippage?)")
    print("=" * 110)
    for name, (trades, stats, worst_dd) in all_results.items():
        print(f"\n{name}:")
        for pair in PAIRS:
            pair_trades = [t for t in trades if t.pair == pair]
            s = pf_stats(pair_trades)
            print(f"  {pair:<10} trades={s['trades']:>4}  win%={s['win_rate']:>5.1f}  PF={s['pf']:>6.3f}")
    print("=" * 110 + "\n")

    # R:R sweep removed from this run - fully recorded in
    # project_forex_london_overlap_breakout_backtest.md (clean fail: a wider
    # fixed target is an early-half-only illusion that fails walk-forward).

    sweep_candidates = [("LONDON", "relvol"), ("LONDON", "range_top20"), ("OVERLAP", "relvol"), ("OVERLAP", "range_top20")]

    # -----------------------------------------------------------------
    # TRAILING EXIT SWEEP - now WITH the realistic cost model. The
    # frictionless version of this (run #5/#6) was the first walk-forward
    # pass in the arc (LONDON_RELVOL@CH=3.0, pooled 1.157, 6/9 pairs). The
    # question this run answers: does that survive spread + slippage? A
    # trailing exit takes many small partial-profit exits, so it's more
    # cost-exposed than a fixed 2:1 target - this is the real test.
    # -----------------------------------------------------------------
    print("\n" + "=" * 110)
    print("TRAILING EXIT SWEEP - REALISTIC COST - chandelier trail distance varies, no fixed target")
    print("(PF shown frictionless / realistic-cost for each cell)")
    print("=" * 110)
    print(f"{'Candidate':<20} " + " ".join(f"CH={c:<10.1f}" for c in CHANDELIER_SWEEP))
    print("-" * 110)
    trail_results = {}           # realistic-cost trade lists, consumed by the sections below
    for label, candidate in sweep_candidates:
        cells = []
        for ch in CHANDELIER_SWEEP:
            free_trades, cost_trades = [], []
            for pair in PAIRS:
                ft, _ = simulate_pair_trailing(pair, per_pair_h4[pair], per_pair_daily[pair], label, candidate, chandelier_mult=ch, cost_mult=0.0)
                ct, _ = simulate_pair_trailing(pair, per_pair_h4[pair], per_pair_daily[pair], label, candidate, chandelier_mult=ch, cost_mult=realistic)
                free_trades.extend(ft)
                cost_trades.extend(ct)
            trail_results[(label, candidate, ch)] = cost_trades
            cells.append(f"{pf_stats(free_trades)['pf']:.3f}/{pf_stats(cost_trades)['pf']:.3f}")
        name = f"{label}_{candidate.upper()}"
        print(f"{name:<20} " + " ".join(f"{c:<13}" for c in cells))
    print("=" * 110)

    # cost-sensitivity: the standout LONDON candidates across all three cost scenarios
    print("\n" + "=" * 110)
    print("COST SENSITIVITY - LONDON candidates, PF at each cost scenario")
    print(f"(scenarios: {', '.join(f'{k}={v}x' for k, v in COST_SCENARIOS.items())}; "
          f"realistic = per-pair spread {min(SPREAD_PIPS.values())}-{max(SPREAD_PIPS.values())}pips + "
          f"{SLIPPAGE_PIPS_PER_SIDE}pip/side slippage)")
    print("=" * 110)
    print(f"{'Candidate':<20} {'CH':>5} " + " ".join(f"{s:>13}" for s in COST_SCENARIOS))
    print("-" * 110)
    for label, candidate in [("LONDON", "relvol"), ("LONDON", "range_top20")]:
        for ch in (2.0, 3.0, 4.0):  # CH=3.0 is the standout; 2.0/4.0 for context
            pfs = []
            for scen, mult in COST_SCENARIOS.items():
                pooled = []
                for pair in PAIRS:
                    tr, _ = simulate_pair_trailing(pair, per_pair_h4[pair], per_pair_daily[pair], label, candidate, chandelier_mult=ch, cost_mult=mult)
                    pooled.extend(tr)
                pfs.append(pf_stats(pooled)["pf"])
            print(f"{label + '_' + candidate.upper():<20} {ch:>5.1f} " + " ".join(f"{p:>13.3f}" for p in pfs))
    print("=" * 110)

    print("\n" + "=" * 110)
    print("TRAILING EXIT SWEEP - WALK-FORWARD CHECK - REALISTIC COST (early/late PF on every cell above)")
    print("=" * 110)
    print(f"{'Candidate':<20} {'CH':>5} {'EarlyTr':>8} {'EarlyPF':>9} {'LateTr':>8} {'LatePF':>9}")
    print("-" * 110)
    for label, candidate in sweep_candidates:
        for ch in CHANDELIER_SWEEP:
            trades = trail_results[(label, candidate, ch)]
            if len(trades) < 10:
                print(f"{label}_{candidate.upper():<12} {ch:>5.1f}  SKIP - too few trades ({len(trades)})")
                continue
            times = sorted(t.entry_time for t in trades)
            mid = times[len(times) // 2]
            early = [t for t in trades if t.entry_time < mid]
            late = [t for t in trades if t.entry_time >= mid]
            es, ls = pf_stats(early), pf_stats(late)
            print(f"{label + '_' + candidate.upper():<20} {ch:>5.1f} {es['trades']:>8} {es['pf']:>9.3f} {ls['trades']:>8} {ls['pf']:>9.3f}")
    print("=" * 110)
    print("Reference (memory, same per-pair-independent methodology): BASELINE PF 1.232, WINNERS (GAPCONFIRM+CURCAP) PF 1.320\n")

    # -----------------------------------------------------------------
    # TRAILING EXIT - PER-PAIR BREAKDOWN (2026-09-10, user follow-up:
    # "run the per-pair check next"). The pooled trailing result -
    # especially LONDON_RELVOL@CH=3.0 (pooled 1.157, early 1.224, late
    # 1.093) - is the first walk-forward pass in this whole research arc.
    # Real question: does it hold across the 9 pairs individually, or is
    # 1-2 pairs carrying the pooled number? Uses the already-computed
    # trade lists in trail_results (each Trade carries its .pair), so no
    # extra simulation - just filter + re-score per pair, with each
    # pair's own trade-time median as its early/late split.
    # -----------------------------------------------------------------
    print("\n" + "=" * 110)
    print("TRAILING EXIT - PER-PAIR BREAKDOWN - REALISTIC COST (does the pooled pass survive costs pair-by-pair?)")
    print("=" * 110)
    for label, candidate in [("LONDON", "relvol"), ("LONDON", "range_top20")]:
        for ch in CHANDELIER_SWEEP:
            all_trades = trail_results[(label, candidate, ch)]
            print(f"\n{label}_{candidate.upper()}  CH={ch:.1f}   (pooled: {pf_stats(all_trades)['trades']} trades, "
                  f"PF {pf_stats(all_trades)['pf']:.3f})")
            print(f"  {'pair':<10} {'trades':>7} {'PF':>8} {'earlyPF':>9} {'latePF':>9}   both halves >1.0?")
            n_pass = 0
            for pair in PAIRS:
                pt = [t for t in all_trades if t.pair == pair]
                s = pf_stats(pt)
                if len(pt) < 10:
                    print(f"  {pair:<10} {s['trades']:>7}  (too few to split)")
                    continue
                times = sorted(t.entry_time for t in pt)
                mid = times[len(times) // 2]
                e = pf_stats([t for t in pt if t.entry_time < mid])
                l = pf_stats([t for t in pt if t.entry_time >= mid])
                ok = (e["pf"] > 1.0 and l["pf"] > 1.0)
                n_pass += ok
                print(f"  {pair:<10} {s['trades']:>7} {s['pf']:>8.3f} {e['pf']:>9.3f} {l['pf']:>9.3f}   {'YES' if ok else 'no'}")
            print(f"  -> {n_pass}/9 pairs pass both walk-forward halves individually")
    print("=" * 110)
    print("Reference (memory): BASELINE PF 1.232, WINNERS (GAPCONFIRM+CURCAP) PF 1.320 - those come from")
    print("golden_cross_portfolio_lab.py, which models only fee=0.00005 (~0.5bp/side, ~0.05-0.1pip) - MUCH tighter than")
    print("this run's per-pair spread+slippage model. So: this run's frictionless column ~ their cost basis; this run's")
    print("realistic column is a deliberately harsher, more deployment-honest test than the reference bot ever faced.\n")


if __name__ == "__main__":
    main()
