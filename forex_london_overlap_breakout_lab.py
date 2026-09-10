"""
Backtests a NEW candidate strategy - enter in the direction of the London
(09:00-13:00 UTC) or London/NY-Overlap (13:00-17:00 UTC) H4 bar whenever it
shows unusually large volume or range - against the deployed golden-cross
forex bot's own baseline numbers, using the SAME risk-management framework
(ATR*2.0 stop, ATR*4.0 target = 2:1 R:R, 1% risk per trade) as
`multi_pair_bot/bot.py` so the comparison is apples-to-apples on
methodology even though the entry signal itself is completely different.

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
trailing candidates (user: "run the per-pair check next") - does the
pooled pass hold up pair-by-pair, or is 1-2 pairs carrying it? Also
trimmed RR_SWEEP to [2.0, 6.0] this run since the full R:R sweep is
already recorded and was slow.

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
# RR_SWEEP trimmed to [2.0, 6.0] for the 2026-09-10 per-pair-check run - the full
# 7-point sweep is already recorded (pooled PF climbs to 1.26 at RR=6 but it's an
# early-half-only illusion that fails walk-forward). Keeping RR=2 (baseline) and
# RR=6 here just as a one-line reminder of that failure mode alongside the trailing
# results, without paying the full sweep's ~2.5min runtime again.
RR_SWEEP = [2.0, 6.0]
CHANDELIER_SWEEP = [2.0, 3.0, 4.0, 5.0]  # trailing distance (x ATR from peak); 3.0 matches the live-bot-family default


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
    d = df.iloc[first_idx[0]:].reset_index(drop=True)
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
            row[f"{label}_entry_bar_idx"] = base + entry_idx
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


def _resolve_trade(h4: pd.DataFrame, atr: pd.Series, entry_idx: int, direction: int, rr_ratio: float = RR_RATIO):
    """Shared exit-scanning logic: enter at h4.iloc[entry_idx]'s open, ATR
    taken from the just-closed prior bar, scan forward from the entry bar
    itself (same-day-resolution fix) for the first stop/target hit. Stop
    distance is always ATR_STOP_MULT*ATR (unchanged); `rr_ratio` sets how
    many multiples of that stop distance the target sits at - this is the
    knob the 2026-09-10 R:R sweep varies, keeping the stop fixed and
    widening ONLY the target, per the user's request ("wider target
    instead of 2:1" - not a wider stop).
    Returns (r_outcome, entry_price, exit_idx) or None if untradeable
    (ATR not yet warmed up)."""
    entry_bar = h4.iloc[entry_idx]
    entry_price = entry_bar["open"]
    a = atr.iloc[entry_idx - 1] if entry_idx > 0 else float("nan")
    if pd.isna(a) or a <= 0:
        return None
    stop_dist = ATR_STOP_MULT * a
    if direction == 1:
        stop_price, target_price = entry_price - stop_dist, entry_price + rr_ratio * stop_dist
    else:
        stop_price, target_price = entry_price + stop_dist, entry_price - rr_ratio * stop_dist

    for j in range(entry_idx, len(h4)):
        bar = h4.iloc[j]
        hit_stop = bar["low"] <= stop_price if direction == 1 else bar["high"] >= stop_price
        hit_target = bar["high"] >= target_price if direction == 1 else bar["low"] <= target_price
        if hit_stop:  # dual-hit -> conservative: stop wins
            return -1.0, entry_price, j
        if hit_target:
            return rr_ratio, entry_price, j

    last = h4.iloc[-1]
    move = (last["close"] - entry_price) if direction == 1 else (entry_price - last["close"])
    return move / stop_dist, entry_price, len(h4) - 1


def simulate_pair(pair: str, h4: pd.DataFrame, daily: pd.DataFrame, label: str, candidate: str, rr_ratio: float = RR_RATIO):
    flag_col = f"{label}_{candidate}"
    dir_col = f"{label}_dir"
    entry_idx_col = f"{label}_entry_bar_idx"
    atr = atr_wilder(h4, ATR_LEN)

    trades = []
    equity_curve = [STARTING_EQUITY]
    equity = STARTING_EQUITY
    blocked_until_idx = -1

    for _, day in daily.iterrows():
        if not bool(day.get(flag_col, False)):
            continue
        direction = day[dir_col]
        if direction == 0:
            continue
        entry_idx = int(day[entry_idx_col])
        if entry_idx <= blocked_until_idx or entry_idx >= len(h4):
            continue
        result = _resolve_trade(h4, atr, entry_idx, direction, rr_ratio)
        if result is None:
            continue
        r_outcome, entry_price, exit_idx = result

        equity *= (1 + RISK_PCT * r_outcome)
        equity_curve.append(equity)
        blocked_until_idx = exit_idx
        trades.append(Trade(pair, label, candidate, direction, h4.iloc[entry_idx]["time"], h4.iloc[exit_idx]["time"], r_outcome))

    return trades, equity_curve


def _resolve_trade_trailing(h4: pd.DataFrame, atr: pd.Series, entry_idx: int, direction: int, chandelier_mult: float):
    """Chandelier trailing exit: no fixed target. The stop starts at the
    same ATR_STOP_MULT*ATR initial risk distance used everywhere else in
    this file (so R-multiples stay comparable to the fixed-R:R results),
    then ratchets to (peak-since-entry - chandelier_mult*ATR) once that
    level becomes MORE protective than the initial stop - i.e.
    stop = max(initial_stop, trailing_level) for a long (min for a short).
    This means risk is capped at exactly 1R from bar one (same as the
    fixed-target version), and the trailing stop only ever tightens once
    the trade is sufficiently in profit for chandelier_mult*ATR of trail
    room to sit inside the initial risk distance.

    ATR is a single snapshot from the bar before entry (not recomputed
    bar-by-bar) - consistent with how the initial stop/target distance is
    computed everywhere else in this file.

    Same look-ahead discipline as _resolve_trade: within a bar, the OLD
    stop (as of entering that bar) is checked against that bar's low/high
    FIRST; only if not hit does the peak (and therefore the trailing
    stop) update using that same bar's extreme, for the NEXT bar's check.
    Doing it the other way around would optimistically assume the bar's
    favorable extreme happened before its unfavorable one.

    Returns (r_outcome, entry_price, exit_idx) or None if untradeable."""
    entry_bar = h4.iloc[entry_idx]
    entry_price = entry_bar["open"]
    a = atr.iloc[entry_idx - 1] if entry_idx > 0 else float("nan")
    if pd.isna(a) or a <= 0:
        return None
    initial_stop_dist = ATR_STOP_MULT * a
    trail_dist = chandelier_mult * a
    peak = entry_price
    stop_price = entry_price - initial_stop_dist if direction == 1 else entry_price + initial_stop_dist

    for j in range(entry_idx, len(h4)):
        bar = h4.iloc[j]
        hit_stop = bar["low"] <= stop_price if direction == 1 else bar["high"] >= stop_price
        if hit_stop:
            move = (stop_price - entry_price) if direction == 1 else (entry_price - stop_price)
            return move / initial_stop_dist, entry_price, j
        if direction == 1:
            peak = max(peak, bar["high"])
            stop_price = max(stop_price, peak - trail_dist)
        else:
            peak = min(peak, bar["low"])
            stop_price = min(stop_price, peak + trail_dist)

    last = h4.iloc[-1]
    move = (last["close"] - entry_price) if direction == 1 else (entry_price - last["close"])
    return move / initial_stop_dist, entry_price, len(h4) - 1


def simulate_pair_trailing(pair: str, h4: pd.DataFrame, daily: pd.DataFrame, label: str, candidate: str, chandelier_mult: float):
    flag_col = f"{label}_{candidate}"
    dir_col = f"{label}_dir"
    entry_idx_col = f"{label}_entry_bar_idx"
    atr = atr_wilder(h4, ATR_LEN)

    trades = []
    equity_curve = [STARTING_EQUITY]
    equity = STARTING_EQUITY
    blocked_until_idx = -1

    for _, day in daily.iterrows():
        if not bool(day.get(flag_col, False)):
            continue
        direction = day[dir_col]
        if direction == 0:
            continue
        entry_idx = int(day[entry_idx_col])
        if entry_idx <= blocked_until_idx or entry_idx >= len(h4):
            continue
        result = _resolve_trade_trailing(h4, atr, entry_idx, direction, chandelier_mult)
        if result is None:
            continue
        r_outcome, entry_price, exit_idx = result

        equity *= (1 + RISK_PCT * r_outcome)
        equity_curve.append(equity)
        blocked_until_idx = exit_idx
        trades.append(Trade(pair, label, candidate, direction, h4.iloc[entry_idx]["time"], h4.iloc[exit_idx]["time"], r_outcome))

    return trades, equity_curve


def simulate_pair_cross(pair: str, h4: pd.DataFrame, daily: pd.DataFrame, candidate: str):
    """Cross-window confluence: require LONDON AND OVERLAP to BOTH show the
    signal on the SAME day AND agree on direction, entering only once
    OVERLAP has closed (the later of the two signal bars). A genuinely
    different idea from relvol_and_range (which combined two signals on
    ONE window and didn't help, because RELVOL and RANGE on a single bar
    are highly correlated by construction) - LONDON and OVERLAP are
    different bars four hours apart, shown to fire on partly independent
    subsets of days in forex_opening_range_lab.py's original screen."""
    flag_a, flag_b = f"LONDON_{candidate}", f"OVERLAP_{candidate}"
    entry_idx_col = "OVERLAP_entry_bar_idx"
    atr = atr_wilder(h4, ATR_LEN)

    trades = []
    equity_curve = [STARTING_EQUITY]
    equity = STARTING_EQUITY
    blocked_until_idx = -1

    for _, day in daily.iterrows():
        if not (bool(day.get(flag_a, False)) and bool(day.get(flag_b, False))):
            continue
        london_dir, overlap_dir = day["LONDON_dir"], day["OVERLAP_dir"]
        if london_dir == 0 or london_dir != overlap_dir:
            continue
        direction = overlap_dir
        entry_idx = int(day[entry_idx_col])
        if entry_idx <= blocked_until_idx or entry_idx >= len(h4):
            continue
        result = _resolve_trade(h4, atr, entry_idx, direction)
        if result is None:
            continue
        r_outcome, entry_price, exit_idx = result

        equity *= (1 + RISK_PCT * r_outcome)
        equity_curve.append(equity)
        blocked_until_idx = exit_idx
        trades.append(Trade(pair, "CROSS", candidate, direction, h4.iloc[entry_idx]["time"], h4.iloc[exit_idx]["time"], r_outcome))

    return trades, equity_curve


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
        per_pair_h4[pair], per_pair_daily[pair] = h4, daily
        print(f"  {pair}: {len(h4)} H4 bars -> {len(daily)} days, {h4['time'].iloc[0]} to {h4['time'].iloc[-1]}")
        time.sleep(0.3)

    print("\n" + "=" * 110)
    print("LONDON / OVERLAP BREAKOUT BACKTEST vs. deployed golden-cross-bot-multi-pair")
    print("Same risk framework: ATR*2.0 stop, ATR*4.0 target (2:1 R:R), 1% risk/trade, per-pair independent equity")
    print("Reference (memory, same per-pair-independent methodology): BASELINE PF 1.232, WINNERS (GAPCONFIRM+CURCAP) PF 1.320")
    print("=" * 110)
    print(f"{'Candidate':<24} {'Trades':>7} {'Win%':>7} {'PF':>8} {'WorstPairDD%':>13}")
    print("-" * 110)

    all_results = {}
    for label, _, _ in OR_WINDOWS:
        for candidate in CANDIDATE_TYPES:
            pooled_trades = []
            dds = []
            for pair in PAIRS:
                trades, curve = simulate_pair(pair, per_pair_h4[pair], per_pair_daily[pair], label, candidate)
                pooled_trades.extend(trades)
                if len(curve) > 1:
                    dds.append(max_dd_pct(curve))
            stats = pf_stats(pooled_trades)
            worst_dd = max(dds) if dds else float("nan")
            name = f"{label}_{candidate.upper()}"
            all_results[name] = (pooled_trades, stats, worst_dd)
            print(f"{name:<24} {stats['trades']:>7} {stats['win_rate']:>6.1f}% {stats['pf']:>8.3f} {worst_dd:>12.2f}%")

    # CROSS-WINDOW confluence: LONDON and OVERLAP both fire AND agree on direction.
    for candidate in ("relvol", "range_top20"):
        pooled_trades = []
        dds = []
        for pair in PAIRS:
            trades, curve = simulate_pair_cross(pair, per_pair_h4[pair], per_pair_daily[pair], candidate)
            pooled_trades.extend(trades)
            if len(curve) > 1:
                dds.append(max_dd_pct(curve))
        stats = pf_stats(pooled_trades)
        worst_dd = max(dds) if dds else float("nan")
        name = f"CROSS_{candidate.upper()}"
        all_results[name] = (pooled_trades, stats, worst_dd)
        print(f"{name:<24} {stats['trades']:>7} {stats['win_rate']:>6.1f}% {stats['pf']:>8.3f} {worst_dd:>12.2f}%")
    print("=" * 110)

    print("\n" + "=" * 110)
    print("WALK-FORWARD CHECK (split at pooled-trade time median per candidate)")
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
    print("PER-PAIR BREAKDOWN (does it hold up pair-by-pair?)")
    print("=" * 110)
    for name, (trades, stats, worst_dd) in all_results.items():
        print(f"\n{name}:")
        for pair in PAIRS:
            pair_trades = [t for t in trades if t.pair == pair]
            s = pf_stats(pair_trades)
            print(f"  {pair:<10} trades={s['trades']:>4}  win%={s['win_rate']:>5.1f}  PF={s['pf']:>6.3f}")
    print("=" * 110 + "\n")

    # -----------------------------------------------------------------
    # R:R SWEEP (2026-09-10, user follow-up): every result above used a
    # fixed 2:1 target (RR_RATIO=2.0, i.e. target = 4x ATR against a 2x
    # ATR stop) inherited from the live bot's own risk framework. Stop
    # distance stays fixed at ATR*2.0 throughout this sweep - only the
    # TARGET multiple varies - testing whether a wider (or narrower)
    # target changes the picture for the two single-window candidates
    # that came closest to breakeven (LONDON_RELVOL, LONDON_RANGE_TOP20)
    # plus the two OVERLAP equivalents for contrast. Skips the
    # relvol_and_range/CROSS variants - they were strictly worse than
    # their single-signal counterparts at RR=2.0, no reason to re-sweep them.
    # -----------------------------------------------------------------
    print("\n" + "=" * 110)
    print("R:R SWEEP - stop fixed at ATR*2.0, target multiple varies (2.0 = the baseline used everywhere above)")
    print("=" * 110)
    sweep_candidates = [("LONDON", "relvol"), ("LONDON", "range_top20"), ("OVERLAP", "relvol"), ("OVERLAP", "range_top20")]
    print(f"{'Candidate':<20} " + " ".join(f"RR={rr:<4.1f}" for rr in RR_SWEEP))
    print("-" * 110)
    sweep_results = {}
    for label, candidate in sweep_candidates:
        row_pf, row_trades = [], []
        for rr in RR_SWEEP:
            pooled_trades = []
            for pair in PAIRS:
                trades, _ = simulate_pair(pair, per_pair_h4[pair], per_pair_daily[pair], label, candidate, rr_ratio=rr)
                pooled_trades.extend(trades)
            stats = pf_stats(pooled_trades)
            sweep_results[(label, candidate, rr)] = pooled_trades
            row_pf.append(stats["pf"])
            row_trades.append(stats["trades"])
        name = f"{label}_{candidate.upper()}"
        print(f"{name:<20} " + " ".join(f"{pf:6.3f}" for pf in row_pf))
        print(f"{'  (n trades)':<20} " + " ".join(f"{n:6d}" for n in row_trades))
    print("=" * 110)

    print("\n" + "=" * 110)
    print("R:R SWEEP - WALK-FORWARD CHECK on every cell above (early/late PF)")
    print("=" * 110)
    print(f"{'Candidate':<20} {'RR':>5} {'EarlyTr':>8} {'EarlyPF':>9} {'LateTr':>8} {'LatePF':>9}")
    print("-" * 110)
    for label, candidate in sweep_candidates:
        for rr in RR_SWEEP:
            trades = sweep_results[(label, candidate, rr)]
            if len(trades) < 10:
                print(f"{label}_{candidate.upper():<12} {rr:>5.1f}  SKIP - too few trades ({len(trades)})")
                continue
            times = sorted(t.entry_time for t in trades)
            mid = times[len(times) // 2]
            early = [t for t in trades if t.entry_time < mid]
            late = [t for t in trades if t.entry_time >= mid]
            es, ls = pf_stats(early), pf_stats(late)
            print(f"{label + '_' + candidate.upper():<20} {rr:>5.1f} {es['trades']:>8} {es['pf']:>9.3f} {ls['trades']:>8} {ls['pf']:>9.3f}")
    print("=" * 110)
    print("Reference (memory, same per-pair-independent methodology): BASELINE PF 1.232, WINNERS (GAPCONFIRM+CURCAP) PF 1.320\n")

    # -----------------------------------------------------------------
    # TRAILING EXIT SWEEP (2026-09-10, user follow-up: "try the trailing
    # exit instead"). No fixed target at all - a chandelier stop trails
    # peak-since-entry by chandelier_mult*ATR, only taking over once it's
    # more protective than the initial ATR*2.0 risk stop (see
    # _resolve_trade_trailing's docstring). The R:R sweep just above
    # showed a wider FIXED target creates an early-half-only illusion of
    # improvement that fails walk-forward - a trailing exit is a
    # genuinely different mechanism (it can bank a smaller profit quickly
    # on a choppy trade, or ride a real trend indefinitely), not just
    # "the same idea with a bigger number," so it's not assumed to fail
    # the same way.
    # -----------------------------------------------------------------
    print("\n" + "=" * 110)
    print("TRAILING EXIT SWEEP - chandelier trail distance varies, no fixed target (vs. fixed-target results above)")
    print("=" * 110)
    print(f"{'Candidate':<20} " + " ".join(f"CH={c:<4.1f}" for c in CHANDELIER_SWEEP))
    print("-" * 110)
    trail_results = {}
    for label, candidate in sweep_candidates:
        row_pf, row_trades = [], []
        for ch in CHANDELIER_SWEEP:
            pooled_trades = []
            for pair in PAIRS:
                trades, _ = simulate_pair_trailing(pair, per_pair_h4[pair], per_pair_daily[pair], label, candidate, chandelier_mult=ch)
                pooled_trades.extend(trades)
            stats = pf_stats(pooled_trades)
            trail_results[(label, candidate, ch)] = pooled_trades
            row_pf.append(stats["pf"])
            row_trades.append(stats["trades"])
        name = f"{label}_{candidate.upper()}"
        print(f"{name:<20} " + " ".join(f"{pf:6.3f}" for pf in row_pf))
        print(f"{'  (n trades)':<20} " + " ".join(f"{n:6d}" for n in row_trades))
    print("=" * 110)

    print("\n" + "=" * 110)
    print("TRAILING EXIT SWEEP - WALK-FORWARD CHECK on every cell above (early/late PF)")
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
    print("TRAILING EXIT - PER-PAIR BREAKDOWN of the LONDON candidates (does the pooled pass hold up pair-by-pair?)")
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
    print("Reference (memory): BASELINE PF 1.232, WINNERS (GAPCONFIRM+CURCAP) PF 1.320. "
          "1% risk/trade, per-pair independent equity, NO spread/slippage modeled.\n")


if __name__ == "__main__":
    main()
