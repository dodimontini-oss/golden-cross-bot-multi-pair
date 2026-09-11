"""
Narrowed-portfolio test of CROSS_RANGE_TOP20, per the deep-dive finding
that the pooled 9-pair edge is really a 3-pair edge carried by GBP_JPY +
GBP_AUD + (GBP_CAD with the fixed target / GBP_CHF with the trailing
exit). Tests exactly those two combinations - does NOT re-optimize or
pick new pairs, since that would just be re-fitting to the same sample.

METHODOLOGY CAVEAT (read before trusting the numbers): these 3 pairs were
selected BECAUSE they passed the per-pair walk-forward check in
forex_cross_range_top20_lab.py, on the SAME 2015-2026 sample tested here.
This is NOT a fresh out-of-sample validation - it mechanically must look
good, since the pairs were chosen for looking good. What this script adds
that IS new information:
  1. A finer THIRDS split (not just the halves already used to pick the
     pairs) - weak evidence, still same-sample, but at least a different
     cut than the one used for selection.
  2. A SHARED-EQITY, single-account, event-driven simulation (entry sizes
     equity at trade-open, P&L realizes at trade-close, trades from
     different pairs can be concurrently open) - the per-pair independent-
     $10k-each backtests never show concurrent-drawdown or correlated-
     exposure risk, and these 3-4 pairs all have GBP as the base currency,
     so a real account running them together could take correlated hits.
  3. Concurrent-exposure / direction-agreement stats - when two of these
     pairs have overlapping open positions, do they agree on GBP direction
     (compounding exposure) or disagree (partial natural hedge)?

Imports the (bugfixed) engine from forex_london_overlap_breakout_lab.py
and the cross+trailing wrapper from forex_cross_range_top20_lab.py.

Run (needs OANDA_API_KEY; must sit alongside both of those files):
    python forex_cross_range_top20_narrowed_lab.py
"""

import time

import numpy as np
import pandas as pd

import forex_london_overlap_breakout_lab as lab
import forex_cross_range_top20_lab as deep

CANDIDATE = "range_top20"
PORTFOLIO_A = ["GBP_JPY", "GBP_AUD", "GBP_CAD"]   # fixed 2:1 target combo
PORTFOLIO_B = ["GBP_JPY", "GBP_AUD", "GBP_CHF"]   # CH=4.0 trailing combo
ALL_PAIRS = sorted(set(PORTFOLIO_A) | set(PORTFOLIO_B))


def thirds_split(trades):
    times = sorted(t.entry_time for t in trades)
    if len(times) < 15:
        return None
    t1, t2 = times[len(times) // 3], times[2 * len(times) // 3]
    return (
        [t for t in trades if t.entry_time < t1],
        [t for t in trades if t1 <= t.entry_time < t2],
        [t for t in trades if t.entry_time >= t2],
    )


def shared_equity_sim(all_trades, starting_equity=10000.0, risk_pct=0.01):
    """Event-driven: each trade's risk-dollar size is locked in at ITS OWN
    entry against equity AT THAT MOMENT; P&L realizes at ITS OWN exit.
    Trades from different pairs can be concurrently open - this is what a
    real one-account bot running all these pairs together would do."""
    events = []
    for t in all_trades:
        events.append((t.entry_time, 0, id(t), t))
        events.append((t.exit_time, 1, id(t), t))
    events.sort(key=lambda e: (e[0], e[1]))

    equity = starting_equity
    pending = {}
    curve = [starting_equity]
    open_set = set()
    max_concurrent = 0
    for _time, kind, tid, t in events:
        if kind == 0:
            pending[tid] = equity * risk_pct
            open_set.add(tid)
            max_concurrent = max(max_concurrent, len(open_set))
        else:
            risk_dollars = pending.pop(tid, equity * risk_pct)
            equity += risk_dollars * t.r_multiple
            open_set.discard(tid)
            curve.append(equity)
    return curve, max_concurrent


def max_dd(curve):
    peak, worst = curve[0], 0.0
    for e in curve:
        peak = max(peak, e)
        worst = max(worst, (peak - e) / peak * 100)
    return worst


def direction_agreement(trades):
    """For every cross-pair pair of trades whose [entry,exit] intervals
    overlap: same GBP direction (both long or both short - all these
    pairs quote GBP as the base, so `direction` is directly comparable)
    vs opposite (partial natural hedge)."""
    trades = list(trades)
    same, opp = 0, 0
    for i in range(len(trades)):
        a = trades[i]
        for j in range(i + 1, len(trades)):
            b = trades[j]
            if a.pair == b.pair:
                continue
            if a.entry_time < b.exit_time and b.entry_time < a.exit_time:
                if a.direction == b.direction:
                    same += 1
                else:
                    opp += 1
    return same, opp


def report_portfolio(name, pairs, trades, per_pair_curve_source):
    print(f"\n{'=' * 110}")
    print(f"{name}: {pairs}")
    print("=" * 110)
    s = lab.pf_stats(trades)
    print(f"Independent-equity pooled: n={s['trades']} win%={s['win_rate']:.1f} PF={s['pf']:.3f}")

    halves = thirds_split(trades)
    # also the halves check (same cut used to select these pairs - for reference, not new evidence)
    times = sorted(t.entry_time for t in trades)
    mid = times[len(times) // 2]
    e_half = lab.pf_stats([t for t in trades if t.entry_time < mid])
    l_half = lab.pf_stats([t for t in trades if t.entry_time >= mid])
    print(f"Halves (same cut used to select these pairs - reference only): "
          f"early PF={e_half['pf']:.3f} (n={e_half['trades']})  late PF={l_half['pf']:.3f} (n={l_half['trades']})")

    if halves:
        t1, t2, t3 = halves
        s1, s2, s3 = lab.pf_stats(t1), lab.pf_stats(t2), lab.pf_stats(t3)
        print(f"THIRDS (a different cut - weak but new evidence): "
              f"1st PF={s1['pf']:.3f} (n={s1['trades']})  2nd PF={s2['pf']:.3f} (n={s2['trades']})  "
              f"3rd PF={s3['pf']:.3f} (n={s3['trades']})")
    else:
        print("Too few trades to split into thirds.")

    print("\nPer-pair (already known from the deep-dive, shown again for this exact combo):")
    for pair in pairs:
        pt = [t for t in trades if t.pair == pair]
        ps = lab.pf_stats(pt)
        print(f"  {pair:<10} n={ps['trades']:>4}  win%={ps['win_rate']:>5.1f}  PF={ps['pf']:>6.3f}")

    curve, max_conc = shared_equity_sim(trades)
    dd = max_dd(curve)
    net_pct = (curve[-1] / curve[0] - 1) * 100
    print(f"\nSHARED-EQUITY single-account simulation (starting $10,000, 1% risk/trade, "
          f"event-driven across all {len(pairs)} pairs at once):")
    print(f"  Final equity=${curve[-1]:,.0f}  net={net_pct:+.1f}%  max drawdown={dd:.1f}%  "
          f"max concurrent open positions={max_conc}")

    same, opp = direction_agreement(trades)
    total = same + opp
    if total > 0:
        print(f"\nCONCURRENT-EXPOSURE check: of {total} overlapping cross-pair trade pairs, "
              f"{same} ({same/total*100:.0f}%) agree on GBP direction (compounding exposure), "
              f"{opp} ({opp/total*100:.0f}%) disagree (partial natural hedge)")
    else:
        print("\nNo overlapping cross-pair trades found (positions never coincided in time).")


def main():
    print(f"Fetching full {lab.FROM_TIME[:10]}-present H4 history for {ALL_PAIRS}...")
    per_pair_A, per_pair_daily = {}, {}
    for pair in ALL_PAIRS:
        h4 = lab.get_candles_range(pair, lab.FROM_TIME)
        daily = lab.add_day_flags(lab.build_daily(h4))
        per_pair_A[pair] = lab._Arrays(h4)
        per_pair_daily[pair] = daily
        print(f"  {pair}: {len(h4)} H4 bars -> {len(daily)} days")
        time.sleep(0.3)

    realistic = lab.COST_SCENARIOS["realistic"]

    # ---- Portfolio A: fixed 2:1 target ----
    trades_a = []
    for pair in PORTFOLIO_A:
        tr, _ = lab.simulate_pair_cross(pair, per_pair_A[pair], per_pair_daily[pair], CANDIDATE, cost_mult=realistic)
        trades_a.extend(tr)
    report_portfolio("PORTFOLIO A - fixed 2:1 target, REALISTIC COST", PORTFOLIO_A, trades_a, per_pair_A)

    # ---- Portfolio B: CH=4.0 trailing exit ----
    trades_b = []
    for pair in PORTFOLIO_B:
        tr, _ = deep.simulate_pair_cross_trailing(pair, per_pair_A[pair], per_pair_daily[pair], CANDIDATE,
                                                  chandelier_mult=4.0, cost_mult=realistic)
        trades_b.extend(tr)
    report_portfolio("PORTFOLIO B - CH=4.0 chandelier trail, REALISTIC COST", PORTFOLIO_B, trades_b, per_pair_A)

    print("\n" + "=" * 110)
    print("CAVEAT: these 3 pairs were selected because they passed the per-pair walk-forward check in")
    print("forex_cross_range_top20_lab.py, on this SAME 2015-2026 sample. The halves numbers above are NOT")
    print("fresh evidence - they're the same check used to pick the pairs. The thirds split is a different")
    print("cut (weak but non-identical evidence). True out-of-sample validation needs future data.")
    print("Reference: deployed golden-cross-bot-multi-pair BASELINE PF 1.232, WINNERS PF 1.320 (tighter cost model).")
    print("=" * 110 + "\n")


if __name__ == "__main__":
    main()
