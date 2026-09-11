"""
Deep dive on CROSS_RANGE_TOP20 - the best-performing candidate to come out
of the (bugfixed) forex_london_overlap_breakout_lab.py sweep: pooled
realistic-cost PF 1.097, walk-forward 1.089/1.104, the most comfortable
double-halves pass of anything in that whole research arc. Everything else
tested there landed 0.90-1.05; this is the one worth a proper look before
calling the line fully dead.

Entry rule (unchanged from the main lab): require LONDON (09:00-13:00 UTC)
AND OVERLAP (13:00-17:00 UTC) H4 bars to BOTH sit in the top-20th-
percentile of their own trailing 60-day range distribution AND agree on
direction; enter at the OPEN of the bar after OVERLAP closes (17:00 UTC),
in the agreed direction.

Imports the (bugfixed, numpy-vectorized) engine from
forex_london_overlap_breakout_lab.py directly rather than reimplementing -
this inherits its correctness (local-vs-global indexing fix, same-day-
resolution exit scan) instead of risking a fresh bug.

Adds five things the main lab's generic sweep had no room for:
  A. Per-pair breakdown + per-pair walk-forward (fixed-target, realistic cost)
  B. Trade anatomy (win%, avg win/loss R, expectancy, hold time)
  C. Cost sensitivity (frictionless / realistic / pessimistic)
  D. Year-by-year pooled stability
  E. A TRAILING-exit variant (chandelier 2.0-5.0x ATR) - no CROSS candidate
     was ever run through the trailing sweep in the main lab (that sweep
     only covered single-window LONDON/OVERLAP)

Run (needs OANDA_API_KEY; must sit alongside forex_london_overlap_
breakout_lab.py in the same repo so the import resolves):
    python forex_cross_range_top20_lab.py
"""

import time

import numpy as np
import pandas as pd

import forex_london_overlap_breakout_lab as lab

CANDIDATE = "range_top20"
CHANDELIER_SWEEP = [2.0, 3.0, 4.0, 5.0]


def simulate_pair_cross_trailing(pair, h4, daily, candidate, chandelier_mult, cost_mult=0.0):
    """CROSS-window trigger (both windows fire + agree), chandelier trailing
    exit - the one combination the main lab never tested. Composed from the
    main lab's own building blocks so it inherits their correctness."""
    return lab._run(pair, "CROSS", candidate, lab._arrays(h4), daily, "OVERLAP_entry_bar_idx",
                    lab._cross_window_dir(candidate),
                    lambda A, ei, d: lab._resolve_trade_trailing(A, ei, d, chandelier_mult), cost_mult)


def anatomy(trades):
    """trades already carry cost in r_multiple (baked in by whatever
    cost_mult was passed to the simulator that produced them)."""
    rs = [t.r_multiple for t in trades]
    wins = [x for x in rs if x > 0]
    losses = [x for x in rs if x <= 0]
    return {
        "n": len(rs),
        "win_pct": len(wins) / len(rs) * 100 if rs else float("nan"),
        "avg_win": np.mean(wins) if wins else float("nan"),
        "avg_loss": np.mean(losses) if losses else float("nan"),
        "expectancy": np.mean(rs) if rs else float("nan"),
        "pf": lab.pf_stats(trades)["pf"],
    }


def bars_held(trades):
    """Trade doesn't store bar count directly - recompute from the entry/
    exit time delta in units of 4h (exact, since H4 bars are evenly spaced)."""
    out = []
    for t in trades:
        delta_hours = (pd.Timestamp(t.exit_time) - pd.Timestamp(t.entry_time)).total_seconds() / 3600
        out.append(round(delta_hours / 4))
    return out


def main():
    print(f"Fetching full {lab.FROM_TIME[:10]}-present H4 history for {len(lab.PAIRS)} pairs (paginated)...")
    per_pair_A, per_pair_daily = {}, {}
    for pair in lab.PAIRS:
        h4 = lab.get_candles_range(pair, lab.FROM_TIME)
        daily = lab.add_day_flags(lab.build_daily(h4))
        per_pair_A[pair] = lab._Arrays(h4)
        per_pair_daily[pair] = daily
        print(f"  {pair}: {len(h4)} H4 bars -> {len(daily)} days")
        time.sleep(0.3)

    realistic = lab.COST_SCENARIOS["realistic"]
    pessimistic = lab.COST_SCENARIOS["pessimistic"]

    # =================================================================
    # A. FIXED-TARGET CROSS_RANGE_TOP20 - pooled, cost scenarios
    # =================================================================
    print("\n" + "=" * 110)
    print("A. CROSS_RANGE_TOP20 (fixed 2:1 target) - pooled across all 9 pairs")
    print("=" * 110)
    trades_by_cost = {}
    for scen, mult in lab.COST_SCENARIOS.items():
        pooled = []
        for pair in lab.PAIRS:
            tr, _ = lab.simulate_pair_cross(pair, per_pair_A[pair], per_pair_daily[pair], CANDIDATE, cost_mult=mult)
            pooled.extend(tr)
        trades_by_cost[scen] = pooled
        s = lab.pf_stats(pooled)
        print(f"  {scen:<14} n={s['trades']:>5}  win%={s['win_rate']:>5.1f}  PF={s['pf']:>6.3f}")

    realistic_trades = trades_by_cost["realistic"]

    print("\nWalk-forward (realistic cost, split at pooled-trade time median):")
    times = sorted(t.entry_time for t in realistic_trades)
    mid = times[len(times) // 2]
    early = [t for t in realistic_trades if t.entry_time < mid]
    late = [t for t in realistic_trades if t.entry_time >= mid]
    es, ls = lab.pf_stats(early), lab.pf_stats(late)
    print(f"  EARLY: n={es['trades']:>4} PF={es['pf']:.3f}   LATE: n={ls['trades']:>4} PF={ls['pf']:.3f}")

    # =================================================================
    # B. PER-PAIR BREAKDOWN + per-pair walk-forward (realistic cost)
    # =================================================================
    print("\n" + "=" * 110)
    print("B. PER-PAIR BREAKDOWN - REALISTIC COST (does the pooled pass hold up pair-by-pair?)")
    print("=" * 110)
    print(f"  {'pair':<10} {'n':>5} {'win%':>7} {'PF':>7} {'earlyPF':>9} {'latePF':>9}   both halves >1.0?")
    n_pass = 0
    for pair in lab.PAIRS:
        pt = [t for t in realistic_trades if t.pair == pair]
        s = lab.pf_stats(pt)
        if len(pt) < 10:
            print(f"  {pair:<10} {s['trades']:>5}  (too few to split)")
            continue
        ptimes = sorted(t.entry_time for t in pt)
        pmid = ptimes[len(ptimes) // 2]
        pe = lab.pf_stats([t for t in pt if t.entry_time < pmid])
        pl = lab.pf_stats([t for t in pt if t.entry_time >= pmid])
        ok = pe["pf"] > 1.0 and pl["pf"] > 1.0
        n_pass += ok
        print(f"  {pair:<10} {s['trades']:>5} {s['win_rate']:>6.1f}% {s['pf']:>7.3f} {pe['pf']:>9.3f} {pl['pf']:>9.3f}   {'YES' if ok else 'no'}")
    print(f"  -> {n_pass}/9 pairs pass both walk-forward halves individually")

    # =================================================================
    # C. TRADE ANATOMY (frictionless vs realistic)
    # =================================================================
    print("\n" + "=" * 110)
    print("C. TRADE ANATOMY - fixed 2:1 target")
    print("=" * 110)
    for scen in ("frictionless", "realistic", "pessimistic"):
        a = anatomy(trades_by_cost[scen])
        held = bars_held(trades_by_cost[scen])
        print(f"  {scen:<14} n={a['n']:>5}  win%={a['win_pct']:>5.1f}  avgWin={a['avg_win']:>6.3f}R  "
              f"avgLoss={a['avg_loss']:>7.3f}R  expectancy={a['expectancy']:>+7.3f}R  PF={a['pf']:>6.3f}  "
              f"medBarsHeld={np.median(held):>4.0f}")

    # =================================================================
    # D. YEAR BY YEAR (pooled, realistic cost)
    # =================================================================
    print("\n" + "=" * 110)
    print("D. YEAR-BY-YEAR STABILITY - pooled across all 9 pairs, REALISTIC COST")
    print("=" * 110)
    by_year = {}
    for t in realistic_trades:
        yr = pd.Timestamp(t.entry_time).year
        by_year.setdefault(yr, []).append(t.r_multiple)
    print(f"  {'year':<6} {'n':>5} {'win%':>7} {'expectancy R':>13} {'PF':>7} {'sum R':>9}")
    for yr in sorted(by_year):
        rs = by_year[yr]
        wp = sum(1 for x in rs if x > 0) / len(rs) * 100
        gw = sum(x for x in rs if x > 0)
        gl = abs(sum(x for x in rs if x <= 0))
        pf = gw / gl if gl > 0 else float("inf")
        print(f"  {yr:<6} {len(rs):>5} {wp:>6.1f}% {np.mean(rs):>13.3f} {pf:>7.3f} {sum(rs):>9.2f}")

    # =================================================================
    # E. TRAILING EXIT VARIANT - not tested on any CROSS candidate before
    # =================================================================
    print("\n" + "=" * 110)
    print("E. TRAILING EXIT SWEEP on CROSS_RANGE_TOP20 (chandelier trail, no fixed target)")
    print("(PF shown frictionless / realistic-cost for each cell)")
    print("=" * 110)
    trail_trades = {}
    for ch in CHANDELIER_SWEEP:
        free_pooled, cost_pooled = [], []
        for pair in lab.PAIRS:
            ft, _ = simulate_pair_cross_trailing(pair, per_pair_A[pair], per_pair_daily[pair], CANDIDATE, ch, cost_mult=0.0)
            ct, _ = simulate_pair_cross_trailing(pair, per_pair_A[pair], per_pair_daily[pair], CANDIDATE, ch, cost_mult=realistic)
            free_pooled.extend(ft)
            cost_pooled.extend(ct)
        trail_trades[ch] = cost_pooled
        fs, cs = lab.pf_stats(free_pooled), lab.pf_stats(cost_pooled)
        print(f"  CH={ch:.1f}   n={cs['trades']:>5}   PF frictionless={fs['pf']:.3f}   PF realistic={cs['pf']:.3f}")

    print("\nWalk-forward per CH (realistic cost):")
    for ch in CHANDELIER_SWEEP:
        trades = trail_trades[ch]
        if len(trades) < 10:
            print(f"  CH={ch:.1f}  SKIP - too few trades")
            continue
        ttimes = sorted(t.entry_time for t in trades)
        tmid = ttimes[len(ttimes) // 2]
        te = lab.pf_stats([t for t in trades if t.entry_time < tmid])
        tl = lab.pf_stats([t for t in trades if t.entry_time >= tmid])
        print(f"  CH={ch:.1f}  early n={te['trades']:>4} PF={te['pf']:.3f}   late n={tl['trades']:>4} PF={tl['pf']:.3f}")

    # per-pair for the best-looking CH only (whichever passes both halves most comfortably)
    best_ch = None
    best_margin = -999
    for ch in CHANDELIER_SWEEP:
        trades = trail_trades[ch]
        if len(trades) < 10:
            continue
        ttimes = sorted(t.entry_time for t in trades)
        tmid = ttimes[len(ttimes) // 2]
        te = lab.pf_stats([t for t in trades if t.entry_time < tmid])
        tl = lab.pf_stats([t for t in trades if t.entry_time >= tmid])
        margin = min(te["pf"], tl["pf"])
        if margin > best_margin:
            best_margin, best_ch = margin, ch

    if best_ch is not None:
        print(f"\nPer-pair breakdown for the best trailing config, CH={best_ch:.1f} (realistic cost):")
        bt = trail_trades[best_ch]
        n_pass_t = 0
        for pair in lab.PAIRS:
            pt = [t for t in bt if t.pair == pair]
            s = lab.pf_stats(pt)
            if len(pt) < 10:
                print(f"  {pair:<10} {s['trades']:>5}  (too few to split)")
                continue
            ptimes = sorted(t.entry_time for t in pt)
            pmid = ptimes[len(ptimes) // 2]
            pe = lab.pf_stats([t for t in pt if t.entry_time < pmid])
            pl = lab.pf_stats([t for t in pt if t.entry_time >= pmid])
            ok = pe["pf"] > 1.0 and pl["pf"] > 1.0
            n_pass_t += ok
            print(f"  {pair:<10} {s['trades']:>5} {s['win_rate']:>6.1f}% {s['pf']:>7.3f} {pe['pf']:>9.3f} {pl['pf']:>9.3f}   {'YES' if ok else 'no'}")
        print(f"  -> {n_pass_t}/9 pairs pass both walk-forward halves individually")

    print("\n" + "=" * 110)
    print("Reference: deployed golden-cross-bot-multi-pair BASELINE PF 1.232, WINNERS PF 1.320 (much tighter cost model).")
    print("=" * 110 + "\n")


if __name__ == "__main__":
    main()
