"""
"Run CROSS_RANGE_TOP20's narrowed portfolios against the live execution
model" - what that turned out to mean once actually checked (2026-09-10):

1. `trading_core/simulation.py` and `trading_core/strategies.py` (the
   engine this project's memory refers to as "the live execution model,"
   with its different next-bar-open fill convention) NO LONGER EXIST in
   the deployed golden-cross-bot-multi-pair repo, and neither does
   golden_cross_portfolio_lab.py (which was the only thing that imported
   them). Confirmed via the GitHub API - trading_core/ now contains only
   execution.py (Alpaca order-reconciliation helpers, irrelevant to OANDA
   forex) and __init__.py. Presumably removed by the user's parallel
   ChatGPT tool (see project_astra_agent_incident.md) - this is a real,
   observable repo change, not something to route around silently.
   CANNOT run anything against that engine because it doesn't exist.

2. What DOES still exist and IS the actual live execution model:
   `bot.py`'s own order-placement code. Two mechanics from it are directly
   testable against the narrowed-portfolio backtests:
     a. CURCAP - blocks a new trade if opening it would push cumulative
        open risk sharing EITHER the base or quote currency above
        MAX_CURRENCY_RISK_PCT (2.5%) of current equity, at RISK_PER_TRADE_
        PCT (1.0%) per trade. ALL FOUR candidate pairs (GBP_JPY, GBP_AUD,
        GBP_CAD, GBP_CHF) have GBP as the base currency - so 3 concurrent
        1%-risk GBP-cross positions (3%) EXCEEDS the 2.5% cap. Both
        narrowed portfolios' earlier shared-equity sim reported max
        concurrent positions = 3, meaning CURCAP would plausibly have
        blocked some of those trades live. This is the one directly,
        mechanically testable piece - implemented below exactly as
        bot.py's check_and_trade() computes it (same constants, same
        base-OR-quote block rule, base_risk/quote_risk tracked from
        currently-open ACCEPTED trades only).
     b. Order type: bot.py places ONE market order with stopLossOnFill +
        takeProfitOnFill attached (a fixed bracket) - there is NO trailing-
        stop update logic anywhere in the file (grepped for "trail" -
        zero matches). This means Portfolio B's chandelier trail (the
        stronger of the two candidates, PF 1.686) has NO live-execution
        analog to test against AS THE BOT CURRENTLY STANDS - it would
        need new code (poll open positions each run, recompute a trailing
        level, PATCH the stop-loss order) that doesn't exist yet. It's
        still run through CURCAP below (the cap logic doesn't care about
        exit style), clearly labeled as "IF a trailing capability existed"
        rather than "what bot.py does today."

Imports the (bugfixed) engine from forex_london_overlap_breakout_lab.py
and the trailing-cross wrapper from forex_cross_range_top20_lab.py -
same trade lists as the narrowed-portfolio report, re-simulated through a
CURCAP-aware event-driven equity curve instead of an uncapped one.

Run (needs OANDA_API_KEY; must sit alongside both of those files):
    python forex_cross_range_top20_curcap_lab.py
"""

import time

import numpy as np

import forex_london_overlap_breakout_lab as lab
import forex_cross_range_top20_lab as deep

CANDIDATE = "range_top20"
PORTFOLIO_A = ["GBP_JPY", "GBP_AUD", "GBP_CAD"]   # fixed 2:1 target - matches bot.py's order style
PORTFOLIO_B = ["GBP_JPY", "GBP_AUD", "GBP_CHF"]   # CH=4.0 trailing - NO live analog, see docstring
ALL_PAIRS = sorted(set(PORTFOLIO_A) | set(PORTFOLIO_B))

# exact constants from bot.py - do not drift from these without a reason
RISK_PER_TRADE_PCT = 1.0
MAX_CURRENCY_RISK_PCT = 2.5


def shared_equity_sim(all_trades, starting_equity=10000.0, risk_pct=0.01):
    """Uncapped baseline (same as forex_cross_range_top20_narrowed_lab.py) -
    reproduced here for a clean side-by-side comparison in one report."""
    events = []
    for t in all_trades:
        events.append((t.entry_time, 0, id(t), t))
        events.append((t.exit_time, 1, id(t), t))
    events.sort(key=lambda e: (e[0], e[1]))
    equity = starting_equity
    pending = {}
    curve = [starting_equity]
    max_concurrent = 0
    open_set = set()
    for _time, kind, tid, t in events:
        if kind == 0:
            pending[tid] = equity * risk_pct
            open_set.add(tid)
            max_concurrent = max(max_concurrent, len(open_set))
        else:
            risk_amount = pending.pop(tid, equity * risk_pct)
            equity += risk_amount * t.r_multiple
            open_set.discard(tid)
            curve.append(equity)
    return curve, max_concurrent


def curcap_shared_equity_sim(all_trades, starting_equity=10000.0,
                             risk_pct=RISK_PER_TRADE_PCT / 100, cap_pct=MAX_CURRENCY_RISK_PCT / 100):
    """Same event-driven engine, but every ENTRY is gated by bot.py's exact
    CURCAP rule before being admitted: blocked if base-currency OR quote-
    currency open risk (from currently-open ACCEPTED trades only) plus this
    trade's own risk would exceed cap_pct of CURRENT equity. A blocked
    signal is simply skipped - matches CROSS_RANGE_TOP20's same-day trigger
    (unlike an SMA crossover, it doesn't persist to retry on a later bar)."""
    events = []
    for t in all_trades:
        events.append((t.entry_time, 0, id(t), t))
        events.append((t.exit_time, 1, id(t), t))
    events.sort(key=lambda e: (e[0], e[1]))

    equity = starting_equity
    open_risk = {}
    pending = {}
    accepted, rejected = [], []
    curve = [starting_equity]
    max_concurrent = 0

    for _time, kind, tid, t in events:
        base, quote = t.pair.split("_")
        if kind == 0:
            risk_amount = equity * risk_pct
            cap_dollars = equity * cap_pct
            base_risk = open_risk.get(base, 0.0)
            quote_risk = open_risk.get(quote, 0.0)
            if base_risk + risk_amount > cap_dollars or quote_risk + risk_amount > cap_dollars:
                rejected.append(t)
                continue
            open_risk[base] = base_risk + risk_amount
            open_risk[quote] = quote_risk + risk_amount
            pending[tid] = (risk_amount, base, quote)
            accepted.append(t)
            max_concurrent = max(max_concurrent, len(pending))
        else:
            if tid not in pending:
                continue  # was rejected at entry - its exit is a no-op
            risk_amount, base, quote = pending.pop(tid)
            open_risk[base] -= risk_amount
            open_risk[quote] -= risk_amount
            equity += risk_amount * t.r_multiple
            curve.append(equity)

    return curve, accepted, rejected, max_concurrent


def max_dd(curve):
    peak, worst = curve[0], 0.0
    for e in curve:
        peak = max(peak, e)
        worst = max(worst, (peak - e) / peak * 100)
    return worst


def report(name, trades, live_analog: bool):
    print(f"\n{'=' * 110}")
    print(f"{name}  ({'matches bot.py order style' if live_analog else 'NO live analog - bot.py has no trailing-stop capability'})")
    print("=" * 110)

    free_curve, free_conc = shared_equity_sim(trades)
    free_dd = max_dd(free_curve)
    print(f"UNCAPPED shared-equity (as reported previously): final=${free_curve[-1]:,.0f} "
          f"net={(free_curve[-1]/free_curve[0]-1)*100:+.1f}%  max DD={free_dd:.1f}%  max concurrent={free_conc}")

    curve, accepted, rejected, conc = curcap_shared_equity_sim(trades)
    dd = max_dd(curve)
    s = lab.pf_stats(accepted)
    print(f"\nCURCAP-GATED (bot.py's actual rule: block if base OR quote currency open risk + 1% > 2.5% of equity):")
    print(f"  {len(rejected)} of {len(trades)} trades BLOCKED that the uncapped backtest took ({len(rejected)/len(trades)*100:.0f}%)")
    print(f"  Accepted trades: n={s['trades']} win%={s['win_rate']:.1f} PF={s['pf']:.3f}")
    print(f"  Final equity=${curve[-1]:,.0f}  net={(curve[-1]/curve[0]-1)*100:+.1f}%  "
          f"max DD={dd:.1f}%  max concurrent={conc}")

    print(f"\n  Impact of CURCAP: net return {(free_curve[-1]/free_curve[0]-1)*100:+.1f}% -> "
          f"{(curve[-1]/curve[0]-1)*100:+.1f}%,  max drawdown {free_dd:.1f}% -> {dd:.1f}%")


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

    trades_a = []
    for pair in PORTFOLIO_A:
        tr, _ = lab.simulate_pair_cross(pair, per_pair_A[pair], per_pair_daily[pair], CANDIDATE, cost_mult=realistic)
        trades_a.extend(tr)
    report(f"PORTFOLIO A - {PORTFOLIO_A}, fixed 2:1 target", trades_a, live_analog=True)

    trades_b = []
    for pair in PORTFOLIO_B:
        tr, _ = deep.simulate_pair_cross_trailing(pair, per_pair_A[pair], per_pair_daily[pair], CANDIDATE,
                                                  chandelier_mult=4.0, cost_mult=realistic)
        trades_b.extend(tr)
    report(f"PORTFOLIO B - {PORTFOLIO_B}, CH=4.0 chandelier trail", trades_b, live_analog=False)

    print("\n" + "=" * 110)
    print("SUMMARY: trading_core.simulation.py / strategies.py / golden_cross_portfolio_lab.py no longer exist in")
    print("the deployed repo - this file's CURCAP replication of bot.py's actual check_and_trade() logic is the")
    print("closest available thing to 'the live execution model.' Portfolio B has no live order-management analog")
    print("at all (no trailing-stop code in bot.py) - its CURCAP numbers describe a bot that doesn't exist yet.")
    print("=" * 110 + "\n")


if __name__ == "__main__":
    main()
