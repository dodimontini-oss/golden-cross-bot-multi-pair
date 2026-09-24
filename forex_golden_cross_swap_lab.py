"""
Swap (financing) re-run of the forex golden-cross review - follow-up to
forex_golden_cross_review_lab.py, which modelled spreads/slippage/gaps but NOT
overnight financing.

How financing is modelled
  - A held FX position earns the base-currency interest rate and pays the quote-
    currency rate on its notional; the broker then takes a markup on top, in BOTH
    directions. Per year, as a decimal on notional:
        long  rate = (r_base - r_quote) - markup
        short rate = -(r_base - r_quote) - markup        (negative = you pay)
  - OANDA's own CURRENT long/short rates come from its API. Their sum gives the
    markup directly (long + short = -2 x markup) and their difference gives the
    current differential ((long - short)/2) - no policy-rate knowledge needed.
  - For history, the differential comes from approximate central-bank policy-rate
    step tables (written from memory, month-level accuracy). They are checked
    against OANDA's current implied differentials below, and from 2026-01-01 the
    OANDA-implied differential is used instead of the table.
  - The markup is held constant over the whole history (OANDA's true historical
    markup is unknown; this is an assumption).
  - Accrued by calendar day held (the Wednesday triple rollover covers the weekend),
    on notional. In R: financing_R = -rate x days/365 / (stop_distance/entry_price),
    because notional/risk = entry/stop_distance. Rate is fixed at the entry date.

Scenarios (all with the previous 'realistic' spreads + slippage, gap fills, CURCAP):
  A no swap (reproduces the previous run - regression check)   B carry + markup (the model)
  C markup only   D carry only   E/F/G flat 1%/2%/3% p.a. drag on notional (sensitivity)

Run (needs OANDA_API_KEY and OANDA_ACCOUNT_ID): python forex_golden_cross_swap_lab.py
"""

import bisect
import os
import sys
import time

import numpy as np
import pandas as pd
import requests

import forex_golden_cross_review_lab as base

DEFAULT_MARKUP = 0.010   # only used if OANDA's financing fields are unavailable

# Approximate central-bank policy rates, percent p.a., (effective date, rate). From memory.
POLICY = {
    "USD": [("2015-01-01", .25), ("2015-12-17", .50), ("2016-12-15", .75), ("2017-03-16", 1.0), ("2017-06-15", 1.25),
            ("2017-12-14", 1.5), ("2018-03-22", 1.75), ("2018-06-14", 2.0), ("2018-09-27", 2.25), ("2018-12-20", 2.5),
            ("2019-08-01", 2.25), ("2019-09-19", 2.0), ("2019-10-31", 1.75), ("2020-03-04", 1.25), ("2020-03-16", .25),
            ("2022-03-17", .5), ("2022-05-05", 1.0), ("2022-06-16", 1.75), ("2022-07-28", 2.5), ("2022-09-22", 3.25),
            ("2022-11-03", 4.0), ("2022-12-15", 4.5), ("2023-02-02", 4.75), ("2023-03-23", 5.0), ("2023-05-04", 5.25),
            ("2023-07-27", 5.5), ("2024-09-19", 5.0), ("2024-11-08", 4.75), ("2024-12-19", 4.5), ("2025-09-18", 4.25),
            ("2025-10-30", 4.0), ("2025-12-11", 3.75)],
    "EUR": [("2015-01-01", -.20), ("2015-12-10", -.30), ("2016-03-16", -.40), ("2019-09-18", -.50), ("2022-07-21", 0.0),
            ("2022-09-08", .75), ("2022-10-27", 1.5), ("2022-12-15", 2.0), ("2023-02-02", 2.5), ("2023-03-16", 3.0),
            ("2023-05-04", 3.25), ("2023-06-15", 3.5), ("2023-07-27", 3.75), ("2023-09-14", 4.0), ("2024-06-06", 3.75),
            ("2024-09-12", 3.5), ("2024-10-17", 3.25), ("2024-12-12", 3.0), ("2025-01-30", 2.75), ("2025-03-06", 2.5),
            ("2025-04-17", 2.25), ("2025-06-05", 2.0)],
    "GBP": [("2015-01-01", .5), ("2016-08-04", .25), ("2017-11-02", .5), ("2018-08-02", .75), ("2020-03-11", .25),
            ("2020-03-19", .10), ("2021-12-16", .25), ("2022-02-03", .5), ("2022-03-17", .75), ("2022-05-05", 1.0),
            ("2022-06-16", 1.25), ("2022-08-04", 1.75), ("2022-09-22", 2.25), ("2022-11-03", 3.0), ("2022-12-15", 3.5),
            ("2023-02-02", 4.0), ("2023-03-23", 4.25), ("2023-05-11", 4.5), ("2023-06-22", 5.0), ("2023-08-03", 5.25),
            ("2024-08-01", 5.0), ("2024-11-07", 4.75), ("2025-02-06", 4.5), ("2025-05-08", 4.25), ("2025-08-07", 4.0),
            ("2025-12-18", 3.75)],
    "JPY": [("2015-01-01", .05), ("2016-01-29", -.10), ("2024-03-19", .05), ("2024-07-31", .25), ("2025-01-24", .5),
            ("2025-12-19", .75)],
    "CHF": [("2015-01-01", -.25), ("2015-01-15", -.75), ("2022-06-16", -.25), ("2022-09-22", .5), ("2022-12-15", 1.0),
            ("2023-03-23", 1.5), ("2023-06-22", 1.75), ("2024-03-21", 1.5), ("2024-06-20", 1.25), ("2024-09-26", 1.0),
            ("2024-12-12", .5), ("2025-03-20", .25), ("2025-06-19", 0.0)],
    "AUD": [("2015-01-01", 2.5), ("2015-02-03", 2.25), ("2015-05-05", 2.0), ("2016-05-03", 1.75), ("2016-08-02", 1.5),
            ("2019-06-04", 1.25), ("2019-07-02", 1.0), ("2019-10-01", .75), ("2020-03-19", .25), ("2020-11-03", .10),
            ("2022-05-03", .35), ("2022-06-07", .85), ("2022-07-05", 1.35), ("2022-08-02", 1.85), ("2022-09-06", 2.35),
            ("2022-10-04", 2.6), ("2022-11-01", 2.85), ("2022-12-06", 3.1), ("2023-02-07", 3.35), ("2023-03-07", 3.6),
            ("2023-05-02", 3.85), ("2023-06-06", 4.1), ("2023-11-07", 4.35), ("2025-02-18", 4.1), ("2025-05-20", 3.85),
            ("2025-08-12", 3.6)],
    "NZD": [("2015-01-01", 3.5), ("2015-06-11", 3.25), ("2015-07-23", 3.0), ("2015-09-10", 2.75), ("2015-12-10", 2.5),
            ("2016-03-10", 2.25), ("2016-08-11", 2.0), ("2016-11-10", 1.75), ("2019-05-08", 1.5), ("2019-08-08", 1.0),
            ("2020-03-16", .25), ("2021-10-06", .5), ("2021-11-24", .75), ("2022-02-23", 1.0), ("2022-04-13", 1.5),
            ("2022-05-25", 2.0), ("2022-07-13", 2.5), ("2022-08-17", 3.0), ("2022-10-05", 3.5), ("2022-11-23", 4.25),
            ("2023-02-22", 4.75), ("2023-04-05", 5.25), ("2023-05-24", 5.5), ("2024-08-14", 5.25), ("2024-10-09", 4.75),
            ("2024-11-27", 4.25), ("2025-02-19", 3.75), ("2025-04-09", 3.5), ("2025-05-28", 3.25), ("2025-08-20", 3.0),
            ("2025-10-08", 2.5), ("2025-11-26", 2.25)],
    "CAD": [("2015-01-01", 1.0), ("2015-01-21", .75), ("2015-07-15", .5), ("2017-07-12", .75), ("2017-09-06", 1.0),
            ("2018-01-17", 1.25), ("2018-07-11", 1.5), ("2018-10-24", 1.75), ("2020-03-04", 1.25), ("2020-03-16", .25),
            ("2022-03-02", .5), ("2022-04-13", 1.0), ("2022-06-01", 1.5), ("2022-07-13", 2.5), ("2022-09-07", 3.25),
            ("2022-10-26", 3.75), ("2022-12-07", 4.25), ("2023-01-25", 4.5), ("2023-06-07", 4.75), ("2023-07-12", 5.0),
            ("2024-06-05", 4.75), ("2024-07-24", 4.5), ("2024-09-11", 4.25), ("2024-10-23", 3.75), ("2024-12-11", 3.25),
            ("2025-03-12", 2.75), ("2025-09-17", 2.5)],
}
_POLICY_TS = {c: ([pd.Timestamp(d, tz="UTC").timestamp() for d, _ in rows], [r for _, r in rows]) for c, rows in POLICY.items()}
T_2026 = pd.Timestamp("2026-01-01", tz="UTC").timestamp()


def policy_rate(ccy, ts):
    times, rates = _POLICY_TS[ccy]
    return rates[max(bisect.bisect_right(times, ts) - 1, 0)] / 100.0


# ---------------------------------------------------------------- OANDA financing / spreads
def _creds():
    return os.environ.get("OANDA_API_KEY"), os.environ.get("OANDA_ACCOUNT_ID")


def parse_financing(payload):
    out = {}
    for ins in payload.get("instruments", []):
        f = ins.get("financing")
        if f and "longRate" in f and "shortRate" in f:
            out[ins["name"]] = dict(long=float(f["longRate"]), short=float(f["shortRate"]))
    return out


def fetch_financing(pairs):
    key, acct = _creds()
    if not key or not acct:
        print("  (OANDA_ACCOUNT_ID missing - cannot read OANDA financing rates; falling back to the table + default markup)")
        return {}
    r = requests.get(f"{base.OANDA_BASE_URL}/v3/accounts/{acct}/instruments", headers={"Authorization": "Bearer " + key},
                     params={"instruments": ",".join(pairs)}, timeout=30)
    r.raise_for_status()
    payload = r.json()
    out = parse_financing(payload)
    if not out:
        ins = payload.get("instruments", [])
        print("  WARNING: no 'financing' field in the instruments response; keys seen:", sorted(ins[0].keys()) if ins else "none")
    return out


def fetch_spreads(pairs):
    key, acct = _creds()
    if not key or not acct:
        return {}
    r = requests.get(f"{base.OANDA_BASE_URL}/v3/accounts/{acct}/pricing", headers={"Authorization": "Bearer " + key},
                     params={"instruments": ",".join(pairs)}, timeout=30)
    r.raise_for_status()
    out = {}
    for pr in r.json().get("prices", []):
        if pr.get("bids") and pr.get("asks"):
            out[pr["instrument"]] = (float(pr["asks"][0]["price"]) - float(pr["bids"][0]["price"])) / base.pip_size(pr["instrument"])
    return out


def implied(api):
    """markup and current differential per pair from OANDA's long/short rates."""
    return ({p: -(v["long"] + v["short"]) / 2 for p, v in api.items()},
            {p: (v["long"] - v["short"]) / 2 for p, v in api.items()})


def make_fin(api, use_carry=True, use_markup=True, flat=None):
    markup, api_diff = implied(api)

    def diff(p, ts):
        if ts >= T_2026 and p in api_diff:
            return api_diff[p]
        b, q = p.split("_")
        return policy_rate(b, ts) - policy_rate(q, ts)

    def fin(p, direction, ts):
        if flat is not None:
            return -flat
        d = diff(p, ts) if use_carry else 0.0
        m = markup.get(p, DEFAULT_MARKUP) if use_markup else 0.0
        return direction * d - m
    return fin


# ---------------------------------------------------------------- report
def yearly(res):
    eq = base.daily_equity(res)
    ye = eq.resample("YE" if base.month_end_rule() == "ME" else "A").last()
    out, prev = {}, base.START_EQ
    for d, v in ye.items():
        out[d.year] = (v / prev - 1) * 100
        prev = v
    return out


def run_swap_report(data, api, spreads):
    pairs = base.LIVE_PAIRS
    print("=" * 128)
    print("0. INPUTS - OANDA's current financing rates vs the policy-rate table")
    print("=" * 128)
    markup, api_diff = implied(api)
    end_ts = pd.Timestamp("2025-12-31", tz="UTC").timestamp()
    print("  pair      OANDA long  OANDA short | implied markup  implied diff (OANDA) | table diff @2025-12-31   gap")
    for p in pairs:
        b, q = p.split("_")
        tbl = policy_rate(b, end_ts) - policy_rate(q, end_ts)
        if p in api:
            print(f"  {p}   {api[p]['long']*100:+8.2f}%   {api[p]['short']*100:+8.2f}%  |   {markup[p]*100:6.2f}%      {api_diff[p]*100:+8.2f}%       |"
                  f"   {tbl*100:+8.2f}%        {abs(tbl-api_diff[p])*100:5.2f}pp")
        else:
            print(f"  {p}   (no OANDA financing data)                                      | table diff {tbl*100:+8.2f}%")
    if markup:
        print(f"  average markup per year, per direction: {np.mean(list(markup.values()))*100:.2f}%  "
              f"(assumed constant over 2015-2026 - OANDA's historical markup is unknown)")
    if spreads:
        print("  live spread snapshot (pips) vs the model's assumption:  " + "  ".join(
            f"{p} {spreads[p]:.1f}/{base.SPREAD_PIPS[p]:.1f}" for p in pairs if p in spreads) +
              "   (snapshot at run time; off-hours spreads run wider than typical)")

    scen = [("A. no swap (== previous 'realistic' run)", None),
            ("B. SWAP MODEL: carry + OANDA markup", make_fin(api)),
            ("C. markup only (no carry)", make_fin(api, use_carry=False)),
            ("D. carry only (no markup)", make_fin(api, use_markup=False)),
            ("E. flat 1% p.a. drag on notional", make_fin(api, flat=0.01)),
            ("F. flat 2% p.a. drag on notional", make_fin(api, flat=0.02)),
            ("G. flat 3% p.a. drag on notional", make_fin(api, flat=0.03))]
    res = {}
    for label, f in scen:
        res[label] = base.simulate(data, pairs, cost_mult=1.0, gap_fills=True, curcap=True, fin=f)

    print("\n" + "=" * 128)
    print("1. RESULTS - 9 pairs, shared equity, CURCAP, 1% risk, realistic spreads+slippage, gap fills")
    print("=" * 128)
    for label, _ in scen:
        base.summarize(label, res[label])
    print("  regression check: scenario A must equal the previous run (n=913, PF(R)=1.146, PF($)=1.119, net +114.4%, maxDD -25.2%)")

    a, b = res[scen[0][0]], res[scen[1][0]]
    tb = b["trades"]
    print("\n" + "=" * 128)
    print("2. WHAT THE SWAP MODEL DID (scenario B)")
    print("=" * 128)
    d = tb["days"]
    print(f"  holding time: median {d.median():.1f} days, mean {d.mean():.1f}, 90th pct {d.quantile(.9):.1f}, longest {d.max():.0f}")
    print(f"  financing per trade: mean {tb['fin_R'].mean():+.3f}R  median {tb['fin_R'].median():+.3f}R  "
          f"(+ = cost). Trades that paid net financing: {(tb['fin_R']>0).mean()*100:.0f}%; that earned: {(tb['fin_R']<0).mean()*100:.0f}%")
    print(f"  edge per trade: no swap {a['trades']['R_net'].mean():+.3f}R  ->  with swap {tb['R_net'].mean():+.3f}R   "
          f"(carry alone {res[scen[3][0]]['trades']['R_net'].mean()-a['trades']['R_net'].mean():+.3f}R, "
          f"markup alone {res[scen[2][0]]['trades']['R_net'].mean()-a['trades']['R_net'].mean():+.3f}R)")
    print("  per pair (scenario B):  pair      n   avg days   avg swap R   PF(R) no swap -> with swap")
    ta = a["trades"]
    for p in pairs:
        sb, sa = tb[tb.pair == p], ta[ta.pair == p]
        print(f"                          {p}  {len(sb):>3}   {sb['days'].mean():6.1f}    {sb['fin_R'].mean():+7.3f}     "
              f"{base.pf_of(sa['R_net']):.2f} -> {base.pf_of(sb['R_net']):.2f}")
    print("  by trade direction:     longs avg swap {:+.3f}R   shorts avg swap {:+.3f}R".format(
        tb[tb.dir == 1]['fin_R'].mean(), tb[tb.dir == -1]['fin_R'].mean()))

    print("\n" + "=" * 128)
    print("3. YEAR BY YEAR: no swap vs swap model")
    print("=" * 128)
    ya, yb = yearly(a), yearly(b)
    print("  year   no swap    with swap   difference")
    for y in sorted(ya):
        print(f"  {y}   {ya[y]:+7.1f}%   {yb.get(y, float('nan')):+7.1f}%   {yb.get(y, float('nan'))-ya[y]:+6.1f}pp")

    print("\n" + "=" * 128)
    print("4. DRAWDOWN / RISK with swap (scenario B)")
    print("=" * 128)
    eq = base.daily_equity(b)
    eps = base.drawdown_episodes(eq)
    print(f"  time below a prior equity high: {(eq < eq.cummax()).mean()*100:.0f}% of days; deepest episodes:")
    for e in sorted(eps, key=lambda x: x["depth"])[:4]:
        rec = e["recover"].date() if e["recover"] is not None else "NOT YET"
        print(f"     peak {e['peak'].date()}  trough {e['trough'].date()}  recovered {str(rec):<10}  {e['depth']*100:6.1f}%  {e['days']:>4} days ({e['days']/30.4:.0f} mo)")
    yrs = (eq.index[-1] - eq.index[0]).days / 365.25
    base.bootstrap(tb["R_net"].to_numpy(), len(tb) / yrs)
    print("  streaks: longest loss run {}; runs >=7: {}; >=10: {}".format(
        max(base.loss_streaks(tb.sort_values('exit_ts')['pnl'].to_numpy())),
        sum(1 for r in base.loss_streaks(tb.sort_values('exit_ts')['pnl'].to_numpy()) if r >= 7),
        sum(1 for r in base.loss_streaks(tb.sort_values('exit_ts')['pnl'].to_numpy()) if r >= 10)))
    print("=" * 128 + "\n")


def main():
    pairs = base.LIVE_PAIRS
    raw = {}
    for p in pairs:
        raw[p] = base.fetch(p)
        print(f"  {p}: {len(raw[p])} H4 bars")
        sys.stdout.flush()
        time.sleep(0.3)
    data = {p: base.PairData(p, df) for p, df in raw.items()}
    api = fetch_financing(pairs)
    try:
        spreads = fetch_spreads(pairs)
    except Exception as e:  # snapshot only - never fatal
        print("  spread snapshot failed:", e)
        spreads = {}
    run_swap_report(data, api, spreads)


if __name__ == "__main__":
    main()
