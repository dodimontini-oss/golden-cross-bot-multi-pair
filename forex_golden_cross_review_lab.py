"""
Pre-real-money review of the LIVE forex golden-cross bot (bot.py).

Replays the bot's rules on the full 2015-2026 H4 history and answers the
questions a real-money decision hinges on, with numbers instead of a single
profit factor:

  A. Faithful replica summary vs the memory-recorded WINNERS backtest
  B. Cost sensitivity - frictionless / realistic per-pair spreads / 2x that
  C. Drawdown episodes, longest time underwater, time spent underwater
  D. Losing-streak distribution
  E. Per-year returns, rolling-window returns, bootstrap of a "bad year"
  F. Gap risk - gap-through-stop fills actually experienced in the sample,
     whether the 2015 SNB shock was ever tradeable, CHF shock stress table
  G. Pair-selection bias - select pairs on the EARLY half exactly the way the
     original 23-pair screen did (PF > 1.15), then look at how they did LATE

REPLICATED LIVE RULES (bot.py): SMA50/SMA200 crossover on H4; GAPCONFIRM (the
signal is only actionable while the crossover is <= 5 bars old AND the fast/
slow gap is >= 0.10 x ATR at the latest bar); stop = 2 x ATR of the crossover
candle, target = 2x the stop distance (2:1); 1% of equity risked per trade;
CURCAP - a new trade is blocked if open risk on either of its two currencies
plus the new trade's risk would exceed 2.5% of equity; one position per pair;
one entry per crossover signal; pairs processed in the bot's list order;
entry at the open of the bar after the confirming bar (no lookahead).

MODELLING NOTES:
  - P&L is tracked in R-multiples scaled by each trade's own risk dollars, so
    no FX conversion is needed: P&L$ = risk$ x (move / stop_distance). This is
    exact, because the bot sizes units so a stop-out costs exactly risk$.
  - Equity is marked to market every bar (open trades included), which is what
    the bot's live sizing (NAV) sees.
  - Same-bar dual hit: stop assumed first. Entry-bar exits are checked.
  - "gap fills": a bar that OPENS beyond the stop fills at the open (worse
    than the stop), beyond the target fills at the open (better). The old
    lab-style convention fills at the stop/target price exactly; both are shown.
  - Costs: per-pair spread + 0.5 pip/side slippage, deducted in R at entry.
  - Bot cadence (hours-late entries from GitHub cron throttling) is NOT
    modelled - H4 bars, entry at the next open.

Run (needs OANDA_API_KEY):  python forex_golden_cross_review_lab.py
"""

import os
import sys
import time
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import requests

OANDA_BASE_URL = "https://api-fxpractice.oanda.com"  # candles/data only - practice host is fine
GRANULARITY = "H4"
FROM_TIME = "2015-01-01T00:00:00Z"

LIVE_PAIRS = ["GBP_CHF", "GBP_CAD", "GBP_USD", "GBP_JPY", "GBP_AUD", "USD_CHF", "EUR_USD", "EUR_AUD", "AUD_NZD"]
ALL_PAIRS = [
    "EUR_USD", "GBP_USD", "USD_JPY", "USD_CHF", "USD_CAD", "AUD_USD", "NZD_USD",
    "EUR_GBP", "EUR_JPY", "EUR_CHF", "EUR_AUD", "EUR_CAD",
    "GBP_JPY", "GBP_CHF", "GBP_AUD", "GBP_CAD",
    "AUD_JPY", "AUD_NZD", "AUD_CAD", "AUD_CHF",
    "CAD_JPY", "CHF_JPY", "NZD_JPY",
]

FAST, SLOW, ATR_LEN = 50, 200, 14
STOP_MULT, RR = 2.0, 2.0
RISK, CAP = 0.01, 0.025
CONF_GAP_ATR, CONF_MAX_BARS = 0.10, 5
START_EQ = 10_000.0

SPREAD_PIPS = {"GBP_CHF": 2.5, "GBP_CAD": 2.6, "GBP_USD": 1.3, "GBP_JPY": 2.2, "GBP_AUD": 2.8,
               "USD_CHF": 1.6, "EUR_USD": 0.9, "EUR_AUD": 1.9, "AUD_NZD": 2.4}
SLIP_PIPS_PER_SIDE = 0.5
EPOCH = pd.Timestamp("1970-01-01", tz="UTC")

MEMORY_REFERENCE = "memory (2015-2026, other engine): WINNERS PF 1.320, +568.3% net, max DD 17.56%, ~900-1000 trades"


def pip_size(pair):
    return 0.01 if pair.endswith("_JPY") else 0.0001


# ---------------------------------------------------------------- data
def fetch(instrument):
    headers = {"Authorization": "Bearer " + os.environ["OANDA_API_KEY"]}
    candles, cur = [], FROM_TIME
    while True:
        r = requests.get(f"{OANDA_BASE_URL}/v3/instruments/{instrument}/candles", headers=headers,
                         params={"granularity": GRANULARITY, "price": "M", "from": cur, "count": 5000}, timeout=30)
        r.raise_for_status()
        batch = r.json()["candles"]
        if not batch:
            break
        candles.extend(c for c in batch if c["complete"])
        if len(batch) < 5000:
            break
        cur = batch[-1]["time"]
        time.sleep(0.2)
    rows = [{"time": pd.Timestamp(c["time"]), "open": float(c["mid"]["o"]), "high": float(c["mid"]["h"]),
             "low": float(c["mid"]["l"]), "close": float(c["mid"]["c"])} for c in candles]
    return pd.DataFrame(rows).drop_duplicates("time").sort_values("time").reset_index(drop=True)


def wilder_atr(h, l, c, n):
    prev_c = np.r_[np.nan, c[:-1]]
    tr = np.fmax.reduce([h - l, np.abs(h - prev_c), np.abs(l - prev_c)])
    atr = np.full(len(c), np.nan)
    if len(c) < n:
        return atr
    atr[n - 1] = tr[:n].mean()
    for i in range(n, len(c)):
        atr[i] = (atr[i - 1] * (n - 1) + tr[i]) / n
    return atr


class PairData:
    """Arrays + the live bot's per-bar signal eligibility for one pair."""

    def __init__(self, name, df):
        self.name = name
        self.sec = ((df["time"] - EPOCH).dt.total_seconds()).astype("int64").to_numpy()
        self.o, self.h = df["open"].to_numpy(float), df["high"].to_numpy(float)
        self.l, self.c = df["low"].to_numpy(float), df["close"].to_numpy(float)
        self.fast = df["close"].rolling(FAST).mean().to_numpy()
        self.slow = df["close"].rolling(SLOW).mean().to_numpy()
        self.atr = wilder_atr(self.h, self.l, self.c, ATR_LEN)
        n = len(self.c)
        cross = np.zeros(n, dtype=np.int8)
        for j in range(1, n):
            if np.isnan(self.slow[j]) or np.isnan(self.slow[j - 1]):
                continue
            if self.fast[j - 1] <= self.slow[j - 1] and self.fast[j] > self.slow[j]:
                cross[j] = 1
            elif self.fast[j - 1] >= self.slow[j - 1] and self.fast[j] < self.slow[j]:
                cross[j] = -1
        last_i = np.full(n, -1, dtype=np.int64)
        cur = -1
        for j in range(n):
            if cross[j] != 0:
                cur = j
            last_i[j] = cur
        elig = np.zeros(n, dtype=bool)
        for k in range(n):
            i = last_i[k]
            if i < 0 or k - i > CONF_MAX_BARS:
                continue
            if not (self.atr[k] > 0) or not (self.atr[i] > 0):
                continue
            if abs(self.fast[k] - self.slow[k]) / self.atr[k] >= CONF_GAP_ATR:
                elig[k] = True
        self.cross, self.last_i, self.elig = cross, last_i, elig
        self.sig_dir = np.where(last_i >= 0, cross[np.maximum(last_i, 0)], 0)


# ---------------------------------------------------------------- simulator
def simulate(data, order, cost_mult=0.0, gap_fills=True, curcap=True, fin=None):
    """fin: optional callable fin(pair, direction, ts_seconds) -> ANNUAL financing rate as a decimal on the
    position's notional (negative = you pay, positive = you earn). Accrued by calendar day held (OANDA's triple
    Wednesday rollover covers the weekend, so calendar days is the right average) and converted to R via
    notional/risk = entry/stop_distance. fin=None reproduces the no-swap results exactly."""
    secs = np.unique(np.concatenate([data[p].sec for p in order]))
    idx = {p: {int(s): i for i, s in enumerate(data[p].sec)} for p in order}
    balance = START_EQ
    eq_prev = START_EQ
    ccy_risk = defaultdict(float)
    open_t, consumed, last_close, trades = {}, {}, {}, []
    eq_t, eq_v = [], []

    for s in secs:
        s = int(s)
        # 1) entries at this bar's open, in the bot's pair order
        for p in order:
            d = data[p]
            j = idx[p].get(s)
            if j is None or j < 1:
                continue
            k = j - 1
            if not d.elig[k] or p in open_t:
                continue
            i = int(d.last_i[k])
            if consumed.get(p) == i:
                continue
            risk_d = RISK * eq_prev
            base, quote = p.split("_")
            if curcap and (ccy_risk[base] + risk_d > CAP * eq_prev or ccy_risk[quote] + risk_d > CAP * eq_prev):
                continue
            direction = int(d.sig_dir[k])
            entry = d.o[j]
            stop_dist = STOP_MULT * d.atr[i]
            cost_price = (SPREAD_PIPS.get(p, 2.5) + 2 * SLIP_PIPS_PER_SIDE) * pip_size(p) * cost_mult
            open_t[p] = dict(pair=p, dir=direction, entry=entry, stop=entry - direction * stop_dist,
                             target=entry + direction * RR * stop_dist, stop_dist=stop_dist, risk_d=risk_d,
                             cost_R=cost_price / stop_dist, entry_ts=s,
                             fin_rate=(fin(p, direction, s) if fin else 0.0), stop_frac=stop_dist / entry)
            consumed[p] = i
            ccy_risk[base] += risk_d
            ccy_risk[quote] += risk_d

        # 2) exits within this bar (entry bar included)
        for p in list(open_t):
            j = idx[p].get(s)
            if j is None:
                continue
            d, tr = data[p], open_t[p]
            o, h, l = d.o[j], d.h[j], d.l[j]
            exit_px = kind = None
            if tr["dir"] == 1:
                if gap_fills and o <= tr["stop"]:
                    exit_px, kind = o, "gap_stop"
                elif gap_fills and o >= tr["target"]:
                    exit_px, kind = o, "gap_target"
                elif l <= tr["stop"]:
                    exit_px, kind = tr["stop"], "stop"
                elif h >= tr["target"]:
                    exit_px, kind = tr["target"], "target"
            else:
                if gap_fills and o >= tr["stop"]:
                    exit_px, kind = o, "gap_stop"
                elif gap_fills and o <= tr["target"]:
                    exit_px, kind = o, "gap_target"
                elif h >= tr["stop"]:
                    exit_px, kind = tr["stop"], "stop"
                elif l <= tr["target"]:
                    exit_px, kind = tr["target"], "target"
            if exit_px is None:
                continue
            r_gross = tr["dir"] * (exit_px - tr["entry"]) / tr["stop_dist"]
            days_held = (s - tr["entry_ts"]) / 86400.0
            fin_R = -tr["fin_rate"] * days_held / 365.0 / tr["stop_frac"]      # +ve = cost, -ve = credit
            pnl = tr["risk_d"] * (r_gross - tr["cost_R"] - fin_R)
            balance += pnl
            base, quote = p.split("_")
            ccy_risk[base] -= tr["risk_d"]
            ccy_risk[quote] -= tr["risk_d"]
            trades.append(dict(pair=p, dir=tr["dir"], entry_ts=tr["entry_ts"], exit_ts=s, entry=tr["entry"],
                               stop_dist=tr["stop_dist"], stop_pct=tr["stop_dist"] / tr["entry"] * 100,
                               R_gross=r_gross, cost_R=tr["cost_R"], fin_R=fin_R, days=days_held,
                               R_net=r_gross - tr["cost_R"] - fin_R,
                               pnl=pnl, risk_d=tr["risk_d"], kind=kind))
            del open_t[p]

        # 3) mark to market at this bar's close
        for p in order:
            j = idx[p].get(s)
            if j is not None:
                last_close[p] = data[p].c[j]
        eq = balance
        for p, tr in open_t.items():
            fin_acc = -tr["fin_rate"] * ((s - tr["entry_ts"]) / 86400.0) / 365.0 / tr["stop_frac"]
            eq += tr["risk_d"] * (tr["dir"] * (last_close[p] - tr["entry"]) / tr["stop_dist"] - tr["cost_R"] - fin_acc)
        eq_prev = eq
        eq_t.append(s)
        eq_v.append(eq)

    tdf = pd.DataFrame(trades)
    equity = pd.Series(eq_v, index=pd.to_datetime(eq_t, unit="s", utc=True))
    return dict(trades=tdf, equity=equity, balance=balance, open_left=len(open_t))


# ---------------------------------------------------------------- metrics
def pf_of(r):
    r = np.asarray(r, dtype=float)
    gw, gl = r[r > 0].sum(), -r[r <= 0].sum()
    return gw / gl if gl > 0 else float("inf")


def month_end_rule():
    try:
        pd.Series([1], index=pd.to_datetime(["2020-01-31"])).resample("ME").last()
        return "ME"
    except Exception:
        return "M"


def daily_equity(res):
    tdf = res["trades"]
    eq = res["equity"].resample("D").last().ffill()
    if len(tdf):
        t0 = pd.to_datetime(tdf["entry_ts"].min(), unit="s", utc=True).normalize()
        eq = eq[eq.index >= t0]
    return eq


def drawdown_episodes(eq):
    eps, peak_v, peak_d, tr_v, tr_d, in_dd = [], eq.iloc[0], eq.index[0], eq.iloc[0], eq.index[0], False
    for d, v in eq.items():
        if v >= peak_v:
            if in_dd:
                eps.append(dict(peak=peak_d, trough=tr_d, recover=d, depth=tr_v / peak_v - 1))
                in_dd = False
            peak_v, peak_d, tr_v, tr_d = v, d, v, d
        else:
            in_dd = True
            if v < tr_v:
                tr_v, tr_d = v, d
    if in_dd:
        eps.append(dict(peak=peak_d, trough=tr_d, recover=None, depth=tr_v / peak_v - 1))
    for e in eps:
        end = e["recover"] if e["recover"] is not None else eq.index[-1]
        e["days"] = (end - e["peak"]).days
    return eps


def loss_streaks(pnl_in_exit_order):
    runs, cur = [], 0
    for x in pnl_in_exit_order:
        if x < 0:
            cur += 1
        else:
            if cur:
                runs.append(cur)
            cur = 0
    if cur:
        runs.append(cur)
    return runs


def summarize(label, res):
    t = res["trades"]
    eq = daily_equity(res)
    years = max((eq.index[-1] - eq.index[0]).days / 365.25, 1e-9)
    wins = t["R_net"] > 0
    dd = (eq / eq.cummax() - 1).min()
    print(f"  {label:<34} n={len(t):>4} win%={wins.mean()*100:5.1f} avgW={t.loc[wins,'R_net'].mean():5.2f}R "
          f"avgL={t.loc[~wins,'R_net'].mean():6.2f}R exp={t['R_net'].mean():+.3f}R PF(R)={pf_of(t['R_net']):.3f} "
          f"PF($)={pf_of(t['pnl']):.3f} net={(eq.iloc[-1]/START_EQ-1)*100:+8.1f}% CAGR={((eq.iloc[-1]/START_EQ)**(1/years)-1)*100:5.1f}% "
          f"maxDD={dd*100:5.1f}%")


def bootstrap(r_net, trades_per_year, n_paths=10000, seed=7):
    rng = np.random.default_rng(seed)
    n = max(int(round(trades_per_year)), 1)
    draws = rng.choice(np.asarray(r_net, dtype=float), size=(n_paths, n), replace=True)
    curves = np.cumprod(1 + RISK * draws, axis=1)
    peaks = np.maximum.accumulate(np.concatenate([np.ones((n_paths, 1)), curves], axis=1), axis=1)[:, 1:]
    maxdd = ((peaks - curves) / peaks).max(axis=1)
    ret = curves[:, -1] - 1
    pct = lambda a, q: np.percentile(a, q) * 100
    print(f"  {n} trades/year, {n_paths} resampled years (iid draws of the realistic-cost trade results, "
          f"1% risk each, concurrency ignored):")
    print(f"    12-month return: 5th {pct(ret,5):+.1f}%  25th {pct(ret,25):+.1f}%  median {pct(ret,50):+.1f}%  "
          f"75th {pct(ret,75):+.1f}%  95th {pct(ret,95):+.1f}%   P(losing year) {np.mean(ret<0)*100:.0f}%")
    print(f"    worst peak-to-trough inside a year: median {pct(maxdd,50):.1f}%  95th pct {pct(maxdd,95):.1f}%   "
          f"P(>=10% dd) {np.mean(maxdd>=.10)*100:.0f}%  P(>=15%) {np.mean(maxdd>=.15)*100:.0f}%  "
          f"P(>=20%) {np.mean(maxdd>=.20)*100:.0f}%")


# ---------------------------------------------------------------- report
def run_report(data_live, data_all):
    order = LIVE_PAIRS
    scen = [("frictionless, fills AT stop (lab-style)", 0.0, False),
            ("frictionless, gap fills", 0.0, True),
            ("REALISTIC spreads+slippage, gap fills", 1.0, True),
            ("2x costs, gap fills", 2.0, True)]
    results = {}
    for label, cm, gf in scen:
        results[label] = simulate(data_live, order, cost_mult=cm, gap_fills=gf, curcap=True)

    print("\n" + "=" * 128)
    print("A/B. LIVE-RULES REPLAY, 9 pairs, shared equity, CURCAP on, 1% risk/trade  (start $10,000)")
    print("=" * 128)
    for label, _, _ in scen:
        summarize(label, results[label])
    print("  reference: " + MEMORY_REFERENCE)
    real = results["REALISTIC spreads+slippage, gap fills"]
    t, eq = real["trades"], daily_equity(real)
    print(f"  realistic scenario: avg cost {t['cost_R'].mean():.3f}R/trade; first trade "
          f"{pd.to_datetime(t['entry_ts'].min(), unit='s', utc=True).date()}; trades still open at the end: {real['open_left']}")
    mid = (t["entry_ts"].min() + t["entry_ts"].max()) / 2
    for nm, sub in (("early half", t[t.entry_ts < mid]), ("late half", t[t.entry_ts >= mid])):
        print(f"    walk-forward {nm}: n={len(sub)} PF(R)={pf_of(sub['R_net']):.3f} win%={(sub['R_net']>0).mean()*100:.1f}")
    print("  per pair (realistic):  " + "  ".join(
        f"{p}:{(t.pair==p).sum()}/{pf_of(t.loc[t.pair==p,'R_net']):.2f}" for p in order) + "   (trades/PF)")
    print("  entries blocked or skipped are not counted; CURCAP means at most ~2 open trades per currency.")

    print("\n" + "=" * 128)
    print("C. DRAWDOWNS (realistic scenario, mark-to-market, daily)")
    print("=" * 128)
    eps = drawdown_episodes(eq)
    under = (eq < eq.cummax()).mean() * 100
    print(f"  time spent below a prior equity high: {under:.0f}% of days")
    print("  deepest 5 episodes:      peak        trough      recovered    depth   days peak->recovery")
    for e in sorted(eps, key=lambda x: x["depth"])[:5]:
        rec = e["recover"].date() if e["recover"] is not None else "NOT YET"
        print(f"                           {e['peak'].date()}  {e['trough'].date()}  {str(rec):<11}  {e['depth']*100:6.1f}%  {e['days']:>5}")
    print("  longest 3 episodes (time without a new equity high):")
    for e in sorted(eps, key=lambda x: -x["days"])[:3]:
        rec = e["recover"].date() if e["recover"] is not None else "NOT YET"
        print(f"                           {e['peak'].date()}  {e['trough'].date()}  {str(rec):<11}  {e['depth']*100:6.1f}%  {e['days']:>5} days ({e['days']/30.4:.0f} months)")

    print("\n" + "=" * 128)
    print("D. LOSING STREAKS (consecutive losing trades in exit order, realistic scenario)")
    print("=" * 128)
    runs = loss_streaks(t.sort_values("exit_ts")["pnl"].to_numpy())
    cnt = Counter(runs)
    p_win = (t["R_net"] > 0).mean()
    print(f"  win rate {p_win*100:.1f}% over {len(t)} trades; longest losing streak: {max(runs)}")
    print("  streak length -> how many times it happened:  " + "  ".join(f"{k}:{cnt[k]}" for k in sorted(cnt)))
    for k in (5, 7, 10):
        print(f"  streaks of >= {k:>2} losses: {sum(1 for r in runs if r >= k)}   (a {k}-loss streak at 1% risk each = about -{k}%)")

    print("\n" + "=" * 128)
    print("E. RETURNS BY PERIOD (realistic scenario)")
    print("=" * 128)
    ye = eq.resample("YE" if month_end_rule() == "ME" else "A").last()
    prev = START_EQ
    tt = t.copy()
    tt["year"] = pd.to_datetime(tt["exit_ts"], unit="s", utc=True).dt.year
    print("  year   return    trades  win%   PF(R)")
    for d, v in ye.items():
        sub = tt[tt.year == d.year]
        print(f"  {d.year}  {(v/prev-1)*100:+7.1f}%  {len(sub):>5}  {(sub['R_net']>0).mean()*100 if len(sub) else 0:5.1f}  "
              f"{pf_of(sub['R_net']) if len(sub) else 0:5.2f}")
        prev = v
    me = eq.resample(month_end_rule()).last()
    m1, m3, m6, m12 = me.pct_change(), me.pct_change(3), me.pct_change(6), me.pct_change(12)
    print(f"  months with a loss: {(m1<0).mean()*100:.0f}%   worst month {m1.min()*100:+.1f}%")
    print(f"  worst rolling 3m {m3.min()*100:+.1f}%  6m {m6.min()*100:+.1f}%  12m {m12.min()*100:+.1f}%;  "
          f"share of 12-month windows that lost money: {(m12.dropna()<0).mean()*100:.0f}%")
    yrs = (eq.index[-1] - eq.index[0]).days / 365.25
    print("  bootstrap of a 'bad year':")
    bootstrap(t["R_net"].to_numpy(), len(t) / yrs)

    print("\n" + "=" * 128)
    print("F. GAP RISK")
    print("=" * 128)
    gap = t[t.kind == "gap_stop"]
    stops = t[t.kind.isin(["stop", "gap_stop"])]
    print(f"  stopped-out trades: {len(stops)}; of those filled through the stop on a gap: {len(gap)} "
          f"({len(gap)/max(len(stops),1)*100:.1f}%)")
    print(f"  stop-loss size in R (gross): median {stops['R_gross'].median():.2f}  5th pct {stops['R_gross'].quantile(.05):.2f}  "
          f"worst {stops['R_gross'].min():.2f}")
    print("  6 worst trades by gross R (a plain stop-out is -1.00R; anything worse than that was a gap fill):")
    for _, r in t.nsmallest(6, "R_gross").iterrows():
        print(f"     {pd.to_datetime(r['exit_ts'], unit='s', utc=True).date()}  {r['pair']}  {r['R_gross']:6.2f}R  "
              f"(= {r['R_gross']*RISK*100:+.1f}% of equity at 1% risk)")
    snb = pd.Timestamp("2015-01-15 09:00", tz="UTC")
    first = pd.to_datetime(t["entry_ts"].min(), unit="s", utc=True)
    open_then = t[(pd.to_datetime(t.entry_ts, unit="s", utc=True) <= snb) & (pd.to_datetime(t.exit_ts, unit="s", utc=True) >= snb)]
    print(f"  SNB shock (2015-01-15): first trade of the whole replay was {first.date()} (SMA200 needs ~33 days of H4 bars); "
          f"trades open on the day: {len(open_then)}  -> the backtest never experienced a franc-style gap.")
    print("  CHF-shock stress (illustrative arithmetic, NOT a forecast): loss to equity if one adverse-side CHF trade "
          "gaps through its stop by X%, at 1% risk")
    print("    pair      median stop%   " + "   ".join(f"{x:>2}% gap" for x in (3, 5, 10, 15, 20)))
    for p in ("GBP_CHF", "USD_CHF"):
        sp = t.loc[t.pair == p, "stop_pct"].median()
        print(f"    {p}   {sp:6.2f}%      " + "   ".join(f"{RISK*(x/sp)*100:6.0f}%" for x in (3, 5, 10, 15, 20)))
    print("    (CURCAP allows two CHF trades open at once; if both were on the same side the losses add.)")

    print("\n" + "=" * 128)
    print("G. PAIR-SELECTION BIAS  (frictionless, gap fills, each pair alone - no CURCAP)")
    print("=" * 128)
    allsec = np.concatenate([d.sec for d in data_all.values()])
    mid_all = (allsec.min() + allsec.max()) / 2
    rows = []
    for p in ALL_PAIRS:
        if p not in data_all:
            continue
        tr = simulate({p: data_all[p]}, [p], cost_mult=0.0, gap_fills=True, curcap=False)["trades"]
        e, l = tr[tr.entry_ts < mid_all], tr[tr.entry_ts >= mid_all]
        rows.append(dict(pair=p, n_e=len(e), pf_e=pf_of(e["R_gross"]) if len(e) else np.nan,
                         n_l=len(l), pf_l=pf_of(l["R_gross"]) if len(l) else np.nan, live=p in LIVE_PAIRS))
    g = pd.DataFrame(rows).sort_values("pf_e", ascending=False)
    print("  pair       early n/PF     late n/PF     (* = one of the 9 live pairs)   split at " +
          str(pd.to_datetime(mid_all, unit="s", utc=True).date()))
    for _, r in g.iterrows():
        print(f"  {r['pair']:<8}{'*' if r['live'] else ' '}   {int(r['n_e']):>3} / {r['pf_e']:5.2f}    {int(r['n_l']):>3} / {r['pf_l']:5.2f}")
    cap = lambda s: np.minimum(s, 3.0)
    sel = g[(g.pf_e > 1.15) & (g.n_e >= 20)]
    rej = g[~g.index.isin(sel.index)]
    print(f"  rule 'PF > 1.15 in the EARLY half (>= 20 trades)' (like the original screen) picks {len(sel)} of {len(g)} pairs.")
    if len(sel):
        print(f"    picked pairs, EARLY PF: mean {cap(sel.pf_e).mean():.2f}   -> same pairs, LATE PF: mean {cap(sel.pf_l).mean():.2f}, "
              f"median {sel.pf_l.median():.2f}, {int((sel.pf_l>1).sum())}/{len(sel)} still above 1.0")
    if len(rej):
        print(f"    not-picked pairs, LATE PF: mean {cap(rej.pf_l).mean():.2f}, median {rej.pf_l.median():.2f}")
    lv = g[g.live]
    print(f"    the 9 live pairs: EARLY PF mean {cap(lv.pf_e).mean():.2f}, LATE PF mean {cap(lv.pf_l).mean():.2f}, median {lv.pf_l.median():.2f}")
    print("    (the live 9 were chosen on the FULL sample, so their late-half number is partly in-sample - the "
          "'picked on early half' row is the honest out-of-sample read.)")
    print("=" * 128 + "\n")


def main():
    pairs = ALL_PAIRS
    raw = {}
    for p in pairs:
        raw[p] = fetch(p)
        print(f"  {p}: {len(raw[p])} H4 bars, {raw[p]['time'].iloc[0].date()} -> {raw[p]['time'].iloc[-1].date()}")
        sys.stdout.flush()
        time.sleep(0.3)
    data_all = {p: PairData(p, df) for p, df in raw.items()}
    data_live = {p: data_all[p] for p in LIVE_PAIRS}
    run_report(data_live, data_all)


if __name__ == "__main__":
    main()
