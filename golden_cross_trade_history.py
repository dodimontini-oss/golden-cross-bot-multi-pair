"""
One-off trade history report for the live multi-pair Golden Cross forex bot.
Not part of the bot's own logic - just reads OANDA's own trade records
(open and closed) and prints a summary. Read-only: no orders are placed.

Environment variables required:
    OANDA_API_KEY
    OANDA_ACCOUNT_ID
"""

import os

import requests

OANDA_API_KEY = os.environ["OANDA_API_KEY"]
OANDA_ACCOUNT_ID = os.environ["OANDA_ACCOUNT_ID"]
OANDA_BASE_URL = "https://api-fxpractice.oanda.com"
HEADERS = {"Authorization": f"Bearer {OANDA_API_KEY}"}


def get_account_summary() -> dict:
    url = f"{OANDA_BASE_URL}/v3/accounts/{OANDA_ACCOUNT_ID}/summary"
    resp = requests.get(url, headers=HEADERS, timeout=10)
    resp.raise_for_status()
    return resp.json()["account"]


def get_all_trades() -> list:
    """OANDA returns newest-first, most recent 500 by default - plenty for
    this bot's trade volume."""
    url = f"{OANDA_BASE_URL}/v3/accounts/{OANDA_ACCOUNT_ID}/trades"
    params = {"state": "ALL", "count": 500}
    resp = requests.get(url, headers=HEADERS, params=params, timeout=15)
    resp.raise_for_status()
    return resp.json()["trades"]


def direction(trade: dict) -> str:
    return "LONG" if float(trade["initialUnits"]) > 0 else "SHORT"


def run():
    account = get_account_summary()
    balance = float(account["balance"])
    currency = account["currency"]
    pl = float(account.get("pl", 0.0))
    unrealized = float(account.get("unrealizedPL", 0.0))

    trades = get_all_trades()
    open_trades = [t for t in trades if t["state"] == "OPEN"]
    closed_trades = [t for t in trades if t["state"] == "CLOSED"]
    # chronological, oldest first, for a readable history
    closed_trades.sort(key=lambda t: t["openTime"])
    open_trades.sort(key=lambda t: t["openTime"])

    print(f"\nOANDA account summary: balance={balance:.2f} {currency}, "
          f"lifetime realized P/L={pl:+.2f}, current unrealized P/L={unrealized:+.2f}\n")

    print(f"=== OPEN POSITIONS ({len(open_trades)}) ===")
    if not open_trades:
        print("  (none)")
    for t in open_trades:
        instrument = t["instrument"]
        entry = float(t["price"])
        units = abs(float(t["currentUnits"]))
        upl = float(t.get("unrealizedPL", 0.0))
        stop = t.get("stopLossOrder", {}).get("price")
        target = t.get("takeProfitOrder", {}).get("price")
        print(f"  [{instrument}] {direction(t):<5} {units:.0f} units @ {entry} "
              f"(opened {t['openTime'][:19]}) stop={stop} target={target} "
              f"unrealized P/L={upl:+.2f} {currency}")

    print(f"\n=== CLOSED TRADES ({len(closed_trades)}) ===")
    if not closed_trades:
        print("  (none)")
    wins = losses = 0
    total_realized = 0.0
    for t in closed_trades:
        instrument = t["instrument"]
        entry = float(t["price"])
        units = abs(float(t["initialUnits"]))
        close_price = t.get("averageClosePrice")
        rpl = float(t.get("realizedPL", 0.0))
        total_realized += rpl
        outcome = "WIN " if rpl > 0 else ("LOSS" if rpl < 0 else "FLAT")
        if rpl > 0:
            wins += 1
        elif rpl < 0:
            losses += 1
        print(f"  [{instrument}] {direction(t):<5} {units:.0f} units @ {entry} -> {close_price} "
              f"| opened {t['openTime'][:19]} closed {t.get('closeTime', '?')[:19]} "
              f"| {outcome} {rpl:+.2f} {currency}")

    print(f"\n=== SUMMARY ===")
    print(f"Closed trades: {len(closed_trades)}  Wins: {wins}  Losses: {losses}"
          + (f"  Win rate: {wins / len(closed_trades) * 100:.1f}%" if closed_trades else ""))
    print(f"Total realized P/L (closed trades): {total_realized:+.2f} {currency}")
    print(f"Open positions: {len(open_trades)}  Current unrealized P/L: {unrealized:+.2f} {currency}")
    print()


if __name__ == "__main__":
    run()
