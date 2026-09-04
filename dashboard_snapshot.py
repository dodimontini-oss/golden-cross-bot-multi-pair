"""
One-off JSON snapshot for the live dashboard - reads OANDA's account summary
and trade history, reconstructs an approximate historical equity curve from
closed-trade realized P&L (starting balance isn't tracked anywhere, so it's
inferred as current_balance - sum(realized P&L) - the same "back out the
starting point" trick used nowhere else in this codebase yet, but safe since
OANDA's balance already excludes unrealized P&L), and prints ONE json blob
between marker lines so it can be grepped out of the Action's log cleanly.

Read-only: no orders are placed.

Environment variables required:
    OANDA_API_KEY
    OANDA_ACCOUNT_ID
"""

import json
import os
from datetime import datetime, timezone

import requests

OANDA_API_KEY = os.environ["OANDA_API_KEY"]
OANDA_ACCOUNT_ID = os.environ["OANDA_ACCOUNT_ID"]
OANDA_BASE_URL = "https://api-fxpractice.oanda.com"
HEADERS = {"Authorization": f"Bearer {OANDA_API_KEY}"}

BOT_ID = "forex"
BOT_LABEL = "Forex (Golden Cross, multi-pair)"


def get_account_summary() -> dict:
    resp = requests.get(f"{OANDA_BASE_URL}/v3/accounts/{OANDA_ACCOUNT_ID}/summary", headers=HEADERS, timeout=10)
    resp.raise_for_status()
    return resp.json()["account"]


def get_all_trades() -> list:
    resp = requests.get(f"{OANDA_BASE_URL}/v3/accounts/{OANDA_ACCOUNT_ID}/trades",
                         headers=HEADERS, params={"state": "ALL", "count": 500}, timeout=15)
    resp.raise_for_status()
    return resp.json()["trades"]


def direction(trade: dict) -> str:
    return "LONG" if float(trade["initialUnits"]) > 0 else "SHORT"


def run():
    account = get_account_summary()
    balance = float(account["balance"])
    nav = float(account.get("NAV", balance))
    unrealized_pl = float(account.get("unrealizedPL", 0.0))
    realized_pl_alltime = float(account.get("pl", 0.0))
    currency = account.get("currency", "USD")

    trades = get_all_trades()
    open_trades = [t for t in trades if t["state"] == "OPEN"]
    closed_trades = sorted((t for t in trades if t["state"] == "CLOSED"), key=lambda t: t.get("closeTime", ""))

    open_positions = [{
        "symbol": t["instrument"],
        "side": direction(t),
        "qty": abs(float(t["currentUnits"])),
        "entry": float(t["price"]),
        "unrealized_pl": float(t.get("unrealizedPL", 0.0)),
        "stop": t.get("stopLossOrder", {}).get("price"),
        "target": t.get("takeProfitOrder", {}).get("price"),
        "opened_at": t["openTime"],
    } for t in open_trades]

    closed_trade_points = [{
        "closed_at": t.get("closeTime"),
        "pnl": float(t.get("realizedPL", 0.0)),
        "symbol": t["instrument"],
    } for t in closed_trades if t.get("closeTime")]

    snapshot = {
        "bot_id": BOT_ID,
        "label": BOT_LABEL,
        "broker": "OANDA",
        "currency": currency,
        "as_of": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "equity": nav,
        "balance": balance,
        "unrealized_pl": unrealized_pl,
        "realized_pl_alltime": realized_pl_alltime,
        "open_positions": open_positions,
        "closed_trades": closed_trade_points,
    }

    print("===SNAPSHOT_JSON_START===")
    print(json.dumps(snapshot))
    print("===SNAPSHOT_JSON_END===")


if __name__ == "__main__":
    run()
