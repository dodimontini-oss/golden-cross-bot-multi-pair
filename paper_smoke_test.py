"""Explicit OANDA practice-account order check for manual workflow runs."""
import hashlib
import os
from decimal import Decimal, ROUND_DOWN

import requests


def oanda_cancel_test(base, account_id, headers, instrument, *, http=requests):
    """Submit an off-market one-unit practice limit order, then cancel it."""
    if base.rstrip("/") != "https://api-fxpractice.oanda.com":
        raise RuntimeError("Smoke tests are restricted to OANDA practice trading")

    run_key = "|".join((
        os.getenv("GITHUB_REPOSITORY", "local"),
        os.getenv("GITHUB_RUN_ID", "manual"),
        os.getenv("GITHUB_RUN_ATTEMPT", "1"),
        instrument,
    ))
    client_id = "smoke-" + hashlib.sha256(run_key.encode()).hexdigest()[:20]
    pricing_url = f"{base}/v3/accounts/{account_id}/pricing"
    response = http.get(pricing_url, headers=headers, params={"instruments": instrument}, timeout=15)
    response.raise_for_status()
    prices = response.json().get("prices", [])
    if not prices or not prices[0].get("bids"):
        raise RuntimeError(f"No executable practice price returned for {instrument}")
    bid = Decimal(prices[0]["bids"][0]["price"])
    decimals = 3 if instrument.endswith("_JPY") else 5
    quantum = Decimal(1).scaleb(-decimals)
    limit_price = (bid * Decimal("0.90")).quantize(quantum, rounding=ROUND_DOWN)

    orders_url = f"{base}/v3/accounts/{account_id}/orders"
    response = http.post(orders_url, headers=headers, json={"order": {
        "type": "LIMIT",
        "instrument": instrument,
        "units": "1",
        "price": format(limit_price, "f"),
        "timeInForce": "GTC",
        "positionFill": "DEFAULT",
        "clientExtensions": {"id": client_id, "tag": "smoke-test"},
    }}, timeout=15)
    response.raise_for_status()
    created = response.json().get("orderCreateTransaction", {})
    order_id = created.get("id")
    if not order_id:
        raise RuntimeError(f"OANDA did not confirm order creation: {response.text[:300]}")

    cancel_url = f"{orders_url}/{order_id}/cancel"
    response = http.put(cancel_url, headers=headers, timeout=15)
    response.raise_for_status()
    canceled = response.json().get("orderCancelTransaction", {})
    if not canceled:
        raise RuntimeError(f"OANDA did not confirm cancellation: {response.text[:300]}")

    print(
        "SMOKE_TEST_OK "
        f"broker=oanda instrument={instrument} order_id={order_id} "
        f"limit_price={limit_price} final_status=canceled"
    )
    return canceled
