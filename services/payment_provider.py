"""Paiements AutoDeal via checkout hébergé.

AutoDeal ne collecte jamais PAN/CVV. Un checkout n'est considéré activable que
si le prestataire ET un webhook de confirmation sont configurés : accepter un
paiement sans chemin fiable de confirmation risquerait de débiter l'utilisateur
sans activer son abonnement.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable

import requests

PLAN_PRICES_TND = {
    "pro": {"monthly": 29, "yearly": 299},
    "business": {"monthly": 79, "yearly": 799},
    "business_plus": {"monthly": 149, "yearly": 1499},
}


@dataclass(frozen=True)
class CheckoutAvailability:
    enabled: bool
    provider: str | None
    currency: str
    reason: str
    methods: tuple[str, ...] = ()


def _secret(name: str) -> str | None:
    value = os.getenv(name)
    if value:
        return value.strip()
    try:
        import streamlit as st

        value = st.secrets.get(name)
        return str(value).strip() if value else None
    except Exception:
        return None


def checkout_availability() -> CheckoutAvailability:
    api_key = _secret("KONNECT_API_KEY")
    wallet_id = _secret("KONNECT_WALLET_ID")
    webhook = _secret("KONNECT_WEBHOOK_URL")
    methods = ("bank_card", "e-DINAR")

    if api_key and wallet_id and webhook:
        return CheckoutAvailability(
            enabled=True,
            provider="Konnect",
            currency="TND",
            reason="Checkout TND + confirmation webhook configurés.",
            methods=methods,
        )

    missing = []
    if not api_key:
        missing.append("KONNECT_API_KEY")
    if not wallet_id:
        missing.append("KONNECT_WALLET_ID")
    if not webhook:
        missing.append("KONNECT_WEBHOOK_URL")
    return CheckoutAvailability(
        enabled=False,
        provider="Konnect",
        currency="TND",
        reason=(
            "Paiement non activé : configuration incomplète ("
            + ", ".join(missing)
            + "). Le webhook de confirmation est obligatoire avant d'accepter "
            "un paiement réel."
        ),
        methods=methods,
    )


def _price(plan: str, billing_cycle: str) -> int:
    cycle = "yearly" if billing_cycle == "yearly" else "monthly"
    if plan not in PLAN_PRICES_TND:
        raise ValueError("Formule payante inconnue.")
    return PLAN_PRICES_TND[plan][cycle]


def _base_url() -> str:
    return (
        _secret("KONNECT_API_BASE_URL") or "https://api.konnect.network/api/v2"
    ).rstrip("/")


def start_checkout(
    *,
    user_id: str,
    plan: str,
    billing_cycle: str = "monthly",
    email: str | None = None,
    full_name: str | None = None,
    accepted_methods: Iterable[str] = ("bank_card", "e-DINAR"),
) -> str:
    """Crée un paiement hébergé Konnect et renvoie l'URL de checkout.

    Le paiement est ponctuel. L'accès ne doit être activé qu'après confirmation
    serveur du prestataire via le webhook configuré.
    """
    availability = checkout_availability()
    if not availability.enabled:
        raise RuntimeError(availability.reason)

    api_key = _secret("KONNECT_API_KEY")
    wallet_id = _secret("KONNECT_WALLET_ID")
    webhook = _secret("KONNECT_WEBHOOK_URL")
    success_url = _secret("PAYMENT_SUCCESS_URL")
    fail_url = _secret("PAYMENT_FAIL_URL")
    amount_tnd = _price(plan, billing_cycle)

    names = (full_name or "").strip().split(maxsplit=1)
    first_name = names[0] if names else ""
    last_name = names[1] if len(names) > 1 else ""

    payload = {
        "receiverWalletId": wallet_id,
        "token": "TND",
        # Konnect attend les TND en millimes.
        "amount": int(amount_tnd * 1000),
        "type": "immediate",
        "description": f"AutoDeal {plan} - {billing_cycle}",
        "acceptedPaymentMethods": list(accepted_methods),
        "lifespan": 30,
        "checkoutForm": True,
        "addPaymentFeesToAmount": False,
        "orderId": f"autodeal:{user_id}:{plan}:{billing_cycle}",
        "webhook": webhook,
        "silentWebhook": webhook,
    }
    if email:
        payload["email"] = email
    if first_name:
        payload["firstName"] = first_name
    if last_name:
        payload["lastName"] = last_name
    if success_url:
        payload["successUrl"] = success_url
    if fail_url:
        payload["failUrl"] = fail_url

    response = requests.post(
        f"{_base_url()}/payments/init-payment",
        headers={"x-api-key": api_key, "Content-Type": "application/json"},
        json=payload,
        timeout=20,
    )
    response.raise_for_status()
    data = response.json()
    url = data.get("payUrl") or data.get("link") or data.get("url")
    if not url:
        raise RuntimeError("Konnect n'a pas renvoyé d'URL de paiement.")
    return str(url)
