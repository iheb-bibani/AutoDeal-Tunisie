"""Paiements AutoDeal via checkout hébergé.

- AutoDeal ne collecte jamais PAN/CVV.
- Konnect est utilisé quand les secrets sont configurés.
- Le moyen e-DINAR peut être proposé dans le checkout. D17 est traité comme
  canal utilisateur autour de l'écosystème e-Dinar/merchant payment; le code
  ne suppose pas qu'un débit D17 récurrent public existe.
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
    if api_key and wallet_id:
        return CheckoutAvailability(
            enabled=True,
            provider="Konnect",
            currency="TND",
            reason="Checkout TND configuré.",
            methods=("bank_card", "e-DINAR"),
        )
    return CheckoutAvailability(
        enabled=False,
        provider="Konnect",
        currency="TND",
        reason=(
            "Paiement non activé : configure KONNECT_API_KEY et KONNECT_WALLET_ID "
            "après validation de votre compte marchand par le prestataire."
        ),
        methods=("bank_card", "e-DINAR"),
    )


def _price(plan: str, billing_cycle: str) -> int:
    cycle = "yearly" if billing_cycle == "yearly" else "monthly"
    if plan not in PLAN_PRICES_TND:
        raise ValueError("Formule payante inconnue.")
    return PLAN_PRICES_TND[plan][cycle]


def _base_url() -> str:
    return (_secret("KONNECT_API_BASE_URL") or "https://api.konnect.network/api/v2").rstrip("/")


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

    Le paiement est ponctuel. La récurrence/renouvellement est gérée côté
    abonnement AutoDeal via webhooks et dates de période. Un débit réellement
    automatique n'est activé que si le contrat/prestataire le supporte.
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
    }
    if email:
        payload["email"] = email
    if first_name:
        payload["firstName"] = first_name
    if last_name:
        payload["lastName"] = last_name
    if webhook:
        payload["webhook"] = webhook
        payload["silentWebhook"] = webhook
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
