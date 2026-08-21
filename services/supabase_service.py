"""Couche Supabase optionnelle pour AutoDeal Tunisie.

Le client utilisateur emploie la publishable/anon key + le JWT de la session.
Le client admin (service role/secret key) est réservé aux scripts CI/backend.

Les politiques RLS restent la barrière de sécurité principale. Les requêtes
utilisateur filtrent aussi explicitement par ``user_id`` : cela rend le contrat
plus clair et évite de dépendre d'un ``limit(1)`` implicite pour les profils,
abonnements et préférences.
"""
from __future__ import annotations

import os
from typing import Any

try:
    from supabase import Client, create_client
except Exception:  # dépendance optionnelle tant que Supabase n'est pas configuré
    Client = Any  # type: ignore
    create_client = None


def _secret(name: str) -> str | None:
    value = os.getenv(name)
    if value:
        return value
    try:
        import streamlit as st

        value = st.secrets.get(name)
        return str(value) if value else None
    except Exception:
        return None


def is_configured() -> bool:
    return bool(
        _secret("SUPABASE_URL")
        and (_secret("SUPABASE_PUBLISHABLE_KEY") or _secret("SUPABASE_ANON_KEY"))
    )


def _public_key() -> str | None:
    return _secret("SUPABASE_PUBLISHABLE_KEY") or _secret("SUPABASE_ANON_KEY")


def create_public_client(
    access_token: str | None = None,
    refresh_token: str | None = None,
):
    if create_client is None:
        raise RuntimeError("Le package 'supabase' n'est pas installé.")
    url, key = _secret("SUPABASE_URL"), _public_key()
    if not url or not key:
        raise RuntimeError(
            "SUPABASE_URL et SUPABASE_PUBLISHABLE_KEY/ANON_KEY sont requis."
        )
    client = create_client(url, key)
    if access_token and refresh_token:
        client.auth.set_session(access_token, refresh_token)
    return client


def create_admin_client():
    """Client backend uniquement. La secret/service-role key contourne les RLS."""
    if create_client is None:
        raise RuntimeError("Le package 'supabase' n'est pas installé.")
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_SECRET_KEY") or os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    if not url or not key:
        raise RuntimeError(
            "SUPABASE_URL et SUPABASE_SECRET_KEY/SERVICE_ROLE_KEY sont requis côté backend."
        )
    return create_client(url, key)


def sign_up(
    email: str,
    password: str,
    full_name: str | None = None,
    account_role: str = "user",
) -> dict[str, Any]:
    client = create_public_client()
    payload = {"email": email.strip(), "password": password}
    role = account_role if account_role in {"user", "samsar", "dealer"} else "user"
    metadata = {"account_role": role}
    if full_name and full_name.strip():
        metadata["full_name"] = full_name.strip()
    payload["options"] = {"data": metadata}
    response = client.auth.sign_up(payload)
    session = getattr(response, "session", None)
    user = getattr(response, "user", None)
    return {
        "user_id": str(user.id) if user else None,
        "email": getattr(user, "email", email),
        "access_token": getattr(session, "access_token", None),
        "refresh_token": getattr(session, "refresh_token", None),
        "needs_confirmation": session is None,
    }


def sign_in(email: str, password: str) -> dict[str, Any]:
    client = create_public_client()
    response = client.auth.sign_in_with_password(
        {"email": email.strip(), "password": password}
    )
    session = response.session
    user = response.user
    return {
        "user_id": str(user.id),
        "email": user.email,
        "access_token": session.access_token,
        "refresh_token": session.refresh_token,
    }


def restore_session(access_token: str, refresh_token: str) -> dict[str, Any]:
    client = create_public_client()
    response = client.auth.set_session(access_token, refresh_token)
    user_resp = client.auth.get_user()
    session = getattr(response, "session", response)
    user = user_resp.user
    return {
        "client": client,
        "user_id": str(user.id),
        "email": user.email,
        "access_token": session.access_token,
        "refresh_token": session.refresh_token,
    }


def user_client_from_state(state: dict[str, Any]):
    access = state.get("sb_access_token")
    refresh = state.get("sb_refresh_token")
    if not access or not refresh:
        raise RuntimeError("Utilisateur non connecté.")
    restored = restore_session(access, refresh)
    state["sb_access_token"] = restored["access_token"]
    state["sb_refresh_token"] = restored["refresh_token"]
    state["sb_user_id"] = restored["user_id"]
    state["sb_user_email"] = restored["email"]
    return restored["client"]


def _current_user_id(client) -> str:
    response = client.auth.get_user()
    user = getattr(response, "user", None)
    user_id = getattr(user, "id", None)
    if not user_id:
        raise RuntimeError("Session Supabase sans utilisateur authentifié.")
    return str(user_id)


def sign_out(state: dict[str, Any]) -> None:
    try:
        client = user_client_from_state(state)
        client.auth.sign_out({"scope": "local"})
    except Exception:
        pass
    for key in (
        "sb_access_token",
        "sb_refresh_token",
        "sb_user_id",
        "sb_user_email",
        "favorite_links",
        "sb_role",
        "sb_plan",
        "sb_subscription_status",
    ):
        state.pop(key, None)


def list_alerts(client) -> list[dict[str, Any]]:
    uid = _current_user_id(client)
    return (
        client.table("alerts")
        .select("*")
        .eq("user_id", uid)
        .order("created_at", desc=True)
        .execute()
        .data
        or []
    )


def create_alert(client, payload: dict[str, Any]) -> dict[str, Any]:
    return client.table("alerts").insert(payload).execute().data[0]


def update_alert(client, alert_id: str, payload: dict[str, Any]) -> None:
    uid = _current_user_id(client)
    (
        client.table("alerts")
        .update(payload)
        .eq("id", alert_id)
        .eq("user_id", uid)
        .execute()
    )


def delete_alert(client, alert_id: str) -> None:
    uid = _current_user_id(client)
    client.table("alerts").delete().eq("id", alert_id).eq("user_id", uid).execute()


def get_notification_settings(client) -> dict[str, Any] | None:
    uid = _current_user_id(client)
    data = (
        client.table("notification_settings")
        .select("*")
        .eq("user_id", uid)
        .limit(1)
        .execute()
        .data
        or []
    )
    return data[0] if data else None


def save_notification_settings(client, payload: dict[str, Any]) -> None:
    uid = _current_user_id(client)
    safe_payload = {**payload, "user_id": uid}
    (
        client.table("notification_settings")
        .upsert(safe_payload, on_conflict="user_id")
        .execute()
    )


def list_favorites(client) -> list[dict[str, Any]]:
    uid = _current_user_id(client)
    return (
        client.table("favorites")
        .select("*")
        .eq("user_id", uid)
        .order("created_at", desc=True)
        .execute()
        .data
        or []
    )


def add_favorite(client, payload: dict[str, Any]) -> None:
    uid = _current_user_id(client)
    safe_payload = {**payload, "user_id": uid}
    (
        client.table("favorites")
        .upsert(safe_payload, on_conflict="user_id,listing_url")
        .execute()
    )


def remove_favorite(client, listing_url: str) -> None:
    uid = _current_user_id(client)
    (
        client.table("favorites")
        .delete()
        .eq("listing_url", listing_url)
        .eq("user_id", uid)
        .execute()
    )


def request_password_reset(email: str, redirect_to: str | None = None) -> None:
    client = create_public_client()
    options = {"redirect_to": redirect_to} if redirect_to else None
    if options:
        client.auth.reset_password_for_email(email.strip(), options=options)
    else:
        client.auth.reset_password_for_email(email.strip())


def verify_recovery_code(email: str, token: str) -> dict[str, Any]:
    client = create_public_client()
    response = client.auth.verify_otp(
        {"email": email.strip(), "token": token.strip(), "type": "recovery"}
    )
    session = response.session
    user = response.user
    return {
        "user_id": str(user.id),
        "email": user.email,
        "access_token": session.access_token,
        "refresh_token": session.refresh_token,
    }


def update_password_from_state(state: dict[str, Any], new_password: str) -> None:
    client = user_client_from_state(state)
    client.auth.update_user({"password": new_password})


def get_profile(client) -> dict[str, Any] | None:
    uid = _current_user_id(client)
    data = (
        client.table("profiles")
        .select("*")
        .eq("user_id", uid)
        .limit(1)
        .execute()
        .data
        or []
    )
    return data[0] if data else None


def get_subscription(client) -> dict[str, Any] | None:
    uid = _current_user_id(client)
    data = (
        client.table("subscriptions")
        .select("*")
        .eq("user_id", uid)
        .limit(1)
        .execute()
        .data
        or []
    )
    return data[0] if data else None


def current_access_context(state: dict[str, Any]) -> dict[str, Any]:
    if not state.get("sb_access_token") or not state.get("sb_refresh_token"):
        return {
            "role": "guest",
            "plan": "free",
            "subscription_status": "inactive",
        }
    client = user_client_from_state(state)
    profile = get_profile(client) or {}
    subscription = get_subscription(client) or {}
    role = profile.get("role") or "user"
    plan = subscription.get("plan") or "free"
    status = subscription.get("status") or (
        "active" if plan == "free" else "inactive"
    )
    state["sb_role"] = role
    state["sb_plan"] = plan
    state["sb_subscription_status"] = status
    return {
        "user_id": state.get("sb_user_id"),
        "email": state.get("sb_user_email"),
        "role": role,
        "plan": plan,
        "subscription_status": status,
        "trial_start": subscription.get("trial_start"),
        "trial_end": subscription.get("trial_end"),
        "current_period_start": subscription.get("current_period_start"),
        "current_period_end": subscription.get("current_period_end"),
        "cancel_at_period_end": bool(
            subscription.get("cancel_at_period_end", False)
        ),
        "billing_cycle": subscription.get("billing_cycle"),
        "payment_provider": subscription.get("payment_provider"),
        "last_payment_status": subscription.get("last_payment_status"),
        "last_payment_at": subscription.get("last_payment_at"),
        "next_payment_due_at": subscription.get("next_payment_due_at"),
    }


def current_user_profile(state: dict[str, Any]) -> dict[str, Any]:
    client = user_client_from_state(state)
    response = client.auth.get_user()
    user = response.user
    metadata = getattr(user, "user_metadata", None) or {}
    profile = get_profile(client) or {}
    subscription = get_subscription(client) or {}
    return {
        "id": str(user.id),
        "email": user.email,
        "full_name": profile.get("full_name")
        or metadata.get("full_name")
        or metadata.get("name")
        or "",
        "role": profile.get("role") or "user",
        "plan": subscription.get("plan") or "free",
        "subscription_status": subscription.get("status") or "active",
        "trial_end": subscription.get("trial_end"),
        "current_period_end": subscription.get("current_period_end"),
        "billing_cycle": subscription.get("billing_cycle"),
        "payment_provider": subscription.get("payment_provider"),
        "last_payment_status": subscription.get("last_payment_status"),
        "next_payment_due_at": subscription.get("next_payment_due_at"),
    }
