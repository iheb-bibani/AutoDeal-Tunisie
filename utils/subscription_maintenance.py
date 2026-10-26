"""Expire les abonnements arrivés à échéance et envoie les notifications.

À lancer côté backend (GitHub Actions ou cron), jamais depuis l'interface
Streamlit. Les noms de secrets sont alignés sur le reste du projet, avec
compatibilité ascendante pour les anciens noms.
"""
from __future__ import annotations

import json
import os
import re
import smtplib
import ssl
import sys
from datetime import datetime, timezone
from email.message import EmailMessage
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import requests

from services.supabase_service import create_admin_client


def _client():
    return create_admin_client()


def _send_email(to: str, subject: str, body: str) -> bool:
    host = os.getenv("SMTP_HOST")
    user = os.getenv("SMTP_USERNAME") or os.getenv("SMTP_USER")
    password = os.getenv("SMTP_PASSWORD")
    sender = os.getenv("SMTP_FROM") or user
    if not (host and user and password and sender and to):
        return False
    msg = EmailMessage()
    msg["From"], msg["To"], msg["Subject"] = sender, to, subject
    msg.set_content(body)
    port = int(os.getenv("SMTP_PORT", "587"))
    with smtplib.SMTP(host, port, timeout=20) as smtp:
        smtp.starttls(context=ssl.create_default_context())
        smtp.login(user, password)
        smtp.send_message(msg)
    return True


class DeliveryError(RuntimeError):
    """Erreur d'envoi SANS détail sensible (jamais d'URL contenant le token)."""


_TOKEN_RE = re.compile(r"bot\d+:[\w-]+")


def _redact(text: str) -> str:
    return _TOKEN_RE.sub("bot<redacted>", str(text))


def _send_telegram(chat_id: str, text: str) -> bool:
    token = os.getenv("TELEGRAM_TOKEN")
    if not token or not chat_id:
        return False
    try:
        response = requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat_id, "text": text},
            timeout=20,
        )
    except requests.RequestException as exc:
        # str(exc) contient l'URL, donc le token : on ne garde que le type.
        raise DeliveryError(f"telegram_network:{type(exc).__name__}") from None
    if not response.ok:
        try:
            description = str(response.json().get("description", ""))[:120]
        except ValueError:
            description = ""
        raise DeliveryError(f"telegram_http_{response.status_code}:{_redact(description)}")
    return True


def _parse_dt(value):
    dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def expire_due_subscriptions(sb):
    """Passe en past_due les abonnements payants échus, et en expired les essais finis.

    La mise à jour est CONDITIONNELLE (status + date) : si le webhook de
    paiement renouvelle l'abonnement entre la lecture et l'écriture, la ligne
    ne correspond plus au filtre et n'est pas écrasée.
    """
    now = datetime.now(timezone.utc)
    now_iso = now.isoformat()
    rows = (
        sb.table("subscriptions")
        .select("user_id,plan,status,current_period_end")
        .eq("status", "active")
        .neq("plan", "free")
        .lte("current_period_end", now_iso)
        .execute()
        .data
        or []
    )
    for row in rows:
        end = row.get("current_period_end")
        try:
            if not end or _parse_dt(end) > now:
                continue
            updated = (
                sb.table("subscriptions")
                .update({
                    "status": "past_due",
                    "last_payment_status": "due",
                    "updated_at": now_iso,
                })
                .eq("user_id", row["user_id"])
                .eq("status", "active")
                .lte("current_period_end", now_iso)
                .execute()
                .data
                or []
            )
            if updated:
                sb.table("subscription_notifications").insert({
                    "user_id": row["user_id"],
                    "kind": "subscription_expired",
                    "payload": {"plan": row.get("plan"), "period_end": end},
                }).execute()
        except Exception as exc:  # une ligne invalide ne bloque pas les autres
            print(f"[maintenance] abonnement {row.get('user_id')} ignoré : {type(exc).__name__}")

    # Essais terminés : l'accès est déjà coupé par trial_end, on aligne le statut
    # affiché (badge, page compte).
    (
        sb.table("subscriptions")
        .update({"status": "expired", "updated_at": now_iso})
        .eq("status", "trialing")
        .lte("trial_end", now_iso)
        .execute()
    )


def _message(kind: str, payload: dict) -> tuple[str, str]:
    if kind == "payment_success":
        return (
            "Paiement AutoDeal confirmé",
            "Votre paiement AutoDeal a été confirmé. Votre accès premium est "
            "actif jusqu'au "
            + str(payload.get("period_end", "prochain renouvellement"))
            + ".",
        )
    if kind == "payment_failed":
        return (
            "Paiement AutoDeal non abouti",
            "Votre paiement AutoDeal n'a pas abouti. Si votre période réglée "
            "est terminée, l'accès premium est suspendu jusqu'à un nouveau "
            "paiement confirmé.",
        )
    if kind == "subscription_expired":
        return (
            "Abonnement AutoDeal arrivé à échéance",
            "Votre période AutoDeal réglée est terminée. L'accès premium est "
            "suspendu jusqu'au renouvellement.",
        )
    return (
        "Information AutoDeal",
        "Une mise à jour concerne votre abonnement AutoDeal.",
    )


def send_pending_notifications(sb):
    notices = (
        sb.table("subscription_notifications")
        .select("*")
        .eq("status", "pending")
        .order("created_at")
        .limit(100)
        .execute()
        .data
        or []
    )
    for notice in notices:
        uid = notice["user_id"]
        kind = notice["kind"]
        payload = notice.get("payload") or {}
        subject, body = _message(kind, payload)

        email = None
        try:
            user = sb.auth.admin.get_user_by_id(uid).user
            email = getattr(user, "email", None)
        except Exception:
            pass

        settings_rows = (
            sb.table("notification_settings")
            .select("*")
            .eq("user_id", uid)
            .limit(1)
            .execute()
            .data
            or []
        )
        settings = settings_rows[0] if settings_rows else {}
        sent = False
        errors = []
        if settings.get("email_enabled", True) and email:
            try:
                sent = _send_email(email, subject, body) or sent
            except Exception as exc:
                # Message lisible par l'utilisateur (RLS) : pas de détail serveur.
                errors.append(f"email:{type(exc).__name__}")
        if settings.get("telegram_enabled") and settings.get("telegram_chat_id"):
            try:
                sent = _send_telegram(
                    str(settings["telegram_chat_id"]), body
                ) or sent
            except DeliveryError as exc:
                errors.append(str(exc))
            except Exception as exc:
                errors.append(f"telegram:{type(exc).__name__}")

        (
            sb.table("subscription_notifications")
            .update({
                "status": "sent" if sent else "failed",
                "sent_at": datetime.now(timezone.utc).isoformat() if sent else None,
                "payload": {**payload, "delivery_errors": errors},
            })
            .eq("id", notice["id"])
            .execute()
        )


def main():
    sb = _client()
    expire_due_subscriptions(sb)
    send_pending_notifications(sb)
    print(
        json.dumps({
            "ok": True,
            "finished_at": datetime.now(timezone.utc).isoformat(),
        })
    )


if __name__ == "__main__":
    main()
