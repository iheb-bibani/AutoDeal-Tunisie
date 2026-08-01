"""Expire les abonnements arrivés à échéance et envoie les notifications en attente.

À lancer côté backend (GitHub Actions ou cron), jamais depuis l'interface Streamlit.
"""
from __future__ import annotations

import json
import os
import smtplib
from datetime import datetime, timezone
from email.message import EmailMessage

import requests
from supabase import create_client


def _client():
    return create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_ROLE_KEY"])


def _send_email(to: str, subject: str, body: str) -> bool:
    host = os.getenv("SMTP_HOST")
    user = os.getenv("SMTP_USER")
    password = os.getenv("SMTP_PASSWORD")
    sender = os.getenv("SMTP_FROM") or user
    if not (host and user and password and sender and to):
        return False
    msg = EmailMessage()
    msg["From"], msg["To"], msg["Subject"] = sender, to, subject
    msg.set_content(body)
    port = int(os.getenv("SMTP_PORT", "587"))
    with smtplib.SMTP(host, port, timeout=20) as smtp:
        smtp.starttls()
        smtp.login(user, password)
        smtp.send_message(msg)
    return True


def _send_telegram(chat_id: str, text: str) -> bool:
    token = os.getenv("TELEGRAM_TOKEN")
    if not token or not chat_id:
        return False
    r = requests.post(
        f"https://api.telegram.org/bot{token}/sendMessage",
        json={"chat_id": chat_id, "text": text},
        timeout=20,
    )
    r.raise_for_status()
    return True


def expire_due_subscriptions(sb):
    now = datetime.now(timezone.utc)
    rows = sb.table("subscriptions").select("user_id,plan,status,current_period_end").in_("status", ["active", "past_due"]).execute().data or []
    for row in rows:
        end = row.get("current_period_end")
        if not end:
            continue
        dt = datetime.fromisoformat(str(end).replace("Z", "+00:00"))
        if dt <= now and row.get("plan") != "free" and row.get("status") != "past_due":
            sb.table("subscriptions").update({
                "status": "past_due",
                "last_payment_status": "due",
                "updated_at": now.isoformat(),
            }).eq("user_id", row["user_id"]).execute()
            sb.table("subscription_notifications").insert({
                "user_id": row["user_id"],
                "kind": "subscription_expired",
                "payload": {"plan": row.get("plan"), "period_end": end},
            }).execute()


def _message(kind: str, payload: dict) -> tuple[str, str]:
    if kind == "payment_success":
        return "Paiement AutoDeal confirmé", "Votre paiement AutoDeal a été confirmé. Votre accès premium est actif jusqu'au " + str(payload.get("period_end", "prochain renouvellement")) + "."
    if kind == "payment_failed":
        return "Paiement AutoDeal non abouti", "Votre paiement AutoDeal n'a pas abouti. Si votre période réglée est terminée, l'accès premium est suspendu jusqu'à un nouveau paiement confirmé."
    if kind == "subscription_expired":
        return "Abonnement AutoDeal arrivé à échéance", "Votre période AutoDeal réglée est terminée. L'accès premium est suspendu jusqu'au renouvellement."
    return "Information AutoDeal", "Une mise à jour concerne votre abonnement AutoDeal."


def send_pending_notifications(sb):
    notices = sb.table("subscription_notifications").select("*").eq("status", "pending").order("created_at").limit(100).execute().data or []
    for notice in notices:
        uid = notice["user_id"]
        kind = notice["kind"]
        payload = notice.get("payload") or {}
        subject, body = _message(kind, payload)

        # L'email vient d'auth.users via l'API admin.
        email = None
        try:
            user = sb.auth.admin.get_user_by_id(uid).user
            email = getattr(user, "email", None)
        except Exception:
            pass

        settings = sb.table("notification_settings").select("*").eq("user_id", uid).limit(1).execute().data or []
        settings = settings[0] if settings else {}
        sent = False
        errors = []
        if settings.get("email_enabled", True) and email:
            try:
                sent = _send_email(email, subject, body) or sent
            except Exception as exc:
                errors.append(f"email:{exc}")
        if settings.get("telegram_enabled") and settings.get("telegram_chat_id"):
            try:
                sent = _send_telegram(str(settings["telegram_chat_id"]), body) or sent
            except Exception as exc:
                errors.append(f"telegram:{exc}")

        sb.table("subscription_notifications").update({
            "status": "sent" if sent else "failed",
            "sent_at": datetime.now(timezone.utc).isoformat() if sent else None,
            "payload": {**payload, "delivery_errors": errors},
        }).eq("id", notice["id"]).execute()


def main():
    sb = _client()
    expire_due_subscriptions(sb)
    send_pending_notifications(sb)
    print(json.dumps({"ok": True, "finished_at": datetime.now(timezone.utc).isoformat()}))


if __name__ == "__main__":
    main()
