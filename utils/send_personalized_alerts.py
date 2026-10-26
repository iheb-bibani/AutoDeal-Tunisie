# isort: skip_file
"""Envoie les alertes personnalisées Supabase (email / Telegram).

Exécuté après detect_deals par GitHub Actions. Le script utilise une clé
Supabase serveur (secret/service-role), lit uniquement les alertes actives,
matche les opportunités fiables, puis écrit alert_deliveries pour dédupliquer.
"""
from __future__ import annotations

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import html  # noqa: E402
import os  # noqa: E402
import smtplib  # noqa: E402
import ssl  # noqa: E402
from email.message import EmailMessage  # noqa: E402
from email.utils import getaddresses  # noqa: E402
import re  # noqa: E402
from functools import partial  # noqa: E402

import pandas as pd  # noqa: E402
import requests  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

from config import PROCESSED_FILES  # noqa: E402
from logger import get_logger  # noqa: E402
from services.supabase_service import create_admin_client  # noqa: E402

load_dotenv()
logger = get_logger(__name__)
DEALS_FILE = PROCESSED_FILES["deals"]


def _num(v, default=None):
    try:
        x = float(v)
        if pd.isna(x):
            return default
        return x
    except (TypeError, ValueError):
        return default


def _match(row, alert):
    if alert.get("brand") and str(row.get("Marque", "")).strip() != str(alert["brand"]).strip():
        return False
    if alert.get("model") and str(row.get("Modèle", "")).strip() != str(alert["model"]).strip():
        return False
    checks = [
        (_num(row.get("Prix")), _num(alert.get("budget_max")), lambda a, b: a <= b),
        (_num(row.get("Kilométrage")), _num(alert.get("max_km")), lambda a, b: a <= b),
        (_num(row.get("Année")), _num(alert.get("min_year")), lambda a, b: a >= b),
        (_num(row.get("Score_Opportunite"), 0) * 100, _num(alert.get("min_gap_pct"), 25), lambda a, b: a >= b),
    ]
    for value, threshold, op in checks:
        if threshold is not None and (value is None or not op(value, threshold)):
            return False
    # garde-fou identique au pipeline : éviter les anomalies extrêmes
    opp = _num(row.get("Score_Opportunite"), 0)
    if opp > 0.55:
        return False
    return True


def _message_text(row, alert_name):
    prix = int(_num(row.get("Prix"), 0))
    theorique = int(_num(row.get("Prix_Theorique"), 0))
    gain = max(0, theorique - prix)
    return (
        f"AutoDeal — {alert_name}\n\n"
        f"{row.get('Marque', '')} {row.get('Modèle', '')} — {row.get('Année', 'N/A')}\n"
        f"Prix : {prix:,} DT | estimation : {theorique:,} DT\n"
        f"Écart estimé : {gain:,} DT | kilométrage : {int(_num(row.get('Kilométrage'), 0)):,} km\n"
        f"Région : {row.get('Localisation', 'N/A')}\n\n"
        f"Annonce : {row.get('Lien', '')}"
    )


_TOKEN_RE = re.compile(r"bot\d+:[\w-]+")
_PAGE = 1000  # plafond PostgREST par défaut


def _redact(text) -> str:
    """Masque le token du bot : les exceptions requests contiennent l'URL complète."""
    return _TOKEN_RE.sub("bot<redacted>", str(text))


def valid_single_email(value) -> str | None:
    """Renvoie l'adresse si `value` contient exactement UNE adresse e-mail valide.

    Le champ est saisi librement par l'utilisateur : sans ce contrôle,
    "a@x.com, b@y.com" ferait envoyer à plusieurs destinataires, et un retour
    à la ligne lèverait une exception qui interrompait tout le run.
    """
    if not value or not isinstance(value, str) or any(c in value for c in "\r\n"):
        return None
    parsed = getaddresses([value])
    if len(parsed) != 1:
        return None
    addr = parsed[0][1].strip()
    if not re.fullmatch(r"[^@\s,;<>]+@[^@\s,;<>]+\.[^@\s,;<>]+", addr):
        return None
    return addr


def _fetch_all(build_query):
    """Pagine une requête PostgREST (`build_query()` renvoie un builder neuf)."""
    rows, start = [], 0
    while True:
        batch = build_query().range(start, start + _PAGE - 1).execute().data or []
        rows.extend(batch)
        if len(batch) < _PAGE:
            return rows
        start += _PAGE


def send_telegram(chat_id, text):
    token = os.getenv("TELEGRAM_TOKEN")
    if not token or not chat_id:
        return False
    try:
        r = requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": str(chat_id), "text": text, "disable_web_page_preview": False},
            timeout=15,
        )
        if r.ok:
            return True
        logger.error("Telegram %s: %s", r.status_code, _redact(r.text[:300]))
    except Exception as exc:
        logger.error("Erreur Telegram: %s", _redact(exc))
    return False


def send_email(to_email, row, alert_name):
    host = os.getenv("SMTP_HOST")
    port = int(os.getenv("SMTP_PORT", "587"))
    username = os.getenv("SMTP_USERNAME")
    password = os.getenv("SMTP_PASSWORD")
    sender = os.getenv("SMTP_FROM") or username
    to_email = valid_single_email(to_email)
    if not all([host, username, password, sender, to_email]):
        return False

    title = f"{row.get('Marque', '')} {row.get('Modèle', '')}".strip() or "Nouvelle opportunité"
    text = _message_text(row, alert_name)
    msg = EmailMessage()
    msg["Subject"] = f"AutoDeal : {title} correspond à votre alerte"
    msg["From"] = sender
    msg["To"] = to_email
    msg.set_content(text)
    link = html.escape(str(row.get("Lien", "")), quote=True)
    msg.add_alternative(
        f"<h2>🚗 {html.escape(title)}</h2>"
        f"<p><strong>Alerte :</strong> {html.escape(str(alert_name))}</p>"
        f"<p>Prix : <strong>{int(_num(row.get('Prix'), 0)):,} DT</strong><br>"
        f"Estimation AutoDeal : {int(_num(row.get('Prix_Theorique'), 0)):,} DT<br>"
        f"Kilométrage : {int(_num(row.get('Kilométrage'), 0)):,} km</p>"
        f"<p><a href=\"{link}\">Voir l'annonce</a></p>",
        subtype="html",
    )
    try:
        context = ssl.create_default_context()
        with smtplib.SMTP(host, port, timeout=20) as smtp:
            smtp.starttls(context=context)
            smtp.login(username, password)
            smtp.send_message(msg)
        return True
    except Exception as exc:
        logger.error("Erreur email vers %s: %s", to_email, exc)
        return False


def _process_alert(db, alert, deals, settings_by_user, account_email) -> int:
    user_id = str(alert["user_id"])
    settings = settings_by_user.get(user_id, {})
    matches = deals[deals.apply(_match, axis=1, args=(alert,))]
    if matches.empty:
        return 0
    delivered = _fetch_all(
        lambda: db.table("alert_deliveries").select("listing_url,channel").eq("alert_id", alert["id"]).order("id")
    )
    done = {(str(x["listing_url"]), str(x["channel"])) for x in delivered}
    alert_name = alert.get("name") or "Mon alerte"
    sent = 0

    for _, row in matches.iterrows():
        link = str(row.get("Lien", "")).strip()
        if not link or link == "nan":
            continue
        channels = []
        if alert.get("telegram_enabled") and settings.get("telegram_enabled") and settings.get("telegram_chat_id"):
            channels.append(("telegram", partial(send_telegram, settings["telegram_chat_id"], _message_text(row, alert_name))))
        if alert.get("email_enabled") and settings.get("email_enabled", True):
            to = account_email(user_id)
            if to:
                channels.append(("email", partial(send_email, to, row, alert_name)))

        for channel, sender in channels:
            if (link, channel) in done:
                continue
            if sender():
                # upsert idempotent : un doublon (run concurrent) ne fait pas planter le run.
                db.table("alert_deliveries").upsert(
                    {"alert_id": alert["id"], "user_id": alert["user_id"], "listing_url": link, "channel": channel},
                    on_conflict="alert_id,listing_url,channel",
                    ignore_duplicates=True,
                ).execute()
                done.add((link, channel))
                sent += 1
    return sent


def main():
    if not os.path.exists(DEALS_FILE):
        logger.warning("Fichier d'opportunités absent: %s", DEALS_FILE)
        return
    try:
        db = create_admin_client()
    except Exception as exc:
        logger.warning("Alertes personnalisées désactivées: %s", exc)
        return

    deals = pd.read_csv(DEALS_FILE, sep=";", encoding="utf-8-sig")
    if "Alerte_Telegram" in deals.columns:
        mask = deals["Alerte_Telegram"].astype(str).str.lower().isin({"true", "1", "vrai"})
        deals = deals[mask]
    if deals.empty:
        logger.info("Aucune opportunité solide à notifier.")
        return

    alerts = _fetch_all(lambda: db.table("alerts").select("*").eq("active", True).order("id"))
    settings_rows = _fetch_all(lambda: db.table("notification_settings").select("*").order("user_id"))
    settings_by_user = {str(x["user_id"]): x for x in settings_rows}
    verified_emails: dict[str, str | None] = {}

    def account_email(user_id: str) -> str | None:
        """Adresse VÉRIFIÉE du compte (auth.users), jamais le champ libre saisi
        dans notification_settings : on n'envoie qu'au titulaire du compte."""
        if user_id not in verified_emails:
            try:
                user = db.auth.admin.get_user_by_id(user_id).user
                confirmed = getattr(user, "email_confirmed_at", None) or getattr(user, "confirmed_at", None)
                verified_emails[user_id] = valid_single_email(getattr(user, "email", None)) if confirmed else None
            except Exception as exc:
                logger.warning("Email du compte %s introuvable : %s", user_id, type(exc).__name__)
                verified_emails[user_id] = None
        return verified_emails[user_id]

    total = 0
    for alert in alerts:
        try:
            total += _process_alert(db, alert, deals, settings_by_user, account_email)
        except Exception as exc:  # une alerte en échec ne bloque pas les autres utilisateurs
            logger.error("Alerte %s ignorée : %s", alert.get("id"), _redact(exc))

    logger.info("%s notification(s) personnalisée(s) envoyée(s).", total)


if __name__ == "__main__":
    main()
