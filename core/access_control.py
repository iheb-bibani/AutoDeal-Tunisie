"""Rôles, abonnements et droits d'accès AutoDeal.

Le rôle décrit le métier de l'utilisateur. Le plan décrit son abonnement.
Un rôle n'accorde jamais à lui seul les fonctions premium : il faut également
un abonnement actif (ou une période d'essai active).
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

ROLE_LABELS = {
    "guest": "Visiteur",
    "user": "Particulier",
    "samsar": "Samsar",
    "dealer": "Concessionnaire",
    "admin": "Administrateur",
}

PLAN_CATALOG = {
    "free": {
        "name": "Gratuit",
        "monthly_tnd": 0,
        "yearly_tnd": 0,
        "roles": {"user", "samsar", "dealer"},
        "features": [
            "Recherche et estimation", "Comparateur", "Favoris",
            "Alertes de base", "Historique et santé du marché",
        ],
    },
    "pro": {
        "name": "Pro Samsar",
        "monthly_tnd": 29,
        "yearly_tnd": 299,
        "roles": {"samsar"},
        "features": [
            "Tout le plan Gratuit", "Espace Samsar complet",
            "Opportunités et radar de deals", "Recherche avancée et carte",
            "Alertes professionnelles", "Analyses achat-revente",
        ],
    },
    "business": {
        "name": "Business Concessionnaire",
        "monthly_tnd": 79,
        "yearly_tnd": 799,
        "roles": {"dealer"},
        "features": [
            "Tout le plan Gratuit", "Espace Concessionnaire complet",
            "Pricing et structure du marché", "Recherche avancée et carte",
            "Benchmark marché", "Analyses professionnelles",
        ],
    },
    "business_plus": {
        "name": "Business+",
        "monthly_tnd": 149,
        "yearly_tnd": 1499,
        "roles": {"dealer"},
        "features": [
            "Tout Business", "Préparation multi-utilisateurs",
            "Reporting avancé", "Exports et automatisations avancées",
        ],
    },
}

PUBLIC_PAGES = {
    "🏠 Accueil", "🛒 Acheter", "⚖️ Comparateur", "📊 Santé du marché",
    "📈 Historique", "🔔 Alertes", "👤 Mon compte", "💳 Tarifs",
    "💰 Calculateur", "🤖 Assistant", "🚘 Détail annonce",
}

ROLE_PAGES = {
    "user": set(),
    "samsar": {"🤝 Samsar", "🗺️ Carte", "🔎 Recherche avancée"},
    "dealer": {"🏢 Concessionnaire", "🗺️ Carte", "🔎 Recherche avancée"},
    "admin": {"🤝 Samsar", "🏢 Concessionnaire", "🗺️ Carte", "🔎 Recherche avancée", "🛠️ Admin"},
}


def _parse_dt(value: Any):
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except Exception:
        return None


def subscription_is_active(access: dict[str, Any]) -> bool:
    if access.get("role") == "admin":
        return True
    status = str(access.get("subscription_status") or "free").lower()
    if status == "active":
        # Une formule payante n'est active que pendant la période effectivement réglée.
        plan = str(access.get("plan") or "free")
        if plan == "free":
            return True
        period_end = _parse_dt(access.get("current_period_end"))
        return bool(period_end and period_end > datetime.now(timezone.utc))
    if status == "trialing":
        trial_end = _parse_dt(access.get("trial_end"))
        return bool(trial_end and trial_end > datetime.now(timezone.utc))
    return False


def effective_plan(access: dict[str, Any]) -> str:
    if access.get("role") == "admin":
        return "business_plus"
    plan = str(access.get("plan") or "free")
    if plan == "free":
        return "free"
    return plan if subscription_is_active(access) else "free"


def has_professional_access(access: dict[str, Any], area: str) -> bool:
    role = access.get("role", "guest")
    if role == "admin":
        return True
    plan = effective_plan(access)
    if area == "samsar":
        return role == "samsar" and plan in {"pro"}
    if area == "dealer":
        return role == "dealer" and plan in {"business", "business_plus"}
    if area == "advanced":
        return (role == "samsar" and plan == "pro") or (role == "dealer" and plan in {"business", "business_plus"})
    return False


def can_open_page(page: str, access: dict[str, Any]) -> bool:
    if page in PUBLIC_PAGES:
        return True
    role = access.get("role", "guest")
    if page not in ROLE_PAGES.get(role, set()):
        return False
    if role == "admin":
        return True
    if page == "🤝 Samsar":
        return has_professional_access(access, "samsar")
    if page == "🏢 Concessionnaire":
        return has_professional_access(access, "dealer")
    if page in {"🗺️ Carte", "🔎 Recherche avancée"}:
        return has_professional_access(access, "advanced")
    return True


def visible_pro_pages(access: dict[str, Any]) -> list[str]:
    role = access.get("role", "guest")
    if role == "admin":
        return ["🏢 Concessionnaire", "🤝 Samsar", "🗺️ Carte", "🔎 Recherche avancée"]
    if role == "samsar" and has_professional_access(access, "samsar"):
        return ["🤝 Samsar", "🗺️ Carte", "🔎 Recherche avancée"]
    if role == "dealer" and has_professional_access(access, "dealer"):
        return ["🏢 Concessionnaire", "🗺️ Carte", "🔎 Recherche avancée"]
    return []


def access_badge(access: dict[str, Any]) -> str:
    role = ROLE_LABELS.get(access.get("role", "guest"), "Utilisateur")
    plan_key = effective_plan(access)
    plan = PLAN_CATALOG.get(plan_key, PLAN_CATALOG["free"])["name"]
    if access.get("subscription_status") == "trialing" and subscription_is_active(access):
        return f"{role} · essai {plan}"
    if access.get("subscription_status") in {"past_due", "expired", "unpaid"}:
        return f"{role} · paiement requis"
    return f"{role} · {plan}"
