"""Canonical frontend routes for AutoDeal.

The label is the historical business identifier used by access-control and
session-state code. ``url_path`` is the stable browser URL exposed by
Streamlit's native router. Keeping both in one place prevents URL drift and
lets the old views migrate progressively without breaking behavior.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Route:
    key: str
    label: str
    title: str
    icon: str
    url_path: str
    section: str
    default: bool = False
    hidden_unless_selected: bool = False


ROUTES: tuple[Route, ...] = (
    Route("home", "🏠 Accueil", "Accueil", "🏠", "accueil", "ACHETER", default=True),
    Route("buy", "🛒 Acheter", "Acheter", "🛒", "acheter", "ACHETER"),
    Route("detail", "🚘 Détail annonce", "Détail annonce", "🚘", "annonce", "ACHETER", hidden_unless_selected=True),
    Route("compare", "⚖️ Comparateur", "Comparateur", "⚖️", "comparateur", "ACHETER"),
    Route("market_health", "📊 Santé du marché", "Santé du marché", "📊", "marche", "ACHETER"),
    Route("history", "📈 Historique", "Historique", "📈", "historique", "ACHETER"),
    Route("alerts", "🔔 Alertes", "Alertes", "🔔", "alertes", "ACHETER"),
    Route("account", "👤 Mon compte", "Mon compte", "👤", "compte", "COMPTE"),
    Route("pricing", "💳 Tarifs", "Tarifs", "💳", "tarifs", "COMPTE"),
    Route("calculator", "💰 Calculateur", "Estimer", "💰", "estimer", "ESTIMER"),
    Route("assistant", "🤖 Assistant", "Assistant", "🤖", "assistant", "ESTIMER"),
    Route("dealer", "🏢 Concessionnaire", "Concessionnaire", "🏢", "pro-concessionnaire", "PRO"),
    Route("advanced_search", "🔎 Recherche avancée", "Recherche avancée", "🔎", "recherche-avancee", "PRO"),
    Route("map", "🗺️ Carte", "Carte", "🗺️", "carte", "PRO"),
    Route("samsar", "🤝 Samsar", "Samsar", "🤝", "pro-samsar", "PRO"),
    Route("admin", "🛠️ Admin", "Admin", "🛠️", "admin", "ADMIN"),
)

BY_KEY = {route.key: route for route in ROUTES}
BY_LABEL = {route.label: route for route in ROUTES}
BY_PATH = {route.url_path: route for route in ROUTES}


def validate_routes() -> None:
    """Fail fast when a developer accidentally reuses a route identifier."""
    if len(BY_KEY) != len(ROUTES):
        raise ValueError("Duplicate frontend route key")
    if len(BY_LABEL) != len(ROUTES):
        raise ValueError("Duplicate frontend route label")
    if len(BY_PATH) != len(ROUTES):
        raise ValueError("Duplicate frontend URL path")
    defaults = [route for route in ROUTES if route.default]
    if len(defaults) != 1:
        raise ValueError("Exactly one frontend route must be the default")


validate_routes()
