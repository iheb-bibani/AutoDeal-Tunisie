"""AutoDeal Tunisie — Streamlit frontend entrypoint.

V2 architecture starts by separating the application shell/routing from the
historical page implementations. The existing views are preserved byte-for-byte
in ``views.legacy_dashboard`` during the migration, while this file now owns:

- canonical browser URLs via ``st.Page`` / ``st.navigation``;
- role-aware navigation;
- shared sidebar status;
- compatibility with older views that still navigate through
  ``st.session_state['page']``.

FastAPI is intentionally a separate backend concern (``backend/main.py``).
Using FastAPI to fake frontend tabs would not change a Streamlit browser URL;
Streamlit's native router is the correct layer for page URLs.
"""
from __future__ import annotations

from collections import defaultdict
from functools import partial

import pandas as pd
import streamlit as st

# IMPORTANT: legacy_dashboard currently owns st.set_page_config + global CSS.
# It must therefore be imported before this entrypoint issues Streamlit UI calls.
from views import legacy_dashboard as legacy  # noqa: E402
from core.access_control import access_badge, can_open_page  # noqa: E402
from services.supabase_service import (  # noqa: E402
    current_access_context,
    is_configured as supabase_is_configured,
)
from web.routes import BY_LABEL, ROUTES, Route  # noqa: E402


PAGE_RENDERERS = {
    "🏠 Accueil": lambda: legacy.page_accueil(legacy.charger_scored()),
    "🛒 Acheter": lambda: legacy.page_acheter(legacy.charger_scored()),
    "🚘 Détail annonce": lambda: legacy.page_detail(legacy.charger_scored(), legacy.charger_modele()),
    "⚖️ Comparateur": lambda: legacy.page_comparateur(legacy.charger_scored()),
    "📊 Santé du marché": lambda: legacy.page_sante_marche(legacy.charger_scored()),
    "📈 Historique": lambda: legacy.page_historique(legacy.charger_scored()),
    "🔔 Alertes": lambda: legacy.page_alertes(legacy.charger_scored(), legacy.charger_deals()),
    "👤 Mon compte": lambda: legacy.page_compte(legacy.charger_scored()),
    "💳 Tarifs": lambda: legacy.page_tarifs(legacy.charger_scored()),
    "💰 Calculateur": lambda: legacy.page_calculateur(legacy.charger_scored(), legacy.charger_modele()),
    "🤖 Assistant": lambda: legacy.page_assistant(legacy.charger_scored(), legacy.charger_modele()),
    "🏢 Concessionnaire": lambda: legacy.page_marche(legacy.charger_scored()),
    "🔎 Recherche avancée": lambda: legacy.page_recherche(legacy.charger_scored()),
    "🗺️ Carte": lambda: legacy.page_carte(legacy.charger_scored()),
    "🤝 Samsar": lambda: legacy.page_samsar(legacy.charger_scored(), legacy.charger_deals()),
    "🛠️ Admin": lambda: legacy.page_admin(legacy.charger_scored(), legacy.charger_deals()),
}


def _access_context() -> dict:
    """Return a safe access context without making public mode fragile."""
    if not supabase_is_configured():
        return {"role": "guest", "plan": "free", "subscription_status": "inactive"}
    try:
        return current_access_context(st.session_state)
    except Exception:
        return {"role": "guest", "plan": "free", "subscription_status": "inactive"}


def _render_route(route: Route) -> None:
    """Render one routed page with defense-in-depth authorization.

    ``page`` remains synchronized for the historical views. This compatibility
    state can disappear once all views use the V2 router directly.
    """
    st.session_state["_active_route_label"] = route.label
    st.session_state["page"] = route.label

    access = _access_context()
    if not can_open_page(route.label, access):
        st.error("Accès non autorisé pour votre profil ou votre abonnement.")
        st.info("Cette URL existe, mais votre compte n'a pas accès à cette fonctionnalité.")
        return

    renderer = PAGE_RENDERERS.get(route.label)
    if renderer is None:
        st.error("Page AutoDeal non configurée.")
        return
    renderer()


def _make_pages() -> dict[str, st.Page]:
    pages: dict[str, st.Page] = {}
    for route in ROUTES:
        pages[route.label] = st.Page(
            partial(_render_route, route),
            title=route.title,
            icon=route.icon,
            url_path=route.url_path,
            default=route.default,
        )
    return pages


def _navigation_sections(access: dict, pages: dict[str, st.Page]) -> dict[str, list[st.Page]]:
    """Build role-aware navigation while keeping direct URLs stable."""
    sections: dict[str, list[st.Page]] = defaultdict(list)
    selected_listing = bool(st.session_state.get("selected_listing"))
    legacy_requested = st.session_state.get("page")
    active = st.session_state.get("_active_route_label")

    for route in ROUTES:
        if route.hidden_unless_selected:
            if not (selected_listing or legacy_requested == route.label or active == route.label):
                continue
        if can_open_page(route.label, access):
            sections[route.section].append(pages[route.label])

    # Keep the UX compact: account and pricing are grouped under one section,
    # while sections with no authorized page disappear automatically.
    ordered = {}
    for name in ("ACHETER", "ESTIMER", "PRO", "COMPTE", "ADMIN"):
        if sections.get(name):
            ordered[name] = sections[name]
    return ordered


def _redirect_legacy_navigation(pages: dict[str, st.Page], access: dict) -> None:
    """Translate old session-state navigation into a real URL transition.

    Several existing page buttons still do:
        session_state['page'] = '...'; st.rerun()

    On the next run we intercept that intent and call ``st.switch_page``.
    This lets us ship clean URLs now and migrate those buttons incrementally.
    """
    requested = st.session_state.get("page")
    active = st.session_state.get("_active_route_label")
    if not active or not requested or requested == active or requested not in pages:
        return

    if not can_open_page(requested, access):
        requested = "💳 Tarifs" if access.get("role") in {"samsar", "dealer"} else "🏠 Accueil"
        st.session_state["page"] = requested
    st.switch_page(pages[requested])


def _render_sidebar_status(access: dict, pages: dict[str, st.Page]) -> None:
    with st.sidebar:
        st.divider()
        st.markdown("### 🚗 AutoDeal Tunisie")
        if st.session_state.get("sb_user_email"):
            st.success(access_badge(access))
            who = st.session_state.get("sb_user_email", "").split("@")[0]
            st.page_link(pages["👤 Mon compte"], label=f"👤 {who}", use_container_width=True)
        else:
            st.page_link(pages["👤 Mon compte"], label="👤 Se connecter", use_container_width=True)

        df_temp = legacy.charger_scored()
        if df_temp is not None and "Annonce-Detectee" in df_temp.columns:
            derniere = df_temp["Annonce-Detectee"].max()
            st.caption(f"Données mises à jour : **{derniere}**")
            st.caption(f"Base : **{len(df_temp):,} annonces**".replace(",", " "))
            try:
                retard = (pd.Timestamp.now().normalize() - pd.to_datetime(derniere)).days
                if retard > 2:
                    st.warning(f"Données vieilles de {retard} jours — vérifie le pipeline nocturne.")
            except Exception:
                pass

        if st.button("🔄 Rafraîchir les données", use_container_width=True):
            st.cache_data.clear()
            st.cache_resource.clear()
            st.rerun()
        st.caption(
            f"Source : {'GitHub' if legacy.LIRE_DEPUIS_GITHUB else 'fichiers locaux'} · "
            f"cache {legacy.DUREE_CACHE // 60} min"
        )


def main() -> None:
    access = _access_context()
    pages = _make_pages()

    # Compatibility redirect must happen before st.navigation chooses a page.
    _redirect_legacy_navigation(pages, access)

    navigation = _navigation_sections(access, pages)
    current_page = st.navigation(navigation, position="sidebar", expanded=True)
    _render_sidebar_status(access, pages)
    current_page.run()


if __name__ == "__main__":
    main()
