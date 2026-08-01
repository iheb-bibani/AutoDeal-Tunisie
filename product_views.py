"""Vues produit grand public pour AutoDeal Tunisie.

Ces vues complètent les écrans métier historiques (Concessionnaire / Samsar)
avec un parcours centré acheteur : accueil, recherche par budget, fiche annonce,
comparateur, historique du marché et alertes.
"""
from __future__ import annotations

import html
import json
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

from core.market_valuation import market_valuation, compare_ml_to_market
from core.product_insights import explain_price, valuation_confidence
from core.access_control import PLAN_CATALOG, ROLE_LABELS, access_badge, effective_plan, subscription_is_active
from services.payment_provider import checkout_availability, start_checkout

from services.supabase_service import (
    add_favorite, create_alert, delete_alert, get_notification_settings,
    is_configured as supabase_is_configured, list_alerts, list_favorites,
    remove_favorite, save_notification_settings, sign_in, sign_out, sign_up,
    update_alert, user_client_from_state, request_password_reset, verify_recovery_code,
    update_password_from_state, current_user_profile, current_access_context,
)

TRACKING_PATH = "data/processed/suivi_annonces.csv"


def _fmt_dt(v):
    try:
        return pd.to_datetime(v).strftime("%d/%m/%Y")
    except Exception:
        return "—"


def _fmt_int(v, suffix=""):
    try:
        if pd.isna(v):
            return "—"
        return f"{int(round(float(v))):,}".replace(",", " ") + suffix
    except Exception:
        return "—"


def _safe(v, fallback="—"):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return fallback
    s = str(v).strip()
    return html.escape(s) if s else fallback


def _freshness_days(row):
    for col in ("Annonce-Detectee", "Annonce-Deposee"):
        if col in row.index and pd.notna(row[col]):
            try:
                return max(0, int((pd.Timestamp.now().normalize() - pd.to_datetime(row[col]).normalize()).days))
            except Exception:
                pass
    return None


def _freshness_label(row):
    days = _freshness_days(row)
    if days is None:
        return "Date inconnue"
    if days == 0:
        return "Détectée aujourd'hui"
    if days == 1:
        return "Détectée hier"
    return f"Détectée il y a {days} j"


def _range_estimation(row):
    est = pd.to_numeric(pd.Series([row.get("Prix_Theorique")]), errors="coerce").iloc[0]
    if pd.isna(est) or est <= 0:
        return None, None, None
    err = pd.to_numeric(pd.Series([row.get("Erreur_Relative_Modele")]), errors="coerce").iloc[0]
    if pd.isna(err) or err <= 0:
        err = 0.12
    # garde-fous : une fourchette utilisateur ne doit pas devenir absurde
    err = float(np.clip(err, 0.08, 0.25))
    return float(est), float(est * (1 - err)), float(est * (1 + err))


def score_autodeal(row):
    """Score explicable 0-100 : prix 55%, confiance 25%, profondeur du marché 20%.

    Il ne remplace pas le score ML. Il transforme des signaux existants en
    indicateur grand public, avec une formule volontairement lisible.
    """
    opp = pd.to_numeric(pd.Series([row.get("Score_Opportunite")]), errors="coerce").iloc[0]
    liq = pd.to_numeric(pd.Series([row.get("Score_Liquidite")]), errors="coerce").iloc[0]
    ncomp = pd.to_numeric(pd.Series([row.get("Nb_Comparables")]), errors="coerce").iloc[0]

    opp = 0 if pd.isna(opp) else float(opp)
    # 25% sous le marché = déjà excellent ; 55%+ est plafonné et potentiellement suspect.
    prix_score = float(np.clip((opp + 0.05) / 0.35, 0, 1) * 100)
    liq_score = float(np.clip(0 if pd.isna(liq) else liq, 0, 1) * 100)
    comp_score = float(np.clip((0 if pd.isna(ncomp) else ncomp) / 15, 0, 1) * 100)

    fiab = str(row.get("Fiabilite_Estimation", "")).lower()
    bonus = 8 if "élev" in fiab or "elev" in fiab else 4 if "moy" in fiab else 0
    confiance = min(100.0, comp_score + bonus)
    total = round(0.55 * prix_score + 0.25 * confiance + 0.20 * liq_score)
    return int(np.clip(total, 0, 100)), {
        "Prix vs marché": round(prix_score),
        "Confiance estimation": round(confiance),
        "Profondeur marché": round(liq_score),
    }


def _badge(score):
    if score >= 80:
        return "🔥 Très intéressant"
    if score >= 65:
        return "✅ Intéressant"
    if score >= 50:
        return "🟡 Correct"
    return "⚪ À comparer"


def _store_auth(result):
    for source, target in (("access_token", "sb_access_token"), ("refresh_token", "sb_refresh_token"),
                           ("user_id", "sb_user_id"), ("email", "sb_user_email")):
        if result.get(source):
            st.session_state[target] = result[source]


def _auth_client(silent=True):
    if not supabase_is_configured() or not st.session_state.get("sb_access_token"):
        return None
    try:
        return user_client_from_state(st.session_state)
    except Exception as exc:
        for key in ("sb_access_token", "sb_refresh_token", "sb_user_id", "sb_user_email"):
            st.session_state.pop(key, None)
        if not silent:
            st.error(f"Session expirée : {exc}")
        return None


def _listing_snapshot(row):
    keys = ["Titre", "Prix", "Lien", "Source", "Marque", "Modèle", "Année", "Kilométrage",
            "Energie", "Boite_Vitesse", "Localisation", "Prix_Theorique", "Score_Opportunite",
            "Nb_Comparables", "Fiabilite_Estimation", "Annonce-Detectee"]
    out = {}
    for key in keys:
        value = row.get(key)
        if pd.isna(value) if not isinstance(value, (list, dict)) else False:
            value = None
        if hasattr(value, "item"):
            value = value.item()
        if isinstance(value, pd.Timestamp):
            value = value.isoformat()
        out[key] = value
    return out


def _set_detail(row):
    st.session_state["selected_listing"] = row.get("Lien")
    st.session_state["page"] = "🚘 Détail annonce"


def render_listing_card(row, key, compare_enabled=True):
    score, _ = score_autodeal(row)
    est, low, high = _range_estimation(row)
    prix = row.get("Prix")
    opp = pd.to_numeric(pd.Series([row.get("Score_Opportunite")]), errors="coerce").iloc[0]
    opp_txt = "—" if pd.isna(opp) else f"{opp * 100:.0f} %"
    title = " ".join(x for x in [_safe(row.get("Marque"), ""), _safe(row.get("Modèle"), "")] if x).strip() or _safe(row.get("Titre"), "Annonce")

    with st.container(border=True):
        c1, c2 = st.columns([3, 1])
        with c1:
            st.markdown(f"### {title}")
            meta = " · ".join([
                _fmt_int(row.get("Année")),
                _fmt_int(row.get("Kilométrage"), " km"),
                _safe(row.get("Energie")),
                _safe(row.get("Boite_Vitesse")),
            ])
            st.caption(meta)
        with c2:
            st.markdown(f"### {_fmt_int(prix, ' DT')}")
            st.caption(_freshness_label(row))

        a, b, c = st.columns(3)
        a.metric("Score AutoDeal", f"{score}/100", help="55% prix vs marché, 25% confiance de l'estimation, 20% liquidité")
        b.metric("Écart marché", opp_txt)
        c.metric("Comparables", _fmt_int(row.get("Nb_Comparables")))
        st.caption(f"{_badge(score)} · Estimation : {_fmt_int(est, ' DT')}" +
                   (f" · fourchette {_fmt_int(low, ' DT')} – {_fmt_int(high, ' DT')}" if low else ""))

        x1, x2, x3, x4 = st.columns([1.0, 1.0, 1.0, 1.7])
        if x1.button("Analyser", key=f"detail_{key}", use_container_width=True):
            _set_detail(row)
            st.rerun()
        lien = row.get("Lien")
        if isinstance(lien, str) and lien.startswith("http"):
            x2.link_button("Voir l'annonce", lien, use_container_width=True)

        client = _auth_client()
        if client and isinstance(lien, str) and lien.startswith("http"):
            favs = st.session_state.get("favorite_links")
            if favs is None:
                try:
                    favs = {f["listing_url"] for f in list_favorites(client)}
                except Exception:
                    favs = set()
                st.session_state["favorite_links"] = favs
            if lien in favs:
                if x3.button("♥ Sauvé", key=f"fav_rm_{key}", use_container_width=True):
                    remove_favorite(client, lien)
                    favs.discard(lien)
                    st.rerun()
            else:
                if x3.button("♡ Favori", key=f"fav_add_{key}", use_container_width=True):
                    add_favorite(client, {
                        "user_id": st.session_state["sb_user_id"],
                        "listing_url": lien,
                        "listing_snapshot": _listing_snapshot(row),
                    })
                    favs.add(lien)
                    st.rerun()
        else:
            if x3.button("♡ Favori", key=f"fav_login_{key}", use_container_width=True,
                         help="Connectez-vous pour sauvegarder cette annonce."):
                st.session_state["auth_return_page"] = st.session_state.get("page", "🛒 Acheter")
                st.session_state["auth_mode"] = "login"
                st.session_state["page"] = "👤 Mon compte"
                st.rerun()

        if compare_enabled:
            selected = st.session_state.setdefault("compare_links", [])
            if lien in selected:
                if x4.button("✓ Retirer du comparateur", key=f"cmp_rm_{key}", use_container_width=True):
                    st.session_state["compare_links"] = [x for x in selected if x != lien]
                    st.rerun()
            elif len(selected) < 4:
                if x4.button("＋ Comparer", key=f"cmp_add_{key}", use_container_width=True):
                    st.session_state["compare_links"] = selected + [lien]
                    st.rerun()


def page_accueil(df):
    st.markdown("# Trouvez une voiture au bon prix en Tunisie 🚗")
    st.markdown("### AutoDeal compare les annonces au marché pour repérer les voitures intéressantes — sans jargon ML.")
    if df is None or df.empty:
        st.info("Données indisponibles.")
        return

    latest = pd.to_datetime(df.get("Annonce-Detectee"), errors="coerce").max()
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Annonces analysées", f"{len(df):,}".replace(",", " "))
    k2.metric("Mise à jour", latest.strftime("%d/%m/%Y") if pd.notna(latest) else "—")
    k3.metric("Opportunités", int(df.get("Score_Opportunite", pd.Series(dtype=float)).between(0.25, 0.55).sum()))
    k4.metric("Sources", df.get("Source", pd.Series(dtype=str)).nunique())

    with st.container(border=True):
        st.markdown("**Pourquoi faire confiance aux chiffres ?**")
        st.caption("AutoDeal confronte deux méthodes indépendantes : le modèle ML et les prix de véhicules réellement comparables. La confiance baisse automatiquement quand il y a peu de comparables, un marché dispersé ou un désaccord entre les deux.")

    st.markdown("## Que voulez-vous faire ?")
    c1, c2, c3 = st.columns(3)
    with c1:
        with st.container(border=True):
            st.markdown("### 🔎 Trouver une voiture")
            st.write("Filtrez le marché et voyez immédiatement le juste prix, l'écart et la confiance.")
            if st.button("Explorer les annonces", key="home_search", use_container_width=True, type="primary"):
                st.session_state.page = "🛒 Acheter"
                st.rerun()
    with c2:
        with st.container(border=True):
            st.markdown("### 💰 Estimer une voiture")
            st.write("Obtenez une estimation, une fourchette réaliste et les facteurs qui expliquent le prix.")
            if st.button("Estimer un véhicule", key="home_calc", use_container_width=True):
                st.session_state.page = "💰 Calculateur"
                st.rerun()
    with c3:
        with st.container(border=True):
            st.markdown("### 🔥 Voir les bonnes affaires")
            st.write("Classez les annonces qui sont sous leur valeur de marché avec suffisamment de comparables.")
            if st.button("Voir les opportunités", key="home_deals", use_container_width=True):
                st.session_state["buy_deals_only"] = True
                st.session_state.page = "🛒 Acheter"
                st.rerun()

    with st.container(border=True):
        h1, h2 = st.columns([3, 1])
        with h1:
            st.markdown("### 📊 Comprendre le marché")
            st.write("Prix, dispersion, profondeur d’offre, modèles qui baissent et durée observée des annonces.")
        with h2:
            if st.button("Voir la santé du marché", key="home_market_health", use_container_width=True):
                st.session_state.page = "📊 Santé du marché"
                st.rerun()

    st.markdown("## Bonnes affaires du moment")
    deals = df[df.get("Score_Opportunite", pd.Series(index=df.index, dtype=float)).between(0.15, 0.55)].copy()
    if "Nb_Comparables" in deals:
        deals = deals[deals["Nb_Comparables"].fillna(0) >= 5]
    deals = deals.sort_values(["Score_Opportunite", "Nb_Comparables"], ascending=False).head(3)
    if deals.empty:
        st.info("Pas assez d'opportunités fiables dans la collecte actuelle.")
    else:
        for idx, row in deals.iterrows():
            render_listing_card(row, f"home_{idx}")


def page_acheter(df):
    st.title("🛒 Trouver une voiture")
    st.caption("Commencez par votre budget. AutoDeal classe ensuite les annonces selon le rapport prix / marché, la confiance et la profondeur du marché.")
    if df is None or df.empty:
        st.info("Données indisponibles.")
        return

    d = df.copy()
    prix = pd.to_numeric(d["Prix"], errors="coerce")
    p99 = int(prix.quantile(.99)) if prix.notna().any() else 200000
    default_budget = min(60000, p99)

    f1, f2, f3, f4 = st.columns(4)
    with f1:
        max_budget = st.number_input("Budget maximum", min_value=3000, max_value=max(p99, 5000), value=max(3000, default_budget), step=1000, format="%d")
    with f2:
        min_year = int(pd.to_numeric(d["Année"], errors="coerce").dropna().quantile(.05)) if d["Année"].notna().any() else 2000
        year = st.number_input("Année minimum", min_value=1980, max_value=pd.Timestamp.now().year + 1, value=max(min_year, 2015), step=1)
    with f3:
        km_max = st.number_input("Kilométrage maximum", min_value=0, max_value=1000000, value=150000, step=10000)
    with f4:
        deals_only = st.checkbox("Bonnes affaires uniquement", value=bool(st.session_state.pop("buy_deals_only", False)))

    s1, s2, s3 = st.columns(3)
    marques = s1.multiselect("Marque", sorted(d["Marque"].dropna().astype(str).unique()))
    energies = s2.multiselect("Énergie", sorted(d["Energie"].dropna().astype(str).unique())) if "Energie" in d else []
    boites = s3.multiselect("Boîte", sorted(d["Boite_Vitesse"].dropna().astype(str).unique())) if "Boite_Vitesse" in d else []

    d = d[pd.to_numeric(d["Prix"], errors="coerce") <= max_budget]
    d = d[pd.to_numeric(d["Année"], errors="coerce").fillna(0) >= year]
    d = d[pd.to_numeric(d["Kilométrage"], errors="coerce").fillna(0) <= km_max]
    if marques:
        d = d[d["Marque"].isin(marques)]
    if energies:
        d = d[d["Energie"].isin(energies)]
    if boites:
        d = d[d["Boite_Vitesse"].isin(boites)]
    if deals_only:
        d = d[d["Score_Opportunite"].between(.15, .55)]

    if len(d):
        d = d.copy()
        d["_score_ui"] = d.apply(lambda r: score_autodeal(r)[0], axis=1)
        d = d.sort_values(["_score_ui", "Nb_Comparables"], ascending=False)

    st.markdown(f"### {len(d):,} véhicules correspondent".replace(",", " "))
    if not len(d):
        st.info("Aucun véhicule ne correspond. Augmentez le budget, l'année ou le kilométrage maximum.")
        return

    top = d.head(30)
    tabs = st.tabs(["⭐ Recommandés", "💸 Moins chers", "🆕 Plus récents"])
    orders = [
        top,
        d.sort_values("Prix").head(30),
        d.sort_values(["Année", "Annonce-Detectee"], ascending=False).head(30),
    ]
    for tab, subset in zip(tabs, orders):
        with tab:
            for idx, row in subset.iterrows():
                render_listing_card(row, f"buy_{tab}_{idx}")


def page_detail(df, bundle=None):
    st.title("🚘 Analyse d'une annonce")
    if df is None or df.empty:
        st.info("Données indisponibles.")
        return
    link = st.session_state.get("selected_listing")
    if not link or "Lien" not in df.columns or not (df["Lien"] == link).any():
        st.info("Choisissez une annonce depuis la page Acheter ou le Comparateur.")
        if st.button("Ouvrir la recherche"):
            st.session_state.page = "🛒 Acheter"
            st.rerun()
        return

    row = df.loc[df["Lien"] == link].iloc[0]
    score, parts = score_autodeal(row)
    ml_value, ml_low, ml_high = _range_estimation(row)
    price = pd.to_numeric(pd.Series([row.get("Prix")]), errors="coerce").iloc[0]
    title = f"{row.get('Marque','')} {row.get('Modèle','')}".strip()

    market, comparables = market_valuation(df, row, min_n=5)
    comparison = compare_ml_to_market(ml_value, market)
    confidence, confidence_label, confidence_parts = valuation_confidence(row, market, comparison)

    st.subheader(title)
    st.caption(" · ".join([_fmt_int(row.get("Année")), _fmt_int(row.get("Kilométrage"), " km"), _safe(row.get("Localisation")), _freshness_label(row)]))

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Prix demandé", _fmt_int(price, " DT"))
    c2.metric("Estimation ML", _fmt_int(ml_value, " DT"))
    c3.metric("Médiane marché", _fmt_int(market.median_price, " DT"))
    c4.metric("Confiance", f"{confidence}/100 · {confidence_label}")

    st.markdown("## Ce que dit le marché")
    if market.n_comparables >= 5:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Comparables proches", market.n_comparables)
        m2.metric("Cœur du marché Q25–Q75", f"{_fmt_int(market.q25, ' DT')} – {_fmt_int(market.q75, ' DT')}")
        m3.metric("Marché élargi P10–P90", f"{_fmt_int(market.p10, ' DT')} – {_fmt_int(market.p90, ' DT')}")
        width = market.relative_width * 100 if market.relative_width is not None else np.nan
        m4.metric("Homogénéité", f"{market.homogeneity} · {width:.1f} %" if np.isfinite(width) else market.homogeneity)

        if comparison.get("inside_market_range") is True:
            st.success("✅ L'estimation ML se situe dans le cœur Q25–Q75 du marché comparable.")
        elif comparison.get("inside_market_range_p10_p90") is True:
            st.warning(f"🟡 L'estimation ML est {comparison.get('market_position', '').lower()}, mais reste dans le marché élargi P10–P90.")
        elif comparison.get("available"):
            st.error(f"⚠️ L'estimation ML est {comparison.get('market_position', '').lower()} et sort aussi du marché élargi P10–P90.")

        plot = comparables.copy()
        plot["Prix"] = pd.to_numeric(plot["Prix"], errors="coerce")
        plot = plot.dropna(subset=["Prix"])
        if len(plot):
            fig = px.histogram(plot, x="Prix", nbins=min(14, max(5, len(plot)//2)), title="Distribution des prix des comparables")
            if market.median_price:
                fig.add_vline(x=market.median_price, line_dash="dash", annotation_text="Médiane marché")
            if ml_value:
                fig.add_vline(x=ml_value, line_dash="dot", annotation_text="ML")
            if pd.notna(price):
                fig.add_vline(x=float(price), annotation_text="Annonce")
            fig.update_layout(height=330, margin=dict(l=10, r=10, t=55, b=10), showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Pas assez de comparables proches pour construire une fourchette de marché robuste. La confiance est donc réduite.")

    st.markdown("## Pourquoi ce prix ?")
    left, right = st.columns([1, 1])
    with left:
        st.markdown("### 🤖 Ce que voit le modèle")
        contrib = explain_price(bundle, row, df) if bundle else None
        if contrib is not None and not contrib.empty:
            top = contrib.head(8).sort_values("impact")
            fig = px.bar(top, x="impact", y="label", orientation="h", title=f"Facteurs principaux ({contrib.attrs.get('method', 'explication')})",
                         labels={"impact": "Impact sur l'estimation", "label": ""})
            fig.update_layout(height=360, margin=dict(l=10, r=10, t=55, b=10))
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Impact positif = pousse l'estimation vers le haut ; impact négatif = la tire vers le bas. Si SHAP est indisponible, l'app affiche une approximation par perturbation.")
        else:
            st.info("Explication locale indisponible pour ce modèle. L'estimation et les comparables restent utilisables.")
    with right:
        st.markdown("### 📊 Ce que montrent les comparables")
        if market.n_comparables >= 5:
            st.write(f"**{market.n_comparables} annonces proches** sélectionnées au niveau **{market.selection_level}**.")
            st.write(f"Médiane observée : **{_fmt_int(market.median_price, ' DT')}**")
            st.write(f"Cœur du marché : **{_fmt_int(market.q25, ' DT')} – {_fmt_int(market.q75, ' DT')}**")
            if comparison.get("gap_vs_market_median_pct") is not None:
                st.write(f"Écart ML ↔ médiane marché : **{comparison['gap_vs_market_median_pct']*100:+.1f} %**")
            cols = [c for c in ["Marque", "Modèle", "Année", "Kilométrage", "Energie", "Boite_Vitesse", "Prix", "Lien"] if c in comparables.columns]
            st.dataframe(comparables.head(8)[cols], hide_index=True, use_container_width=True,
                         column_config={"Prix": st.column_config.NumberColumn(format="%d DT"), "Lien": st.column_config.LinkColumn("Annonce", display_text="ouvrir")})
        else:
            st.info("Les comparables proches sont insuffisants pour une lecture marché fiable.")

    st.markdown("## Confiance de la valorisation")
    st.progress(confidence / 100.0, text=f"{confidence}/100 — Confiance {confidence_label.lower()}")
    q1, q2, q3, q4 = st.columns(4)
    q1.metric("Historique ML", f"{confidence_parts['Historique ML']}/100")
    q2.metric("Comparables", f"{confidence_parts['Comparables']}/100")
    q3.metric("Homogénéité", f"{confidence_parts['Homogénéité']}/100")
    q4.metric("Accord ML ↔ marché", f"{confidence_parts['Accord ML ↔ marché']}/100")
    st.caption("Ce score mesure la crédibilité de la valorisation, pas l'intérêt de l'annonce. Une forte décote avec confiance faible doit être vérifiée davantage.")

    st.markdown("## Potentiel de l'annonce")
    p1, p2, p3, p4 = st.columns(4)
    p1.metric("Score AutoDeal", f"{score}/100")
    p2.metric("Prix vs estimation", f"{parts['Prix vs marché']}/100")
    p3.metric("Confiance historique", f"{parts['Confiance estimation']}/100")
    p4.metric("Profondeur marché", f"{parts['Profondeur marché']}/100")
    st.caption("Le Score AutoDeal mesure l'intérêt potentiel de l'annonce. Il est volontairement séparé du score de confiance ci-dessus.")

    if ml_low:
        st.caption(f"Intervalle ML indicatif : {_fmt_int(ml_low, ' DT')} – {_fmt_int(ml_high, ' DT')} · Fourchette marché empirique : {_fmt_int(market.q25, ' DT')} – {_fmt_int(market.q75, ' DT')} quand disponible.")

    if isinstance(link, str) and link.startswith("http"):
        st.link_button("Voir l'annonce originale", link, type="primary")

def page_comparateur(df):
    st.title("⚖️ Comparateur AutoDeal")
    st.caption("Comparez jusqu'à 4 annonces sur les critères qui comptent vraiment.")
    if df is None or df.empty:
        st.info("Données indisponibles.")
        return
    selected = [x for x in st.session_state.get("compare_links", []) if x in set(df["Lien"].dropna())]

    # Permet aussi d'ajouter directement depuis cette page.
    pool = df.sort_values("Score_Opportunite", ascending=False).head(250).copy()
    pool["_label"] = pool.apply(lambda r: f"{r.get('Marque','')} {r.get('Modèle','')} · {_fmt_int(r.get('Année'))} · {_fmt_int(r.get('Prix'),' DT')}", axis=1)
    labels = dict(zip(pool["_label"], pool["Lien"]))
    defaults = [lab for lab, lk in labels.items() if lk in selected]
    picks = st.multiselect("Véhicules à comparer", list(labels.keys()), default=defaults, max_selections=4)
    selected = [labels[p] for p in picks]
    st.session_state["compare_links"] = selected
    if len(selected) < 2:
        st.info("Sélectionnez au moins 2 véhicules. Vous pouvez aussi les ajouter depuis la page Acheter.")
        return

    cmp = df[df["Lien"].isin(selected)].copy()
    cmp["Score AutoDeal"] = cmp.apply(lambda r: score_autodeal(r)[0], axis=1)
    cmp["Écart marché %"] = (cmp["Score_Opportunite"] * 100).round(0)
    cmp["Fourchette basse"] = cmp.apply(lambda r: _range_estimation(r)[1], axis=1)
    cmp["Fourchette haute"] = cmp.apply(lambda r: _range_estimation(r)[2], axis=1)
    cmp["Véhicule"] = cmp.apply(lambda r: f"{r.get('Marque','')} {r.get('Modèle','')}", axis=1)

    display = cmp[["Véhicule", "Prix", "Année", "Kilométrage", "Prix_Theorique", "Fourchette basse", "Fourchette haute", "Écart marché %", "Score AutoDeal", "Nb_Comparables", "Score_Liquidite"]].set_index("Véhicule").T
    st.dataframe(display, use_container_width=True)

    winner = cmp.sort_values(["Score AutoDeal", "Nb_Comparables"], ascending=False).iloc[0]
    st.success(f"🏆 Meilleur compromis selon AutoDeal : **{winner['Véhicule']}** — score {winner['Score AutoDeal']}/100.")
    for idx, row in cmp.iterrows():
        if st.button(f"Analyser {row['Véhicule']}", key=f"cmp_detail_{idx}"):
            _set_detail(row)
            st.rerun()


def _load_tracking():
    try:
        return pd.read_csv(TRACKING_PATH, sep=";", encoding="utf-8-sig")
    except Exception:
        return None


def page_historique(df):
    st.title("📈 Historique & évolution du marché")
    st.caption("Suivez les baisses de prix observées et l'évolution des cohortes d'annonces. L'historique actuel conserve prix initial et dernier prix, pas chaque changement intermédiaire.")
    tracking = _load_tracking()
    if tracking is None or tracking.empty:
        st.info("Historique indisponible.")
        return

    t = tracking.copy()
    t["Prix_Initial"] = pd.to_numeric(t["Prix_Initial"], errors="coerce")
    t["Prix_Dernier"] = pd.to_numeric(t["Prix_Dernier"], errors="coerce")
    t["Variation_DT"] = t["Prix_Dernier"] - t["Prix_Initial"]
    t["Variation_pct"] = np.where(t["Prix_Initial"] > 0, t["Variation_DT"] / t["Prix_Initial"] * 100, np.nan)
    baisses = t[t["Variation_DT"] < 0]
    c1, c2, c3 = st.columns(3)
    c1.metric("Annonces suivies", f"{len(t):,}".replace(",", " "))
    c2.metric("Baisses de prix détectées", len(baisses))
    c3.metric("Baisse médiane", f"{abs(baisses['Variation_pct'].median()):.1f} %" if len(baisses) else "—")

    brands = sorted(t["Marque"].dropna().astype(str).unique())
    h1, h2 = st.columns(2)
    sel = h1.selectbox("Marque", ["Toutes"] + brands)
    subset_brand = t if sel == "Toutes" else t[t["Marque"] == sel]
    models = sorted(subset_brand["Modèle"].dropna().astype(str).unique()) if "Modèle" in subset_brand else []
    sel_model = h2.selectbox("Modèle", ["Tous"] + models)
    x = subset_brand if sel_model == "Tous" else subset_brand[subset_brand["Modèle"].astype(str) == sel_model]
    x = x.copy()
    x["Premiere_Vue"] = pd.to_datetime(x["Premiere_Vue"], errors="coerce")
    daily = x.dropna(subset=["Premiere_Vue", "Prix_Initial"]).groupby(x["Premiere_Vue"].dt.date).agg(prix_median=("Prix_Initial", "median"), annonces=("Lien", "count")).reset_index()
    if len(daily) >= 2:
        fig = px.line(daily, x="Premiere_Vue", y="prix_median", markers=True, title="Prix médian des nouvelles annonces par date de première observation", labels={"Premiere_Vue":"", "prix_median":"Prix médian (DT)"})
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Il faut davantage de jours de suivi pour tracer une vraie tendance temporelle.")

    st.markdown("### Plus fortes baisses observées")
    cols = [c for c in ["Marque", "Modèle", "Année", "Prix_Initial", "Prix_Dernier", "Variation_DT", "Variation_pct", "Premiere_Vue", "Derniere_Vue", "Lien"] if c in t]
    st.dataframe(t.sort_values("Variation_pct").head(30)[cols], hide_index=True, use_container_width=True,
                 column_config={"Lien": st.column_config.LinkColumn("Annonce", display_text="ouvrir"), "Variation_pct": st.column_config.NumberColumn("Variation", format="%.1f %%")})


def _filter_alert_matches(df, marque="Toutes", modele="Tous", budget=50000, max_km=150000, min_year=2015, seuil=25):
    x = df.copy()
    x = x[pd.to_numeric(x["Prix"], errors="coerce") <= budget]
    if marque not in ("Toutes", "", None):
        x = x[x["Marque"].astype(str) == str(marque)]
    if modele not in ("Tous", "", None):
        x = x[x["Modèle"].astype(str) == str(modele)]
    x = x[pd.to_numeric(x["Kilométrage"], errors="coerce").fillna(10**9) <= max_km]
    x = x[pd.to_numeric(x["Année"], errors="coerce").fillna(0) >= min_year]
    x = x[pd.to_numeric(x["Score_Opportunite"], errors="coerce") >= seuil / 100]
    x = x[pd.to_numeric(x["Score_Opportunite"], errors="coerce") <= .55]
    if "Nb_Comparables" in x:
        x = x[pd.to_numeric(x["Nb_Comparables"], errors="coerce").fillna(0) >= 5]
    return x.sort_values("Annonce-Detectee", ascending=False)



def page_sante_marche(df):
    st.title("📊 Santé du marché automobile")
    st.caption("Un observatoire des annonces collectées : niveau de prix, dispersion, profondeur d'offre et signaux issus du suivi. Il ne s'agit pas des immatriculations officielles du marché tunisien.")
    if df is None or df.empty:
        st.info("Données indisponibles.")
        return

    d = df.copy()
    d["Prix"] = pd.to_numeric(d["Prix"], errors="coerce")
    d["Année"] = pd.to_numeric(d.get("Année"), errors="coerce")
    d["Kilométrage"] = pd.to_numeric(d.get("Kilométrage"), errors="coerce")
    d = d[d["Prix"].gt(0)].copy()

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Annonces observées", f"{len(d):,}".replace(",", " "))
    k2.metric("Prix médian", _fmt_int(d["Prix"].median(), " DT"))
    k3.metric("Modèles suivis", d[["Marque", "Modèle"]].dropna().drop_duplicates().shape[0])
    k4.metric("Sources", d.get("Source", pd.Series(dtype=str)).nunique())

    st.markdown("## Profondeur d'offre")
    grp = (d.dropna(subset=["Marque", "Modèle"])
             .groupby(["Marque", "Modèle"])
             .agg(n=("Prix", "size"), prix_median=("Prix", "median"), q25=("Prix", lambda x: x.quantile(.25)), q75=("Prix", lambda x: x.quantile(.75)))
             .reset_index())
    grp = grp[grp["n"] >= 5].copy()
    grp["dispersion_pct"] = np.where(grp["prix_median"] > 0, (grp["q75"] - grp["q25"]) / grp["prix_median"] * 100, np.nan)
    if len(grp):
        top_supply = grp.sort_values("n", ascending=False).head(15)
        fig = px.bar(top_supply.sort_values("n"), x="n", y=top_supply["Marque"] + " " + top_supply["Modèle"], orientation="h",
                     title="Modèles les plus présents dans les annonces observées", labels={"n": "Nombre d'annonces", "y": ""})
        fig.update_layout(height=470, margin=dict(l=10, r=10, t=55, b=10))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("## Où les prix sont-ils les plus homogènes ?")
    if len(grp):
        view = grp[grp["n"] >= 8].sort_values("dispersion_pct").head(20).copy()
        view["Véhicule"] = view["Marque"] + " " + view["Modèle"]
        fig = px.bar(view.sort_values("dispersion_pct", ascending=False), x="dispersion_pct", y="Véhicule", orientation="h",
                     title="Dispersion Q25–Q75 relative à la médiane", labels={"dispersion_pct": "Dispersion (%)", "Véhicule": ""})
        fig.update_layout(height=520, margin=dict(l=10, r=10, t=55, b=10))
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Une faible dispersion signifie que les annonces d'un même modèle sont relativement regroupées en prix. Cela facilite une valorisation par comparables.")

    tracking = _load_tracking()
    if tracking is None or tracking.empty:
        st.info("Le suivi historique n'est pas encore assez disponible pour analyser les baisses et durées d'annonces.")
        return

    t = tracking.copy()
    for col in ["Prix_Initial", "Prix_Dernier"]:
        if col in t:
            t[col] = pd.to_numeric(t[col], errors="coerce")
    if "Prix_Initial" in t and "Prix_Dernier" in t:
        t["Variation_pct"] = np.where(t["Prix_Initial"] > 0, (t["Prix_Dernier"] - t["Prix_Initial"]) / t["Prix_Initial"] * 100, np.nan)
        g = (t.dropna(subset=["Marque", "Modèle", "Variation_pct"])
               .groupby(["Marque", "Modèle"])
               .agg(n=("Variation_pct", "size"), variation_mediane=("Variation_pct", "median"))
               .reset_index())
        g = g[g["n"] >= 5].sort_values("variation_mediane")
        if len(g):
            st.markdown("## Mouvements de prix observés pendant le suivi")
            left, right = st.columns(2)
            with left:
                low = g.head(12).copy()
                low["Véhicule"] = low["Marque"] + " " + low["Modèle"]
                fig = px.bar(low.sort_values("variation_mediane", ascending=False), x="variation_mediane", y="Véhicule", orientation="h",
                             title="Baisses médianes les plus fortes", labels={"variation_mediane": "Variation (%)", "Véhicule": ""})
                fig.update_layout(height=430, margin=dict(l=10, r=10, t=55, b=10))
                st.plotly_chart(fig, use_container_width=True)
            with right:
                high = g[g["variation_mediane"] > 0].sort_values("variation_mediane", ascending=False).head(12).copy()
                if len(high):
                    high["Véhicule"] = high["Marque"] + " " + high["Modèle"]
                    fig = px.bar(high.sort_values("variation_mediane"), x="variation_mediane", y="Véhicule", orientation="h",
                                 title="Hausses médianes observées", labels={"variation_mediane": "Variation (%)", "Véhicule": ""})
                    fig.update_layout(height=430, margin=dict(l=10, r=10, t=55, b=10))
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Aucun modèle avec au moins 5 annonces suivies ne présente actuellement une hausse médiane positive.")
            st.caption("Ces signaux comparent le premier et le dernier prix vus sur les annonces suivies ; ils ne constituent pas un indice officiel d'évolution des prix du marché.")

    if "Premiere_Vue" in t.columns and "Derniere_Vue" in t.columns:
        first = pd.to_datetime(t["Premiere_Vue"], errors="coerce")
        last = pd.to_datetime(t["Derniere_Vue"], errors="coerce")
        t["Duree_Observee_J"] = (last - first).dt.days
        ended = t[t["Duree_Observee_J"].ge(0)].copy()
        if len(ended):
            st.markdown("## Durée observée des annonces")
            d1, d2, d3 = st.columns(3)
            d1.metric("Médiane", f"{ended['Duree_Observee_J'].median():.0f} j")
            d2.metric("25e percentile", f"{ended['Duree_Observee_J'].quantile(.25):.0f} j")
            d3.metric("75e percentile", f"{ended['Duree_Observee_J'].quantile(.75):.0f} j")
            st.caption("Tant que le pipeline ne distingue pas parfaitement une annonce vendue d'une annonce retirée/expirée, cette durée est une durée de présence observée, pas une durée de vente certifiée.")

def page_alertes(df, deals):
    st.title("🔔 Alertes personnalisées")
    st.caption("Enregistrez vos critères : le pipeline nocturne pourra vous prévenir par email et/ou Telegram lorsqu'une nouvelle opportunité correspond.")
    if df is None or df.empty:
        st.info("Données indisponibles.")
        return
    if not supabase_is_configured():
        st.warning("Supabase n'est pas encore configuré. Ouvrez `supabase/schema.sql`, créez le projet Supabase puis renseignez les secrets Streamlit.")
        return
    client = _auth_client(silent=False)
    if client is None:
        st.info("Connectez-vous via **👤 Mon compte** pour créer des alertes persistantes.")
        if st.button("Aller à Mon compte", type="primary"):
            st.session_state.page = "👤 Mon compte"
            st.rerun()
        return

    marques = ["Toutes"] + sorted(df["Marque"].dropna().astype(str).unique())
    c1, c2 = st.columns([2, 1])
    with c1:
        st.markdown("### Créer une alerte")
        name = st.text_input("Nom de l'alerte", value="Ma recherche")
        a1, a2, a3 = st.columns(3)
        marque = a1.selectbox("Marque", marques, key="alert_brand")
        modeles = ["Tous"]
        if marque != "Toutes":
            modeles += sorted(df.loc[df["Marque"].astype(str) == marque, "Modèle"].dropna().astype(str).unique())
        modele = a2.selectbox("Modèle", modeles, key="alert_model")
        budget = a3.number_input("Budget max", min_value=3000, max_value=max(5000, int(pd.to_numeric(df["Prix"], errors="coerce").quantile(.99))), value=50000, step=1000, key="alert_budget")
        b1, b2, b3 = st.columns(3)
        max_km = b1.number_input("Kilométrage max", min_value=0, max_value=1000000, value=150000, step=10000)
        min_year = b2.number_input("Année minimum", min_value=1980, max_value=pd.Timestamp.now().year + 1, value=2015)
        seuil = b3.slider("Minimum sous le marché", 10, 40, 25, step=5, key="alert_gap")
        n1, n2 = st.columns(2)
        email_enabled = n1.checkbox("📧 Email", value=True)
        telegram_enabled = n2.checkbox("✈️ Telegram", value=False)
        if st.button("🔔 Enregistrer cette alerte", type="primary", use_container_width=True):
            if telegram_enabled:
                settings = get_notification_settings(client) or {}
                if not settings.get("telegram_chat_id"):
                    st.error("Ajoutez d'abord votre Chat ID Telegram dans Mon compte.")
                else:
                    create_alert(client, {"user_id": st.session_state["sb_user_id"], "name": name.strip() or "Ma recherche",
                        "brand": None if marque == "Toutes" else marque, "model": None if modele == "Tous" else modele,
                        "budget_max": int(budget), "max_km": int(max_km), "min_year": int(min_year),
                        "min_gap_pct": float(seuil), "email_enabled": email_enabled, "telegram_enabled": telegram_enabled, "active": True})
                    st.success("Alerte enregistrée.")
                    st.rerun()
            else:
                create_alert(client, {"user_id": st.session_state["sb_user_id"], "name": name.strip() or "Ma recherche",
                    "brand": None if marque == "Toutes" else marque, "model": None if modele == "Tous" else modele,
                    "budget_max": int(budget), "max_km": int(max_km), "min_year": int(min_year),
                    "min_gap_pct": float(seuil), "email_enabled": email_enabled, "telegram_enabled": telegram_enabled, "active": True})
                st.success("Alerte enregistrée.")
                st.rerun()
    with c2:
        preview = _filter_alert_matches(df, marque, modele, budget, max_km, min_year, seuil)
        st.metric("Correspondances actuelles", len(preview))
        st.caption("L'alerte n'enverra que les nouvelles correspondances non déjà livrées.")

    st.divider()
    alerts = list_alerts(client)
    st.markdown(f"### Mes alertes ({len(alerts)})")
    if not alerts:
        st.info("Aucune alerte enregistrée.")
    for alert in alerts:
        with st.container(border=True):
            z1, z2, z3 = st.columns([3, 1, 1])
            z1.markdown(f"**{_safe(alert.get('name'), 'Alerte')}**")
            desc = " · ".join([str(x) for x in [alert.get("brand") or "Toutes marques", alert.get("model") or "Tous modèles",
                    f"≤ {_fmt_int(alert.get('budget_max'), ' DT')}", f"≤ {_fmt_int(alert.get('max_km'), ' km')}",
                    f"≥ {_fmt_int(alert.get('min_year'))}", f"≥ {float(alert.get('min_gap_pct') or 0):.0f}% sous marché"]])
            z1.caption(desc + f" · {'Email ' if alert.get('email_enabled') else ''}{'Telegram' if alert.get('telegram_enabled') else ''}")
            active = z2.toggle("Active", value=bool(alert.get("active")), key=f"active_{alert['id']}")
            if active != bool(alert.get("active")):
                update_alert(client, alert["id"], {"active": active})
                st.rerun()
            if z3.button("Supprimer", key=f"del_{alert['id']}", use_container_width=True):
                delete_alert(client, alert["id"])
                st.rerun()

    st.divider()
    st.markdown("### Aperçu de la recherche en cours")
    for idx, row in preview.head(10).iterrows():
        render_listing_card(row, f"alert_preview_{idx}", compare_enabled=False)


def page_compte(df=None):
    st.title("👤 Mon compte AutoDeal")
    st.caption("Un compte vous permet de conserver vos favoris, créer des alertes et retrouver vos préférences sur tous vos appareils.")
    if not supabase_is_configured():
        st.warning("Supabase n'est pas configuré. Consultez `SUPABASE_SETUP.md` dans le projet.")
        return

    client = _auth_client()
    if client is None:
        mode = st.session_state.get("auth_mode", "login")
        tabs = st.tabs(["🔐 Connexion", "✨ Créer un compte", "🔑 Mot de passe oublié"])

        with tabs[0]:
            st.markdown("### Bon retour 👋")
            st.write("Connectez-vous pour retrouver vos favoris et vos alertes AutoDeal.")
            with st.form("login_form"):
                email = st.text_input("Email", placeholder="vous@exemple.com")
                password = st.text_input("Mot de passe", type="password")
                submit = st.form_submit_button("Se connecter", type="primary", use_container_width=True)
            if submit:
                if not email.strip() or not password:
                    st.error("Renseignez votre email et votre mot de passe.")
                else:
                    try:
                        result = sign_in(email, password)
                        _store_auth(result)
                        st.session_state.pop("favorite_links", None)
                        st.success("Connexion réussie.")
                        target = st.session_state.pop("auth_return_page", None)
                        if target:
                            st.session_state["page"] = target
                        st.rerun()
                    except Exception as exc:
                        msg = str(exc).lower()
                        if "invalid login" in msg or "invalid credentials" in msg:
                            st.error("Email ou mot de passe incorrect.")
                        elif "email not confirmed" in msg:
                            st.error("Votre adresse email n'est pas encore confirmée. Ouvrez l'email envoyé par AutoDeal puis réessayez.")
                        else:
                            st.error("Connexion impossible pour le moment. Vérifiez vos informations puis réessayez.")

        with tabs[1]:
            st.markdown("### Créer votre compte")
            st.write("L'inscription est gratuite. Vos données personnelles restent séparées des autres utilisateurs grâce aux règles RLS Supabase.")
            with st.form("signup_form"):
                full_name = st.text_input("Nom affiché", placeholder="Votre prénom ou votre nom")
                role_label = st.selectbox(
                    "Je suis",
                    ["Particulier", "Samsar", "Concessionnaire"],
                    help="Ce profil détermine les espaces professionnels visibles. Le rôle Administrateur n'est jamais attribuable à l'inscription.",
                )
                role_map = {"Particulier": "user", "Samsar": "samsar", "Concessionnaire": "dealer"}
                account_role = role_map[role_label]
                email2 = st.text_input("Email", key="signup_email", placeholder="vous@exemple.com")
                password2 = st.text_input("Mot de passe", type="password", key="signup_password", help="8 caractères minimum.")
                password3 = st.text_input("Confirmer le mot de passe", type="password", key="signup_password_confirm")
                accept = st.checkbox("J'accepte que mon compte soit utilisé pour enregistrer mes favoris et alertes.")
                submit2 = st.form_submit_button("Créer mon compte", type="primary", use_container_width=True)
            if submit2:
                if not full_name.strip() or not email2.strip():
                    st.error("Renseignez votre nom et votre email.")
                elif len(password2) < 8:
                    st.error("Choisissez un mot de passe d'au moins 8 caractères.")
                elif password2 != password3:
                    st.error("Les deux mots de passe ne correspondent pas.")
                elif not accept:
                    st.error("Vous devez accepter l'enregistrement de vos favoris et alertes pour créer le compte.")
                else:
                    try:
                        result = sign_up(email2, password2, full_name, account_role=account_role)
                        if result.get("needs_confirmation"):
                            st.success("Compte créé ✅ Un email de confirmation vient de vous être envoyé.")
                            st.info("Ouvrez l'email, confirmez votre adresse, puis revenez ici et connectez-vous. Les profils Samsar et Concessionnaire démarrent avec 14 jours d'essai de leur espace professionnel.")
                        else:
                            _store_auth(result)
                            st.success("Compte créé et connecté.")
                            st.rerun()
                    except Exception as exc:
                        msg = str(exc).lower()
                        if "already registered" in msg or "already been registered" in msg or "user already" in msg:
                            st.error("Un compte existe déjà avec cette adresse email. Utilisez l'onglet Connexion ou Mot de passe oublié.")
                        else:
                            st.error("Création du compte impossible pour le moment. Vérifiez l'adresse email et réessayez.")

        with tabs[2]:
            st.markdown("### Réinitialiser votre mot de passe")
            st.write("Demandez un code de récupération, puis saisissez-le ici pour choisir un nouveau mot de passe.")
            recovery_email = st.text_input("Email du compte", key="recovery_email")
            if st.button("Envoyer le code de récupération", use_container_width=True):
                if not recovery_email.strip():
                    st.error("Renseignez votre adresse email.")
                else:
                    try:
                        request_password_reset(recovery_email)
                        st.session_state["recovery_email_saved"] = recovery_email.strip()
                        st.success("Si cette adresse correspond à un compte, Supabase vient d'envoyer un email de récupération.")
                        st.caption("Le modèle d'email Supabase doit afficher le code `{{ .Token }}` ; la procédure est expliquée dans SUPABASE_SETUP.md.")
                    except Exception:
                        st.error("Impossible d'envoyer l'email de récupération pour le moment.")
            with st.form("recovery_code_form"):
                rmail = st.text_input("Email", value=st.session_state.get("recovery_email_saved", ""), key="recovery_email_code")
                code = st.text_input("Code reçu par email")
                npass = st.text_input("Nouveau mot de passe", type="password")
                npass2 = st.text_input("Confirmer le nouveau mot de passe", type="password")
                reset = st.form_submit_button("Changer mon mot de passe", type="primary", use_container_width=True)
            if reset:
                if len(npass) < 8:
                    st.error("Le nouveau mot de passe doit contenir au moins 8 caractères.")
                elif npass != npass2:
                    st.error("Les deux mots de passe ne correspondent pas.")
                elif not rmail.strip() or not code.strip():
                    st.error("Renseignez l'email et le code de récupération.")
                else:
                    try:
                        result = verify_recovery_code(rmail, code)
                        _store_auth(result)
                        update_password_from_state(st.session_state, npass)
                        st.success("Mot de passe modifié. Vous êtes maintenant connecté.")
                        st.rerun()
                    except Exception:
                        st.error("Code invalide ou expiré. Demandez un nouveau code puis réessayez.")
        return

    try:
        profile = current_user_profile(st.session_state)
    except Exception:
        profile = {"email": st.session_state.get("sb_user_email", ""), "full_name": ""}
    display_name = profile.get("full_name") or (profile.get("email") or "Utilisateur").split("@")[0]

    h1, h2 = st.columns([4, 1])
    with h1:
        st.success(f"Connecté en tant que **{display_name}** · {profile.get('email','')}")
    with h2:
        if st.button("🚪 Déconnexion", use_container_width=True):
            sign_out(st.session_state)
            st.rerun()

    tab_fav, tab_alert, tab_pref, tab_sub, tab_sec = st.tabs(["❤️ Mes favoris", "🔔 Mes alertes", "⚙️ Préférences", "💳 Abonnement", "🔐 Sécurité"])

    with tab_fav:
        favs = list_favorites(client)
        st.session_state["favorite_links"] = {f["listing_url"] for f in favs}
        st.markdown(f"### {len(favs)} favori{'s' if len(favs) != 1 else ''}")
        if not favs:
            st.info("Aucun favori pour l'instant. Cliquez sur ♡ Favori depuis une annonce.")
        for fav in favs:
            snap = fav.get("listing_snapshot") or {}
            with st.container(border=True):
                a, b = st.columns([4, 1])
                a.markdown(f"**{_safe(snap.get('Marque'), '')} {_safe(snap.get('Modèle'), '')}** — {_fmt_int(snap.get('Prix'), ' DT')}")
                a.caption(f"{_fmt_int(snap.get('Année'))} · {_fmt_int(snap.get('Kilométrage'), ' km')} · {_safe(snap.get('Localisation'))}")
                if b.button("Retirer", key=f"account_unfav_{fav['id']}", use_container_width=True):
                    remove_favorite(client, fav["listing_url"])
                    st.session_state.pop("favorite_links", None)
                    st.rerun()
                if isinstance(fav.get("listing_url"), str):
                    st.link_button("Voir l'annonce", fav["listing_url"])

    with tab_alert:
        alerts = list_alerts(client)
        c1, c2 = st.columns([3, 1])
        c1.markdown(f"### {len(alerts)} alerte{'s' if len(alerts) != 1 else ''}")
        if c2.button("＋ Nouvelle alerte", use_container_width=True, type="primary"):
            st.session_state["page"] = "🔔 Alertes"
            st.rerun()
        if not alerts:
            st.info("Vous n'avez pas encore créé d'alerte.")
        for alert in alerts:
            with st.container(border=True):
                x1, x2, x3 = st.columns([4, 1, 1])
                x1.markdown(f"**{_safe(alert.get('name'), 'Alerte')}**")
                x1.caption(" · ".join([str(v) for v in [alert.get("brand") or "Toutes marques", alert.get("model") or "Tous modèles", f"≤ {_fmt_int(alert.get('budget_max'), ' DT')}"]]))
                active = x2.toggle("Active", value=bool(alert.get("active")), key=f"account_active_{alert['id']}")
                if active != bool(alert.get("active")):
                    update_alert(client, alert["id"], {"active": active})
                    st.rerun()
                if x3.button("Supprimer", key=f"account_del_{alert['id']}"):
                    delete_alert(client, alert["id"])
                    st.rerun()

    with tab_pref:
        st.markdown("### Notifications")
        settings = get_notification_settings(client) or {}
        with st.form("notif_settings"):
            email = st.text_input("Email de notification", value=settings.get("email") or profile.get("email", ""))
            telegram_chat_id = st.text_input("Chat ID Telegram", value=settings.get("telegram_chat_id") or "", help="Identifiant du chat où le bot AutoDeal doit envoyer vos alertes.")
            c1, c2 = st.columns(2)
            email_enabled = c1.checkbox("Activer les emails", value=bool(settings.get("email_enabled", True)))
            telegram_enabled = c2.checkbox("Activer Telegram", value=bool(settings.get("telegram_enabled", False)))
            save = st.form_submit_button("Enregistrer mes préférences", type="primary")
        if save:
            save_notification_settings(client, {"user_id": st.session_state["sb_user_id"], "email": email.strip() or None,
                "telegram_chat_id": telegram_chat_id.strip() or None, "email_enabled": email_enabled, "telegram_enabled": telegram_enabled})
            st.success("Préférences enregistrées.")
            st.rerun()

    with tab_sub:
        access = current_access_context(st.session_state)
        plan_key = effective_plan(access)
        plan = PLAN_CATALOG.get(plan_key, PLAN_CATALOG["free"])
        st.markdown("### Votre accès AutoDeal")
        c1, c2, c3 = st.columns(3)
        c1.metric("Profil", ROLE_LABELS.get(access.get("role", "user"), "Particulier"))
        c2.metric("Formule", plan["name"])
        c3.metric("Statut", "Essai" if access.get("subscription_status") == "trialing" and subscription_is_active(access) else str(access.get("subscription_status", "active")).replace("_", " ").title())
        if access.get("trial_end") and access.get("subscription_status") == "trialing":
            st.info(f"Votre période d'essai se termine le **{_fmt_dt(access.get('trial_end'))}**.")
        st.caption("Le profil métier et l'abonnement sont séparés : être Samsar ou Concessionnaire ne suffit pas à débloquer les fonctions premium si l'abonnement n'est plus actif.")
        if st.button("Voir les formules", key="account_to_pricing", type="primary", use_container_width=True):
            st.session_state["page"] = "💳 Tarifs"
            st.rerun()
        if access.get("role") == "admin":
            st.warning("Le rôle Administrateur est attribué uniquement manuellement dans Supabase et n'est jamais proposé à l'inscription.")

    with tab_sec:
        st.markdown("### Modifier mon mot de passe")
        with st.form("change_password_form"):
            new_password = st.text_input("Nouveau mot de passe", type="password")
            confirm_password = st.text_input("Confirmer", type="password")
            change = st.form_submit_button("Modifier le mot de passe")
        if change:
            if len(new_password) < 8:
                st.error("Le mot de passe doit contenir au moins 8 caractères.")
            elif new_password != confirm_password:
                st.error("Les deux mots de passe ne correspondent pas.")
            else:
                try:
                    update_password_from_state(st.session_state, new_password)
                    st.success("Mot de passe modifié.")
                except Exception:
                    st.error("Impossible de modifier le mot de passe pour le moment.")



def page_tarifs(df=None):
    st.title("💳 Formules AutoDeal")
    st.caption("Des offres simples en dinars tunisiens. L'encaissement CB reste désactivé tant qu'un compte marchand adapté n'est pas configuré.")

    access = current_access_context(st.session_state) if supabase_is_configured() else {"role": "guest", "plan": "free", "subscription_status": "inactive"}
    current = effective_plan(access)
    availability = checkout_availability()

    cycle = st.radio("Facturation", ["Mensuelle", "Annuelle"], horizontal=True, key="pricing_cycle")
    annual = cycle == "Annuelle"
    cols = st.columns(4)
    ordered = ["free", "pro", "business", "business_plus"]
    role_targets = {"free": "Particulier", "pro": "Samsar", "business": "Concessionnaire", "business_plus": "Concessionnaire"}
    for col, key in zip(cols, ordered):
        plan = PLAN_CATALOG[key]
        price = plan["yearly_tnd"] if annual else plan["monthly_tnd"]
        with col:
            with st.container(border=True):
                st.markdown(f"### {plan['name']}")
                st.caption(role_targets[key])
                if price:
                    suffix = "/an" if annual else "/mois"
                    st.markdown(f"## {price} DT{suffix}")
                else:
                    st.markdown("## 0 DT")
                for feat in plan["features"]:
                    st.markdown(f"✓ {feat}")
                if key == current and access.get("role") != "guest":
                    st.success("Votre formule actuelle")
                elif key == "free":
                    st.caption("Disponible sans paiement.")
                else:
                    if st.button("Choisir cette formule", key=f"choose_{key}_{'y' if annual else 'm'}", use_container_width=True):
                        if access.get("role") == "guest":
                            st.session_state["auth_return_page"] = "💳 Tarifs"
                            st.session_state["page"] = "👤 Mon compte"
                            st.rerun()
                        else:
                            try:
                                profile = current_user_profile(st.session_state)
                                url = start_checkout(
                                    user_id=str(access.get("user_id")),
                                    plan=key,
                                    billing_cycle="yearly" if annual else "monthly",
                                    email=profile.get("email"),
                                    full_name=profile.get("full_name"),
                                )
                                st.link_button("Continuer vers le paiement", url)
                            except Exception as exc:
                                st.info(str(exc))

    st.divider()
    st.markdown("### Paiement et sécurité")
    if availability.enabled:
        st.success(f"Checkout activé via {availability.provider} · devise {availability.currency}.")
    else:
        st.warning(availability.reason)
    st.write("La carte bancaire est saisie sur la page sécurisée du prestataire : **AutoDeal ne stocke jamais de numéro de carte ni de CVV**.")
    st.write("Moyens prévus : **CB tunisienne/internationale** et **e-Dinar**. D17 peut servir côté utilisateur lorsqu'il permet le paiement marchand/e-Dinar, mais AutoDeal ne suppose pas un prélèvement D17 automatique chaque mois.")
    st.caption("L'accès premium est synchronisé avec le paiement : à l'expiration de la période réglée ou en cas d'impayé, l'espace Pro/Business est verrouillé jusqu'au prochain paiement confirmé.")
