"""
AutoDeal Tunisie — point d'entrée du dashboard.

Découpé en modules :
  - ui/        : thème (couleurs, CSS) et helpers de rendu (Plotly, format).
  - services/  : accès données, modèle de prix, analyses agrégées.
  - views/     : une page par fichier (concessionnaire, samsar, ...).

Note : le dossier des pages s'appelle `views/` et non `pages/`, car `pages/`
est réservé par Streamlit (génération automatique d'une navigation multipage
qui doublonnerait la sidebar personnalisée ci-dessous).
"""
import pandas as pd
import streamlit as st

from ui import theme
from services.data_service import charger_scored, charger_deals, LIRE_DEPUIS_GITHUB, DUREE_CACHE
from services.model_service import charger_modele
from views.concessionnaire import page_marche
from views.samsar import page_samsar
from views.calculateur import page_calculateur
from views.recherche import page_recherche
from views.carte import page_carte
from views.assistant import page_assistant
from views.admin import page_admin

theme.configurer_page()  # DOIT être la première commande Streamlit


def main():
    # Navigation par RÔLE : deux personas métier (Concessionnaire, Samsar),
    # une boîte à outils transverse, et l'espace technique. Bouton actif surligné.
    NAV = {
        "CONCESSIONNAIRE": ["🏢 Concessionnaire", "💰 Calculateur"],
        "SAMSAR": ["🤝 Samsar"],
        "EXPLORER": ["🔎 Recherche", "🗺️ Carte", "🤖 Assistant"],
        "ADMIN": ["🛠️ Admin"],
    }
    if "page" not in st.session_state:
        st.session_state.page = "🏢 Concessionnaire"

    with st.sidebar:
        st.markdown("## 🚗 AutoDeal Tunisie")
        st.caption("Intelligence du marché de l'occasion — automobile.tn · tayara.tn · "
                   "automax.tn · sayyaratn.com")
        for section, pages in NAV.items():
            st.markdown(
                f"<div style='font-size:0.68rem;letter-spacing:0.09em;color:#8A97A3;"
                f"font-weight:700;margin:0.9rem 0 0.15rem'>{section}</div>",
                unsafe_allow_html=True,
            )
            for p in pages:
                actif = st.session_state.page == p
                if st.button(p, key=f"nav_{p}", use_container_width=True,
                             type="primary" if actif else "secondary"):
                    st.session_state.page = p
                    st.rerun()
        page = st.session_state.page
        st.divider()
        df_temp = charger_scored()
        if df_temp is not None and "Annonce-Detectee" in df_temp.columns:
            derniere = df_temp["Annonce-Detectee"].max()
            st.caption(f"Dernière collecte : **{derniere}**")
            st.caption(f"Base : **{len(df_temp):,} annonces**".replace(",", " "))
            try:
                retard = (pd.Timestamp.now().normalize() - pd.to_datetime(derniere)).days
                if retard > 2:
                    st.warning(f"Données vieilles de {retard} jours — le scraping nocturne "
                               "a peut-être échoué. Vérifie l'onglet Actions du dépôt.")
            except Exception:
                pass
        if st.button("🔄 Rafraîchir les données"):
            st.cache_data.clear()
            st.cache_resource.clear()
            st.rerun()
        st.caption(f"Source : {'GitHub' if LIRE_DEPUIS_GITHUB else 'fichiers locaux'} · "
                   f"cache {DUREE_CACHE // 60} min")

    df_scored = charger_scored()
    df_deals = charger_deals()
    bundle = charger_modele()

    if page == "🏢 Concessionnaire":
        page_marche(df_scored)
    elif page == "🔎 Recherche":
        page_recherche(df_scored)
    elif page == "🗺️ Carte":
        page_carte(df_scored)
    elif page == "🤝 Samsar":
        page_samsar(df_scored, df_deals)
    elif page == "💰 Calculateur":
        page_calculateur(df_scored, bundle)
    elif page == "🤖 Assistant":
        page_assistant(df_scored, bundle)
    else:
        page_admin(df_scored, df_deals)


if __name__ == "__main__":
    main()
