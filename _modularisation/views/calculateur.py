"""Page calculateur."""
import numpy as np
import pandas as pd
import streamlit as st
from config import MAX_DAYS_OLD
import plotly.graph_objects as go

from ui.theme import (C_ASPHALTE, C_GAIN, C_ALERTE)
from ui.charts import style_figure, fmt_dt
from services.data_service import SCORED_PATH
from services.model_service import (_tenter_charger_modele, expliquer_prix, _ligne_modele, table_fiabilite_prix,
    erreur_pour_prix)
from services.analytics_service import MARQUES_LUXE


def page_calculateur(df, bundle):
    st.title("💰 Calculateur de Juste Prix")
    st.caption(f"Estimation par le modèle entraîné sur les annonces récentes (≤ {MAX_DAYS_OLD} jours), "
               "confrontée à des annonces réellement proches (âge, km, énergie).")

    if df is None:
        st.error(
            f"`{SCORED_PATH}` introuvable ou illisible. Lance `python main.py`, "
            "puis pousse `data/` sur GitHub (l'app lit raw.githubusercontent en premier)."
        )
        return
    if bundle is None:
        _, raison = _tenter_charger_modele()
        st.warning("Le modèle n'a pas pu être chargé.")
        if raison:
            st.code(raison, language="text")
        st.caption(
            "Causes fréquentes : (1) version de scikit-learn différente entre "
            "l'entraînement du `.pkl` et l'app (voir `requirements.txt`) — "
            "réentraîne avec `python core/modele_prediction.py` dans le même "
            "environnement, puis `git push` ; (2) modèle pas encore poussé sur GitHub."
        )
        return

    col1, col2, col3 = st.columns(3)
    with col1:
        marques = sorted(df["Marque"].dropna().unique())
        marque = st.selectbox("Marque", marques, index=marques.index("Volkswagen") if "Volkswagen" in marques else 0)
        modeles_m = sorted(df[df["Marque"] == marque]["Modèle"].dropna().astype(str).unique())
        modele = st.selectbox("Modèle", modeles_m) if modeles_m else st.text_input("Modèle")
        annee = st.number_input("Année", min_value=1990, max_value=pd.Timestamp.now().year + 1, value=2020)
    with col2:
        km = st.number_input("Kilométrage", min_value=0, max_value=500000, value=80000, step=5000)
        cv = st.number_input("Puissance fiscale (CV)", min_value=2, max_value=30, value=7)
        zone = st.selectbox("Zone", ["Grand Tunis", "Province"])
    with col3:
        energies = sorted(df["Energie"].dropna().unique())
        energie = st.selectbox("Énergie", energies)
        # Cylindrée optionnelle : renseignée -> utilisée par le modèle ;
        # laissée à 0 -> transmise en NaN. Réduit l'écart de précision entre ce
        # calculateur et le scoring automatique du pipeline.
        cyl = st.number_input("Cylindrée (L) — optionnel", min_value=0.0, max_value=8.0, value=0.0, step=0.1,
                              help="0 = inconnu. Ex : 1.6. Améliore l'estimation si renseigné.")

    segment = "Luxe" if marque in MARQUES_LUXE else "Standard"
    annee_courante = pd.Timestamp.now().year
    presque_neuve = annee >= annee_courante - 1
    st.caption(f"Segment déduit : **{segment}** — Presque neuve : **{'Oui' if presque_neuve else 'Non'}**")

    if st.button("Calculer le prix théorique", type="primary"):
        colonnes = bundle.get("features_numeriques", []) + bundle.get("features_categorielles", [])
        saisie = {
            "Kilométrage": km, "Log_Kilometrage": float(np.log1p(km)),
            "Age_Vehicule": max(annee_courante - annee, 0),
            "Age_Carre": float(max(annee_courante - annee, 0) ** 2),
            "Km_Par_An": float(min(km / max(max(annee_courante - annee, 0), 1), 100000)),
            "Puissance_Fiscale": cv,
            "Segment_Vehicule": int(segment == "Luxe"),
            "Zone_Economique": int(zone == "Grand Tunis"),
            "Marque": marque, "Modèle": modele, "Energie": energie,
            # Cylindrée (optionnelle) : NaN si laissée à 0, sinon la valeur saisie.
            "Cylindree": np.nan if cyl == 0 else float(cyl),
            # Transmission non saisie dans le formulaire.
            "Transmission": np.nan,
        }
        X = _ligne_modele(bundle, saisie)
        try:
            prix_log = bundle["pipeline"].predict(X)[0]
            prix_theorique = float(np.expm1(prix_log))
        except Exception as e:
            st.error(f"Erreur lors de la prédiction : {str(e)[:120]}")
            st.info("Relance `python core/modele_prediction.py` pour réentraîner le modèle "
                    "avec ta version de scikit-learn.")
            return

        if np.isnan(prix_theorique) or prix_theorique <= 0:
            st.error("Le modèle n'a pas pu produire une estimation cohérente pour cette combinaison.")
            return

        # ---- Résultat + comparables réels --------------------------------
        comparables = df[(df["Marque"] == marque) & (df["Modèle"].astype(str) == str(modele))]

        gr_fia = table_fiabilite_prix(df)
        err_pct = erreur_pour_prix(gr_fia, prix_theorique)

        r1, r2, r3 = st.columns(3)
        r1.metric("Prix théorique estimé", fmt_dt(prix_theorique))
        if err_pct is not None:
            bas = prix_theorique * (1 - err_pct / 100)
            haut = prix_theorique * (1 + err_pct / 100)
            niveau = "bonne" if err_pct <= 10 else ("moyenne" if err_pct <= 18 else "faible")
            r1.caption(f"Fourchette ± {err_pct:.0f} % : {fmt_dt(bas)} – {fmt_dt(haut)}  ·  fiabilité {niveau} sur cette gamme")
        if len(comparables) >= 3:
            r2.metric(f"Médiane des {len(comparables)} annonces comparables", fmt_dt(comparables["Prix"].median()))
            r3.metric("Fourchette observée",
                      f"{comparables['Prix'].quantile(.25):,.0f} – {comparables['Prix'].quantile(.75):,.0f} DT".replace(",", " "))

            fig = go.Figure()
            fig.add_trace(go.Box(
                x=comparables["Prix"], name="", marker_color=C_ASPHALTE,
                boxpoints="all", jitter=0.4, pointpos=0,
                hovertemplate="%{x:,.0f} DT<extra></extra>",
            ))
            fig.add_vline(x=prix_theorique, line_color=C_GAIN, line_width=3,
                          annotation_text="Estimation", annotation_font_color=C_GAIN)
            fig.update_layout(title=f"Ton estimation face aux {len(comparables)} annonces "
                                    f"{marque} {modele} du marché")
            fig.update_xaxes(title="Prix (DT)")
            fig.update_yaxes(visible=False)
            st.plotly_chart(style_figure(fig, 280), width="stretch")
        else:
            st.info("Moins de 3 annonces comparables sur le marché récent — l'estimation repose "
                    "surtout sur des véhicules proches, à prendre avec plus de prudence.")

        # ---- Pourquoi ce prix ? (décomposition SHAP ou repli) ------------
        contrib = expliquer_prix(bundle, saisie, df)
        if contrib is not None and len(contrib):
            methode = contrib.attrs.get("methode", "SHAP")
            st.markdown("**Pourquoi ce prix ?**")
            top = contrib.head(6).iloc[::-1]  # plus fort en haut du graphe horizontal
            fig = go.Figure(go.Bar(
                x=top["shap"], y=top["label"], orientation="h",
                marker_color=[C_GAIN if v > 0 else C_ALERTE for v in top["shap"]],
                hovertemplate="%{y}<extra></extra>", showlegend=False,
            ))
            fig.update_layout(title="Ce qui pousse le prix vers le haut (vert) ou le bas (rouge)")
            fig.update_xaxes(title="contribution (échelle log-prix)")
            st.plotly_chart(style_figure(fig, 300), width="stretch")
            methode_txt = ("Décomposition SHAP de cette estimation" if methode == "SHAP"
                           else "Effet marginal de chaque variable (approximation, SHAP indisponible)")
            st.caption(f"{methode_txt} : vert = tire le prix vers le haut, rouge = vers le bas. "
                       "Échelle logarithmique (le modèle prédit le log du prix).")

        st.caption("💡 Une annonce réelle nettement sous ce prix ? Regarde la page Samsar : "
                   "elle y est probablement déjà signalée.")
