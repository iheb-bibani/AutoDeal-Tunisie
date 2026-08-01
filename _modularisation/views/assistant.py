"""Page assistant."""
import re
import unicodedata
import numpy as np
import pandas as pd
import streamlit as st

from services.model_service import (_ligne_modele, table_fiabilite_prix, erreur_pour_prix)
from services.analytics_service import _match_gouvernorat


def page_assistant(df, bundle):
    st.title("🤖 Assistant")
    st.caption("Pose ta question en langage naturel. L'assistant s'appuie sur TON modèle "
               "de prix et TES données scorées — pas un chatbot générique.")
    if df is None or df.empty:
        st.info("Données indisponibles.")
        return

    exemples = ("Exemples : « prix d'une Volkswagen Golf 2018 à 120000 km » · "
                "« affaires Peugeot sous 30000 DT à Sfax » · « quel gouvernorat est le moins cher »")
    st.caption(exemples)
    q = st.text_input("Ta question", placeholder="prix d'une Golf 2018 120000 km …")
    if not q:
        return

    qn = "".join(c for c in unicodedata.normalize("NFD", q.lower())
                 if unicodedata.category(c) != "Mn")

    # Extraction d'entités
    marques = sorted(df["Marque"].dropna().unique(), key=len, reverse=True)
    marque = next((m for m in marques
                   if "".join(c for c in unicodedata.normalize("NFD", str(m).lower())
                              if unicodedata.category(c) != "Mn") in qn), None)
    gouv = _match_gouvernorat(qn)
    annee = next((int(x) for x in re.findall(r"\b(19\d{2}|20\d{2})\b", qn)), None)
    nombres = [int(n.replace(" ", "")) for n in re.findall(r"\d[\d\s]{2,}", qn)]
    budget = next((n for n in sorted(nombres, reverse=True) if n >= 3000), None)
    km = next((n for n in nombres if 1000 <= n <= 500000 and n != budget), None)
    veut_affaires = any(w in qn for w in ["affaire", "opportunit", "deal", "bon plan", "moins cher que"])
    veut_prix = any(w in qn for w in ["prix", "combien", "estim", "vaut", "cote"])

    # Intention 1 : estimer un prix (modèle)
    if veut_prix and marque and not veut_affaires and bundle is not None:
        sous = df[df["Marque"] == marque]
        modele = None
        for md in sorted(sous["Modèle"].dropna().unique(), key=len, reverse=True):
            if "".join(c for c in unicodedata.normalize("NFD", str(md).lower())
                       if unicodedata.category(c) != "Mn") in qn:
                modele = md
                break
        annee = annee or int(sous["Année"].median()) if sous["Année"].notna().any() else 2018
        saisie = {
            "Kilométrage": km or 100000, "Age_Vehicule": max(pd.Timestamp.now().year - annee, 0),
            "Puissance_Fiscale": sous["Puissance_Fiscale"].median() if "Puissance_Fiscale" in sous else 6,
            "Segment_Vehicule": int(sous["Segment_Vehicule"].median()) if "Segment_Vehicule" in sous else 0,
            "Zone_Economique": 1 if gouv in ("Tunis", "Ariana", "Ben Arous", "Manouba") else 0,
            "Cylindree": sous["Cylindree"].median() if "Cylindree" in sous else np.nan,
            "Marque": marque, "Modèle": modele or "Autre",
            "Energie": (sous["Energie"].mode().iloc[0] if sous["Energie"].notna().any() else "Essence"),
            "Transmission": np.nan,
        }
        try:
            prix = float(np.expm1(bundle["pipeline"].predict(_ligne_modele(bundle, saisie))[0]))
            gr = table_fiabilite_prix(df)
            err = erreur_pour_prix(gr, prix) or 12
            st.success(f"**{marque} {modele or ''} {annee}**, {km or 100000:,} km".replace(",", " "))
            st.metric("Prix estimé par le modèle", f"{prix:,.0f} DT".replace(",", " "))
            st.caption(f"Fourchette ± {err:.0f} % : {prix*(1-err/100):,.0f} – {prix*(1+err/100):,.0f} DT."
                       .replace(",", " "))
            st.caption("Pour la décomposition « pourquoi ce prix » (SHAP) et les annonces "
                       "comparables réelles, ouvre le Calculateur :")
            if st.button("💰 Ouvrir dans le Calculateur", key="assist_to_calc"):
                st.session_state.page = "💰 Calculateur"
                st.rerun()
        except Exception:
            st.warning("Je n'ai pas pu estimer ce véhicule précis. Essaie avec marque + modèle + année.")
        return

    # Intention 2 : trouver des affaires
    if veut_affaires or budget or gouv or marque:
        res = df.copy()
        if marque:
            res = res[res["Marque"] == marque]
        if gouv and "Localisation" in res.columns:
            res = res[res["Localisation"].map(_match_gouvernorat) == gouv]
        if budget:
            res = res[res["Prix"] <= budget]
        if veut_affaires:
            res = res[res["Score_Opportunite"].between(0.25, 0.55)]
        res = res.sort_values("Score_Opportunite", ascending=False).head(20)
        crit = ", ".join(filter(None, [marque, f"≤ {budget:,} DT".replace(",", " ") if budget else None,
                                       gouv, "opportunités" if veut_affaires else None])) or "tout le marché"
        st.markdown(f"**{len(res)} résultat(s)** — {crit}")
        if len(res):
            aff = res.copy()
            aff["Écart"] = (aff["Score_Opportunite"] * 100).round(0)
            cols = [c for c in ["Marque", "Modèle", "Année", "Kilométrage", "Prix",
                                "Prix_Theorique", "Écart", "Localisation", "Lien"] if c in aff.columns]
            st.dataframe(aff[cols], hide_index=True, width="stretch",
                         column_config={
                             "Prix": st.column_config.NumberColumn("Prix", format="%d DT"),
                             "Prix_Theorique": st.column_config.NumberColumn("Estimé", format="%d DT"),
                             "Écart": st.column_config.NumberColumn("Écart", format="%d %%"),
                             "Lien": st.column_config.LinkColumn("Annonce", display_text="ouvrir"),
                         })
        else:
            st.info("Aucune annonce ne correspond. Élargis le budget ou la région.")
        return

    st.info("Je n'ai pas saisi la demande. Précise une marque, un budget, une région, "
            "ou demande un prix (« prix d'une Clio 2019 »).")
