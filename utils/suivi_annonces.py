"""
suivi_annonces.py
Suit chaque annonce dans le temps : quand elle est apparue, quand elle a
disparu, combien de temps elle est restée en ligne.

C'est la seule source de vérité du projet. Tout le reste (prix théorique,
score d'opportunité, liquidité) est une prédiction sur des prix DEMANDÉS.
Ici on observe enfin un fait : l'annonce a disparu du site.

Pourquoi pas de requêtes HTTP
-----------------------------
La version précédente (utils/tracker_ventes.py) vérifiait chaque annonce par
une requête HEAD : plusieurs milliers de requêtes par nuit, bloquées par
Cloudflare sur automobile.tn, et un simple incident réseau suffisait à
marquer une annonce « vendue » définitivement.

Les scrapers parcourent déjà l'intégralité du catalogue de chaque site à
chaque exécution. Une annonce présente hier et absente aujourd'hui a donc
été retirée — l'information est déjà là, gratuitement et sans risque de
faux positif réseau.

Ce que « disparue » veut dire, et ne veut pas dire
--------------------------------------------------
Une annonce disparue N'EST PAS forcément une vente. Elle peut avoir été
retirée, avoir expiré, ou été supprimée par le vendeur. Le délai avant
disparition est un *proxy* de vitesse d'écoulement, à traiter comme tel :
il est utile pour comparer des modèles entre eux, pas pour affirmer qu'une
voiture précise s'est vendue.

Garde-fou important
-------------------
Si un scraping échoue à mi-parcours, des centaines d'annonces encore en
ligne sembleraient avoir disparu, et cette erreur serait enregistrée
définitivement. Le script refuse donc de conclure quoi que ce soit quand le
volume du jour s'effondre par rapport à la dernière exécution.
"""

import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from config import PROCESSED_FILES

FICHIER_SUIVI = "data/processed/suivi_annonces.csv"
FICHIER_ALERTES = "data/processed/alertes_bonnes_affaires.csv"

# En dessous de cette proportion du volume précédent, on considère que le
# scraping du jour est incomplet et on ne marque AUCUNE disparition.
SEUIL_VOLUME_SUSPECT = 0.60

COLONNES_SNAPSHOT = ["Source", "Marque", "Modèle", "Année", "Localisation"]


def charger_suivi():
    if os.path.exists(FICHIER_SUIVI):
        suivi = pd.read_csv(FICHIER_SUIVI, sep=";", encoding="utf-8-sig")
        # Tant qu'aucune annonce n'a disparu, Date_Disparition ne contient que
        # des valeurs vides et pandas la relit en float64 -- qui refuse ensuite
        # d'accueillir une date. On force le type texte dès le chargement.
        for col in ["Date_Disparition", "Premiere_Vue", "Derniere_Vue", "Statut"]:
            if col in suivi.columns:
                suivi[col] = suivi[col].astype(object)
        return suivi
    return pd.DataFrame(columns=[
        "Lien", "Source", "Marque", "Modèle", "Année", "Localisation",
        "Prix_Initial", "Prix_Dernier", "Premiere_Vue", "Derniere_Vue",
        "Statut", "Date_Disparition", "Jours_En_Ligne", "Nb_Reapparitions",
    ])


def mettre_a_jour():
    aujourd_hui = datetime.now().strftime("%Y-%m-%d")

    merged = pd.read_csv(PROCESSED_FILES["merged"], sep=";", encoding="utf-8-sig")
    merged = merged.dropna(subset=["Lien"]).drop_duplicates(subset=["Lien"], keep="last")
    liens_du_jour = set(merged["Lien"].astype(str))
    print(f"Annonces vues aujourd'hui : {len(liens_du_jour)}")

    suivi = charger_suivi()
    connus = set(suivi["Lien"].astype(str)) if len(suivi) else set()

    # ---- Garde-fou : scraping manifestement incomplet -------------------
    actifs_avant = int((suivi["Statut"] == "Active").sum()) if len(suivi) else 0
    scraping_suspect = False
    if actifs_avant > 0 and len(liens_du_jour) < actifs_avant * SEUIL_VOLUME_SUSPECT:
        scraping_suspect = True
        print(f"⚠️  Volume du jour ({len(liens_du_jour)}) très inférieur aux "
              f"{actifs_avant} annonces actives connues : scraping probablement "
              "incomplet. Les disparitions ne seront PAS enregistrées cette fois "
              "(les nouvelles annonces le sont normalement).")

    # ---- Annonces déjà suivies et revues aujourd'hui --------------------
    if len(suivi):
        vue = suivi["Lien"].astype(str).isin(liens_du_jour)
        prix_du_jour = merged.set_index(merged["Lien"].astype(str))["Prix"]
        suivi.loc[vue, "Derniere_Vue"] = aujourd_hui
        suivi.loc[vue, "Prix_Dernier"] = (
            suivi.loc[vue, "Lien"].astype(str).map(prix_du_jour).values
        )
        # Une annonce peut réapparaître (republiée par le vendeur, ou absente
        # d'un scraping partiel) : on la remet active et on compte l'événement.
        revenues = vue & (suivi["Statut"] == "Disparue")
        if revenues.sum():
            suivi.loc[revenues, "Statut"] = "Active"
            suivi.loc[revenues, "Date_Disparition"] = pd.NA
            suivi.loc[revenues, "Jours_En_Ligne"] = pd.NA
            suivi.loc[revenues, "Nb_Reapparitions"] = (
                pd.to_numeric(suivi.loc[revenues, "Nb_Reapparitions"], errors="coerce").fillna(0) + 1
            )
            print(f"↩️  {int(revenues.sum())} annonce(s) réapparue(s) — remises en actif.")

    # ---- Disparitions ---------------------------------------------------
    nb_disparues = 0
    if len(suivi) and not scraping_suspect:
        disparues = (~suivi["Lien"].astype(str).isin(liens_du_jour)) & (suivi["Statut"] == "Active")
        if disparues.sum():
            suivi.loc[disparues, "Statut"] = "Disparue"
            suivi.loc[disparues, "Date_Disparition"] = aujourd_hui
            duree = (
                pd.to_datetime(aujourd_hui)
                - pd.to_datetime(suivi.loc[disparues, "Premiere_Vue"], errors="coerce")
            ).dt.days
            suivi.loc[disparues, "Jours_En_Ligne"] = duree.values
            nb_disparues = int(disparues.sum())

    # ---- Nouvelles annonces --------------------------------------------
    nouveaux = merged[~merged["Lien"].astype(str).isin(connus)].copy()
    if len(nouveaux):
        ajout = pd.DataFrame({
            "Lien": nouveaux["Lien"].astype(str),
            "Prix_Initial": nouveaux["Prix"],
            "Prix_Dernier": nouveaux["Prix"],
            "Premiere_Vue": aujourd_hui,
            "Derniere_Vue": aujourd_hui,
            "Statut": "Active",
            "Date_Disparition": pd.NA,
            "Jours_En_Ligne": pd.NA,
            "Nb_Reapparitions": 0,
        })
        for col in COLONNES_SNAPSHOT:
            ajout[col] = nouveaux[col].values if col in nouveaux.columns else pd.NA
        suivi = pd.concat([suivi, ajout], ignore_index=True)

    # ---- Marquage des annonces signalées comme opportunités -------------
    # C'est le cœur de la validation : on saura si les annonces que le
    # système a signalées disparaissent plus vite que les autres.
    if os.path.exists(FICHIER_ALERTES):
        try:
            alertes = pd.read_csv(FICHIER_ALERTES, sep=";", encoding="utf-8-sig")
            if len(alertes) and "Lien" in alertes.columns:
                if "Etait_Opportunite" not in suivi.columns:
                    suivi["Etait_Opportunite"] = False
                est_deal = suivi["Lien"].astype(str).isin(alertes["Lien"].astype(str))
                suivi.loc[est_deal, "Etait_Opportunite"] = True
        except (pd.errors.EmptyDataError, KeyError):
            pass
    if "Etait_Opportunite" not in suivi.columns:
        suivi["Etait_Opportunite"] = False
    suivi["Etait_Opportunite"] = suivi["Etait_Opportunite"].fillna(False).astype(bool)

    os.makedirs(os.path.dirname(FICHIER_SUIVI), exist_ok=True)
    suivi.to_csv(FICHIER_SUIVI, index=False, sep=";", encoding="utf-8-sig")

    actives = int((suivi["Statut"] == "Active").sum())
    disparues_total = int((suivi["Statut"] == "Disparue").sum())
    print("-" * 30)
    print(f"Nouvelles annonces suivies : {len(nouveaux)}")
    print(f"Disparues aujourd'hui      : {nb_disparues}")
    print(f"Total suivi                : {len(suivi)} ({actives} actives, {disparues_total} disparues)")
    mesurables = pd.to_numeric(suivi["Jours_En_Ligne"], errors="coerce").dropna()
    if len(mesurables) >= 10:
        print(f"Durée médiane en ligne     : {mesurables.median():.0f} jours "
              f"(sur {len(mesurables)} annonces disparues)")
    else:
        print("Durée médiane en ligne     : pas encore assez d'historique "
              "(il faut plusieurs jours de collecte).")
    print(f"Fichier : {FICHIER_SUIVI}")
    print("-" * 30)


if __name__ == "__main__":
    mettre_a_jour()
