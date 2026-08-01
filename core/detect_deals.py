"""
detect_deals.py
Filtre data/tunisia-cars-scored.csv (produit par modele_prediction.py) pour
n'garder que les vraies opportunités, et écrit data/alertes_bonnes_affaires.csv
-- le fichier que lisent utils/send_telegram.py et app.py.

Ce script remplace l'ancienne version qui lisait "data/voitures_clean.csv"
(un fichier qui n'existe nulle part dans le pipeline réel) avec des noms de
colonnes d'une génération antérieure du schéma (Prix_DT, Date_Detection).

Point important ajouté ici, absent de toutes les versions précédentes :
un PLAFOND de plausibilité. Sur les vraies données, les "meilleures
opportunités" à >60% sous le prix théorique sont presque toujours des erreurs
(prix mal saisi, annonce pour pièces détachées ayant échappé au filtre de
bruit) plutôt que de vraies affaires -- les envoyer sur Telegram ferait plus
de mal que de bien à la confiance dans l'outil.
"""

import os
import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

# Seuils importés depuis config (source unique). Le détail du pourquoi
# (SEUIL_MIN=0.25 ≈ 2x l'erreur relative médiane du modèle) est documenté
# dans config.py.
from config import (
    SEUIL_DEAL_MIN as SEUIL_MIN,
    SEUIL_DEAL_MAX as SEUIL_MAX,
    COMPARABLES_MIN_POUR_ALERTE,
)

IN_FICHIER = "data/processed/tunisia-cars-scored.csv"
FICHIER_ALERTES = "data/processed/alertes_bonnes_affaires.csv"

# Règles métier d'exclusion : ces annonces ne doivent JAMAIS être présentées
# comme des affaires, quel que soit leur prix. Une voiture accidentée / en
# pièces / au moteur HS EST légitimement moins chère -- ce n'est pas une bonne
# affaire, et la signaler détruit la crédibilité de l'outil. Le signal est dans
# le titre (les titres tunisiens sont courts, donc le rappel est faible ~1%,
# mais le coût est nul et ça élimine les cas les plus embarrassants).
# Insensible aux accents et à la casse.
MOTIFS_EXCLUSION = (
    r"accident|pour piece|pieces detach|a piece|pr piece|"
    r"moteur hs|boite hs|moteur a refaire|moteur fatigue|"
    r"sans papier|sans carte|non dedouan|"
    r"\bexport\b|dedouan|"
    r"a refaire|endommag|epave|pour ferrailleur"
)


def _sans_accents(s):
    return "".join(c for c in unicodedata.normalize("NFD", str(s)) if unicodedata.category(c) != "Mn")


def annonces_exclues(titres: pd.Series) -> pd.Series:
    """True pour les annonces à écarter des affaires (accidentée, pièces, HS,
    sans papiers, export...). Basé sur le titre, insensible aux accents/casse."""
    t = titres.fillna("").map(lambda s: _sans_accents(s).lower())
    return t.str.contains(MOTIFS_EXCLUSION, regex=True, na=False)


def calculer_argus_et_liquidite():
    if not os.path.exists(IN_FICHIER):
        print(f"❌ Fichier introuvable : {IN_FICHIER} (lance merging_files.py -> nettoyer_base.py -> enrichir_base_avance.py -> modele_prediction.py avant)")
        return

    df = pd.read_csv(IN_FICHIER, sep=";", encoding="utf-8-sig")
    print(f"📊 {len(df)} annonces scorées chargées depuis {IN_FICHIER}")

    if df.empty:
        print("⚠️ Fichier vide -- rien à filtrer.")
        return

    # Règle métier d'exclusion, AVANT toute détection d'affaire.
    if "Titre" in df.columns:
        exclues = annonces_exclues(df["Titre"])
        if exclues.any():
            print(f"🚫 {int(exclues.sum())} annonce(s) écartée(s) par règle métier "
                  f"(accidentée, pièces, HS, sans papiers, export).")
        df = df[~exclues]

    deals = df[(df["Score_Opportunite"] >= SEUIL_MIN) & (df["Score_Opportunite"] <= SEUIL_MAX)]

    exclues_trop_belles = (df["Score_Opportunite"] > SEUIL_MAX).sum()
    if exclues_trop_belles:
        print(f"⚠️ {exclues_trop_belles} annonces avec un score > {SEUIL_MAX:.0%} écartées "
              f"(quasi certainement des erreurs de prix, pas de vraies affaires).")

    print(f"🔎 {len(deals)} opportunité(s) retenue(s) entre {SEUIL_MIN:.0%} et {SEUIL_MAX:.0%} sous le prix théorique.")

    if "Nb_Comparables" in deals.columns:
        deals = deals.copy()
        deals["Alerte_Telegram"] = deals["Nb_Comparables"] >= COMPARABLES_MIN_POUR_ALERTE
        nb_solides = int(deals["Alerte_Telegram"].sum())
        print(f"   dont {nb_solides} appuyée(s) sur au moins {COMPARABLES_MIN_POUR_ALERTE} annonces "
              f"comparables (les seules qui déclencheront une alerte Telegram) ; "
              f"{len(deals) - nb_solides} reposent sur trop peu de comparables pour être fiables.")
        # Les plus fiables en premier, puis le gain décroissant
        deals = deals.sort_values(["Alerte_Telegram", "Score_Opportunite"], ascending=[False, False])

    # Toujours écrire le fichier, même vide (avec les en-têtes) : sinon
    # l'ancien fichier reste en place et l'app + Telegram continuent
    # d'afficher/envoyer des opportunités périmées d'un scraping précédent.
    if "Alerte_Telegram" not in deals.columns:
        deals = deals.sort_values("Score_Opportunite", ascending=False)
    deals.to_csv(FICHIER_ALERTES, index=False, sep=";", encoding="utf-8-sig")
    if len(deals):
        print(f"✅ Fichier écrit : {FICHIER_ALERTES}")
    else:
        print(f"⚠️ Aucun deal dans la fourchette retenue -> {FICHIER_ALERTES} écrit vide (en-têtes seuls).")


if __name__ == "__main__":
    calculer_argus_et_liquidite()