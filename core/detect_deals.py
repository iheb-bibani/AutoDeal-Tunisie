"""
detect_deals.py
Filtre le fichier scoré pour ne garder que les opportunités plausibles et
écrit `data/processed/alertes_bonnes_affaires.csv`, consommé par l'app et les
notifications.

Deux garde-fous sont volontairement distincts du modèle ML :
- une fenêtre de décote configurée dans `config.py` ;
- des exclusions métier (épave, accident, pièces, papiers, export...).

Le plafond de décote évite de présenter comme « affaire » une valeur tellement
extrême qu'elle est plus probablement due à une erreur de saisie ou à un
véhicule non comparable.
"""

import os
import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

from config import (
    COMPARABLES_MIN_POUR_ALERTE,
    SEUIL_DEAL_MAX as SEUIL_MAX,
    SEUIL_DEAL_MIN as SEUIL_MIN,
)

IN_FICHIER = "data/processed/tunisia-cars-scored.csv"
FICHIER_ALERTES = "data/processed/alertes_bonnes_affaires.csv"

# Insensible aux accents/casse après normalisation. Important : on n'exclut PAS
# le mot « dédouanée » tout seul. Une voiture correctement dédouanée est une
# annonce valide ; seules les formulations négatives (non/pas/sans dédouanement)
# sont incompatibles avec une opportunité standard.
MOTIFS_EXCLUSION = (
    r"accident|pour piece|pieces detach|a piece|pr piece|"
    r"moteur hs|boite hs|moteur a refaire|moteur fatigue|"
    r"sans papier|sans carte|"
    r"(?:non|pas)\s+dedouan|sans\s+dedouan|"
    r"\bexport\b|"
    r"a refaire|endommag|epave|pour ferrailleur"
)


def _sans_accents(s):
    return "".join(
        c for c in unicodedata.normalize("NFD", str(s))
        if unicodedata.category(c) != "Mn"
    )


def annonces_exclues(titres: pd.Series) -> pd.Series:
    """True pour les annonces à écarter des affaires.

    La fonction ne juge pas l'état d'une voiture à partir du prix : elle ne
    bloque que les formulations explicitement incompatibles avec une voiture
    standard roulante/immatriculable.
    """
    t = titres.fillna("").map(lambda s: _sans_accents(s).lower())
    return t.str.contains(MOTIFS_EXCLUSION, regex=True, na=False)


def calculer_argus_et_liquidite():
    if not os.path.exists(IN_FICHIER):
        print(
            f"❌ Fichier introuvable : {IN_FICHIER} "
            "(lance merging_files.py -> nettoyer_base.py -> "
            "enrichir_base_avance.py -> modele_prediction.py avant)"
        )
        return

    df = pd.read_csv(IN_FICHIER, sep=";", encoding="utf-8-sig")
    print(f"📊 {len(df)} annonces scorées chargées depuis {IN_FICHIER}")

    if df.empty:
        print("⚠️ Fichier vide -- rien à filtrer.")
        return

    if "Titre" in df.columns:
        exclues = annonces_exclues(df["Titre"])
        if exclues.any():
            print(
                f"🚫 {int(exclues.sum())} annonce(s) écartée(s) par règle métier "
                "(accidentée, pièces, HS, papiers, export)."
            )
        df = df[~exclues]

    deals = df[
        (df["Score_Opportunite"] >= SEUIL_MIN)
        & (df["Score_Opportunite"] <= SEUIL_MAX)
    ]

    exclues_trop_belles = (df["Score_Opportunite"] > SEUIL_MAX).sum()
    if exclues_trop_belles:
        print(
            f"⚠️ {exclues_trop_belles} annonces avec un score > {SEUIL_MAX:.0%} "
            "écartées (signal trop extrême pour une alerte automatique)."
        )

    print(
        f"🔎 {len(deals)} opportunité(s) retenue(s) entre {SEUIL_MIN:.0%} "
        f"et {SEUIL_MAX:.0%} sous le prix théorique."
    )

    if "Nb_Comparables" in deals.columns:
        deals = deals.copy()
        deals["Alerte_Telegram"] = (
            deals["Nb_Comparables"] >= COMPARABLES_MIN_POUR_ALERTE
        )
        nb_solides = int(deals["Alerte_Telegram"].sum())
        print(
            f"   dont {nb_solides} appuyée(s) sur au moins "
            f"{COMPARABLES_MIN_POUR_ALERTE} annonces comparables ; "
            f"{len(deals) - nb_solides} restent visibles mais ne déclenchent "
            "pas d'alerte Telegram."
        )
        deals = deals.sort_values(
            ["Alerte_Telegram", "Score_Opportunite"],
            ascending=[False, False],
        )

    # Toujours écrire le fichier, même vide : sinon un ancien fichier de deals
    # resterait visible après un run sans opportunité.
    if "Alerte_Telegram" not in deals.columns:
        deals = deals.sort_values("Score_Opportunite", ascending=False)
    deals.to_csv(FICHIER_ALERTES, index=False, sep=";", encoding="utf-8-sig")
    if len(deals):
        print(f"✅ Fichier écrit : {FICHIER_ALERTES}")
    else:
        print(
            f"⚠️ Aucun deal dans la fourchette retenue -> {FICHIER_ALERTES} "
            "écrit vide (en-têtes seuls)."
        )


if __name__ == "__main__":
    calculer_argus_et_liquidite()
