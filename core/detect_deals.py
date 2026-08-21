"""
detect_deals.py
Filtre le fichier scoré pour ne garder que les opportunités plausibles et
écrit `data/processed/alertes_bonnes_affaires.csv`, consommé par l'app et les
notifications.

Trois garde-fous sont volontairement distincts du score ML :
- une fenêtre de décote configurée dans `config.py` ;
- des exclusions métier (épave, accident, pièces, papiers, export...) ;
- pour les alertes automatiques, une source dont le holdout dépasse 20 % de
  MdAPE est temporairement mise en quarantaine. Les annonces restent visibles
  dans l'application mais ne déclenchent pas de notification automatique.
"""

import json
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
DIAGNOSTICS_FICHIER = "data/processed/diagnostics_modele.json"
SEUIL_MDAPE_SOURCE_POUR_ALERTE = 0.20
MIN_TEST_SOURCE = 30

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
    """True pour les annonces à écarter des affaires."""
    t = titres.fillna("").map(lambda s: _sans_accents(s).lower())
    return t.str.contains(MOTIFS_EXCLUSION, regex=True, na=False)


def sources_trop_risquees(diagnostics: dict | None = None) -> set[str]:
    """Sources à ne pas notifier automatiquement selon le source-holdout.

    On exige au moins 30 lignes de test pour éviter de mettre une source en
    quarantaine sur un échantillon anecdotique. Si le diagnostic est absent,
    aucune source n'est bloquée : le filtre des comparables reste actif.
    """
    if diagnostics is None:
        try:
            diagnostics = json.loads(Path(DIAGNOSTICS_FICHIER).read_text(encoding="utf-8"))
        except Exception:
            return set()
    rows = (
        diagnostics.get("validation_robuste", {}).get("source_holdout", [])
        if isinstance(diagnostics, dict)
        else []
    )
    return {
        str(row.get("source"))
        for row in rows
        if int(row.get("n_test") or 0) >= MIN_TEST_SOURCE
        and float(row.get("mdape") or 0) > SEUIL_MDAPE_SOURCE_POUR_ALERTE
    }


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
        source_ok = pd.Series(True, index=deals.index)
        risquees = sources_trop_risquees()
        if risquees and "Source" in deals.columns:
            source_ok = ~deals["Source"].astype(str).isin(risquees)
            n_bloquees = int((~source_ok).sum())
            print(
                "⚠️ Sources en quarantaine pour notifications automatiques "
                f"(source-holdout > {SEUIL_MDAPE_SOURCE_POUR_ALERTE:.0%}) : "
                + ", ".join(sorted(risquees))
                + f" — {n_bloquees} deal(s) gardé(s) dans l'app sans notification."
            )
        deals["Alerte_Telegram"] = (
            (deals["Nb_Comparables"] >= COMPARABLES_MIN_POUR_ALERTE)
            & source_ok
        )
        nb_solides = int(deals["Alerte_Telegram"].sum())
        print(
            f"   {nb_solides} opportunité(s) autorisée(s) pour alerte automatique "
            f"(≥ {COMPARABLES_MIN_POUR_ALERTE} comparables + source fiable)."
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
