"""Prix ACTUEL des annonces (et non le premier prix jamais vu).

Les CSV bruts sont cumulatifs : un scraper n'ajoute que les annonces
nouvelles et ne revisite pas les anciennes. Le prix stocké est donc celui du
premier passage : une baisse de prix du vendeur n'était jamais vue, le score
d'opportunité était calculé sur un prix périmé, et une voiture passée de
52 000 à 44 000 DT n'apparaissait jamais comme affaire.

Sources du prix actuel, par priorité :
1. le prix affiché aujourd'hui sur les pages de liste (scrapers/liens_vus) ;
2. le dernier prix connu du suivi (data/processed/suivi_annonces.csv).
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from scrapers.liens_vus import charger_catalogues
from scrapers.parsing import parser_prix_tnd

FICHIER_SUIVI = "data/processed/suivi_annonces.csv"
# Une variation au-delà de ces bornes est plus probablement une erreur de
# saisie ou de parsing qu'une vraie révision de prix : ignorée.
RATIO_MIN, RATIO_MAX = 0.4, 1.6
SEUIL_BAISSE_SIGNIFICATIVE = 0.03


def prix_du_jour() -> dict[str, float]:
    prix: dict[str, float] = {}
    for cat in charger_catalogues().values():
        for lien, p in cat.get("liens", {}).items():
            v = parser_prix_tnd(f"{p} DT") if p not in (None, "", "N/A") else None
            if v:
                prix[str(lien)] = float(v)
    return prix


def prix_suivi(chemin: str = FICHIER_SUIVI) -> dict[str, float]:
    if not os.path.exists(chemin):
        return {}
    try:
        s = pd.read_csv(chemin, sep=";", encoding="utf-8-sig", usecols=["Lien", "Prix_Dernier"])
    except (ValueError, OSError, pd.errors.ParserError):
        return {}
    s["Prix_Dernier"] = pd.to_numeric(s["Prix_Dernier"], errors="coerce")
    s = s.dropna()
    return dict(zip(s["Lien"].astype(str), s["Prix_Dernier"].astype(float), strict=True))


def appliquer_prix_actuel(df: pd.DataFrame, du_jour: dict | None = None, suivi: dict | None = None) -> pd.DataFrame:
    """Ajoute Prix_Initial, remplace Prix par le prix actuel plausible, et
    calcule Variation_Prix_Pct (négatif = baisse) et Baisse_Prix (bool)."""
    if "Prix" not in df or "Lien" not in df:
        return df
    du_jour = prix_du_jour() if du_jour is None else du_jour
    suivi = prix_suivi() if suivi is None else suivi
    df = df.copy()
    initial = pd.to_numeric(df["Prix"], errors="coerce")
    df["Prix_Initial"] = initial
    liens = df["Lien"].astype(str)
    actuel = liens.map(du_jour).fillna(liens.map(suivi))
    ratio = actuel / initial
    plausible = actuel.notna() & initial.notna() & ratio.between(RATIO_MIN, RATIO_MAX)
    df["Prix"] = initial.where(~plausible, actuel)
    variation = (df["Prix"] - initial) / initial
    df["Variation_Prix_Pct"] = (variation * 100).round(1).where(initial.notna())
    df["Baisse_Prix"] = (variation <= -SEUIL_BAISSE_SIGNIFICATIVE).fillna(False)
    n = int((plausible & (np.abs(variation) > 1e-9)).sum())
    if n:
        print(f"💱 {n} prix mis à jour (dont {int(df['Baisse_Prix'].sum())} baisse(s) ≥ "
              f"{SEUIL_BAISSE_SIGNIFICATIVE:.0%}) d'après les pages de liste et le suivi.")
    return df
