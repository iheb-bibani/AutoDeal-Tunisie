"""Règles textuelles partagées (bruit, état du véhicule).

Centralisées pour que la fusion, le nettoyage, l'enrichissement et la
détection des affaires appliquent exactement les mêmes définitions.
"""
from __future__ import annotations

import re
import unicodedata

import pandas as pd

from config import (
    LONGUEUR_TITRE_COURT,
    MARQUES_ALIAS,
    MODELES_PAR_MARQUE_REFERENTIEL,
    MOTS_BRUIT_PARTOUT,
    MOTS_BRUIT_TITRE_COURT,
)


def sans_accents(s) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFD", str(s)) if unicodedata.category(c) != "Mn"
    )


def normaliser(s) -> str:
    """minuscules, sans accents, apostrophe typographique -> ASCII."""
    return sans_accents(s).lower().replace("’", "'")


def _motif_mots(mots) -> str:
    # Frontières "maison" : (?<!\w) / (?!\w) fonctionnent aussi pour "m²".
    # Pluriel simple toléré (pièce -> pièces).
    alts = "|".join(re.escape(normaliser(m).strip()) for m in mots if str(m).strip())
    return rf"(?<!\w)(?:{alts})(?:s|x)?(?!\w)"


MOTIF_BRUIT_PARTOUT = _motif_mots(MOTS_BRUIT_PARTOUT)
MOTIF_BRUIT_TITRE_COURT = _motif_mots(MOTS_BRUIT_TITRE_COURT)


def masque_bruit(titres: pd.Series) -> pd.Series:
    """True pour les titres qui ne sont pas des ventes de voitures."""
    t = titres.fillna("").astype(str).map(normaliser)
    partout = t.str.contains(MOTIF_BRUIT_PARTOUT, regex=True, na=False)
    court = t.str.len() <= LONGUEUR_TITRE_COURT
    pieces = court & t.str.contains(MOTIF_BRUIT_TITRE_COURT, regex=True, na=False)
    return partout | pieces


# « non accidentée », « jamais été accidentée », « zéro accident », « sans
# accident », « aucun accident »… : arguments de vente, pas des défauts.
NEGATION_ACCIDENT = (
    r"(?:non|jamais|aucun\w*|pas|sans|zero|0)\s+(?:(?:ete|eu|d')\s*)*(?:d'\s*)?(?:un\s+)?accident\w*"
)
MOTIF_ACCIDENT = r"accident\w*"


def masques_accident(texte: pd.Series) -> tuple[pd.Series, pd.Series]:
    """(mention_accident_reelle, mention_non_accidentee) sur un texte libre."""
    t = texte.fillna("").astype(str).map(normaliser)
    saine = t.str.contains(NEGATION_ACCIDENT, regex=True, na=False)
    reste = t.str.replace(NEGATION_ACCIDENT, " ", regex=True)
    reelle = reste.str.contains(MOTIF_ACCIDENT, regex=True, na=False)
    return reelle, saine


def _marques_valides() -> dict[str, str]:
    """minuscule -> libellé canonique, à partir du référentiel de config."""
    connues = {k.lower(): v for k, v in MARQUES_ALIAS.items()}
    for v in list(MARQUES_ALIAS.values()) + list(MODELES_PAR_MARQUE_REFERENTIEL):
        connues.setdefault(str(v).lower(), v)
    return connues


MARQUES_VALIDES = _marques_valides()
