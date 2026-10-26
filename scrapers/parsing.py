"""Parsing robuste des valeurs numériques affichées par les sites tunisiens.

Fonctions pures, sans réseau, testées dans tests/test_scraper_parsing.py.

Pourquoi : concaténer tous les chiffres d'un libellé (ancienne méthode)
donnait des prix faux d'un facteur 1000 ou plus :
- "35.000,000 DT" (millimes affichés)  -> 35000000 au lieu de 35000
- "42 mille DT" / "42 MD"              -> 42 au lieu de 42000
- un numéro de téléphone collé au prix -> 52149505
"""
from __future__ import annotations

import re
import unicodedata

PRIX_MIN_PLAUSIBLE = 500
PRIX_MAX_PLAUSIBLE = 2_000_000

_ESPACES = "    "
# Nombre avec séparateurs de milliers (espace, point, virgule) et éventuelle
# partie décimale de 1 à 3 chiffres. Jamais de saut de ligne dans un nombre.
_NOMBRE = rf"\d{{1,3}}(?:[{_ESPACES}.,]\d{{3}})+(?:[.,]\d{{1,3}})?|\d+(?:[.,]\d{{1,3}})?"
_UNITE_MILLE = r"(?:mille|milles|md|mdt|k)\b"
_DEVISE = r"(?:dt|tnd|dinars?)\b"


def _sans_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")


def _vers_nombre(brut: str) -> float | None:
    """'35.000,000' -> 35000 ; '35 000' -> 35000 ; '35,5' -> 35.5 ; '9,000' -> 9000."""
    s = brut.strip()
    for e in _ESPACES:
        s = s.replace(e, " ")
    s = s.replace(" ", "")
    if not s:
        return None
    seps = [c for c in s if c in ".,"]
    if not seps:
        return float(s)
    groupes = re.split(r"[.,]", s)
    # Deux séparateurs différents : le dernier est décimal ("35.000,000").
    if len(set(seps)) == 2:
        entier = "".join(groupes[:-1])
        return float(f"{entier}.{groupes[-1]}")
    # Un seul type de séparateur répété -> milliers ("1.250.000").
    if len(seps) > 1:
        return float("".join(groupes))
    # Un seul séparateur : groupe final de 3 chiffres = milliers ("35.000",
    # "9,000") ; sinon décimal ("35,5").
    if len(groupes[-1]) == 3:
        return float("".join(groupes))
    return float(f"{groupes[0]}.{groupes[1]}")


def parser_prix_tnd(texte) -> int | None:
    """Prix en dinars (entier) depuis un libellé libre, ou None si non fiable."""
    if texte is None:
        return None
    t = _sans_accents(str(texte)).lower()
    if not t.strip():
        return None
    # Priorité au nombre explicitement suivi d'une unité monétaire.
    m = re.search(rf"({_NOMBRE})\s*(?:{_UNITE_MILLE}\s*)?(?:{_DEVISE})", t) or re.search(
        rf"({_NOMBRE})\s*{_UNITE_MILLE}", t
    )
    if m is None:
        m = re.search(rf"({_NOMBRE})", t)
    if m is None:
        return None
    valeur = _vers_nombre(m.group(1))
    if valeur is None:
        return None
    suite = t[m.end(1): m.end(1) + 12]
    if re.match(rf"\s*{_UNITE_MILLE}", suite) or (valeur < 1000 and re.match(r"\s*(?:m\b|mille)", suite)):
        valeur *= 1000
    # Millimes concaténés (35000000 pour 35 000 DT) : division par 1000
    # seulement si le résultat redevient plausible et que la valeur finit par 000.
    if valeur > PRIX_MAX_PLAUSIBLE and float(valeur).is_integer() and int(valeur) % 1000 == 0:
        valeur /= 1000
    valeur = int(round(valeur))
    if not (PRIX_MIN_PLAUSIBLE <= valeur <= PRIX_MAX_PLAUSIBLE):
        return None
    return valeur


def parser_entier(texte, *, maxi: int | None = None) -> int | None:
    """Premier nombre entier d'un libellé ('9,000 km' -> 9000, '120 000 km' -> 120000).

    Ne traverse jamais un saut de ligne ("2020\\n15000 km" -> 2020).
    """
    if texte is None:
        return None
    m = re.search(rf"({_NOMBRE})", str(texte))
    if m is None:
        return None
    v = _vers_nombre(m.group(1))
    if v is None:
        return None
    v = int(round(v))
    if maxi is not None and v > maxi:
        return None
    return v


def parser_annee(texte) -> int | None:
    """'03/2019' -> 2019 ; 'Modèle 2015' -> 2015."""
    if texte is None:
        return None
    m = re.search(r"\b(19[89]\d|20\d\d)\b", str(texte))
    return int(m.group(1)) if m else None
