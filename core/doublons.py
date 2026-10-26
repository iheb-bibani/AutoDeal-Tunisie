"""Clé de doublon « même véhicule » entre sources.

La fusion déduplique par lien d'annonce ; or la même voiture est souvent
publiée sur plusieurs sites (automobile.tn + AutoMax) ou republiée avec un
nouveau lien. Ces copies ont des liens différents mais les mêmes
caractéristiques et le même prix.

Cette clé sert à :
- garder les copies dans le même fold de validation croisée (sinon le modèle
  « mémorise » leur prix et l'erreur publiée est trop optimiste) ;
- ne pas compter une copie comme comparable indépendant ;
- ne pas notifier deux fois la même affaire.
"""
from __future__ import annotations

import pandas as pd

COLONNES_CLE = ("Marque", "Modèle", "Année", "Kilométrage", "Prix")


def _normaliser(col: pd.Series) -> pd.Series:
    num = pd.to_numeric(col, errors="coerce")
    if num.notna().sum() >= max(1, int(col.notna().sum() * 0.9)):
        # 2019 / 2019.0 / "2019" -> "2019" ; NaN -> "?"
        return num.round().astype("Int64").astype(str).replace("<NA>", "?")
    return col.astype(str).str.strip().str.casefold().replace({"nan": "?", "none": "?", "": "?"})


def cle_doublon(df: pd.DataFrame, colonnes=COLONNES_CLE) -> pd.Series:
    """Série (même index que ``df``) identifiant un même véhicule.

    Une ligne sans prix ni kilométrage reçoit une clé unique (son index) :
    l'absence d'information ne doit pas fusionner des voitures différentes.
    """
    presentes = [c for c in colonnes if c in df.columns]
    if not presentes:
        return pd.Series(df.index.astype(str), index=df.index)
    parts = [_normaliser(df[c]) for c in presentes]
    cle = parts[0]
    for p in parts[1:]:
        cle = cle + "|" + p
    info_faible = pd.Series(True, index=df.index)
    for c in ("Prix", "Kilométrage"):
        if c in df.columns:
            info_faible &= pd.to_numeric(df[c], errors="coerce").isna()
    return cle.where(~info_faible, "idx:" + df.index.astype(str))
