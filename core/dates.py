"""Dates « métier » d'AutoDeal, toujours à l'heure de Tunis.

Le pipeline tourne sur GitHub Actions en UTC et démarre à 23:00 UTC (minuit
à Tunis) : avec datetime.now(), les annonces d'un même run recevaient deux
dates de détection différentes selon qu'elles étaient vues avant ou après
minuit UTC, et l'âge des annonces était décalé d'un jour.
"""
from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd

FUSEAU = ZoneInfo("Africa/Tunis")


def maintenant() -> datetime:
    """Heure locale de Tunis, naïve (compatible avec les dates des CSV)."""
    return datetime.now(FUSEAU).replace(tzinfo=None)


def aujourd_hui() -> str:
    return maintenant().strftime("%Y-%m-%d")


def aujourd_hui_ts() -> pd.Timestamp:
    return pd.Timestamp(maintenant()).normalize()


def annee_courante() -> int:
    return maintenant().year
