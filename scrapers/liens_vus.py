"""Liens réellement observés par chaque scraper lors du run courant.

Les fichiers bruts (data/raw/*.csv) sont CUMULATIFS : un scraper n'ajoute que
les annonces nouvelles et saute celles déjà connues. Ils ne disent donc pas
quelles annonces sont encore en ligne aujourd'hui. Avant ce module, le suivi
(core/suivi_annonces.py) déduisait les annonces « vues aujourd'hui » du
fichier fusionné… qui contient tout l'historique : aucune annonce ne pouvait
jamais disparaître, et la validation longitudinale ne se déclenchait pas.

Chaque scraper enregistre ici l'ensemble des liens vus sur ses pages de
liste (y compris ceux déjà présents dans son CSV), avec le prix affiché quand
il est disponible. Le fichier est daté : un fichier d'un autre jour est ignoré.
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from core.dates import aujourd_hui

DOSSIER_VUS = Path(__file__).resolve().parents[1] / "data" / "raw" / "vus"

# Préfixe du champ "Source" des annonces fusionnées, par clé de scraper.
SOURCE_LABELS = {
    "tayara": "tayara.tn",
    "automobile": "automobile.tn",
    "automax": "AutoMax",
    "sayyarat": "sayyaratn.com",
    "autocentral": "autocentral.tn",
}


def enregistrer_liens_vus(
    source: str,
    liens_prix: dict[str, object],
    dossier: Path = DOSSIER_VUS,
    *,
    complet: bool = True,
    nouvelles: int | None = None,
    pages: int | None = None,
) -> None:
    """Écrit (atomiquement) les liens vus ce jour par `source`.

    ``liens_prix`` : {lien: prix affiché ou None}.
    ``complet`` : True si la pagination est allée jusqu'au bout du catalogue.
    Un catalogue partiel (plafond de pages, erreur en cours de route) reste
    exploitable par le suivi, mais seulement sur la fenêtre de dates couverte.
    """
    dossier.mkdir(parents=True, exist_ok=True)
    payload = {
        "source": source,
        "date": aujourd_hui(),
        "n": len(liens_prix),
        "complet": bool(complet),
        "nouvelles": nouvelles,
        "pages": pages,
        "liens": {str(k): (None if v is None else str(v)) for k, v in liens_prix.items() if k},
    }
    cible = dossier / f"{source}.json"
    fd, tmp = tempfile.mkstemp(prefix=f".{source}.", suffix=".tmp", dir=dossier)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)
        os.replace(tmp, cible)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def charger_catalogues(date: str | None = None, dossier: Path = DOSSIER_VUS) -> dict[str, dict]:
    """{source: payload complet} pour les fichiers datés de `date` (défaut : aujourd'hui)."""
    date = date or aujourd_hui()
    resultat: dict[str, dict] = {}
    if not dossier.exists():
        return resultat
    for chemin in dossier.glob("*.json"):
        try:
            data = json.loads(chemin.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if data.get("date") == date and isinstance(data.get("liens"), dict):
            data.setdefault("complet", True)
            resultat[str(data.get("source") or chemin.stem)] = data
    return resultat


def charger_liens_vus(date: str | None = None, dossier: Path = DOSSIER_VUS) -> dict[str, dict[str, str | None]]:
    """{source: {lien: prix}} pour les catalogues COMPLETS du jour."""
    return {k: v["liens"] for k, v in charger_catalogues(date, dossier).items() if v.get("complet")}


def resume_catalogues(date: str | None = None, dossier: Path = DOSSIER_VUS) -> dict[str, dict]:
    """Résumé léger (sans les liens) pour le manifeste du pipeline."""
    return {
        k: {"liens_vus": v.get("n"), "complet": v.get("complet"), "nouvelles": v.get("nouvelles"), "pages": v.get("pages")}
        for k, v in charger_catalogues(date, dossier).items()
    }
