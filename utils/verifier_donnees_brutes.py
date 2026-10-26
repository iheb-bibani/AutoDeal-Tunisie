"""Vérifie les CSV bruts avant leur publication par GitHub Actions.

Les fichiers data/raw/*.csv sont cumulatifs et non reconstructibles : un
fichier tronqué (scraper tué au timeout) ou corrompu publié sur main devient
la base de tous les runs suivants. Avant chaque commit automatique, on vérifie
pour chaque fichier modifié :
- qu'il se relit avec pandas ;
- qu'il contient la colonne `Lien` ;
- que son nombre de lignes n'a pas baissé par rapport au dernier commit.

Un fichier invalide est restauré depuis HEAD (`git checkout -- fichier`).
Code retour 0 dans tous les cas : la restauration EST la correction ; le
détail est affiché en annotation GitHub.
"""
from __future__ import annotations

import io
import subprocess
import sys
from pathlib import Path

import pandas as pd

RACINE = Path(__file__).resolve().parents[1]
DOSSIER_RAW = RACINE / "data" / "raw"
TOLERANCE_BAISSE = 0  # lignes ; le dédoublonnage par Lien peut retirer quelques doublons
TOLERANCE_BAISSE_RELATIVE = 0.02


def _lire(contenu: bytes | str | Path) -> pd.DataFrame:
    source = io.BytesIO(contenu) if isinstance(contenu, bytes) else contenu
    return pd.read_csv(source, sep=";", encoding="utf-8-sig", low_memory=False)


def _version_head(chemin_relatif: str) -> bytes | None:
    try:
        return subprocess.run(
            ["git", "show", f"HEAD:{chemin_relatif}"],
            cwd=RACINE, check=True, capture_output=True,
        ).stdout
    except subprocess.CalledProcessError:
        return None  # fichier nouveau


def verifier_fichier(chemin: Path) -> str | None:
    """Renvoie la raison du rejet, ou None si le fichier est publiable."""
    try:
        df = _lire(chemin)
    except Exception as exc:
        return f"illisible ({type(exc).__name__})"
    if "Lien" not in df.columns:
        return "colonne Lien absente"
    ancien = _version_head(chemin.relative_to(RACINE).as_posix())
    if ancien is None:
        return None
    try:
        n_ancien = len(_lire(ancien))
    except Exception:
        return None  # l'ancienne version était déjà illisible : la nouvelle est meilleure
    seuil = n_ancien - max(TOLERANCE_BAISSE, int(n_ancien * TOLERANCE_BAISSE_RELATIVE))
    if len(df) < seuil:
        return f"{len(df)} lignes au lieu de {n_ancien} (fichier tronqué ?)"
    return None


def main() -> int:
    for chemin in sorted(DOSSIER_RAW.glob("*.csv")):
        raison = verifier_fichier(chemin)
        if raison is None:
            print(f"✅ {chemin.name} publiable")
            continue
        print(f"::warning::{chemin.name} non publié : {raison} — restauration de la version précédente.")
        subprocess.run(["git", "checkout", "--", str(chemin)], cwd=RACINE, check=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
