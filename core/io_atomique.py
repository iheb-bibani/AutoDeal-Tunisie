"""Écritures atomiques des fichiers de données.

Écrire directement sur le fichier cible laisse un fichier tronqué si le
process est tué en cours d'écriture (timeout du pipeline, runner GitHub
arrêté) : l'historique de suivi ou le modèle deviennent alors illisibles.
On écrit dans un fichier temporaire du même dossier puis ``os.replace``,
qui est atomique sur un même système de fichiers.
"""
from __future__ import annotations

import json
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def _fichier_temporaire(chemin):
    chemin = Path(chemin)
    chemin.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{chemin.name}.", suffix=".tmp", dir=chemin.parent)
    os.close(fd)
    try:
        yield tmp
        os.chmod(tmp, 0o644)  # mkstemp crée en 0600
        os.replace(tmp, chemin)
    except BaseException:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
        raise


def ecrire_csv_atomique(df, chemin, *, sep=";", encoding="utf-8-sig", index=False, **kwargs) -> None:
    with _fichier_temporaire(chemin) as tmp:
        df.to_csv(tmp, sep=sep, encoding=encoding, index=index, **kwargs)


def ecrire_json_atomique(chemin, data, *, indent=2) -> None:
    with _fichier_temporaire(chemin) as tmp:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=indent, default=str)


def ecrire_texte_atomique(chemin, texte: str) -> None:
    with _fichier_temporaire(chemin) as tmp:
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(texte)


def dump_joblib_atomique(obj, chemin) -> None:
    import joblib

    with _fichier_temporaire(chemin) as tmp:
        joblib.dump(obj, tmp)
