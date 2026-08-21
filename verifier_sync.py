"""Contrôles rapides de cohérence du dépôt AutoDeal Tunisie.

À lancer depuis la racine avec ``python verifier_sync.py``. Ce script ne
modifie rien : il vérifie les contrats importants entre configuration, modèle,
pipeline, tests et documentation sans figer des valeurs historiques devenues
fausses.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

import config

ROOT = Path(__file__).resolve().parent


def lire(chemin: str) -> str:
    try:
        return (ROOT / chemin).read_text(encoding="utf-8")
    except Exception:
        return ""


def requirements_version(package: str) -> str | None:
    """Retourne la version ``==`` d'un package si elle est explicitement figée."""
    match = re.search(
        rf"(?mi)^\s*{re.escape(package)}\s*==\s*([^\s#]+)",
        lire("requirements.txt"),
    )
    return match.group(1) if match else None


def modele_et_config_coherents() -> bool:
    """Vérifie le contrat de features du pickle si le modèle est présent."""
    chemin = Path(config.MODEL_PATH)
    if not chemin.exists():
        return True
    try:
        import joblib

        bundle = joblib.load(chemin)
        return (
            bundle.get("features_numeriques") == config.FEATURES_NUMERIQUES
            and bundle.get("features_categorielles") == config.FEATURES_CATEGORIELLES
        )
    except Exception:
        return False


CHECKS = [
    (
        "config.py : features de production déclarées",
        lambda: bool(config.FEATURES_NUMERIQUES) and bool(config.FEATURES_CATEGORIELLES),
        "Les listes FEATURES_NUMERIQUES / FEATURES_CATEGORIELLES sont absentes ou vides.",
    ),
    (
        "modèle pickle cohérent avec config.py",
        modele_et_config_coherents,
        "Réentraîne le modèle : le pickle et config.py n'utilisent pas les mêmes features.",
    ),
    (
        "core/detect_deals.py : règles métier d'exclusion",
        lambda: "MOTIFS_EXCLUSION" in lire("core/detect_deals.py"),
        "La détection des deals n'a plus ses garde-fous accident/pièces/HS.",
    ),
    (
        "core/modele_prediction.py : scoring out-of-fold",
        lambda: "cross_val_predict" in lire("core/modele_prediction.py"),
        "Le scoring publié doit rester hors-échantillon.",
    ),
    (
        "core/modele_prediction.py : diagnostics de généralisation",
        lambda: all(x in lire("core/modele_prediction.py") for x in ["GroupKFold", "time_holdout_20pct", "source_holdout"]),
        "Les stress-tests GroupKFold / temporel / source sont incomplets.",
    ),
    (
        "core/suivi_performance.py : monitoring ML enrichi",
        lambda: all(x in lire("core/suivi_performance.py") for x in ["p90_erreur_pct", "dans_10pct", "biais_median_pct"]),
        "Le monitoring doit conserver P90, couverture et biais en plus du MdAPE.",
    ),
    (
        "requirements.txt : versions ML compatibles avec le pickle",
        lambda: requirements_version("scikit-learn") == "1.8.0" and requirements_version("numpy") == "2.3.5",
        "Le modèle embarqué attend actuellement scikit-learn 1.8.0 et NumPy 2.3.5.",
    ),
    (
        "requirements.txt : SHAP et lifelines",
        lambda: bool(re.search(r"(?mi)^\s*shap\b", lire("requirements.txt")))
        and bool(re.search(r"(?mi)^\s*lifelines\b", lire("requirements.txt"))),
        "Les dépendances d'interprétabilité/validation sont manquantes.",
    ),
    (
        "tests principaux présents",
        lambda: all(
            (ROOT / p).exists()
            for p in [
                "tests/test_exclusion.py",
                "tests/test_suivi.py",
                "tests/test_validation.py",
                "tests/test_access_control.py",
                "tests/test_market_valuation.py",
            ]
        ),
        "Une partie des tests de non-régression a disparu.",
    ),
    (
        "README.md documente installation et architecture",
        lambda: all(x in lire("README.md") for x in ["## Architecture", "## Installation", "## Tests"]),
        "README.md est incomplet ou désynchronisé.",
    ),
    ("LICENSE présent", lambda: (ROOT / "LICENSE").exists(), "Le fichier LICENSE est absent."),
]


def main() -> int:
    print("=" * 64)
    print("Vérification de cohérence — AutoDeal Tunisie")
    print("=" * 64)
    erreurs = []
    for libelle, test, aide in CHECKS:
        try:
            ok = bool(test())
        except Exception as exc:
            ok = False
            aide = f"{aide} ({type(exc).__name__}: {exc})"
        print(f"  {'✅' if ok else '❌'}  {libelle}")
        if not ok:
            erreurs.append((libelle, aide))

    print("-" * 64)
    if not erreurs:
        print("✅ Dépôt cohérent.")
        return 0

    print(f"❌ {len(erreurs)} incohérence(s) détectée(s) :")
    for libelle, aide in erreurs:
        print(f"   • {libelle}\n     → {aide}")
    print("\nAprès correction, relance :")
    print("   python -m pytest tests/ -q")
    print("   python verifier_sync.py")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
