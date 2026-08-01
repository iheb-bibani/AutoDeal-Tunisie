"""
Non-régression du bug #1 : config.py se présente comme la source unique de
vérité, mais ses constantes avaient divergé de celles réellement utilisées
(features du modèle, seuils de deals). Ces tests échouent dès qu'une valeur
recommence à être redéfinie localement au lieu d'être importée de config.
"""
import os

import joblib
import pandas as pd
import pytest

import config
from config import MODEL_PATH, PROCESSED_FILES


def test_detect_deals_utilise_les_seuils_de_config():
    import core.detect_deals as d
    assert d.SEUIL_MIN == config.SEUIL_DEAL_MIN
    assert d.SEUIL_MAX == config.SEUIL_DEAL_MAX
    assert d.COMPARABLES_MIN_POUR_ALERTE == config.COMPARABLES_MIN_POUR_ALERTE


def test_modele_utilise_les_features_de_config():
    import core.modele_prediction as m
    # Doivent être littéralement les listes de config, pas des copies locales
    assert m.FEATURES_NUMERIQUES is config.FEATURES_NUMERIQUES
    assert m.FEATURES_CATEGORIELLES is config.FEATURES_CATEGORIELLES


def test_mots_bruit_partages_entre_modules():
    import core.nettoyer_base as n
    import core.merging_files  # noqa: F401  (importé pour vérifier qu'il charge)
    assert n.MOTS_BRUIT is config.MOTS_BRUIT


def test_features_config_presentes_dans_le_fichier_enrichi():
    """Le vrai piège du bug #1 : config listait des features absentes du
    fichier (ou en oubliait). On vérifie que chaque feature déclarée existe
    bien comme colonne produite en amont."""
    chemin = PROCESSED_FILES["enriched"]
    if not os.path.exists(chemin):
        pytest.skip("fichier enrichi absent (pipeline non exécuté)")
    colonnes = set(pd.read_csv(chemin, sep=";", encoding="utf-8-sig", nrows=1).columns)
    attendues = set(config.FEATURES_NUMERIQUES + config.FEATURES_CATEGORIELLES)
    manquantes = attendues - colonnes
    assert not manquantes, f"features déclarées mais absentes des données : {manquantes}"


def test_features_config_coherentes_avec_le_modele_sauvegarde():
    """Le modèle picklé embarque la liste de features utilisée à
    l'entraînement. Elle doit correspondre à config, sinon l'app et le
    pipeline ne parlent plus des mêmes colonnes."""
    if not os.path.exists(MODEL_PATH):
        pytest.skip("modèle sauvegardé absent")
    bundle = joblib.load(MODEL_PATH)
    assert bundle["features_numeriques"] == config.FEATURES_NUMERIQUES
    assert bundle["features_categorielles"] == config.FEATURES_CATEGORIELLES
