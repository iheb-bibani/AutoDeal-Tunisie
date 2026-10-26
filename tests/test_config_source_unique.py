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
    import core.merging_files as mf
    import core.nettoyer_base as n
    import core.regles_texte as rt
    # Fusion et nettoyage utilisent LE MÊME filtre, construit depuis config.
    assert n.masque_bruit is rt.masque_bruit
    assert mf.masque_bruit is rt.masque_bruit
    assert config.MOTS_BRUIT == config.MOTS_BRUIT_PARTOUT + config.MOTS_BRUIT_TITRE_COURT


def test_bruit_en_mot_entier_et_pieces_sur_titre_court():
    import pandas as pd
    from core.regles_texte import masque_bruit

    titres = pd.Series([
        "Location voiture",
        "PIÈCES VOITURE",
        "Clio 4 sécurité et électricité ok",
        "Honda City 2021 69 mille km (entretien maison Honda), jantes alu, sièges cuir",
        "terrain 500 m²",
    ])
    assert masque_bruit(titres).tolist() == [True, True, False, False, True]


def test_negations_accident():
    import pandas as pd
    from core.regles_texte import masques_accident

    reelle, saine = masques_accident(pd.Series(["jamais été accidentée", "voiture accidentée", "zéro accident"]))
    assert reelle.tolist() == [False, True, False]
    assert saine.tolist() == [True, False, True]


def test_modele_ne_confond_pas_annee_et_modele():
    from core.nettoyer_base import construire_devineur_modele

    deviner = construire_devineur_modele({"Peugeot": ["301", "2008", "3008"]})
    assert deviner("Peugeot", "Peugeot 301 2008 diesel") == "301"
    assert deviner("Peugeot", "Peugeot 2008 GT line") == "2008"


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
