"""
Tests de core/nettoyer_base.py -- le module le plus fragile du pipeline
(inférence marque/modèle depuis le titre) et celui qui a historiquement
concentré le plus de bugs. Les entrées sont des titres, les sorties sont
déterministes : c'est le code le plus rentable à tester.
"""
import numpy as np
import pytest

from core.nettoyer_base import (
    normaliser_pour_comparaison,
    est_valeur_chiffre_repete,
    construire_devineur_marque,
    construire_devineur_modele,
)


# ---------------------------------------------------------------------------
# normaliser_pour_comparaison
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("a,b", [
    ("Golf 7", "Golf-7"),
    ("Golf 7", "Golf7"),
    ("Grand i10", "grand-i10"),
    ("C-Elysée", "c elysée"),
])
def test_normalisation_ignore_espaces_et_tirets(a, b):
    assert normaliser_pour_comparaison(a) == normaliser_pour_comparaison(b)


def test_normalisation_distingue_modeles_differents():
    assert normaliser_pour_comparaison("Golf") != normaliser_pour_comparaison("Polo")


# ---------------------------------------------------------------------------
# est_valeur_chiffre_repete (valeurs de remplissage type 11111, 999999)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("valeur", [11111, 99999, 999999, 1111, 22222])
def test_chiffre_repete_detecte(valeur):
    assert est_valeur_chiffre_repete(valeur) is True


@pytest.mark.parametrize("valeur", [
    123456,   # chiffres variés
    100000,   # un vrai kilométrage rond
    5000,     # un vrai prix
    111,      # trois chiffres : trop court pour être un faux flagrant
    0,
])
def test_chiffre_repete_ne_flague_pas_vraies_valeurs(valeur):
    assert est_valeur_chiffre_repete(valeur) is False


def test_chiffre_repete_tolere_nan():
    assert est_valeur_chiffre_repete(np.nan) is False


# ---------------------------------------------------------------------------
# construire_devineur_marque : titre -> marque
# ---------------------------------------------------------------------------

@pytest.fixture
def deviner_marque():
    # On simule une base qui connaît déjà quelques marques structurées.
    return construire_devineur_marque(["Peugeot", "Renault"])


def test_marque_connue_dans_le_titre(deviner_marque):
    assert deviner_marque("Peugeot 208 toutes options") == "Peugeot"


def test_marque_via_alias_faute_orthographe(deviner_marque):
    # "Volswagen" (faute) est un alias connu de config.MARQUES_ALIAS
    assert deviner_marque("Volswagen golf 6") == "Volkswagen"


def test_marque_via_modele_implique(deviner_marque):
    # Le titre ne contient QUE le modèle : "clio" implique Renault
    assert deviner_marque("vends clio 4 essence") == "Renault"
    # "golf" implique Volkswagen même si VW n'est pas dans la base
    assert deviner_marque("Golf 7 toutes options") == "Volkswagen"


def test_marque_absente_retourne_none(deviner_marque):
    assert deviner_marque("terrain à vendre zone industrielle") is None


# ---------------------------------------------------------------------------
# construire_devineur_modele : (marque, titre) -> modèle
# ---------------------------------------------------------------------------

@pytest.fixture
def deviner_modele():
    # Base qui connaît quelques modèles ; le référentiel config complète.
    return construire_devineur_modele({"Renault": ["Clio", "Megane"]})


def test_modele_depuis_catalogue(deviner_modele):
    assert deviner_modele("Volkswagen", "Golf 7 toutes options") == "Golf"


def test_modele_bmw_code_moteur(deviner_modele):
    # Règle dédiée : 320d -> Série 3 (le code moteur implique la série)
    assert deviner_modele("Bmw", "BMW 320d pack M sport") == "Série 3"


def test_modele_mercedes_classe(deviner_modele):
    # Règle dédiée : C220 -> Classe C
    assert deviner_modele("Mercedes-Benz", "Mercedes C220 cdi avantgarde") == "Classe C"


def test_modele_premium_bloque_mot_apres_marque(deviner_modele):
    # Pour les premium allemandes, le mot après la marque est une
    # motorisation, pas un modèle : ne rien inventer -> None
    assert deviner_modele("Audi", "Audi Quattro sport occasion") is None


def test_modele_ignore_mots_parasites(deviner_modele):
    # "occasion" est dans MOTS_A_IGNORER : ne doit pas devenir un modèle
    assert deviner_modele("Fiat", "Fiat occasion propre") is None


def test_modele_marque_nan_retourne_none(deviner_modele):
    assert deviner_modele(np.nan, "un titre quelconque") is None
