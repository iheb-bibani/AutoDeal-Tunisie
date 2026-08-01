"""
Tests des normalisations de core/merging_files.py -- chacune corrige un bug
de fragmentation de catégories confirmé sur données réelles (ex: 'Tunis' et
'tunis' comptés séparément). Ces tests verrouillent ces corrections.
"""
import pandas as pd

from core.merging_files import (
    normaliser_marque,
    normaliser_energie,
    normaliser_localisation,
)


def test_energie_fusionne_casse():
    # "Hybride Diesel" et "Hybride diesel" doivent devenir une seule catégorie
    df = pd.DataFrame({"Energie": ["Hybride Diesel", "Hybride diesel", "ESSENCE"]})
    out = normaliser_energie(df)["Energie"].tolist()
    assert out[0] == out[1] == "Hybride diesel"
    assert out[2] == "Essence"


def test_localisation_title_case_gouvernorats():
    df = pd.DataFrame({"Localisation": ["tunis", "Tunis", "ben arous"]})
    out = normaliser_localisation(df)["Localisation"].tolist()
    assert out[0] == out[1] == "Tunis"
    assert out[2] == "Ben Arous"


def test_marque_correction_citroen():
    df = pd.DataFrame({"Marque": ["citroen", "CITROEN"]})
    out = normaliser_marque(df)["Marque"].tolist()
    assert out[0] == out[1] == "Citroën"


def test_marque_autres_devient_na():
    # "Autres" chez tayara = champ non rempli, pas une marque -> doit être vide
    df = pd.DataFrame({"Marque": ["autres", "Peugeot"]})
    out = normaliser_marque(df)
    assert pd.isna(out["Marque"].iloc[0])
    assert out["Marque"].iloc[1] == "Peugeot"


def test_marque_inconnue_conservee():
    # Une marque non listée dans les corrections doit être gardée (title-case)
    df = pd.DataFrame({"Marque": ["toyota"]})
    assert normaliser_marque(df)["Marque"].iloc[0] == "Toyota"
