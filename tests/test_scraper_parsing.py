import pytest

from scrapers.parsing import parser_annee, parser_entier, parser_prix_tnd


@pytest.mark.parametrize("texte, attendu", [
    ("35 000 DT", 35000),
    ("35.000 DT", 35000),
    ("35.000,000 DT", 35000),
    ("35,000 TND", 35000),
    ("Prix : 125 500 DT", 125500),
    ("42 mille DT", 42000),
    ("42 MD", 42000),
    ("35000000", 35000),           # millimes concaténés
    ("52149505", None),            # numéro de téléphone collé
    ("0", None),
    ("Prix sur demande", None),
    (None, None),
    ("9,000\n45 000 DT", 45000),   # la devise prime, pas de traversée de ligne
])
def test_parser_prix_tnd(texte, attendu):
    assert parser_prix_tnd(texte) == attendu


def test_parser_entier_ne_traverse_pas_les_lignes():
    assert parser_entier("2020\n15000 km") == 2020
    assert parser_entier("120 000 km") == 120000
    assert parser_entier("9,000 km") == 9000
    assert parser_entier("1 500 000 km", maxi=1_000_000) is None


def test_parser_annee():
    assert parser_annee("03/2019") == 2019
    assert parser_annee("Modèle 2015") == 2015
    assert parser_annee("32019") is None


def test_carte_autocentral_ligne_par_ligne():
    from scrapers.scraper_autocentral import parser_carte

    carte = {
        "source": "FACEBOOK", "idPost": "FACEBOOK-1", "titre": "Golf 7 TDI",
        "texte": "Golf 7 TDI\n2018 Volkswagen Golf 7\n150 000 km\n6cv Diesel\nManuelle\n45 500 DT\nSfax\nVoir l'annonce →",
    }
    car = parser_carte(carte)
    assert (car["Marque"], car["Modèle"], car["Année"]) == ("Volkswagen", "Golf 7", 2018)
    assert (car["Prix_DT"], car["Kilométrage"], car["Puissance_Fiscale"]) == (45500, 150000, 6)
    assert car["Localisation"] == "Sfax"


def test_carte_autocentral_marque_inconnue_non_inventee():
    from scrapers.scraper_autocentral import parser_carte

    car = parser_carte({"source": "FACEBOOK", "idPost": "F-2", "titre": "x", "texte": "2011 DSG 7 BLUE\n45000 DT\nManuelle"})
    assert car["Marque"] is None
    assert car["Localisation"] is None


def test_assainir_autocentral_ecarte_les_contenus_dupliques():
    import pandas as pd
    from core.merging_files import assainir_autocentral

    dup = {"Marque": "DSG", "Modèle": "7", "Année": 2011, "Prix_DT": 45000, "Kilométrage": None}
    df = pd.DataFrame([dup, dup, dup, {"Marque": "Volkswagen", "Modèle": "Golf", "Année": 2018, "Prix_DT": 45500, "Kilométrage": 150000},
                       {"Marque": "Diesel", "Modèle": "x", "Année": 2015, "Prix_DT": 30000, "Kilométrage": 90000}])
    out = assainir_autocentral(df)
    assert len(out) == 2
    assert out["Marque"].isna().sum() == 1
