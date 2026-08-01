"""
Tests d'intégration des garde-fous de core/nettoyer_base.process_data() :
prix / kilométrage / puissance fiscale / année hors plage, valeurs de
remplissage (chiffres répétés), bruit et "Autres". process_data lit/écrit
via config.PROCESSED_FILES ; on redirige ces chemins vers des fichiers
temporaires et on vérifie le CSV produit.
"""
from datetime import date

import pandas as pd
import pytest

import core.nettoyer_base as nb

AUJ = date.today().isoformat()  # dans la fenêtre MAX_DAYS_OLD

COLONNES = [
    "Source", "Titre", "Marque", "Modèle", "Année", "Prix", "Kilométrage",
    "Energie", "Boite_Vitesse", "Localisation", "Puissance_Fiscale",
    "Etat_Vehicule", "Annonce-Deposee", "Annonce-Detectee", "Statut", "Lien",
]


def _ligne(lien, titre="Peugeot 208", marque="Peugeot", modele="208",
           annee=2018, prix=25000, km=90000, pf=6, depose=AUJ):
    return {
        "Source": "test", "Titre": titre, "Marque": marque, "Modèle": modele,
        "Année": annee, "Prix": prix, "Kilométrage": km, "Energie": "Diesel",
        "Boite_Vitesse": "Manuelle", "Localisation": "Tunis",
        "Puissance_Fiscale": pf, "Etat_Vehicule": "", "Annonce-Deposee": depose,
        "Annonce-Detectee": AUJ, "Statut": "Active", "Lien": lien,
    }


@pytest.fixture
def resultat(tmp_path, monkeypatch):
    """Construit un CSV merged de test, lance process_data, renvoie le CSV recent."""
    lignes = [
        _ligne("L_valide"),                                  # doit rester, propre
        _ligne("L_prix_bas", prix=1000),                     # prix < 3000 -> écarté
        _ligne("L_prix_haut", prix=600000),                  # prix > 500000 -> écarté
        _ligne("L_prix_repete", prix=111111),                # chiffre répété -> écarté
        _ligne("L_km_repete", km=999999),                    # km chiffre répété -> NaN
        _ligne("L_km_haut", km=700000),                      # km > 500000 -> NaN
        _ligne("L_pf_haut", pf=99),                          # pf > 30 -> NaN
        _ligne("L_annee_futur", annee=2050),                 # année future -> NaN
        _ligne("L_bruit", titre="Terrain à vendre zone", marque="Autres", modele="Autres"),
        _ligne("L_sans_marque", titre="voiture propre à vendre", marque="Autres", modele="Autres"),
    ]
    merged = tmp_path / "merged.csv"
    recent = tmp_path / "recent.csv"
    pd.DataFrame(lignes)[COLONNES].to_csv(merged, sep=";", index=False, encoding="utf-8-sig")

    monkeypatch.setitem(nb.PROCESSED_FILES, "merged", str(merged))
    monkeypatch.setitem(nb.PROCESSED_FILES, "recent", str(recent))
    nb.process_data()
    return pd.read_csv(recent, sep=";", encoding="utf-8-sig").set_index("Lien")


def test_ligne_valide_conservee(resultat):
    assert "L_valide" in resultat.index


@pytest.mark.parametrize("lien", ["L_prix_bas", "L_prix_haut", "L_prix_repete"])
def test_prix_hors_plage_ecarte(resultat, lien):
    assert lien not in resultat.index


@pytest.mark.parametrize("lien", ["L_km_repete", "L_km_haut"])
def test_km_aberrant_mis_a_vide(resultat, lien):
    assert lien in resultat.index                 # la ligne reste
    assert pd.isna(resultat.loc[lien, "Kilométrage"])  # mais le km est vidé


def test_puissance_fiscale_aberrante_vidée(resultat):
    assert pd.isna(resultat.loc["L_pf_haut", "Puissance_Fiscale"])


def test_annee_future_vidée(resultat):
    assert pd.isna(resultat.loc["L_annee_futur", "Année"])


def test_bruit_immobilier_ecarté(resultat):
    assert "L_bruit" not in resultat.index


def test_annonce_sans_marque_ecartée(resultat):
    # "Autres" -> NA, aucune marque inférable depuis "voiture propre à vendre"
    assert "L_sans_marque" not in resultat.index
