import pandas as pd

import config
from core.modele_prediction import calculer_nb_comparables_locaux


def test_zone_exclue_du_modele_prix_et_features_usage_presentes():
    assert "Zone_Economique" not in config.FEATURES_NUMERIQUES
    for f in ["Log_Kilometrage", "Age_Carre", "Km_Par_An"]:
        assert f in config.FEATURES_NUMERIQUES


def test_comparables_sont_locaux_pas_simple_volume_modele():
    df = pd.DataFrame([
        {"Marque": "Peugeot", "Modèle": "208", "Age_Vehicule": 4, "Kilométrage": 60000, "Energie": "Essence", "Transmission": "Traction"},
        {"Marque": "Peugeot", "Modèle": "208", "Age_Vehicule": 5, "Kilométrage": 70000, "Energie": "Essence", "Transmission": "Traction"},
        # Même modèle mais beaucoup trop vieux / kilométré : ne doit pas compter.
        {"Marque": "Peugeot", "Modèle": "208", "Age_Vehicule": 12, "Kilométrage": 220000, "Energie": "Essence", "Transmission": "Traction"},
        # Modèle différent : jamais comparable.
        {"Marque": "Peugeot", "Modèle": "308", "Age_Vehicule": 4, "Kilométrage": 60000, "Energie": "Essence", "Transmission": "Traction"},
    ])
    n = calculer_nb_comparables_locaux(df)
    assert n.iloc[0] == 1
    assert n.iloc[1] == 1
    assert n.iloc[2] == 0
    assert n.iloc[3] == 0


def test_doublons_inter_sources_ne_sont_pas_des_comparables():
    base = {"Marque": "Peugeot", "Modèle": "208", "Age_Vehicule": 4, "Année": 2022,
            "Energie": "Essence", "Boite_Vitesse": "Manuelle"}
    df = pd.DataFrame([
        {**base, "Kilométrage": 60000, "Prix": 50000, "Lien": "https://a/1"},
        # Même voiture publiée sur un 2e site : même km, même prix, autre lien.
        {**base, "Kilométrage": 60000, "Prix": 50000, "Lien": "https://b/9"},
        {**base, "Kilométrage": 65000, "Prix": 48000, "Lien": "https://a/2"},
    ])
    n = calculer_nb_comparables_locaux(df)
    assert n.tolist() == [1, 1, 2]


def test_cle_doublon_normalise_les_types():
    from core.doublons import cle_doublon

    df = pd.DataFrame([
        {"Marque": "Peugeot", "Modèle": "208", "Année": 2020, "Kilométrage": 60000.0, "Prix": 50000},
        {"Marque": "peugeot ", "Modèle": "208", "Année": "2020", "Kilométrage": "60000", "Prix": 50000.0},
        {"Marque": "Peugeot", "Modèle": "208", "Année": None, "Kilométrage": None, "Prix": None},
        {"Marque": "Peugeot", "Modèle": "208", "Année": None, "Kilométrage": None, "Prix": None},
    ])
    k = cle_doublon(df)
    assert k.iloc[0] == k.iloc[1]
    # Sans prix ni km : jamais fusionnées.
    assert k.iloc[2] != k.iloc[3]


def test_inference_regroupe_comme_l_entrainement_et_garde_les_nan():
    import numpy as np
    from core.inference import preparer_lignes

    bundle = {
        "features_numeriques": ["Kilométrage"],
        "features_categorielles": ["Marque", "Modèle", "Boite_Vitesse"],
        "marques_connues": ["Peugeot"],
        "modeles_connus": ["208", "Autre_modele"],
    }
    X = preparer_lignes(bundle, {"Kilométrage": "60000", "Marque": "Lada", "Modèle": "Niva", "Boite_Vitesse": None})
    assert X.loc[0, "Marque"] == "Autre"
    assert X.loc[0, "Modèle"] == "Autre_modele"
    assert X.loc[0, "Boite_Vitesse"] is np.nan or pd.isna(X.loc[0, "Boite_Vitesse"])
    assert X.loc[0, "Kilométrage"] == 60000


def test_ecriture_atomique_ne_laisse_pas_de_fichier_partiel(tmp_path):
    import pytest
    from core.io_atomique import ecrire_csv_atomique

    cible = tmp_path / "f.csv"
    ecrire_csv_atomique(pd.DataFrame({"a": [1]}), cible)

    class Boom(pd.DataFrame):
        def to_csv(self, *a, **k):
            raise RuntimeError("kill")

    with pytest.raises(RuntimeError):
        ecrire_csv_atomique(Boom({"a": [2]}), cible)
    assert pd.read_csv(cible, sep=";", encoding="utf-8-sig")["a"].tolist() == [1]
    assert [p.name for p in tmp_path.iterdir()] == ["f.csv"]


def test_selection_quasi_ex_aequo_departagee_par_la_stabilite():
    from core.modele_prediction import choisir_modele

    stab = {"HGB": {"moyenne_pct": 9.4, "ecart_type_pct": 0.2}, "XGB": {"moyenne_pct": 9.9, "ecart_type_pct": 0.4}}
    # XGB a un score robuste à peine meilleur, sous le bruit (0,4 pt) : HGB l'emporte.
    gagnant, raison = choisir_modele({"HGB": 0.1343, "XGB": 0.1323}, stab)
    assert gagnant == "HGB" and "ex-aequo" in raison
    # Écart net (> bruit) : le meilleur score robuste l'emporte.
    gagnant, _ = choisir_modele({"HGB": 0.150, "XGB": 0.130}, stab)
    assert gagnant == "XGB"


def test_score_robuste_pondere_les_sources_par_volume():
    from core.modele_prediction import score_robuste

    gen = {"groupkfold_marque_modele": {"mdape": 0.16},
           "source_holdout": [{"source": "petite", "n_test": 100, "mdape": 0.50},
                              {"source": "grande", "n_test": 900, "mdape": 0.10}]}
    s = score_robuste(gen, {"moyenne_pct": 10.0})
    # sources : 0,5*0,1 + 0,1*0,9 = 0,14 (et non 0,30 en moyenne simple)
    assert abs(s - (0.10 + 0.16 + 0.14) / 3) < 1e-9
