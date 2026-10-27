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


def _gen(gkf, sources):
    return {"groupkfold_marque_modele": {"mdape": gkf},
            "source_holdout": [{"source": n, "n_test": k, "mdape": m} for n, k, m in sources]}


def test_selection_production_d_abord_avec_garde_fou_stress():
    """Cas réel du 27/10 : HGB meilleur en production, XGB un peu meilleur aux
    stress-tests mais dans la tolérance -> HGB retenu."""
    from core.modele_prediction import choisir_modele

    stab = {"HGB": {"moyenne_pct": 9.25, "ecart_type_pct": 0.27}, "XGB": {"moyenne_pct": 9.86, "ecart_type_pct": 0.30}}
    sources = lambda a, b, c: [("autocentral", 96, a), ("automobile", 1886, b), ("tayara", 1376, c)]  # noqa: E731
    gen = {"HGB": _gen(0.1677, sources(0.485, 0.148, 0.129)), "XGB": _gen(0.150, sources(0.492, 0.138, 0.113))}
    gagnant, raison = choisir_modele(stab, gen)
    assert gagnant == "HGB", raison


def test_selection_ecarte_un_modele_qui_generalise_mal():
    from core.modele_prediction import choisir_modele

    stab = {"A": {"moyenne_pct": 9.0, "ecart_type_pct": 0.2}, "B": {"moyenne_pct": 9.5, "ecart_type_pct": 0.2}}
    gen = {"A": _gen(0.30, [("s", 100, 0.30)]), "B": _gen(0.15, [("s", 100, 0.15)])}
    gagnant, raison = choisir_modele(stab, gen)
    assert gagnant == "B" and "écarté" in raison


def test_selection_ex_aequo_departage_par_les_stress_tests():
    from core.modele_prediction import choisir_modele

    stab = {"A": {"moyenne_pct": 9.40, "ecart_type_pct": 0.30}, "B": {"moyenne_pct": 9.50, "ecart_type_pct": 0.30}}
    gen = {"A": _gen(0.17, [("s", 100, 0.14)]), "B": _gen(0.155, [("s", 100, 0.13)])}
    gagnant, raison = choisir_modele(stab, gen)
    assert gagnant == "B" and "ex-aequo" in raison


def test_selection_gain_de_stress_non_significatif_ne_renverse_pas():
    """Run du 27/10 10h : HGB meilleur de 0,45 pt en production (bruit 0,66),
    XGB meilleur de 0,23 pt seulement aux stress-tests -> HGB reste retenu."""
    from core.modele_prediction import choisir_modele

    stab = {"HGB": {"moyenne_pct": 9.88, "ecart_type_pct": 0.58}, "XGB": {"moyenne_pct": 10.33, "ecart_type_pct": 0.66}}
    gen = {"HGB": _gen(0.1607, []), "XGB": _gen(0.1584, [])}
    gagnant, raison = choisir_modele(stab, gen)
    assert gagnant == "HGB", raison


def test_erreur_stress_pondere_les_sources_par_volume():
    from core.modele_prediction import erreur_stress

    gen = _gen(0.16, [("petite", 100, 0.50), ("grande", 900, 0.10)])
    # sources : 0,5*0,1 + 0,1*0,9 = 0,14 (et non 0,30 en moyenne simple)
    assert abs(erreur_stress(gen) - (0.16 + 0.14) / 2) < 1e-9


def test_lien_canonique_tayara_ignore_la_ville():
    from core.doublons import lien_canonique

    liens = pd.Series([
        "https://www.tayara.tn/item/voitures/ariana/la-soukra/peugeot-208/6aaed51326fc672960d7947d/",
        "https://www.tayara.tn/item/voitures/tunis/l-aouina/peugeot-208/6aaed51326fc672960d7947d/",
        "https://www.automobile.tn/fr/occasion/peugeot/208/130657",
    ])
    c = lien_canonique(liens)
    assert c.iloc[0] == c.iloc[1] != c.iloc[2]
