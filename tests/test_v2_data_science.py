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
