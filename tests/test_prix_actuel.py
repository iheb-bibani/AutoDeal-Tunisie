import pandas as pd

from core.prix_actuel import appliquer_prix_actuel


def test_prix_du_jour_prioritaire_et_baisse_detectee():
    df = pd.DataFrame({"Lien": ["a", "b", "c", "d"], "Prix": [52000, 30000, 40000, 25000]})
    out = appliquer_prix_actuel(
        df,
        du_jour={"a": 44000.0, "c": 400.0},          # c : prix aberrant -> ignoré
        suivi={"a": 50000.0, "b": 28500.0},          # b : seul le suivi connaît la baisse
    )
    assert out["Prix"].tolist() == [44000, 28500, 40000, 25000]
    assert out["Prix_Initial"].tolist() == [52000, 30000, 40000, 25000]
    assert out["Baisse_Prix"].tolist() == [True, True, False, False]
    assert out.loc[0, "Variation_Prix_Pct"] == -15.4


def test_sans_source_le_prix_reste_inchange():
    df = pd.DataFrame({"Lien": ["a"], "Prix": [52000]})
    out = appliquer_prix_actuel(df, du_jour={}, suivi={})
    assert out["Prix"].tolist() == [52000]
    assert not out["Baisse_Prix"].any()


def test_prix_suivi_retrouve_une_annonce_tayara_qui_change_de_ville():
    ancien = "https://www.tayara.tn/item/voitures/ariana/x/peugeot-208/6aaed51326fc672960d7947d/"
    nouveau = "https://www.tayara.tn/item/voitures/tunis/y/peugeot-208/6aaed51326fc672960d7947d/"
    df = pd.DataFrame({"Lien": [nouveau], "Prix": [40000]})
    out = appliquer_prix_actuel(df, du_jour={}, suivi={ancien: 36000.0})
    assert out["Prix"].tolist() == [36000]
