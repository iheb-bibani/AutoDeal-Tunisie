import pandas as pd

import utils.verifier_donnees_brutes as v


def _csv(tmp_path, nom, df):
    chemin = tmp_path / nom
    df.to_csv(chemin, sep=";", index=False, encoding="utf-8-sig")
    return chemin


def test_fichier_sans_lien_refuse(tmp_path, monkeypatch):
    monkeypatch.setattr(v, "RACINE", tmp_path)
    monkeypatch.setattr(v, "_version_head", lambda _: None)
    assert v.verifier_fichier(_csv(tmp_path, "a.csv", pd.DataFrame({"x": [1]}))) == "colonne Lien absente"


def test_fichier_tronque_refuse(tmp_path, monkeypatch):
    monkeypatch.setattr(v, "RACINE", tmp_path)
    ancien = pd.DataFrame({"Lien": [f"L{i}" for i in range(100)]}).to_csv(sep=";", index=False).encode("utf-8-sig")
    monkeypatch.setattr(v, "_version_head", lambda _: ancien)
    raison = v.verifier_fichier(_csv(tmp_path, "a.csv", pd.DataFrame({"Lien": ["L1"]})))
    assert raison and "tronqué" in raison
    assert v.verifier_fichier(_csv(tmp_path, "b.csv", pd.DataFrame({"Lien": [f"L{i}" for i in range(105)]}))) is None
