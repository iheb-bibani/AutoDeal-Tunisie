"""Tests des garde-fous de suivi des annonces."""
import pandas as pd
import pytest

import core.suivi_annonces as sa


def _merged(tmp_path, liens, sources=None):
    """Écrit un CSV merged minimal avec les colonnes lues par le suivi."""
    sources = sources or ["test"] * len(liens)
    df = pd.DataFrame({
        "Lien": liens,
        "Prix": [20000] * len(liens),
        "Source": sources,
        "Marque": ["Peugeot"] * len(liens),
        "Modèle": ["208"] * len(liens),
        "Année": [2018] * len(liens),
        "Localisation": ["Tunis"] * len(liens),
    })
    chemin = tmp_path / "merged.csv"
    df.to_csv(chemin, sep=";", index=False, encoding="utf-8-sig")
    return str(chemin)


@pytest.fixture
def env(tmp_path, monkeypatch):
    suivi = tmp_path / "suivi.csv"
    monkeypatch.setattr(sa, "FICHIER_SUIVI", str(suivi))
    monkeypatch.setattr(
        sa, "FICHIER_ALERTES", str(tmp_path / "inexistant_alertes.csv")
    )
    return tmp_path, monkeypatch, suivi


def _charger(suivi_path):
    return pd.read_csv(
        suivi_path, sep=";", encoding="utf-8-sig"
    ).set_index("Lien")


def test_annonce_absente_devient_disparue(env):
    tmp_path, monkeypatch, suivi = env
    monkeypatch.setitem(
        sa.PROCESSED_FILES, "merged", _merged(tmp_path, ["A", "B", "C"])
    )
    sa.mettre_a_jour()

    monkeypatch.setitem(
        sa.PROCESSED_FILES, "merged", _merged(tmp_path, ["A", "B"])
    )
    sa.mettre_a_jour()

    df = _charger(suivi)
    assert df.loc["C", "Statut"] == "Disparue"
    assert pd.notna(df.loc["C", "Date_Disparition"])
    assert df.loc["A", "Statut"] == "Active"
    assert df.loc["B", "Statut"] == "Active"


def test_garde_fou_volume_bloque_les_disparitions(env):
    tmp_path, monkeypatch, suivi = env
    monkeypatch.setitem(
        sa.PROCESSED_FILES,
        "merged",
        _merged(tmp_path, ["A", "B", "C", "D", "E"]),
    )
    sa.mettre_a_jour()

    monkeypatch.setitem(
        sa.PROCESSED_FILES, "merged", _merged(tmp_path, ["A"])
    )
    sa.mettre_a_jour()

    df = _charger(suivi)
    assert (df["Statut"] == "Disparue").sum() == 0
    assert (df["Statut"] == "Active").sum() == 5


def test_garde_fou_par_source_bloque_une_panne_partielle(env):
    """Une source peut tomber sans faire chuter le total sous 60 %."""
    tmp_path, monkeypatch, suivi = env
    liens = ["A1", "A2", "A3", "B1", "B2", "B3"]
    sources = ["source-a"] * 3 + ["source-b"] * 3
    monkeypatch.setitem(
        sa.PROCESSED_FILES,
        "merged",
        _merged(tmp_path, liens, sources),
    )
    sa.mettre_a_jour()

    # 4/6 = 67 % : le garde-fou global ne se déclenche pas. Mais source-b
    # tombe à 1/3 = 33 %, donc B2/B3 ne doivent pas être déclarées disparues.
    monkeypatch.setitem(
        sa.PROCESSED_FILES,
        "merged",
        _merged(
            tmp_path,
            ["A1", "A2", "A3", "B1"],
            ["source-a", "source-a", "source-a", "source-b"],
        ),
    )
    sa.mettre_a_jour()

    df = _charger(suivi)
    assert df.loc["B2", "Statut"] == "Active"
    assert df.loc["B3", "Statut"] == "Active"
    assert (df["Statut"] == "Disparue").sum() == 0
