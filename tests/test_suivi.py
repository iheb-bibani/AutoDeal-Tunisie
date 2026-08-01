"""
Tests de core/suivi_annonces.mettre_a_jour() :
  - une annonce présente hier et absente aujourd'hui passe à "Disparue" ;
  - mais si le volume du jour s'effondre (scraping incomplet), AUCUNE
    disparition n'est enregistrée -- c'est le garde-fou anti-faux-positif.
On redirige les fichiers vers un dossier temporaire et on enchaîne deux
exécutions.
"""
import pandas as pd
import pytest

import core.suivi_annonces as sa


def _merged(tmp_path, liens):
    """Écrit un CSV merged minimal avec les colonnes que lit mettre_a_jour."""
    df = pd.DataFrame({
        "Lien": liens,
        "Prix": [20000] * len(liens),
        "Source": ["test"] * len(liens),
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
    monkeypatch.setattr(sa, "FICHIER_ALERTES", str(tmp_path / "inexistant_alertes.csv"))
    return tmp_path, monkeypatch, suivi


def _charger(suivi_path):
    return pd.read_csv(suivi_path, sep=";", encoding="utf-8-sig").set_index("Lien")


def test_annonce_absente_devient_disparue(env):
    tmp_path, monkeypatch, suivi = env

    # Jour 1 : A, B, C actives
    monkeypatch.setitem(sa.PROCESSED_FILES, "merged", _merged(tmp_path, ["A", "B", "C"]))
    sa.mettre_a_jour()

    # Jour 2 : C a disparu (volume 2 vs 3 actives -> au-dessus du seuil suspect)
    monkeypatch.setitem(sa.PROCESSED_FILES, "merged", _merged(tmp_path, ["A", "B"]))
    sa.mettre_a_jour()

    df = _charger(suivi)
    assert df.loc["C", "Statut"] == "Disparue"
    assert pd.notna(df.loc["C", "Date_Disparition"])
    assert df.loc["A", "Statut"] == "Active"
    assert df.loc["B", "Statut"] == "Active"


def test_garde_fou_volume_bloque_les_disparitions(env):
    tmp_path, monkeypatch, suivi = env

    # Jour 1 : 5 annonces actives
    monkeypatch.setitem(sa.PROCESSED_FILES, "merged", _merged(tmp_path, ["A", "B", "C", "D", "E"]))
    sa.mettre_a_jour()

    # Jour 2 : une seule annonce (1 < 5 * 0.60) -> scraping jugé incomplet
    monkeypatch.setitem(sa.PROCESSED_FILES, "merged", _merged(tmp_path, ["A"]))
    sa.mettre_a_jour()

    df = _charger(suivi)
    # Aucune disparition ne doit avoir été enregistrée malgré l'absence de B..E
    assert (df["Statut"] == "Disparue").sum() == 0
    assert (df["Statut"] == "Active").sum() == 5
