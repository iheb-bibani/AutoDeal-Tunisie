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


def _cats(complet=True, **sources):
    return {k: {"liens": v, "complet": complet} for k, v in sources.items()}


@pytest.fixture
def env(tmp_path, monkeypatch):
    suivi = tmp_path / "suivi.csv"
    monkeypatch.setattr(sa, "FICHIER_SUIVI", str(suivi))
    monkeypatch.setattr(
        sa, "FICHIER_ALERTES", str(tmp_path / "inexistant_alertes.csv")
    )
    # Par défaut : aucun fichier de liens vus (mode historique).
    monkeypatch.setattr(sa, "charger_catalogues", lambda: {})
    monkeypatch.setattr(sa, "MANIFESTE", str(tmp_path / "pas_de_manifeste.json"))
    # Les tests historiques exercent les garde-fous de l'ancien mode.
    monkeypatch.setenv("AUTODEAL_SUIVI_HISTORIQUE", "1")
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


def test_liens_vus_permettent_les_disparitions_malgre_un_fichier_cumulatif(env):
    """Le fichier fusionné est cumulatif : seules les pages de liste disent
    ce qui est encore en ligne."""
    tmp_path, monkeypatch, suivi = env
    liens = [f"L{i}" for i in range(10)]
    merged = _merged(tmp_path, liens, ["tayara.tn"] * 10)
    monkeypatch.setitem(sa.PROCESSED_FILES, "merged", merged)
    monkeypatch.setattr(sa, "charger_catalogues", lambda: _cats(tayara={l: "20000" for l in liens}))
    sa.mettre_a_jour()

    # Le lendemain : le fichier fusionné contient toujours les 10 liens, mais
    # la page de liste n'en montre plus que 9 (L0 vendue, prix de L1 baissé).
    vus = {l: "20000" for l in liens[1:]}
    vus["L1"] = "18 500"
    monkeypatch.setattr(sa, "charger_catalogues", lambda: _cats(tayara=vus))
    sa.mettre_a_jour()
    df = _charger(suivi)
    assert df.loc["L0", "Statut"] == "Disparue"
    assert df.loc["L1", "Statut"] == "Active"
    assert float(df.loc["L1", "Prix_Dernier"]) == 18500


def test_source_non_observee_ou_en_echec_ne_fait_rien_disparaitre(env):
    tmp_path, monkeypatch, suivi = env
    liens_t = [f"T{i}" for i in range(5)]
    liens_a = [f"A{i}" for i in range(5)]
    merged = _merged(tmp_path, liens_t + liens_a, ["tayara.tn"] * 5 + ["automobile.tn"] * 5)
    monkeypatch.setitem(sa.PROCESSED_FILES, "merged", merged)
    monkeypatch.setattr(sa, "charger_catalogues", lambda: _cats(
        tayara={l: None for l in liens_t}, automobile={l: None for l in liens_a}))
    sa.mettre_a_jour()

    # automobile n'a rien publié aujourd'hui ; tayara a échoué selon le manifeste.
    import json
    manifeste = tmp_path / "run.json"
    manifeste.write_text(json.dumps({"scrapers": {"tayara": {"status": "timeout"}, "automax": {"status": "success"}}}))
    monkeypatch.setattr(sa, "MANIFESTE", str(manifeste))
    monkeypatch.setattr(sa, "charger_catalogues", lambda: _cats(tayara={}, automax={"X": None}))
    sa.mettre_a_jour()
    df = _charger(suivi)
    assert (df["Statut"] == "Active").all()


def test_catalogue_partiel_ne_juge_que_la_fenetre_couverte(env):
    """tayara plafonné : les annonces récentes non revues disparaissent, les
    anciennes (au-delà des pages vues) restent actives."""
    tmp_path, monkeypatch, suivi = env
    recents = [f"R{i}" for i in range(80)]
    anciens = [f"A{i}" for i in range(20)]
    liens = recents + anciens
    df = pd.DataFrame({
        "Lien": liens, "Prix": 20000, "Source": "tayara.tn", "Marque": "Peugeot",
        "Modèle": "208", "Année": 2018, "Localisation": "Tunis",
        # Récentes étalées du 25/10 au 06/10 (R0 la plus récente), anciennes en juin.
        "Annonce-Deposee": [str((pd.Timestamp("2026-10-25") - pd.Timedelta(days=i // 4)).date()) for i in range(80)]
        + ["2026-06-01"] * 20,
    })
    chemin = tmp_path / "merged.csv"
    df.to_csv(chemin, sep=";", index=False, encoding="utf-8-sig")
    monkeypatch.setitem(sa.PROCESSED_FILES, "merged", str(chemin))
    monkeypatch.setattr(sa, "charger_catalogues", lambda: _cats(tayara={l: None for l in liens}))
    sa.mettre_a_jour()

    # Run partiel : seules les pages récentes sont vues, R0 n'y est plus.
    vus = {l: None for l in recents[1:]}
    monkeypatch.setattr(sa, "charger_catalogues", lambda: _cats(complet=False, tayara=vus))
    sa.mettre_a_jour()
    res = _charger(suivi)
    assert res.loc["R0", "Statut"] == "Disparue"
    assert (res.loc[anciens, "Statut"] == "Active").all()


def test_sans_catalogue_du_jour_aucune_disparition(env):
    """Fichier fusionné cumulatif : une annonce qui en sort (filtre,
    dédoublonnage) n'a pas été vendue."""
    tmp_path, monkeypatch, suivi = env
    monkeypatch.delenv("AUTODEAL_SUIVI_HISTORIQUE")
    monkeypatch.setitem(sa.PROCESSED_FILES, "merged", _merged(tmp_path, ["A", "B", "C"]))
    sa.mettre_a_jour()
    monkeypatch.setitem(sa.PROCESSED_FILES, "merged", _merged(tmp_path, ["A", "B"]))
    sa.mettre_a_jour()
    assert (_charger(suivi)["Statut"] == "Active").all()


def test_sans_catalogue_une_disparition_reelle_n_est_pas_annulee(env):
    """Une disparition détectée un jour avec catalogue ne doit pas être
    « annulée » un jour sans catalogue parce que le lien reste dans le
    fichier cumulatif."""
    tmp_path, monkeypatch, suivi = env
    monkeypatch.delenv("AUTODEAL_SUIVI_HISTORIQUE")
    liens = [f"L{i}" for i in range(10)]
    monkeypatch.setitem(sa.PROCESSED_FILES, "merged", _merged(tmp_path, liens, ["tayara.tn"] * 10))
    monkeypatch.setattr(sa, "charger_catalogues", lambda: _cats(tayara={l: None for l in liens}))
    sa.mettre_a_jour()
    monkeypatch.setattr(sa, "charger_catalogues", lambda: _cats(tayara={l: None for l in liens[1:]}))
    sa.mettre_a_jour()
    monkeypatch.setattr(sa, "charger_catalogues", lambda: {})
    sa.mettre_a_jour()
    assert _charger(suivi).loc["L0", "Statut"] == "Disparue"
