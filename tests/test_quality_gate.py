from datetime import datetime, timezone

import pandas as pd

from core.quality_gate import age_derniere_detection_heures, scraper_a_reussi


def test_age_derniere_detection_heures():
    df = pd.DataFrame({"Annonce-Detectee": ["2026-08-20", "2026-08-21"]})
    now = datetime(2026, 8, 21, 12, 0, tzinfo=timezone.utc)
    assert age_derniere_detection_heures(df, now) == 12.0


def test_age_derniere_detection_absente():
    assert age_derniere_detection_heures(pd.DataFrame({"Prix": [10000]})) is None


def test_age_derniere_detection_ignore_dates_invalides():
    df = pd.DataFrame({"Annonce-Detectee": ["invalide", "2026-08-20T18:00:00Z"]})
    now = datetime(2026, 8, 21, 0, 0, tzinfo=timezone.utc)
    assert age_derniere_detection_heures(df, now) == 6.0


def test_scraper_success_comes_from_current_run_manifest():
    manifest = {
        "scrapers": {
            "automax": {"status": "success"},
            "tayara": {"status": "timeout"},
        }
    }
    assert scraper_a_reussi(manifest, "automax") is True
    assert scraper_a_reussi(manifest, "tayara") is False
    assert scraper_a_reussi(manifest, "automobile") is None


def test_old_listing_date_does_not_mean_scraper_failed():
    # L'ancien bug confondait l'âge de la dernière nouvelle annonce et l'état
    # opérationnel du scraper. Ces deux notions doivent rester indépendantes.
    df = pd.DataFrame({"Annonce-Detectee": ["2026-07-13"]})
    now = datetime(2026, 8, 22, 0, 0, tzinfo=timezone.utc)
    assert age_derniere_detection_heures(df, now) > 36
    manifest = {"scrapers": {"automax": {"status": "success"}}}
    assert scraper_a_reussi(manifest, "automax") is True


def _gate_env(tmp_path, monkeypatch, catalogues, detect_status="success"):
    import core.quality_gate as qg

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    raw = {}
    for nom in ["tayara", "automax", "automobile", "sayyarat", "autocentral"]:
        chemin = tmp_path / f"{nom}.csv"
        pd.DataFrame({"Lien": [f"{nom}{i}" for i in range(500)], "Annonce-Detectee": today}).to_csv(
            chemin, sep=";", index=False, encoding="utf-8-sig")
        raw[nom] = str(chemin)
    scored = tmp_path / "scored.csv"
    pd.DataFrame({"Lien": [f"L{i}" for i in range(400)], "Prix": 40000, "Marque": "Kia", "Modèle": "Rio"}).to_csv(
        scored, sep=";", index=False, encoding="utf-8-sig")
    manifest = {"scrapers": {n: {"status": "success"} for n in raw},
                "processing": {"core/detect_deals.py": {"status": detect_status}},
                "catalogues": catalogues}
    monkeypatch.setattr(qg, "SCRAPERS", raw)
    monkeypatch.setattr(qg, "PROCESSED_FILES", {"scored": str(scored)})
    monkeypatch.setattr(qg, "BASELINE_PATH", tmp_path / "baseline.json")
    monkeypatch.setattr(qg, "_charger_manifest_courant", lambda *a, **k: manifest)
    monkeypatch.setattr(qg.sys, "argv", ["quality_gate.py"])
    return qg


def test_gate_utilise_le_volume_du_jour_et_non_le_csv_cumulatif(tmp_path, monkeypatch):
    # CSV bruts bien remplis (cumulatifs) mais 2 sources directes n'ont rien vu aujourd'hui.
    qg = _gate_env(tmp_path, monkeypatch, {"tayara": {"liens_vus": 0}, "automobile": {"liens_vus": 3}})
    assert qg.controler() == 1


def test_gate_passe_avec_des_catalogues_du_jour_corrects(tmp_path, monkeypatch):
    qg = _gate_env(tmp_path, monkeypatch, {"tayara": {"liens_vus": 2000}, "automobile": {"liens_vus": 1500}})
    assert qg.controler() == 0


def test_gate_bloque_si_detect_deals_a_echoue(tmp_path, monkeypatch):
    qg = _gate_env(tmp_path, monkeypatch, {}, detect_status="failed")
    assert qg.controler() == 1
