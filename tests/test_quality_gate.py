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
