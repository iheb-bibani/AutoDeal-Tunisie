from datetime import datetime, timezone

import pandas as pd

from core.quality_gate import age_derniere_detection_heures


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
