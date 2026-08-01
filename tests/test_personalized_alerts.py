import pandas as pd

from utils.send_personalized_alerts import _match


def row(**overrides):
    base = {
        "Marque": "Peugeot", "Modèle": "208", "Prix": 45000,
        "Kilométrage": 80000, "Année": 2021, "Score_Opportunite": 0.30,
    }
    base.update(overrides)
    return pd.Series(base)


def alert(**overrides):
    base = {
        "brand": "Peugeot", "model": "208", "budget_max": 50000,
        "max_km": 100000, "min_year": 2019, "min_gap_pct": 25,
    }
    base.update(overrides)
    return base


def test_match_all_criteria():
    assert _match(row(), alert()) is True


def test_match_rejects_budget_or_gap():
    assert _match(row(Prix=55000), alert()) is False
    assert _match(row(Score_Opportunite=0.20), alert()) is False


def test_match_accepts_optional_brand_model():
    assert _match(row(), alert(brand=None, model=None)) is True
