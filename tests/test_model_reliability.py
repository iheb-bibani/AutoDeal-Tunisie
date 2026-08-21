import pandas as pd

from core.model_reliability import (
    confidence_label,
    metrics,
    prepare_errors,
    reliability_by_price,
    reliability_by_segments,
)


def _sample(n=30):
    rows = []
    for i in range(n):
        price = 40_000 + (i % 5) * 1_000
        pred = price * (1.05 if i % 2 == 0 else 0.95)
        rows.append({
            "Prix": price,
            "Prix_Theorique": pred,
            "Marque": "Peugeot",
            "Modèle": "208",
            "Age_Vehicule": 5,
            "Kilométrage": 80_000,
            "Energie": "Essence",
        })
    return pd.DataFrame(rows)


def test_prepare_errors_and_metrics():
    d = prepare_errors(_sample())
    m = metrics(d)
    assert len(d) == 30
    assert 4.9 <= m["mdape_pct"] <= 5.1
    assert m["within_10_pct"] == 100.0
    assert m["p90_pct"] <= 5.1


def test_reliability_by_price_contains_volume_and_tail_metrics():
    result = reliability_by_price(_sample(), min_n=10)
    assert len(result) == 1
    assert result.iloc[0]["Tranche prix"] == "35–50k"
    assert result.iloc[0]["n"] == 30
    assert {"mdape_pct", "mdae_dt", "p90_pct", "within_10_pct", "bias_median_pct"}.issubset(result.columns)


def test_reliability_by_segments_exposes_expected_dimensions():
    result = reliability_by_segments(_sample())
    assert {"Marque", "Modèle", "Énergie", "Âge", "Kilométrage"}.issubset(result)


def test_confidence_label_uses_error_and_sample_size():
    assert confidence_label(6, 100) == "Élevée"
    assert confidence_label(9, 100) == "Bonne"
    assert confidence_label(12, 100) == "Modérée"
    assert confidence_label(18, 100) == "Faible"
    assert confidence_label(6, 10) == "Modérée"
