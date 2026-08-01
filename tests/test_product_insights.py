import pandas as pd
from core.market_valuation import MarketValuation
from core.product_insights import valuation_confidence


def test_confidence_high_when_market_and_model_agree():
    market = MarketValuation(20, 50000, 45000, 47000, 53000, 56000, 6000, 0.12, "élevée", "élevée", "proche")
    row = {"Erreur_Relative_Modele": 0.08}
    score, label, parts = valuation_confidence(row, market, {
        "available": True,
        "inside_market_range": True,
        "inside_market_range_p10_p90": True,
    })
    assert score >= 80
    assert label == "Élevée"
    assert parts["Accord ML ↔ marché"] == 100


def test_confidence_falls_with_sparse_dispersion_and_disagreement():
    market = MarketValuation(3, 50000, 30000, 35000, 70000, 80000, 35000, 0.70, "faible", "faible", "élargi")
    row = {"Erreur_Relative_Modele": 0.22}
    score, label, _ = valuation_confidence(row, market, {
        "available": True,
        "inside_market_range": False,
        "inside_market_range_p10_p90": False,
    })
    assert score < 60
    assert label == "Faible"
