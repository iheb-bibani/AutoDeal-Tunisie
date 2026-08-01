import pandas as pd

from core.market_valuation import market_valuation, compare_ml_to_market, evaluate_market_ml_coverage


def _df():
    rows = []
    for i, price in enumerate([45_000, 47_000, 49_000, 50_000, 52_000, 54_000, 56_000, 90_000]):
        rows.append({
            "Lien": f"https://x/{i}", "Marque": "Peugeot", "Modèle": "208",
            "Année": 2021 + (i % 2), "Age_Vehicule": 5 - (i % 2),
            "Kilométrage": 60_000 + i * 2_000, "Energie": "Essence",
            "Boite_Vitesse": "Manuelle", "Prix": price, "Prix_Theorique": 51_000,
        })
    rows.append({"Lien":"https://x/golf", "Marque":"Volkswagen", "Modèle":"Golf", "Année":2021,
                 "Age_Vehicule":5, "Kilométrage":65_000, "Energie":"Essence", "Boite_Vitesse":"Manuelle",
                 "Prix":100_000, "Prix_Theorique":100_000})
    return pd.DataFrame(rows)


def test_market_is_independent_and_excludes_other_models():
    df = _df()
    target = df.iloc[0]
    market, comp = market_valuation(df, target, min_n=5)
    assert market.n_comparables >= 5
    assert set(comp["Modèle"]) == {"208"}
    assert "https://x/0" not in set(comp["Lien"])
    assert market.p10 <= market.q25 < market.median_price < market.q75 <= market.p90


def test_ml_market_agreement():
    df = _df()
    market, _ = market_valuation(df, df.iloc[0], min_n=5)
    a = compare_ml_to_market(51_000, market)
    assert a["available"] is True
    assert a["inside_market_range"] is True
    assert a["inside_market_range_p10_p90"] is True
    assert a["market_position"] == "Dans Q25–Q75"


def test_ml_market_directional_position():
    df = _df()
    market, _ = market_valuation(df, df.iloc[0], min_n=5)
    low = compare_ml_to_market(market.q25 - 1, market)
    high = compare_ml_to_market(market.q75 + 1, market)
    assert low["market_position"] == "Sous Q25"
    assert high["market_position"] == "Au-dessus Q75"


def test_coverage_is_leave_one_out():
    result = evaluate_market_ml_coverage(_df(), min_n=5)
    assert len(result) >= 7
    assert "ML_In_Market_Range" in result.columns
    assert "ML_In_Market_Range_P10_P90" in result.columns
    assert "ML_Market_Position" in result.columns
    assert {"Market_P10", "Market_P90"}.issubset(result.columns)
    assert result["Market_N"].min() >= 5
