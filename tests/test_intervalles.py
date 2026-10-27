import numpy as np
import pandas as pd

from core.intervalles import calibrer, couverture, intervalle, valider_temporellement


def _donnees(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    pred = rng.uniform(np.log1p(15000), np.log1p(150000), n)
    # Erreur plus forte en haut de gamme, comme en réalité.
    bruit = rng.normal(0, 0.12 + 0.10 * (pred > np.log1p(75000)), n)
    return pred + bruit, pred


def test_couverture_proche_du_niveau_et_largeur_adaptee_a_la_tranche():
    y, pred = _donnees()
    calib = calibrer(y, pred)
    cov = couverture(np.expm1(y), pred, calib)
    assert 78 <= cov["couverture_pct"] <= 83
    b1, h1 = intervalle(30000.0, calib)
    b2, h2 = intervalle(120000.0, calib)
    assert (h2 - b2) / 120000 > (h1 - b1) / 30000   # plus large en haut de gamme
    assert b1 < 30000 < h1


def test_validation_temporelle_mesure_hors_echantillon():
    y, pred = _donnees(seed=1)
    dates = pd.Series(pd.date_range("2026-08-01", periods=len(y), freq="h"))
    res = valider_temporellement(dates, y, pred, np.expm1(y))
    assert res is not None and 75 <= res["couverture_pct"] <= 86


def test_sans_calibration_renvoie_nan():
    b, h = intervalle(30000.0, None)
    assert np.isnan(b) and np.isnan(h)
