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


def test_couverture_prospective_sur_annonces_futures():
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import GroupKFold

    from core.intervalles import couverture_prospective

    rng = np.random.default_rng(3)
    n = 1200
    X = pd.DataFrame({"x": rng.uniform(0, 1, n)})
    y = pd.Series(10 + X["x"] * 1.5 + rng.normal(0, 0.15, n))
    dates = pd.Series(pd.date_range("2026-07-01", periods=n, freq="2h"))
    groupes = pd.Series(np.arange(n).astype(str))
    res = couverture_prospective(LinearRegression(), X, y, np.expm1(y), dates, groupes,
                                 cv_factory=lambda: GroupKFold(n_splits=5))
    assert res is not None and 72 <= res["couverture_pct"] <= 88 and res["n"] >= 200


def test_petite_tranche_regroupee_avec_ses_voisines():
    y, pred = _donnees(n=3000)
    # Quelques voitures très bon marché seulement.
    pred = np.concatenate([pred, np.log1p(np.full(20, 9000.0))])
    y = np.concatenate([y, pred[-20:] + np.random.default_rng(0).normal(0, 0.4, 20)])
    calib = calibrer(y, pred)
    assert "< 15k" in calib["par_tranche"]
    assert "regroupe_avec" in calib["par_tranche"]["< 15k"]
