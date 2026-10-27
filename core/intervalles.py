"""Intervalles de prix calibrés (prédiction conforme sur résidus out-of-fold).

L'application affichait « estimation ± erreur médiane » : par construction,
une telle fourchette ne contient le vrai prix qu'environ une fois sur deux,
quelle que soit la voiture. Ici, on apprend sur les prédictions out-of-fold
la distribution réelle des écarts (en log-prix), par tranche de prix estimé :
l'intervalle [q10 ; q90] des résidus donne une fourchette qui contient le prix
observé dans ~80 % des cas, plus large là où le modèle se trompe davantage
(haut de gamme, véhicules rares) et asymétrique si nécessaire.

La couverture réelle est contrôlée sur un holdout temporel (quantiles appris
sur les annonces anciennes, mesurés sur les plus récentes) et publiée dans
les diagnostics.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

NIVEAU = 0.80
from config import BORNES_TRANCHES_PRIX as BORNES_TRANCHES  # noqa: E402
from config import LIBELLES_TRANCHES_PRIX as LIBELLES_TRANCHES  # noqa: E402
MIN_PAR_TRANCHE = 80


def _quantiles(residus: np.ndarray, niveau: float) -> dict:
    alpha = (1 - niveau) / 2
    n = len(residus)
    # Correction conforme en échantillon fini : quantile ceil((n+1)(1-a))/n.
    hi = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    lo = max(0.0, np.floor((n + 1) * alpha) / n)
    return {
        "q_bas": float(np.quantile(residus, lo)),
        "q_haut": float(np.quantile(residus, hi)),
        "n": int(n),
    }


def tranche_de(prix) -> pd.Series:
    return pd.cut(pd.Series(prix, dtype=float), bins=BORNES_TRANCHES, labels=LIBELLES_TRANCHES).astype(str)


def calibrer(y_log, pred_log, niveau: float = NIVEAU) -> dict:
    """Quantiles des résidus log (observé − prédit), globaux et par tranche
    de prix PRÉDIT (la tranche réelle est inconnue à l'inférence)."""
    y_log = np.asarray(y_log, dtype=float)
    pred_log = np.asarray(pred_log, dtype=float)
    ok = np.isfinite(y_log) & np.isfinite(pred_log)
    residus = y_log[ok] - pred_log[ok]
    tranches = tranche_de(np.expm1(pred_log[ok])).to_numpy()
    calib = {"niveau": niveau, "echelle": "log-prix", "global": _quantiles(residus, niveau), "par_tranche": {}}
    # Tranche trop petite : regroupée avec ses voisines (et non remplacée par
    # les quantiles globaux, trop étroits pour les extrêmes : les voitures très
    # bon marché ont les écarts relatifs les plus forts).
    for i, t in enumerate(LIBELLES_TRANCHES):
        if not (tranches == t).any():
            continue
        lo = hi = i
        masque = tranches == t
        while masque.sum() < MIN_PAR_TRANCHE and (lo > 0 or hi < len(LIBELLES_TRANCHES) - 1):
            lo, hi = max(lo - 1, 0), min(hi + 1, len(LIBELLES_TRANCHES) - 1)
            masque = np.isin(tranches, LIBELLES_TRANCHES[lo:hi + 1])
        if masque.sum() >= MIN_PAR_TRANCHE:
            q = _quantiles(residus[masque], niveau)
            if hi > lo:
                q["regroupe_avec"] = LIBELLES_TRANCHES[lo:hi + 1]
            calib["par_tranche"][t] = q
    return calib


def intervalle(prix_estime, calib: dict | None):
    """(bas, haut) en dinars pour un ou plusieurs prix estimés. NaN si pas de calibration."""
    prix = np.atleast_1d(np.asarray(prix_estime, dtype=float))
    if not calib:
        nan = np.full_like(prix, np.nan)
        return (nan, nan) if prix.size > 1 else (float("nan"), float("nan"))
    tranches = tranche_de(prix).to_numpy()
    q_bas = np.array([calib["par_tranche"].get(t, calib["global"])["q_bas"] for t in tranches])
    q_haut = np.array([calib["par_tranche"].get(t, calib["global"])["q_haut"] for t in tranches])
    log_p = np.log1p(prix)
    bas, haut = np.expm1(log_p + q_bas), np.expm1(log_p + q_haut)
    if prix.size == 1 and np.ndim(prix_estime) == 0:
        return float(bas[0]), float(haut[0])
    return bas, haut


def couverture(y_brut, pred_log, calib: dict) -> dict:
    """Part des prix observés tombant dans l'intervalle, et largeur relative médiane."""
    y = np.asarray(y_brut, dtype=float)
    pred = np.expm1(np.asarray(pred_log, dtype=float))
    bas, haut = intervalle(pred, calib)
    dedans = (y >= bas) & (y <= haut)
    largeur = (haut - bas) / np.maximum(pred, 1)
    return {
        "couverture_pct": round(float(np.mean(dedans)) * 100, 1),
        "largeur_relative_mediane_pct": round(float(np.median(largeur)) * 100, 1),
        "n": int(len(y)),
    }


def couverture_prospective(pipeline, X, y_log, y_brut, dates, groupes, cv_factory,
                           frac_test: float = 0.2, niveau: float = NIVEAU) -> dict | None:
    """Couverture RÉELLE sur des annonces futures.

    Le modèle est entraîné, et les quantiles calibrés (CV interne), sur les
    seules annonces antérieures à la coupure ; la couverture est mesurée sur
    les annonces postérieures (doublons du train retirés). C'est le scénario
    de production. ``valider_temporellement`` réutilisait des prédictions OOF
    issues de modèles ayant vu des annonces récentes : chiffre optimiste.
    """
    from sklearn.base import clone
    from sklearn.model_selection import cross_val_predict

    dates = pd.to_datetime(dates, errors="coerce")
    if dates.notna().sum() < 300:
        return None
    coupure = dates.dropna().quantile(1 - frac_test)
    train = dates.index[dates < coupure]
    test = dates.index[dates >= coupure]
    vus = set(groupes.loc[train])
    test = pd.Index([i for i in test if groupes.loc[i] not in vus])
    if len(train) < 300 or len(test) < 50:
        return None
    oof_train = cross_val_predict(clone(pipeline), X.loc[train], y_log.loc[train],
                                  cv=cv_factory(), groups=groupes.loc[train])
    calib = calibrer(y_log.loc[train], oof_train, niveau)
    modele = clone(pipeline).fit(X.loc[train], y_log.loc[train])
    res = couverture(y_brut.loc[test], modele.predict(X.loc[test]), calib)
    res["coupure"] = str(pd.Timestamp(coupure).date())
    return res


def valider_temporellement(dates: pd.Series, y_log, pred_log, y_brut, frac_test: float = 0.2,
                           niveau: float = NIVEAU) -> dict | None:
    """Apprend les quantiles sur les annonces anciennes, mesure la couverture
    sur les plus récentes (coupure sur une date)."""
    dates = pd.to_datetime(dates, errors="coerce")
    if dates.notna().sum() < 200:
        return None
    coupure = dates.dropna().quantile(1 - frac_test)
    train = (dates < coupure).to_numpy()
    test = (dates >= coupure).to_numpy()
    if train.sum() < 200 or test.sum() < 50:
        return None
    y_log, pred_log, y_brut = map(lambda a: np.asarray(a, dtype=float), (y_log, pred_log, y_brut))
    calib = calibrer(y_log[train], pred_log[train], niveau)
    return couverture(y_brut[test], pred_log[test], calib)
