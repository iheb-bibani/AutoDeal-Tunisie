"""Préparation des entrées et explications du modèle de prix.

Source UNIQUE pour transformer une saisie (calculateur, fiche annonce,
assistant) en ligne de modèle : mêmes regroupements de catégories rares et
même gestion des valeurs manquantes qu'à l'entraînement.

Avant ce module, chaque écran convertissait les catégorielles avec
``astype(str)`` : une valeur manquante devenait la chaîne ``"nan"`` (inconnue
de l'encodeur, au lieu d'être imputée), et un modèle rare n'était pas ramené à
``Autre_modele``. Les estimations affichées divergeaient donc des prix
théoriques publiés pour la même voiture.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

VALEURS_MANQUANTES = {"", "nan", "none", "null", "<na>"}


def _est_manquant(v) -> bool:
    if v is None:
        return True
    if isinstance(v, float) and np.isnan(v):
        return True
    return isinstance(v, str) and v.strip().lower() in VALEURS_MANQUANTES


def regrouper_categories(X: pd.DataFrame, marques_connues=None, modeles_connus=None) -> pd.DataFrame:
    """Applique le regroupement des catégories rares appris à l'entraînement.

    ``marques_connues`` / ``modeles_connus`` : ensembles conservés tels quels ;
    toute autre valeur non manquante devient ``Autre`` / ``Autre_modele``.
    Sans ensembles (anciens bundles), les valeurs sont laissées telles quelles.
    """
    X = X.copy()
    if "Marque" in X and marques_connues is not None:
        connues = set(marques_connues)
        X["Marque"] = X["Marque"].where(X["Marque"].isna() | X["Marque"].isin(connues), "Autre")
    if "Modèle" in X:
        X["Modèle"] = X["Modèle"].fillna("Inconnu")
        if modeles_connus is not None:
            connus = set(modeles_connus)
            X["Modèle"] = X["Modèle"].where(X["Modèle"].isin(connus), "Autre_modele")
    return X


def preparer_lignes(bundle: dict, lignes) -> pd.DataFrame:
    """DataFrame prêt pour ``bundle["pipeline"].predict``.

    ``lignes`` : dict, Series, liste de dicts ou DataFrame.
    Numériques -> float (NaN si invalide) ; catégorielles -> object avec
    ``np.nan`` pour les manquants (imputés par le pipeline comme à l'entraînement).
    """
    nums = list(bundle.get("features_numeriques", []))
    cats = list(bundle.get("features_categorielles", []))
    noms = nums + cats
    if isinstance(lignes, pd.DataFrame):
        brut = lignes
    elif isinstance(lignes, (dict, pd.Series)):
        brut = pd.DataFrame([{c: lignes.get(c) for c in noms}])
    else:
        brut = pd.DataFrame(list(lignes))
    X = pd.DataFrame(index=brut.index)
    for c in nums:
        X[c] = pd.to_numeric(brut[c], errors="coerce") if c in brut else np.nan
    for c in cats:
        col = brut[c] if c in brut else pd.Series([None] * len(brut), index=brut.index)
        X[c] = pd.Series(
            [np.nan if _est_manquant(v) else str(v).strip() for v in col],
            index=brut.index, dtype=object,
        )
    X = regrouper_categories(X, bundle.get("marques_connues"), bundle.get("modeles_connus"))
    return X[noms]


def _feature_origine(nom_sortie: str, noms: list[str]) -> str | None:
    """Relie une colonne transformée (``cat__Marque_Peugeot``) à sa feature."""
    base = nom_sortie.split("__", 1)[1] if "__" in nom_sortie else nom_sortie
    candidats = [n for n in noms if base == n or base.startswith(n + "_")]
    return max(candidats, key=len) if candidats else None


def shap_par_feature(pipeline, X: pd.DataFrame, noms: list[str]) -> np.ndarray:
    """Valeurs SHAP (échelle log-prix) agrégées par feature d'origine.

    Les contributions des colonnes one-hot d'une même variable sont sommées,
    ce qui rend le calcul valable quel que soit l'encodage du pipeline.
    Lève une exception si SHAP est indisponible ou incompatible.
    """
    import shap

    prep, model = pipeline.named_steps["prep"], pipeline.named_steps["model"]
    Xt = prep.transform(X[noms])
    if hasattr(Xt, "toarray"):
        Xt = Xt.toarray()
    Xt = np.asarray(Xt, dtype=float)
    sv = np.asarray(shap.TreeExplainer(model).shap_values(Xt))
    if sv.ndim == 1:
        sv = sv.reshape(1, -1)
    try:
        sorties = list(prep.get_feature_names_out())
    except Exception:
        sorties = list(noms) if Xt.shape[1] == len(noms) else None
    if sorties is None or len(sorties) != sv.shape[1]:
        raise ValueError("Impossible de relier les colonnes transformées aux features.")
    agg = np.zeros((sv.shape[0], len(noms)))
    for j, nom in enumerate(sorties):
        f = _feature_origine(nom, noms)
        if f is None:
            raise ValueError(f"Colonne transformée non reconnue : {nom}")
        agg[:, noms.index(f)] += sv[:, j]
    return agg
