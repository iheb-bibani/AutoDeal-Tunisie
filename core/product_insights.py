"""Insights produit pour AutoDeal : confiance et explication locale du prix.

Ce module garde les calculs de confiance séparés du Score AutoDeal :
- Score AutoDeal = intérêt potentiel de l'annonce ;
- Score de confiance = crédibilité de la valorisation affichée.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from core.inference import preparer_lignes, shap_par_feature

LABELS_FEATURES = {
    "Log_Kilometrage": "Kilométrage (échelle log)",
    "Km_Par_An": "Kilométrage annuel",
    "Age_Carre": "Effet non linéaire de l'âge",
    "Age_Vehicule": "Âge du véhicule",
    "Puissance_Fiscale": "Puissance fiscale (CV)",
    "Kilométrage": "Kilométrage",
    "Marque": "Marque",
    "Segment_Vehicule": "Segment",
    "Cylindree": "Cylindrée",
    "Année": "Année",
    "Boite_Vitesse": "Boîte de vitesses",
    "Modèle": "Modèle",
    "Energie": "Énergie",
}


def _model_row(bundle: dict, values: dict | pd.Series) -> pd.DataFrame:
    """Ligne de modèle préparée exactement comme à l'entraînement."""
    return preparer_lignes(bundle, values)


def _tableau_contrib(names, impacts, method):
    out = pd.DataFrame({"feature": names, "impact": impacts})
    out["label"] = out["feature"].map(LABELS_FEATURES).fillna(out["feature"])
    out["abs"] = out["impact"].abs()
    out = out[out["abs"] > 1e-9].sort_values("abs", ascending=False)
    out.attrs["method"] = method
    return out


def explain_price(bundle: dict | None, values: dict | pd.Series, df: pd.DataFrame | None = None):
    """Retourne les contributions locales au prix.

    - SHAP (méthode "SHAP") : impact en **log-prix**, colonnes one-hot
      réagrégées par variable d'origine ;
    - repli par perturbation (méthode "perturbation") : impact en **dinars**
      (prix prédit − prix prédit avec la variable ramenée à la médiane/au mode).
    L'unité est indiquée dans ``out.attrs["unit"]``.
    """
    if not bundle or "pipeline" not in bundle:
        return None
    nums = list(bundle.get("features_numeriques", []))
    cats = list(bundle.get("features_categorielles", []))
    names = nums + cats
    pipe = bundle["pipeline"]
    row = _model_row(bundle, values)

    try:
        vals = shap_par_feature(pipe, row, names)[0]
        out = _tableau_contrib(names, vals, "SHAP")
        out.attrs["unit"] = "log"
        return out
    except Exception:
        pass

    if df is None or df.empty:
        return None
    try:
        base = float(np.expm1(pipe.predict(row)[0]))
        rows = []
        source = dict(values)
        for feature in names:
            if feature not in df.columns or df[feature].dropna().empty:
                continue
            if feature in nums:
                baseline = pd.to_numeric(df[feature], errors="coerce").median()
            else:
                mode = df[feature].dropna().mode()
                if mode.empty:
                    continue
                baseline = mode.iloc[0]
            alt = dict(source)
            alt[feature] = baseline
            pred_alt = float(np.expm1(pipe.predict(_model_row(bundle, alt))[0]))
            rows.append((feature, base - pred_alt))
        if not rows:
            return None
        out = _tableau_contrib([r[0] for r in rows], [r[1] for r in rows], "perturbation")
        out.attrs["unit"] = "DT"
        return out
    except Exception:
        return None


def valuation_confidence(row: pd.Series | dict, market, comparison: dict | None = None):
    """Score de confiance 0-100, indépendant du potentiel de bonne affaire.

    Pondérations :
      30 % performance historique du ML
      25 % nombre de comparables
      25 % homogénéité du marché comparable
      20 % accord ML ↔ marché
    """
    err = pd.to_numeric(pd.Series([row.get("Erreur_Relative_Modele")]), errors="coerce").iloc[0]
    if pd.isna(err):
        model_score = 55.0
    else:
        # ~100 à 5 %, ~71 à 12 %, ~20 à 25 %.
        model_score = float(np.clip(120 - 400 * float(err), 20, 100))

    n = int(getattr(market, "n_comparables", 0) or 0)
    comp_score = float(np.clip(n / 15.0, 0, 1) * 100)

    rel = getattr(market, "relative_width", None)
    if rel is None or not np.isfinite(rel):
        hom_score = 25.0
    elif rel <= 0.15:
        hom_score = 100.0
    elif rel <= 0.30:
        hom_score = 70.0
    elif rel <= 0.45:
        hom_score = 45.0
    else:
        hom_score = 20.0

    comparison = comparison or {}
    if comparison.get("inside_market_range") is True:
        agreement_score = 100.0
    elif comparison.get("inside_market_range_p10_p90") is True:
        agreement_score = 70.0
    elif comparison.get("available"):
        agreement_score = 30.0
    else:
        agreement_score = 40.0

    score = round(0.30 * model_score + 0.25 * comp_score + 0.25 * hom_score + 0.20 * agreement_score)
    score = int(np.clip(score, 0, 100))
    if score >= 80:
        label = "Élevée"
    elif score >= 60:
        label = "Moyenne"
    else:
        label = "Faible"
    return score, label, {
        "Historique ML": round(model_score),
        "Comparables": round(comp_score),
        "Homogénéité": round(hom_score),
        "Accord ML ↔ marché": round(agreement_score),
    }
