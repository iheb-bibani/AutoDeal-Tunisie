"""Analyses de fiabilité hors-échantillon pour le dashboard AutoDeal."""
from __future__ import annotations

import numpy as np
import pandas as pd

PRICE_BINS = [0, 15000, 25000, 35000, 50000, 75000, 100000, np.inf]
PRICE_LABELS = ["< 15k", "15–25k", "25–35k", "35–50k", "50–75k", "75–100k", "100k +"]


def prepare_errors(df: pd.DataFrame) -> pd.DataFrame:
    """Retourne les observations évaluables avec erreurs absolue, relative et signée."""
    if df is None or not {"Prix", "Prix_Theorique"}.issubset(df.columns):
        return pd.DataFrame()
    d = df.dropna(subset=["Prix", "Prix_Theorique"]).copy()
    d["Prix"] = pd.to_numeric(d["Prix"], errors="coerce")
    d["Prix_Theorique"] = pd.to_numeric(d["Prix_Theorique"], errors="coerce")
    d = d[(d["Prix"] > 0) & (d["Prix_Theorique"] > 0)]
    d["Erreur_DT"] = (d["Prix_Theorique"] - d["Prix"]).abs()
    d["Erreur_Relative"] = d["Erreur_DT"] / d["Prix"]
    d["Biais_Relatif"] = (d["Prix_Theorique"] - d["Prix"]) / d["Prix"]
    return d


def metrics(d: pd.DataFrame) -> dict:
    if d is None or d.empty:
        return {}
    e = d["Erreur_Relative"]
    return {
        "n": int(len(d)),
        "mdape_pct": float(100 * e.median()),
        "mdae_dt": float(d["Erreur_DT"].median()),
        "mae_dt": float(d["Erreur_DT"].mean()),
        "p90_pct": float(100 * e.quantile(.90)),
        "within_5_pct": float(100 * (e <= .05).mean()),
        "within_10_pct": float(100 * (e <= .10).mean()),
        "within_15_pct": float(100 * (e <= .15).mean()),
        "bias_median_pct": float(100 * d["Biais_Relatif"].median()),
    }


def reliability_by_price(df: pd.DataFrame, min_n: int = 10) -> pd.DataFrame:
    d = prepare_errors(df)
    if d.empty:
        return pd.DataFrame()
    d["Tranche prix"] = pd.cut(d["Prix"], PRICE_BINS, labels=PRICE_LABELS)
    rows = []
    for label, g in d.groupby("Tranche prix", observed=True):
        if len(g) < min_n:
            continue
        rows.append({"Tranche prix": str(label), **metrics(g)})
    return pd.DataFrame(rows)


def reliability_by_segments(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Diagnostics ciblés pour localiser les poches d'erreur."""
    d = prepare_errors(df)
    if d.empty:
        return {}
    out = {}

    def grouped(name, cols, min_n):
        rows = []
        for key, g in d.groupby(cols, observed=True):
            if len(g) < min_n:
                continue
            vals = key if isinstance(key, tuple) else (key,)
            row = {c: v for c, v in zip(cols, vals)}
            row.update(metrics(g))
            rows.append(row)
        if rows:
            out[name] = pd.DataFrame(rows).sort_values("mdape_pct", ascending=False)

    if "Marque" in d: grouped("Marque", ["Marque"], 20)
    if {"Marque", "Modèle"}.issubset(d.columns): grouped("Modèle", ["Marque", "Modèle"], 15)
    if "Energie" in d: grouped("Énergie", ["Energie"], 20)

    if "Age_Vehicule" in d:
        d["Tranche âge"] = pd.cut(pd.to_numeric(d["Age_Vehicule"], errors="coerce"),
                                  [-1, 2, 5, 8, 12, 20, np.inf],
                                  labels=["0–2", "3–5", "6–8", "9–12", "13–20", "20+"])
        grouped("Âge", ["Tranche âge"], 15)
    if "Kilométrage" in d:
        d["Tranche km"] = pd.cut(pd.to_numeric(d["Kilométrage"], errors="coerce"),
                                 [-1, 30000, 60000, 100000, 150000, 200000, np.inf],
                                 labels=["≤30k", "30–60k", "60–100k", "100–150k", "150–200k", "200k+"])
        grouped("Kilométrage", ["Tranche km"], 15)
    return out


def confidence_label(error_pct: float | None, n: int | None = None) -> str:
    """Libellé utilisateur : l'erreur empirique ET le volume comptent."""
    if error_pct is None or not np.isfinite(error_pct):
        return "Inconnue"
    if n is not None and n < 20:
        return "Modérée"
    if error_pct <= 7:
        return "Élevée"
    if error_pct <= 10:
        return "Bonne"
    if error_pct <= 15:
        return "Modérée"
    return "Faible"
