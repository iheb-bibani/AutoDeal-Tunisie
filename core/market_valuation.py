"""Valorisation indépendante par comparables de marché.

Ce module ne dépend PAS du modèle ML. Il construit un voisinage de véhicules
réellement comparables à partir des annonces observées, puis calcule une
fourchette empirique (Q25-Q75), une médiane et des indicateurs de dispersion.

L'objectif est de disposer de deux avis indépendants :
- Market Comparable Valuation : ce que disent les annonces comparables ;
- ML Valuation : ce que prédit le modèle de prix.

Les deux peuvent ensuite être confrontés sans circularité.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MarketValuation:
    n_comparables: int
    median_price: float | None
    p10: float | None
    q25: float | None
    q75: float | None
    p90: float | None
    iqr: float | None
    relative_width: float | None
    homogeneity: str
    confidence: str
    selection_level: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _num(value):
    try:
        v = float(value)
        return v if np.isfinite(v) else np.nan
    except Exception:
        return np.nan


def _text(value):
    if value is None or pd.isna(value):
        return ""
    return str(value).strip().casefold()


def _age_from_row(row: pd.Series | dict) -> float:
    age = _num(row.get("Age_Vehicule"))
    if np.isfinite(age):
        return age
    year = _num(row.get("Année"))
    if np.isfinite(year):
        return max(pd.Timestamp.now().year - year, 0)
    return np.nan


def _exclude_target(df: pd.DataFrame, target: pd.Series | dict) -> pd.DataFrame:
    out = df
    link = target.get("Lien")
    if isinstance(link, str) and link.strip() and "Lien" in out.columns:
        out = out[out["Lien"].astype(str) != link]
    return out


def select_market_comparables(
    df: pd.DataFrame,
    target: pd.Series | dict,
    *,
    min_n: int = 5,
    max_n: int = 40,
) -> tuple[pd.DataFrame, str]:
    """Retourne des comparables empiriques, sans utiliser Prix_Theorique.

    Marque + modèle sont obligatoires. Le voisinage est ensuite relâché par
    paliers uniquement si l'échantillon est trop petit. À l'intérieur du palier,
    les annonces les plus proches en âge/km sont conservées.
    """
    if df is None or len(df) == 0:
        return pd.DataFrame(), "aucun"

    brand = _text(target.get("Marque"))
    model = _text(target.get("Modèle"))
    if not brand or not model or "Marque" not in df.columns or "Modèle" not in df.columns:
        return pd.DataFrame(), "aucun"

    work = df.copy()
    price = pd.to_numeric(work.get("Prix"), errors="coerce")
    work = work[price.gt(0) & price.notna()].copy()
    work = _exclude_target(work, target)
    work = work[
        work["Marque"].astype(str).str.strip().str.casefold().eq(brand)
        & work["Modèle"].astype(str).str.strip().str.casefold().eq(model)
    ].copy()
    if work.empty:
        return work, "aucun"

    target_age = _age_from_row(target)
    target_km = _num(target.get("Kilométrage"))
    target_energy = _text(target.get("Energie"))
    target_trans = _text(target.get("Boite_Vitesse"))

    if "Age_Vehicule" in work.columns:
        work["_age"] = pd.to_numeric(work["Age_Vehicule"], errors="coerce")
    elif "Année" in work.columns:
        work["_age"] = pd.Timestamp.now().year - pd.to_numeric(work["Année"], errors="coerce")
    else:
        work["_age"] = np.nan
    work["_km"] = pd.to_numeric(work.get("Kilométrage"), errors="coerce")

    trans_col = "Boite_Vitesse" if "Boite_Vitesse" in work.columns else None

    # (nom, tolérance âge, tolérance km absolue min, tolérance km relative,
    #  énergie stricte, transmission stricte)
    levels = [
        ("très proche", 1, 20_000, 0.25, True, True),
        ("proche", 2, 30_000, 0.35, True, False),
        ("élargi", 3, 50_000, 0.50, False, False),
    ]

    chosen = pd.DataFrame()
    chosen_name = "aucun"
    for name, age_tol, km_min, km_ratio, energy_strict, trans_strict in levels:
        cand = work.copy()
        if np.isfinite(target_age):
            cand = cand[cand["_age"].isna() | (cand["_age"] - target_age).abs().le(age_tol)]
        if np.isfinite(target_km):
            km_tol = max(float(km_min), float(km_ratio) * max(target_km, 40_000.0))
            cand = cand[cand["_km"].isna() | (cand["_km"] - target_km).abs().le(km_tol)]
        if energy_strict and target_energy and "Energie" in cand.columns:
            e = cand["Energie"].fillna("").astype(str).str.strip().str.casefold()
            # Les valeurs manquantes restent admises ; une valeur connue contradictoire non.
            cand = cand[e.eq("") | e.eq(target_energy)]
        if trans_strict and target_trans and trans_col:
            t = cand[trans_col].fillna("").astype(str).str.strip().str.casefold()
            strict = cand[t.eq("") | t.eq(target_trans)]
            # N'impose la transmission que si cela laisse un échantillon exploitable.
            if len(strict) >= min_n:
                cand = strict

        if len(cand) >= min_n:
            chosen, chosen_name = cand, name
            break
        if len(cand) > len(chosen):
            chosen, chosen_name = cand, name

    if chosen.empty:
        return chosen, chosen_name

    # Distance uniquement pour classer des comparables déjà admissibles.
    age_distance = (chosen["_age"] - target_age).abs().fillna(1.5) if np.isfinite(target_age) else pd.Series(0.0, index=chosen.index)
    if np.isfinite(target_km):
        km_scale = max(target_km, 40_000.0)
        km_distance = ((chosen["_km"] - target_km).abs() / km_scale).fillna(0.5)
    else:
        km_distance = pd.Series(0.0, index=chosen.index)
    chosen = chosen.assign(_market_distance=age_distance + km_distance).sort_values("_market_distance")
    return chosen.head(max_n).copy(), chosen_name


def summarize_market(comparables: pd.DataFrame, selection_level: str = "") -> MarketValuation:
    if comparables is None or comparables.empty or "Prix" not in comparables.columns:
        return MarketValuation(0, None, None, None, None, None, None, None, "indisponible", "faible", selection_level or "aucun")

    prices = pd.to_numeric(comparables["Prix"], errors="coerce")
    prices = prices[(prices > 0) & prices.notna()]
    n = int(len(prices))
    if n == 0:
        return MarketValuation(0, None, None, None, None, None, None, None, "indisponible", "faible", selection_level or "aucun")

    median = float(prices.median())
    p10 = float(prices.quantile(0.10))
    q25 = float(prices.quantile(0.25))
    q75 = float(prices.quantile(0.75))
    p90 = float(prices.quantile(0.90))
    iqr = q75 - q25
    rel = iqr / median if median > 0 else np.nan

    if rel <= 0.15:
        hom = "élevée"
    elif rel <= 0.30:
        hom = "modérée"
    else:
        hom = "faible"

    # La confiance combine taille d'échantillon et dispersion observée.
    if n >= 12 and rel <= 0.25:
        conf = "élevée"
    elif n >= 5 and rel <= 0.45:
        conf = "moyenne"
    else:
        conf = "faible"

    return MarketValuation(n, median, p10, q25, q75, p90, iqr, float(rel), hom, conf, selection_level or "")


def market_valuation(df: pd.DataFrame, target: pd.Series | dict, *, min_n: int = 5, max_n: int = 40):
    comparables, level = select_market_comparables(df, target, min_n=min_n, max_n=max_n)
    return summarize_market(comparables, level), comparables


def compare_ml_to_market(ml_value: float | None, market: MarketValuation) -> dict[str, Any]:
    ml = _num(ml_value)
    if not np.isfinite(ml) or market.median_price is None:
        return {
            "available": False,
            "inside_market_range": None,
            "inside_market_range_p10_p90": None,
            "market_position": None,
            "gap_vs_market_median": None,
            "gap_vs_market_median_pct": None,
        }

    inside = bool(market.q25 <= ml <= market.q75) if market.q25 is not None and market.q75 is not None else None
    inside_wide = bool(market.p10 <= ml <= market.p90) if market.p10 is not None and market.p90 is not None else None
    if market.q25 is None or market.q75 is None:
        position = None
    elif ml < market.q25:
        position = "Sous Q25"
    elif ml > market.q75:
        position = "Au-dessus Q75"
    else:
        position = "Dans Q25–Q75"

    gap = float(ml - market.median_price)
    gap_pct = gap / market.median_price if market.median_price else np.nan
    return {
        "available": True,
        "inside_market_range": inside,
        "inside_market_range_p10_p90": inside_wide,
        "market_position": position,
        "gap_vs_market_median": gap,
        "gap_vs_market_median_pct": float(gap_pct),
    }


def evaluate_market_ml_coverage(
    df: pd.DataFrame,
    *,
    ml_col: str = "Prix_Theorique",
    min_n: int = 5,
    max_rows: int | None = None,
) -> pd.DataFrame:
    """Évalue, en leave-one-out, l'accord entre ML et fourchette de marché.

    Chaque ligne est exclue de ses propres comparables via son URL. Ainsi le
    KPI de couverture ne bénéficie pas artificiellement du prix de l'annonce
    qu'il cherche à contrôler.
    """
    if df is None or df.empty or ml_col not in df.columns:
        return pd.DataFrame()

    source = df.copy()
    if max_rows and len(source) > max_rows:
        source = source.sample(max_rows, random_state=42)

    # Pré-groupe marque/modèle : la comparaison impose de toute façon ce couple,
    # ce qui évite de rescanner tout le marché pour chaque annonce.
    group_map = {}
    if {"Marque", "Modèle"}.issubset(df.columns):
        keys = (df["Marque"].fillna("").astype(str).str.strip().str.casefold()
                + "||" + df["Modèle"].fillna("").astype(str).str.strip().str.casefold())
        for key, idxs in keys.groupby(keys).groups.items():
            group_map[key] = df.loc[list(idxs)]

    rows = []
    for idx, row in source.iterrows():
        ml = _num(row.get(ml_col))
        if not np.isfinite(ml) or ml <= 0:
            continue
        key = _text(row.get("Marque")) + "||" + _text(row.get("Modèle"))
        pool = group_map.get(key, df)
        market, _ = market_valuation(pool, row, min_n=min_n)
        if market.n_comparables < min_n or market.median_price is None:
            continue
        agreement = compare_ml_to_market(ml, market)
        actual = _num(row.get("Prix"))
        rows.append({
            "index": idx,
            "Marque": row.get("Marque"),
            "Modèle": row.get("Modèle"),
            "Année": row.get("Année"),
            "Age_Vehicule": _age_from_row(row),
            "Kilométrage": row.get("Kilométrage"),
            "Prix": actual,
            "Prix_Theorique": ml,
            "Market_Median": market.median_price,
            "Market_P10": market.p10,
            "Market_Q25": market.q25,
            "Market_Q75": market.q75,
            "Market_P90": market.p90,
            "Market_N": market.n_comparables,
            "Market_Relative_Width": market.relative_width,
            "Market_Homogeneity": market.homogeneity,
            "Market_Confidence": market.confidence,
            "ML_In_Market_Range": agreement["inside_market_range"],
            "ML_In_Market_Range_P10_P90": agreement["inside_market_range_p10_p90"],
            "ML_Market_Position": agreement["market_position"],
            "ML_Market_Gap_Pct": agreement["gap_vs_market_median_pct"],
        })
    return pd.DataFrame(rows)
