"""Suivi longitudinal de la fiabilité ML et du drift AutoDeal Tunisie.

Chaque run nocturne enregistre des métriques hors-échantillon exploitables dans
le dashboard Admin. Le MdAPE reste la métrique centrale, mais il est complété
par des métriques métier : erreur en DT, queue d'erreur (P90), couverture à
±5/10/15 %, biais signé et taille des segments.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from config import PROCESSED_FILES, PROCESSED_DATA_DIR
except Exception:  # pragma: no cover
    ROOT = Path(__file__).resolve().parents[1]
    PROCESSED_DATA_DIR = ROOT / "data" / "processed"
    PROCESSED_FILES = {"scored": str(PROCESSED_DATA_DIR / "tunisia-cars-scored.csv")}

HISTORIQUE_PATH = Path(PROCESSED_DATA_DIR) / "historique_performance.json"
BORNES_PRIX = [0, 15000, 25000, 35000, 50000, 75000, 100000, float("inf")]
LIBELLES_PRIX = ["< 15k", "15-25k", "25-35k", "35-50k", "50-75k", "75-100k", "100k +"]
K_REFERENCE = 7
SEUIL_DRIFT_PRIX = 0.15
SEUIL_DRIFT_VOLUME = 0.35
SEUIL_DRIFT_MIX_SOURCE = 0.15
SEP, ENC = ";", "utf-8-sig"


def _metriques_erreur(g: pd.DataFrame) -> dict:
    """Carte d'identité de performance d'un échantillon hors-échantillon."""
    rel = g["_err_rel"]
    abs_dt = g["_err_dt"]
    signed = g["_err_signed"]
    return {
        "n": int(len(g)),
        "mdape_pct": round(100 * rel.median(), 2),
        "mdae_dt": round(float(abs_dt.median()), 0),
        "mae_dt": round(float(abs_dt.mean()), 0),
        "p90_erreur_pct": round(100 * rel.quantile(0.90), 2),
        "dans_5pct": round(100 * (rel <= 0.05).mean(), 1),
        "dans_10pct": round(100 * (rel <= 0.10).mean(), 1),
        "dans_15pct": round(100 * (rel <= 0.15).mean(), 1),
        "biais_median_pct": round(100 * signed.median(), 2),
    }


def _performance(df: pd.DataFrame) -> tuple[dict | None, dict]:
    """Métriques globales + par tranche de prix."""
    if "Prix_Theorique" not in df.columns:
        return None, {}
    d = df.dropna(subset=["Prix", "Prix_Theorique"]).copy()
    d = d[(d["Prix"] > 0) & (d["Prix_Theorique"] > 0)]
    if len(d) < 50:
        return None, {}
    d["_err_dt"] = (d["Prix_Theorique"] - d["Prix"]).abs()
    d["_err_rel"] = d["_err_dt"] / d["Prix"]
    d["_err_signed"] = (d["Prix_Theorique"] - d["Prix"]) / d["Prix"]
    global_metrics = _metriques_erreur(d)
    d["_tr"] = pd.cut(d["Prix"], bins=BORNES_PRIX, labels=LIBELLES_PRIX)
    segments = {}
    for tr, g in d.groupby("_tr", observed=True):
        if len(g) >= 10:
            segments[str(tr)] = _metriques_erreur(g)
    return global_metrics, segments


def _instantane(df: pd.DataFrame) -> dict:
    snap = {"n": int(len(df))}
    for col, cle in [("Prix", "prix_median"), ("Age_Vehicule", "age_median"), ("Kilométrage", "km_median")]:
        if col in df.columns:
            v = pd.to_numeric(df[col], errors="coerce").median()
            snap[cle] = None if pd.isna(v) else round(float(v), 1)
    if "Source" in df.columns and len(df):
        parts = (df["Source"].value_counts(normalize=True) * 100).round(1)
        snap["mix_sources"] = {str(k): float(v) for k, v in parts.items()}
    snap["completude"] = {c: round(float(df[c].notna().mean() * 100), 1)
                          for c in ["Prix", "Marque", "Modèle"] if c in df.columns}
    return snap


def _charger_historique() -> list[dict]:
    if HISTORIQUE_PATH.exists():
        try:
            return json.loads(HISTORIQUE_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass
    return []


def _detecter_drift(courant: dict, historique: list[dict]) -> list[str]:
    refs = [h["instantane"] for h in historique[-K_REFERENCE:] if "instantane" in h]
    if len(refs) < 3:
        return []
    alertes = []

    def med(cle):
        vals = [r[cle] for r in refs if r.get(cle) is not None]
        return float(np.median(vals)) if vals else None

    ref_prix, cur_prix = med("prix_median"), courant.get("prix_median")
    if ref_prix and cur_prix and abs(cur_prix - ref_prix) / ref_prix > SEUIL_DRIFT_PRIX:
        alertes.append(f"Prix médian {cur_prix:.0f} vs réf {ref_prix:.0f} ({(cur_prix-ref_prix)/ref_prix:+.0%})")
    ref_n, cur_n = med("n"), courant.get("n")
    if ref_n and cur_n and abs(cur_n - ref_n) / ref_n > SEUIL_DRIFT_VOLUME:
        alertes.append(f"Volume {cur_n} vs réf {ref_n:.0f} ({(cur_n-ref_n)/ref_n:+.0%})")
    cur_mix, ref_mix = courant.get("mix_sources", {}), {}
    for r in refs:
        for s, p in r.get("mix_sources", {}).items():
            ref_mix.setdefault(s, []).append(p)
    for s in set(cur_mix) | set(ref_mix):
        cur_p, ref_p = cur_mix.get(s, 0.0), float(np.median(ref_mix.get(s, [0.0])))
        if abs(cur_p - ref_p) > SEUIL_DRIFT_MIX_SOURCE * 100:
            alertes.append(f"Source {s} : {cur_p:.0f}% vs réf {ref_p:.0f}%")
    return alertes


def enregistrer(strict: bool = False) -> int:
    print("=" * 60)
    print("  SUIVI PERFORMANCE — AutoDeal Tunisie")
    print("=" * 60)
    scored_path = Path(PROCESSED_FILES.get("scored", ""))
    if not scored_path.exists():
        print("::warning::Fichier scoré introuvable — rien à enregistrer.")
        return 0
    df = pd.read_csv(scored_path, sep=SEP, encoding=ENC, low_memory=False)
    perf, segments = _performance(df)
    snap = _instantane(df)
    historique = _charger_historique()
    drift = _detecter_drift(snap, historique)

    enreg = {
        "date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        # Compatibilité dashboard historique :
        "mdape_global": None if perf is None else perf["mdape_pct"],
        "mdape_par_tranche": {k: v["mdape_pct"] for k, v in segments.items()},
        # Nouveau contrat riche :
        "performance": perf,
        "performance_par_tranche": segments,
        "instantane": snap,
        "drift": drift,
    }
    historique.append(enreg)
    try:
        HISTORIQUE_PATH.write_text(json.dumps(historique, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as e:
        print(f"::warning::Écriture de l'historique impossible : {e}")

    print(f"  Date         : {enreg['date']}")
    print(f"  Volume       : {snap.get('n')}")
    if perf:
        print(f"  MdAPE        : {perf['mdape_pct']} %")
        print(f"  MdAE         : {perf['mdae_dt']:.0f} DT")
        print(f"  P90 erreur   : {perf['p90_erreur_pct']} %")
        print(f"  <=10 %       : {perf['dans_10pct']} % des annonces")
        print(f"  Biais médian : {perf['biais_median_pct']:+.2f} %")
    print(f"  Historique   : {len(historique)} run(s)")
    if drift:
        for a in drift:
            print(f"::warning::DRIFT — {a}")
        if strict:
            return 1
    else:
        print("  Drift        : aucun (ou historique trop court)")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(enregistrer(strict="--strict" in sys.argv))
