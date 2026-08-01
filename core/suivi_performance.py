"""
Suivi des performances du modèle & drift des données, dans le TEMPS.

À lancer à chaque run du pipeline (après le scoring). Chaque exécution ajoute
un enregistrement horodaté à `data/processed/historique_performance.json` :

  - qualité du modèle : MdAPE global + par tranche de prix ;
  - instantané des données : volume, prix / âge / km médians, mix des sources,
    complétude des champs clés ;
  - drift : écart de l'instantané courant vs la médiane des K derniers runs.

Au bout de quelques semaines, l'historique permet de tracer « la MdAPE
reste-t-elle stable ? » et « la composition du marché dérive-t-elle ? » — le
prochain grand saut du projet (montrer la fiabilité *dans la durée*, pas juste
sur un snapshot).

Usage :
    python core/suivi_performance.py
    python core/suivi_performance.py --strict   # sort en erreur si drift détecté
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from config import PROCESSED_FILES, PROCESSED_DATA_DIR, SCRAPERS
except Exception:  # pragma: no cover
    ROOT = Path(__file__).resolve().parents[1]
    PROCESSED_DATA_DIR = ROOT / "data" / "processed"
    RAW = ROOT / "data" / "raw"
    PROCESSED_FILES = {"scored": str(PROCESSED_DATA_DIR / "tunisia-cars-scored.csv")}
    SCRAPERS = {n: str(RAW / f"{n}.csv") for n in
                ["tayara", "automax", "automobile", "sayyarat", "autocentral"]}

HISTORIQUE_PATH = Path(PROCESSED_DATA_DIR) / "historique_performance.json"

# Tranches de prix (mêmes bornes que la courbe de fiabilité du dashboard)
BORNES_PRIX = [0, 15000, 25000, 35000, 50000, 75000, 100000, float("inf")]
LIBELLES_PRIX = ["< 15k", "15-25k", "25-35k", "35-50k", "50-75k", "75-100k", "100k +"]

# Drift : nombre de runs de référence + seuils d'alerte
K_REFERENCE = 7
SEUIL_DRIFT_PRIX = 0.15       # ±15 % sur le prix médian
SEUIL_DRIFT_VOLUME = 0.35     # ±35 % sur le volume
SEUIL_DRIFT_MIX_SOURCE = 0.15  # ±15 pts sur la part d'une source

SEP, ENC = ";", "utf-8-sig"


# --------------------------------------------------------------------------
def _mdape_par_tranche(df: pd.DataFrame) -> tuple[float | None, dict]:
    """MdAPE global et par tranche de prix (erreur relative médiane hors-échantillon)."""
    if "Prix_Theorique" not in df.columns:
        return None, {}
    d = df.dropna(subset=["Prix", "Prix_Theorique"]).copy()
    d = d[(d["Prix"] > 0) & (d["Prix_Theorique"] > 0)]
    if len(d) < 50:
        return None, {}
    d["_err"] = (d["Prix_Theorique"] - d["Prix"]).abs() / d["Prix"]
    global_mdape = round(100 * d["_err"].median(), 2)
    d["_tr"] = pd.cut(d["Prix"], bins=BORNES_PRIX, labels=LIBELLES_PRIX)
    par_tranche = {}
    for tr, g in d.groupby("_tr", observed=True):
        if len(g) >= 10:
            par_tranche[str(tr)] = round(100 * g["_err"].median(), 2)
    return global_mdape, par_tranche


def _instantane(df: pd.DataFrame) -> dict:
    """Instantané des données : volume, médianes, mix des sources, complétude."""
    snap: dict = {"n": int(len(df))}
    for col, cle in [("Prix", "prix_median"), ("Age_Vehicule", "age_median"),
                     ("Kilométrage", "km_median")]:
        if col in df.columns:
            v = pd.to_numeric(df[col], errors="coerce").median()
            snap[cle] = None if pd.isna(v) else round(float(v), 1)
    if "Source" in df.columns and len(df):
        parts = (df["Source"].value_counts(normalize=True) * 100).round(1)
        snap["mix_sources"] = {str(k): float(v) for k, v in parts.items()}
    snap["completude"] = {
        c: round(float(df[c].notna().mean() * 100), 1)
        for c in ["Prix", "Marque", "Modèle"] if c in df.columns
    }
    return snap


def _charger_historique() -> list[dict]:
    if HISTORIQUE_PATH.exists():
        try:
            return json.loads(HISTORIQUE_PATH.read_text(encoding="utf-8"))
        except Exception:
            return []
    return []


def _detecter_drift(courant: dict, historique: list[dict]) -> list[str]:
    """Compare l'instantané courant à la MÉDIANE des K derniers runs enregistrés."""
    refs = [h["instantane"] for h in historique[-K_REFERENCE:] if "instantane" in h]
    if len(refs) < 3:  # pas assez d'historique pour un référentiel crédible
        return []
    alertes = []

    def med(cle):
        vals = [r[cle] for r in refs if r.get(cle) is not None]
        return float(np.median(vals)) if vals else None

    # Prix médian
    ref_prix, cur_prix = med("prix_median"), courant.get("prix_median")
    if ref_prix and cur_prix and abs(cur_prix - ref_prix) / ref_prix > SEUIL_DRIFT_PRIX:
        alertes.append(f"Prix médian {cur_prix:.0f} vs réf {ref_prix:.0f} "
                       f"({(cur_prix - ref_prix) / ref_prix:+.0%})")
    # Volume
    ref_n, cur_n = med("n"), courant.get("n")
    if ref_n and cur_n and abs(cur_n - ref_n) / ref_n > SEUIL_DRIFT_VOLUME:
        alertes.append(f"Volume {cur_n} vs réf {ref_n:.0f} "
                       f"({(cur_n - ref_n) / ref_n:+.0%})")
    # Mix des sources (variation de part la plus forte)
    cur_mix = courant.get("mix_sources", {})
    ref_mix: dict = {}
    for r in refs:
        for s, p in r.get("mix_sources", {}).items():
            ref_mix.setdefault(s, []).append(p)
    for s in set(cur_mix) | set(ref_mix):
        cur_p = cur_mix.get(s, 0.0)
        ref_p = float(np.median(ref_mix.get(s, [0.0])))
        if abs(cur_p - ref_p) > SEUIL_DRIFT_MIX_SOURCE * 100:
            alertes.append(f"Source {s} : {cur_p:.0f}% vs réf {ref_p:.0f}%")
    return alertes


# --------------------------------------------------------------------------
def enregistrer(strict: bool = False) -> int:
    print("=" * 60)
    print("  SUIVI PERFORMANCE — AutoDeal Tunisie")
    print("=" * 60)

    scored_path = Path(PROCESSED_FILES.get("scored", ""))
    if not scored_path.exists():
        print("::warning::Fichier scoré introuvable — rien à enregistrer.")
        return 0
    df = pd.read_csv(scored_path, sep=SEP, encoding=ENC, low_memory=False)

    mdape, mdape_tr = _mdape_par_tranche(df)
    snap = _instantane(df)
    historique = _charger_historique()
    drift = _detecter_drift(snap, historique)

    enreg = {
        "date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "mdape_global": mdape,
        "mdape_par_tranche": mdape_tr,
        "instantane": snap,
        "drift": drift,
    }
    historique.append(enreg)
    try:
        HISTORIQUE_PATH.write_text(
            json.dumps(historique, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as e:
        print(f"::warning::Écriture de l'historique impossible : {e}")

    # Rapport
    print(f"  Date         : {enreg['date']}")
    print(f"  Volume       : {snap.get('n')}")
    print(f"  MdAPE global : {mdape} %" if mdape is not None else "  MdAPE global : n/a")
    if mdape_tr:
        print("  MdAPE / tranche : "
              + ", ".join(f"{k} {v}%" for k, v in mdape_tr.items()))
    print(f"  Prix médian  : {snap.get('prix_median')} DT")
    print(f"  Historique   : {len(historique)} run(s) enregistré(s)")

    if drift:
        for a in drift:
            print(f"::warning::DRIFT — {a}")
        if strict:
            print("=" * 60)
            return 1
    else:
        print("  Drift        : aucun (ou historique trop court)")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(enregistrer(strict="--strict" in sys.argv))
