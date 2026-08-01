"""
Quality gate du pipeline AutoDeal.

Remplace le contrôle naïf « au moins 100 annonces » par des contrôles
*relatifs* et *par dimension*, à lancer APRÈS le pipeline (dans le workflow
de scraping) pour empêcher la publication de données dégradées :

  - **Volume total** : ne doit pas s'effondrer sous X % d'une référence
    glissante (la dernière exécution réussie). Détecte « 5 800 → 720 ».
  - **Par source** : chaque source directe garde un plancher minimal.
  - **Complétude** : Prix / Marque / Modèle renseignés au-dessus d'un seuil.
  - **Plausibilité** : part de prix hors bornes réalistes bornée.
  - **Doublons** : part de liens dupliqués bornée.

Sortie : code 0 si tout passe, 1 si un contrôle DUR échoue (le workflow
s'arrête et ne publie pas processed/models). Les alertes non bloquantes
sont émises en `::warning::`. En cas de succès, la référence glissante est
mise à jour.

Usage :
    python core/quality_gate.py
    python core/quality_gate.py --strict   # transforme les warnings en erreurs
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

try:
    from config import SCRAPERS, PROCESSED_FILES, PROCESSED_DATA_DIR
except Exception:  # pragma: no cover - exécuté hors contexte projet
    ROOT = Path(__file__).resolve().parents[1]
    PROCESSED_DATA_DIR = ROOT / "data" / "processed"
    RAW = ROOT / "data" / "raw"
    SCRAPERS = {n: str(RAW / f"{n}.csv") for n in
                ["tayara", "automax", "automobile", "sayyarat", "autocentral"]}
    PROCESSED_FILES = {"scored": str(PROCESSED_DATA_DIR / "tunisia-cars-scored.csv")}

# --------------------------------------------------------------------------
# Paramètres (ajustables)
# --------------------------------------------------------------------------
BASELINE_PATH = Path(PROCESSED_DATA_DIR) / "quality_baseline.json"

# Sources directes (planchers durs) vs optionnelles (planchers souples)
SOURCES_DIRECTES = ["automobile", "tayara", "automax"]
SOURCES_OPTIONNELLES = ["sayyarat", "autocentral"]
PLANCHER_SOURCE_DIRECTE = 30          # une source directe sous ce seuil = suspect
PLANCHER_TOTAL_ABSOLU = 300           # jamais publier moins que ça
FRACTION_MIN_VS_REFERENCE = 0.60      # total >= 60 % de la dernière réf réussie

# Complétude minimale (part de valeurs renseignées)
SEUIL_PRIX_RENSEIGNE = 0.95
SEUIL_MARQUE_RENSEIGNEE = 0.95
SEUIL_MODELE_RENSEIGNE = 0.80

# Plausibilité des prix (DT)
PRIX_MIN, PRIX_MAX = 1_000, 3_000_000
SEUIL_PRIX_PLAUSIBLE = 0.98

# Doublons
SEUIL_DOUBLONS = 0.02

SEP, ENC = ";", "utf-8-sig"


# --------------------------------------------------------------------------
def _log(niveau: str, msg: str) -> None:
    """Émet un message compatible GitHub Actions (::error:: / ::warning::)."""
    prefix = {"error": "::error::", "warning": "::warning::", "ok": "  ✓ ", "info": "  • "}
    print(f"{prefix.get(niveau, '')}{msg}")


def _lire_csv(chemin: str) -> pd.DataFrame | None:
    p = Path(chemin)
    if not p.exists():
        return None
    try:
        return pd.read_csv(p, sep=SEP, encoding=ENC, low_memory=False)
    except Exception as e:  # fichier illisible
        _log("warning", f"Lecture impossible de {p.name} : {e}")
        return None


def _charger_reference() -> dict:
    if BASELINE_PATH.exists():
        try:
            return json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _ecrire_reference(total: int, par_source: dict) -> None:
    ref = {"total": int(total),
           "par_source": {k: int(v) for k, v in par_source.items()},
           "updated": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    try:
        BASELINE_PATH.write_text(json.dumps(ref, ensure_ascii=False, indent=2), encoding="utf-8")
        _log("info", f"Référence glissante mise à jour ({total} annonces).")
    except Exception as e:
        _log("warning", f"Impossible d'écrire la référence : {e}")


# --------------------------------------------------------------------------
def controler() -> int:
    strict = "--strict" in sys.argv
    echecs: list[str] = []
    alertes: list[str] = []

    print("=" * 64)
    print("  QUALITY GATE — AutoDeal Tunisie")
    print("=" * 64)

    # ---- 1. Volumes par source (data/raw) --------------------------------
    par_source: dict[str, int] = {}
    for nom, chemin in SCRAPERS.items():
        df = _lire_csv(chemin)
        n = 0 if df is None else len(df)
        par_source[nom] = n
        if nom in SOURCES_DIRECTES:
            if df is None:
                echecs.append(f"Source directe absente : {nom}")
                _log("error", f"{nom} : fichier absent")
            elif n < PLANCHER_SOURCE_DIRECTE:
                echecs.append(f"{nom} : {n} annonces (< {PLANCHER_SOURCE_DIRECTE})")
                _log("error", f"{nom} : seulement {n} annonces (plancher {PLANCHER_SOURCE_DIRECTE})")
            else:
                _log("ok", f"{nom} : {n} annonces")
        else:  # optionnelles : simple info / warning si vide
            if n == 0:
                alertes.append(f"Source optionnelle vide : {nom}")
                _log("warning", f"{nom} (optionnelle) : 0 annonce")
            else:
                _log("ok", f"{nom} (optionnelle) : {n} annonces")

    # ---- 2. Fichier scoré final ------------------------------------------
    scored = _lire_csv(PROCESSED_FILES.get("scored", ""))
    if scored is None or scored.empty:
        _log("error", "Fichier scoré introuvable ou vide — publication bloquée.")
        print("=" * 64)
        return 1
    total = len(scored)

    # ---- 3. Volume total : absolu + relatif à la référence ---------------
    if total < PLANCHER_TOTAL_ABSOLU:
        echecs.append(f"Total {total} < plancher absolu {PLANCHER_TOTAL_ABSOLU}")
        _log("error", f"Total scoré {total} sous le plancher absolu {PLANCHER_TOTAL_ABSOLU}")
    else:
        _log("ok", f"Total scoré : {total} annonces")

    ref = _charger_reference()
    ref_total = ref.get("total")
    if ref_total:
        seuil = int(ref_total * FRACTION_MIN_VS_REFERENCE)
        if total < seuil:
            echecs.append(f"Total {total} < {FRACTION_MIN_VS_REFERENCE:.0%} de la référence "
                          f"({ref_total} → seuil {seuil})")
            _log("error", f"Effondrement du volume : {total} vs référence {ref_total} "
                          f"(seuil {seuil})")
        else:
            _log("ok", f"Volume vs référence : {total} / {ref_total} "
                       f"({total / ref_total:.0%})")
    else:
        _log("info", "Pas de référence antérieure — contrôle relatif ignoré (1re exécution).")

    # ---- 4. Complétude ----------------------------------------------------
    def part_renseignee(col: str) -> float:
        if col not in scored.columns:
            return 0.0
        return float(scored[col].notna().mean())

    for col, seuil, dur in [("Prix", SEUIL_PRIX_RENSEIGNE, True),
                            ("Marque", SEUIL_MARQUE_RENSEIGNEE, True),
                            ("Modèle", SEUIL_MODELE_RENSEIGNE, True)]:
        p = part_renseignee(col)
        if p < seuil:
            msg = f"{col} renseigné à {p:.0%} (< {seuil:.0%})"
            (echecs if dur else alertes).append(msg)
            _log("error" if dur else "warning", msg)
        else:
            _log("ok", f"{col} renseigné : {p:.0%}")

    # ---- 5. Plausibilité des prix ----------------------------------------
    if "Prix" in scored.columns:
        prix = pd.to_numeric(scored["Prix"], errors="coerce").dropna()
        if len(prix):
            part_ok = float(((prix >= PRIX_MIN) & (prix <= PRIX_MAX)).mean())
            if part_ok < SEUIL_PRIX_PLAUSIBLE:
                echecs.append(f"Prix plausibles {part_ok:.1%} (< {SEUIL_PRIX_PLAUSIBLE:.0%})")
                _log("error", f"Prix dans [{PRIX_MIN}, {PRIX_MAX}] : {part_ok:.1%} "
                              f"(seuil {SEUIL_PRIX_PLAUSIBLE:.0%})")
            else:
                _log("ok", f"Prix plausibles : {part_ok:.1%}")

    # ---- 6. Doublons ------------------------------------------------------
    cle = "Lien" if "Lien" in scored.columns else None
    if cle:
        taux_dup = float(scored[cle].duplicated().mean())
        if taux_dup > SEUIL_DOUBLONS:
            alertes.append(f"Doublons {taux_dup:.1%} (> {SEUIL_DOUBLONS:.0%})")
            _log("warning", f"Taux de doublons ({cle}) : {taux_dup:.1%} "
                            f"(seuil {SEUIL_DOUBLONS:.0%})")
        else:
            _log("ok", f"Doublons ({cle}) : {taux_dup:.1%}")

    # ---- Verdict ----------------------------------------------------------
    print("-" * 64)
    if strict and alertes:
        echecs.extend(alertes)
        alertes = []

    if echecs:
        _log("error", f"QUALITY GATE ÉCHOUÉ — {len(echecs)} contrôle(s) dur(s) : "
                      + " | ".join(echecs))
        print("=" * 64)
        return 1

    for a in alertes:
        _log("warning", a)
    _log("info", "Tous les contrôles durs sont passés.")
    _ecrire_reference(total, par_source)
    print("=" * 64)
    return 0


if __name__ == "__main__":
    sys.exit(controler())
