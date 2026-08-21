"""Quality gate du pipeline AutoDeal.

À lancer après le pipeline et avant la publication des données. Les contrôles
portent sur le volume, la fraîcheur, la complétude, la plausibilité des prix et
les doublons. Une source principale peut conserver un gros fichier CSV alors
que son scraper vient de tomber : la fraîcheur est donc contrôlée séparément du
volume.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

try:
    from config import PROCESSED_DATA_DIR, PROCESSED_FILES, SCRAPERS
except Exception:  # pragma: no cover
    ROOT = Path(__file__).resolve().parents[1]
    PROCESSED_DATA_DIR = ROOT / "data" / "processed"
    RAW = ROOT / "data" / "raw"
    SCRAPERS = {
        n: str(RAW / f"{n}.csv")
        for n in ["tayara", "automax", "automobile", "sayyarat", "autocentral"]
    }
    PROCESSED_FILES = {
        "scored": str(PROCESSED_DATA_DIR / "tunisia-cars-scored.csv")
    }

BASELINE_PATH = Path(PROCESSED_DATA_DIR) / "quality_baseline.json"

SOURCES_DIRECTES = ["automobile", "tayara", "automax"]
SOURCES_OPTIONNELLES = ["sayyarat", "autocentral"]
PLANCHER_SOURCE_DIRECTE = 30
PLANCHER_TOTAL_ABSOLU = 300
FRACTION_MIN_VS_REFERENCE = 0.60

# Le workflow est quotidien. Une détection vieille de plus de 36 h signifie
# normalement que le scraper principal n'a pas produit de données au run courant.
MAX_AGE_SOURCE_DIRECTE_HEURES = 36
MAX_AGE_SOURCE_OPTIONNELLE_HEURES = 72

SEUIL_PRIX_RENSEIGNE = 0.95
SEUIL_MARQUE_RENSEIGNEE = 0.95
SEUIL_MODELE_RENSEIGNE = 0.80

PRIX_MIN, PRIX_MAX = 1_000, 3_000_000
SEUIL_PRIX_PLAUSIBLE = 0.98
SEUIL_DOUBLONS = 0.02

SEP, ENC = ";", "utf-8-sig"


def _log(niveau: str, msg: str) -> None:
    prefix = {
        "error": "::error::",
        "warning": "::warning::",
        "ok": "  ✓ ",
        "info": "  • ",
    }
    print(f"{prefix.get(niveau, '')}{msg}")


def _lire_csv(chemin: str) -> tuple[pd.DataFrame | None, str | None]:
    p = Path(chemin)
    if not p.exists():
        return None, "absent"
    try:
        return pd.read_csv(p, sep=SEP, encoding=ENC, low_memory=False), None
    except Exception as exc:
        _log("error", f"Lecture impossible de {p.name} : {exc}")
        return None, f"illisible: {exc}"


def age_derniere_detection_heures(
    df: pd.DataFrame,
    maintenant: datetime | pd.Timestamp | None = None,
) -> float | None:
    """Âge en heures de la détection la plus récente d'un CSV brut."""
    if df is None or df.empty or "Annonce-Detectee" not in df.columns:
        return None
    dates = pd.to_datetime(df["Annonce-Detectee"], errors="coerce", utc=True)
    latest = dates.max()
    if pd.isna(latest):
        return None
    now = pd.Timestamp(maintenant or datetime.now(timezone.utc))
    if now.tzinfo is None:
        now = now.tz_localize("UTC")
    else:
        now = now.tz_convert("UTC")
    return max(0.0, float((now - latest).total_seconds() / 3600.0))


def _charger_reference() -> dict:
    if BASELINE_PATH.exists():
        try:
            return json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _ecrire_reference(total: int, par_source: dict) -> None:
    ref = {
        "total": int(total),
        "par_source": {k: int(v) for k, v in par_source.items()},
        "updated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    try:
        BASELINE_PATH.write_text(
            json.dumps(ref, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        _log("info", f"Référence glissante mise à jour ({total} annonces).")
    except Exception as exc:
        _log("warning", f"Impossible d'écrire la référence : {exc}")


def controler() -> int:
    strict = "--strict" in sys.argv
    echecs: list[str] = []
    alertes: list[str] = []

    print("=" * 64)
    print("  QUALITY GATE — AutoDeal Tunisie")
    print("=" * 64)

    # 1. Sources brutes : volume + fraîcheur.
    par_source: dict[str, int] = {}
    for nom, chemin in SCRAPERS.items():
        df, erreur_lecture = _lire_csv(chemin)
        n = 0 if df is None else len(df)
        par_source[nom] = n
        directe = nom in SOURCES_DIRECTES

        if directe:
            if df is None:
                raison = (
                    "fichier absent"
                    if erreur_lecture == "absent"
                    else "fichier illisible/corrompu"
                )
                echecs.append(f"Source directe indisponible : {nom} ({raison})")
                _log("error", f"{nom} : {raison}")
                continue
            if n < PLANCHER_SOURCE_DIRECTE:
                echecs.append(
                    f"{nom} : {n} annonces (< {PLANCHER_SOURCE_DIRECTE})"
                )
                _log(
                    "error",
                    f"{nom} : seulement {n} annonces "
                    f"(plancher {PLANCHER_SOURCE_DIRECTE})",
                )
            else:
                _log("ok", f"{nom} : {n} annonces")
        else:
            if n == 0:
                alertes.append(f"Source optionnelle vide : {nom}")
                _log("warning", f"{nom} (optionnelle) : 0 annonce")
                continue
            _log("ok", f"{nom} (optionnelle) : {n} annonces")

        age_h = age_derniere_detection_heures(df)
        limite = (
            MAX_AGE_SOURCE_DIRECTE_HEURES
            if directe
            else MAX_AGE_SOURCE_OPTIONNELLE_HEURES
        )
        if age_h is None:
            msg = f"{nom} : fraîcheur impossible à vérifier (Annonce-Detectee absente/invalide)"
            if directe:
                echecs.append(msg)
                _log("error", msg)
            else:
                alertes.append(msg)
                _log("warning", msg)
        elif age_h > limite:
            msg = f"{nom} : dernière détection vieille de {age_h:.1f} h (> {limite} h)"
            if directe:
                echecs.append(msg)
                _log("error", msg)
            else:
                alertes.append(msg)
                _log("warning", msg)
        else:
            _log("ok", f"{nom} : fraîcheur {age_h:.1f} h")

    # 2. Fichier scoré final.
    scored, _ = _lire_csv(PROCESSED_FILES.get("scored", ""))
    if scored is None or scored.empty:
        _log("error", "Fichier scoré introuvable ou vide — publication bloquée.")
        print("=" * 64)
        return 1
    total = len(scored)

    # 3. Volume total : absolu + relatif à la référence.
    if total < PLANCHER_TOTAL_ABSOLU:
        echecs.append(f"Total {total} < plancher absolu {PLANCHER_TOTAL_ABSOLU}")
        _log(
            "error",
            f"Total scoré {total} sous le plancher absolu {PLANCHER_TOTAL_ABSOLU}",
        )
    else:
        _log("ok", f"Total scoré : {total} annonces")

    ref = _charger_reference()
    ref_total = ref.get("total")
    if ref_total:
        seuil = int(ref_total * FRACTION_MIN_VS_REFERENCE)
        if total < seuil:
            echecs.append(
                f"Total {total} < {FRACTION_MIN_VS_REFERENCE:.0%} de la référence "
                f"({ref_total} → seuil {seuil})"
            )
            _log(
                "error",
                f"Effondrement du volume : {total} vs référence {ref_total} "
                f"(seuil {seuil})",
            )
        else:
            _log(
                "ok",
                f"Volume vs référence : {total} / {ref_total} "
                f"({total / ref_total:.0%})",
            )
    else:
        _log("info", "Pas de référence antérieure — contrôle relatif ignoré.")

    # 4. Complétude.
    def part_renseignee(col: str) -> float:
        if col not in scored.columns:
            return 0.0
        return float(scored[col].notna().mean())

    for col, seuil in [
        ("Prix", SEUIL_PRIX_RENSEIGNE),
        ("Marque", SEUIL_MARQUE_RENSEIGNEE),
        ("Modèle", SEUIL_MODELE_RENSEIGNE),
    ]:
        part = part_renseignee(col)
        if part < seuil:
            msg = f"{col} renseigné à {part:.0%} (< {seuil:.0%})"
            echecs.append(msg)
            _log("error", msg)
        else:
            _log("ok", f"{col} renseigné : {part:.0%}")

    # 5. Plausibilité des prix.
    if "Prix" in scored.columns:
        prix = pd.to_numeric(scored["Prix"], errors="coerce").dropna()
        if len(prix):
            part_ok = float(((prix >= PRIX_MIN) & (prix <= PRIX_MAX)).mean())
            if part_ok < SEUIL_PRIX_PLAUSIBLE:
                echecs.append(
                    f"Prix plausibles {part_ok:.1%} (< {SEUIL_PRIX_PLAUSIBLE:.0%})"
                )
                _log(
                    "error",
                    f"Prix dans [{PRIX_MIN}, {PRIX_MAX}] : {part_ok:.1%} "
                    f"(seuil {SEUIL_PRIX_PLAUSIBLE:.0%})",
                )
            else:
                _log("ok", f"Prix plausibles : {part_ok:.1%}")

    # 6. Doublons.
    if "Lien" in scored.columns:
        taux_dup = float(scored["Lien"].duplicated().mean())
        if taux_dup > SEUIL_DOUBLONS:
            alertes.append(
                f"Doublons {taux_dup:.1%} (> {SEUIL_DOUBLONS:.0%})"
            )
            _log(
                "warning",
                f"Taux de doublons (Lien) : {taux_dup:.1%} "
                f"(seuil {SEUIL_DOUBLONS:.0%})",
            )
        else:
            _log("ok", f"Doublons (Lien) : {taux_dup:.1%}")

    print("-" * 64)
    if strict and alertes:
        echecs.extend(alertes)
        alertes = []

    if echecs:
        _log(
            "error",
            f"QUALITY GATE ÉCHOUÉ — {len(echecs)} contrôle(s) dur(s) : "
            + " | ".join(echecs),
        )
        print("=" * 64)
        return 1

    for alerte in alertes:
        _log("warning", alerte)
    _log("info", "Tous les contrôles durs sont passés.")
    _ecrire_reference(total, par_source)
    print("=" * 64)
    return 0


if __name__ == "__main__":
    sys.exit(controler())
