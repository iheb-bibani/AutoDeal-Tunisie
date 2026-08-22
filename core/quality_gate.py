"""Quality gate du pipeline AutoDeal.

Le gate valide la qualité des fichiers publiables ET l'état du run courant.
La fraîcheur d'une source ne doit pas être déduite uniquement de
``Annonce-Detectee`` : si un scraper s'exécute correctement mais ne découvre
aucune nouvelle annonce, cette date reste ancienne. Le manifeste écrit par
``main.py`` est donc la source de vérité pour savoir si le scraper du run
courant a réellement abouti.
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
    PROCESSED_FILES = {"scored": str(PROCESSED_DATA_DIR / "tunisia-cars-scored.csv")}

BASELINE_PATH = Path(PROCESSED_DATA_DIR) / "quality_baseline.json"
RUN_MANIFEST_PATH = Path(PROCESSED_DATA_DIR) / "pipeline_run.json"

SOURCES_DIRECTES = ["automobile", "tayara", "automax"]
SOURCES_OPTIONNELLES = ["sayyarat", "autocentral"]
MIN_SOURCES_DIRECTES_OK = 2
PLANCHER_SOURCE_DIRECTE = 30
PLANCHER_TOTAL_ABSOLU = 300
FRACTION_MIN_VS_REFERENCE = 0.60

# Ces seuils décrivent l'âge de la DERNIÈRE NOUVELLE ANNONCE observée. Ils sont
# désormais informatifs ; l'état réel du scraper vient du manifeste du run.
MAX_AGE_SOURCE_DIRECTE_HEURES = 36
MAX_AGE_SOURCE_OPTIONNELLE_HEURES = 72
MAX_AGE_MANIFEST_HEURES = 12

SEUIL_PRIX_RENSEIGNE = 0.95
SEUIL_MARQUE_RENSEIGNEE = 0.95
SEUIL_MODELE_RENSEIGNE = 0.80
PRIX_MIN, PRIX_MAX = 1_000, 3_000_000
SEUIL_PRIX_PLAUSIBLE = 0.98
SEUIL_DOUBLONS = 0.02
SEP, ENC = ";", "utf-8-sig"


def _log(niveau: str, msg: str) -> None:
    prefix = {"error": "::error::", "warning": "::warning::", "ok": "  ✓ ", "info": "  • "}
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
    """Âge de la dernière NOUVELLE annonce détectée, pas du dernier run scraper."""
    if df is None or df.empty or "Annonce-Detectee" not in df.columns:
        return None
    dates = pd.to_datetime(df["Annonce-Detectee"], errors="coerce", utc=True)
    latest = dates.max()
    if pd.isna(latest):
        return None
    now = pd.Timestamp(maintenant or datetime.now(timezone.utc))
    now = now.tz_localize("UTC") if now.tzinfo is None else now.tz_convert("UTC")
    return max(0.0, float((now - latest).total_seconds() / 3600.0))


def _charger_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _charger_reference() -> dict:
    return _charger_json(BASELINE_PATH)


def _charger_manifest_courant(
    maintenant: datetime | pd.Timestamp | None = None,
) -> dict:
    manifest = _charger_json(RUN_MANIFEST_PATH)
    finished = manifest.get("finished_at") or manifest.get("started_at")
    if not finished:
        return {}
    ts = pd.to_datetime(finished, errors="coerce", utc=True)
    if pd.isna(ts):
        return {}
    now = pd.Timestamp(maintenant or datetime.now(timezone.utc))
    now = now.tz_localize("UTC") if now.tzinfo is None else now.tz_convert("UTC")
    age_h = float((now - ts).total_seconds() / 3600.0)
    if age_h < -1 or age_h > MAX_AGE_MANIFEST_HEURES:
        return {}
    return manifest


def scraper_a_reussi(manifest: dict, source: str) -> bool | None:
    """True/False pour un manifeste courant, None si l'information manque."""
    if not manifest:
        return None
    info = (manifest.get("scrapers") or {}).get(source)
    if not isinstance(info, dict):
        return None
    return info.get("status") == "success"


def _ecrire_reference(total: int, par_source: dict) -> None:
    ref = {
        "total": int(total),
        "par_source": {k: int(v) for k, v in par_source.items()},
        "updated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    try:
        BASELINE_PATH.write_text(json.dumps(ref, ensure_ascii=False, indent=2), encoding="utf-8")
        _log("info", f"Référence glissante mise à jour ({total} annonces).")
    except Exception as exc:
        _log("warning", f"Impossible d'écrire la référence : {exc}")


def controler() -> int:
    strict = "--strict" in sys.argv
    echecs: list[str] = []
    alertes: list[str] = []
    manifest = _charger_manifest_courant()

    print("=" * 64)
    print("  QUALITY GATE — AutoDeal Tunisie")
    print("=" * 64)

    if manifest:
        _log("info", f"Manifeste run courant : {manifest.get('status', 'inconnu')}")
    else:
        _log("warning", "Manifeste de run absent/ancien : fallback sur les données disponibles.")

    par_source: dict[str, int] = {}
    sources_directes_ok = 0

    for nom, chemin in SCRAPERS.items():
        df, erreur_lecture = _lire_csv(chemin)
        n = 0 if df is None else len(df)
        par_source[nom] = n
        directe = nom in SOURCES_DIRECTES
        run_ok = scraper_a_reussi(manifest, nom)

        fichier_valide = df is not None and (not directe or n >= PLANCHER_SOURCE_DIRECTE)
        if df is None:
            raison = "fichier absent" if erreur_lecture == "absent" else "fichier illisible/corrompu"
            msg = f"{nom} : {raison}"
            (alertes if directe else alertes).append(msg)
            _log("warning", msg)
        elif directe and n < PLANCHER_SOURCE_DIRECTE:
            msg = f"{nom} : seulement {n} annonces (plancher {PLANCHER_SOURCE_DIRECTE})"
            alertes.append(msg)
            _log("warning", msg)
        else:
            suffix = "" if directe else " (optionnelle)"
            _log("ok", f"{nom}{suffix} : {n} annonces")

        # Santé opérationnelle du scraper : le manifeste est prioritaire.
        if run_ok is True:
            _log("ok", f"{nom} : scraper du run courant terminé avec succès")
        elif run_ok is False:
            msg = f"{nom} : scraper du run courant en échec/timeout"
            alertes.append(msg)
            _log("warning", msg)
        else:
            _log("info", f"{nom} : statut du scraper courant indisponible")

        if directe and fichier_valide and run_ok is not False:
            sources_directes_ok += 1

        # L'âge de la dernière nouvelle annonce reste utile comme signal, mais
        # ne doit plus faire échouer le pipeline à lui seul.
        if df is not None and not df.empty:
            age_h = age_derniere_detection_heures(df)
            limite = MAX_AGE_SOURCE_DIRECTE_HEURES if directe else MAX_AGE_SOURCE_OPTIONNELLE_HEURES
            if age_h is None:
                msg = f"{nom} : date de dernière nouvelle annonce invérifiable"
                alertes.append(msg)
                _log("warning", msg)
            elif age_h > limite:
                msg = f"{nom} : aucune nouvelle annonce depuis {age_h:.1f} h (> {limite} h)"
                alertes.append(msg)
                _log("warning", msg)
            else:
                _log("ok", f"{nom} : dernière nouvelle annonce il y a {age_h:.1f} h")

    if sources_directes_ok < MIN_SOURCES_DIRECTES_OK:
        echecs.append(
            f"Seulement {sources_directes_ok}/{len(SOURCES_DIRECTES)} sources directes exploitables "
            f"(minimum {MIN_SOURCES_DIRECTES_OK})"
        )

    scored, _ = _lire_csv(PROCESSED_FILES.get("scored", ""))
    if scored is None or scored.empty:
        _log("error", "Fichier scoré introuvable ou vide — publication bloquée.")
        return 1
    total = len(scored)

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
            echecs.append(f"Total {total} < {FRACTION_MIN_VS_REFERENCE:.0%} de la référence ({ref_total} → seuil {seuil})")
            _log("error", f"Effondrement du volume : {total} vs référence {ref_total} (seuil {seuil})")
        else:
            _log("ok", f"Volume vs référence : {total} / {ref_total} ({total / ref_total:.0%})")
    else:
        _log("info", "Pas de référence antérieure — contrôle relatif ignoré.")

    def part_renseignee(col: str) -> float:
        return float(scored[col].notna().mean()) if col in scored.columns else 0.0

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

    if "Prix" in scored.columns:
        prix = pd.to_numeric(scored["Prix"], errors="coerce").dropna()
        if len(prix):
            part_ok = float(((prix >= PRIX_MIN) & (prix <= PRIX_MAX)).mean())
            if part_ok < SEUIL_PRIX_PLAUSIBLE:
                echecs.append(f"Prix plausibles {part_ok:.1%} (< {SEUIL_PRIX_PLAUSIBLE:.0%})")
                _log("error", f"Prix dans [{PRIX_MIN}, {PRIX_MAX}] : {part_ok:.1%}")
            else:
                _log("ok", f"Prix plausibles : {part_ok:.1%}")

    if "Lien" in scored.columns:
        taux_dup = float(scored["Lien"].duplicated().mean())
        if taux_dup > SEUIL_DOUBLONS:
            alertes.append(f"Doublons {taux_dup:.1%} (> {SEUIL_DOUBLONS:.0%})")
            _log("warning", f"Taux de doublons (Lien) : {taux_dup:.1%}")
        else:
            _log("ok", f"Doublons (Lien) : {taux_dup:.1%}")

    print("-" * 64)
    if strict and alertes:
        echecs.extend(alertes)
        alertes = []

    if echecs:
        _log("error", f"QUALITY GATE ÉCHOUÉ — {len(echecs)} contrôle(s) dur(s) : " + " | ".join(echecs))
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
