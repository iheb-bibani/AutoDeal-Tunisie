"""Orchestrateur du pipeline AutoDeal Tunisie.

Principes :
- un scraper isolé ne doit pas bloquer toute la collecte ;
- aucun subprocess ne peut rester bloqué indéfiniment ;
- chaque run écrit un manifeste exploitable par le quality gate ;
- une étape de traitement critique arrête le pipeline.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJET = Path(__file__).parent
MANIFEST_PATH = PROJET / "data" / "processed" / "pipeline_run.json"

SCRAPERS = [
    "scrapers/scraper_tayara.py",
    "scrapers/scraper_automobile.py",
    "scrapers/scraper_automax.py",
    "scrapers/scraper_sayyarat.py",
    "scrapers/scraper_autocentral.py",
]

# Un scraper réseau ne doit jamais monopoliser le runner pendant des heures.
SCRAPER_TIMEOUT_SECONDS = int(os.getenv("AUTODEAL_SCRAPER_TIMEOUT_SECONDS", "2700"))
DEFAULT_STEP_TIMEOUT_SECONDS = int(os.getenv("AUTODEAL_STEP_TIMEOUT_SECONDS", "1200"))
MODEL_TIMEOUT_SECONDS = int(os.getenv("AUTODEAL_MODEL_TIMEOUT_SECONDS", "5400"))

# (script, critique, timeout)
ETAPES_TRAITEMENT = [
    ("core/merging_files.py", True, DEFAULT_STEP_TIMEOUT_SECONDS),
    ("core/nettoyer_base.py", True, DEFAULT_STEP_TIMEOUT_SECONDS),
    ("core/enrichir_base_avance.py", True, DEFAULT_STEP_TIMEOUT_SECONDS),
    ("core/modele_prediction.py", True, MODEL_TIMEOUT_SECONDS),
    ("core/detect_deals.py", False, DEFAULT_STEP_TIMEOUT_SECONDS),
    ("core/suivi_annonces.py", False, DEFAULT_STEP_TIMEOUT_SECONDS),
]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _write_manifest(manifest: dict) -> None:
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def lancer(script: str, timeout_seconds: int) -> dict:
    """Lance un script et retourne un statut structuré.

    Le timeout protège GitHub Actions contre un scraper/site externe qui reste
    bloqué. Pour les scrapers, un timeout/échec est toléré par l'orchestrateur ;
    le quality gate décide ensuite si assez de sources fiables ont abouti.
    """
    chemin = PROJET / script
    started = time.monotonic()
    result = {
        "script": script,
        "status": "missing",
        "returncode": None,
        "duration_seconds": 0.0,
        "finished_at": None,
    }
    if not chemin.exists():
        print(f"⚠️ Script manquant : {script}")
        result["finished_at"] = _utc_now()
        return result

    print(f"\n-> Lancement de {script} (timeout {timeout_seconds}s)...")
    try:
        proc = subprocess.run(
            [sys.executable, str(chemin)],
            cwd=str(PROJET),
            timeout=timeout_seconds,
        )
        result["returncode"] = proc.returncode
        result["status"] = "success" if proc.returncode == 0 else "failed"
        if proc.returncode != 0:
            print(f"❌ {script} a échoué (code {proc.returncode}).")
    except subprocess.TimeoutExpired:
        result["status"] = "timeout"
        print(f"⏱️ {script} interrompu après {timeout_seconds}s (timeout).")
    except Exception as exc:
        result["status"] = "error"
        result["error"] = f"{type(exc).__name__}: {exc}"
        print(f"❌ {script} a levé une exception : {exc}")
    finally:
        result["duration_seconds"] = round(time.monotonic() - started, 1)
        result["finished_at"] = _utc_now()
    return result


def _source_key(script: str) -> str:
    return Path(script).stem.removeprefix("scraper_")


def executer_pipeline_auto() -> None:
    manifest = {
        "started_at": _utc_now(),
        "finished_at": None,
        "status": "running",
        "scrapers": {},
        "processing": {},
    }
    _write_manifest(manifest)

    print("=" * 50)
    print("🤖 LANCEMENT DU SYSTEME MULTI-SOURCES AUTO")
    print("=" * 50)

    print("\n🛰️ Étape 1 : Collecte multi-sources...")
    for script in SCRAPERS:
        status = lancer(script, SCRAPER_TIMEOUT_SECONDS)
        manifest["scrapers"][_source_key(script)] = status
        _write_manifest(manifest)
        time.sleep(2)

    scrapers_ok = sum(
        x.get("status") == "success" for x in manifest["scrapers"].values()
    )
    if scrapers_ok == 0:
        manifest["status"] = "failed_no_scraper"
        manifest["finished_at"] = _utc_now()
        _write_manifest(manifest)
        print("❌ Aucun scraper n'a abouti -- arrêt (rien de nouveau à traiter).")
        raise SystemExit(1)

    print(f"\n✅ {scrapers_ok}/{len(SCRAPERS)} scrapers ont abouti.")
    print("🧹 Étape 2-3 : Nettoyage, enrichissement, modèle et deals...")

    for script, critique, timeout_seconds in ETAPES_TRAITEMENT:
        status = lancer(script, timeout_seconds)
        manifest["processing"][script] = status
        _write_manifest(manifest)
        if status["status"] != "success" and critique:
            manifest["status"] = "failed_processing"
            manifest["failed_step"] = script
            manifest["finished_at"] = _utc_now()
            _write_manifest(manifest)
            print(
                f"❌ Étape critique en échec ({script}) -- arrêt du pipeline "
                "pour ne pas produire de fichiers incohérents en aval."
            )
            raise SystemExit(1)

    manifest["status"] = "success"
    manifest["finished_at"] = _utc_now()
    _write_manifest(manifest)
    print("\n✅ Pipeline terminé avec succès !")


if __name__ == "__main__":
    executer_pipeline_auto()
