"""
main.py
Orchestrateur du pipeline complet :

  1. Scrapers (tayara, automobile.tn, automax) -> data/raw/*.csv
  2. Fusion + nettoyage + enrichissement       -> data/processed/*.csv
  3. Modèle de prix + détection des deals      -> scored + alertes

Chaque étape est lancée via subprocess (pas os.system) : la sortie reste
visible, le code retour est vérifié, et le pipeline S'ARRÊTE si une étape de
traitement échoue -- avant, un échec silencieux au milieu laissait les étapes
suivantes travailler sur des fichiers périmés sans aucun avertissement.

Un scraper qui échoue n'arrête PAS le pipeline (les deux autres sources
suffisent), mais l'échec est signalé clairement.
"""

import subprocess
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJET = Path(__file__).parent

SCRAPERS = [
    "scrapers/scraper_tayara.py",
    "scrapers/scraper_automobile.py",
    "scrapers/scraper_automax.py",
]

# (script, critique) -- si un script critique échoue, tout s'arrête
ETAPES_TRAITEMENT = [
    ("core/merging_files.py", True),         # fusion des 3 sources
    ("core/nettoyer_base.py", True),         # nettoyage + inférence Marque/Modèle
    ("core/enrichir_base_avance.py", True),  # variables dérivées
    ("core/modele_prediction.py", True),     # modèle de prix + scoring
    ("core/detect_deals.py", False),         # filtrage des opportunités
]


def lancer(script: str) -> bool:
    """Lance un script Python et retourne True s'il a réussi."""
    chemin = PROJET / script
    if not chemin.exists():
        print(f"⚠️ Script manquant : {script}")
        return False
    print(f"\n-> Lancement de {script}...")
    resultat = subprocess.run([sys.executable, str(chemin)], cwd=str(PROJET))
    if resultat.returncode != 0:
        print(f"❌ {script} a échoué (code {resultat.returncode}).")
        return False
    return True


def executer_pipeline_auto():
    print("=" * 50)
    print("🤖 LANCEMENT DU SYSTEME MULTI-SOURCES AUTO")
    print("=" * 50)

    print("\n🛰️ Étape 1 : Collecte multi-sources...")
    scrapers_ok = 0
    for script in SCRAPERS:
        if lancer(script):
            scrapers_ok += 1
        time.sleep(2)
    if scrapers_ok == 0:
        print("❌ Aucun scraper n'a abouti -- arrêt (rien de nouveau à traiter).")
        sys.exit(1)

    print("\n🧹 Étape 2-3 : Nettoyage, enrichissement, modèle et deals...")
    for script, critique in ETAPES_TRAITEMENT:
        ok = lancer(script)
        if not ok and critique:
            print(f"❌ Étape critique en échec ({script}) -- arrêt du pipeline pour ne pas "
                  "produire de fichiers incohérents en aval.")
            sys.exit(1)

    print("\n✅ Pipeline terminé avec succès !")


if __name__ == "__main__":
    executer_pipeline_auto()
