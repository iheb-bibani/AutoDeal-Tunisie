"""
Scraper autocentral.tn — 5e source pour AutoDeal, mais avec une subtilité clé.

autocentral.tn est un AGRÉGATEUR : il republie les annonces d'automobile.tn,
tayara.tn et Facebook. Scraper tout dupliquerait massivement les sources qu'on
récupère déjà en direct. La SOURCE d'origine de chaque annonce est lisible dans
l'URL de sa photo :

    storage.googleapis.com/.../car-posts/AUTOMOBILETN-audi-a3-128145/...  -> déjà chez nous
    storage.googleapis.com/.../car-posts/TAYARA-6a65e833.../...            -> déjà chez nous
    storage.googleapis.com/.../car-posts/FACEBOOK-1213760287544755/...     -> NOUVEAU

=> On ne garde QUE les annonces dont le préfixe n'est PAS dans SOURCES_DEJA_VUES.
   C'est un dédup par provenance, fiable (pas de fuzzy matching).

Le site est rendu côté client + pagination par bouton « Charger plus » -> Playwright.

⚠️ Les sélecteurs de carte/détail viennent des captures fournies ; à vérifier sur
   le site live. Le cœur (détection de source + filtrage) est robuste car il ne
   dépend que de l'URL de la photo.

Usage :
    python scrapers/scraper_autocentral.py
"""
import os
import re
import sys
import time
import random
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from playwright.sync_api import sync_playwright

BASE = "https://www.autocentral.tn"
FICHIER = "data/raw/autocentral.csv"

# Sources déjà scrapées en direct -> on les IGNORE ici pour ne pas dupliquer.
# (On garde FACEBOOK et tout autre préfixe inconnu = valeur nouvelle.)
SOURCES_DEJA_VUES = {"AUTOMOBILETN", "TAYARA", "AUTOMAX", "SAYYARAT", "SAYYARATN"}

# Nombre de clics sur « Charger plus » (chaque clic ~ +20 annonces).
MAX_CHARGEMENTS = 40

COLONNES_FINALES = [
    "Source", "Titre", "Marque", "Modèle", "Année", "Prix_DT",
    "Kilométrage", "Energie", "Boite", "Localisation",
    "Puissance_Fiscale", "Etat_Vehicule", "Description", "Annonce-Deposee",
    "Annonce-Detectee", "Statut", "Lien",
]

# Extracteur DOM : pour chaque carte-annonce, lit l'image (source+id), puis les
# champs texte. On repère les cartes par leur vignette car-posts/<SOURCE>-<id>.
EXTRACTEUR_JS = r"""
() => {
  const cartes = [];
  const imgs = [...document.querySelectorAll('img[src*="car-posts/"]')];
  for (const img of imgs) {
    const m = img.src.match(/car-posts\/([A-Z]+)-([^\/]+)\//);
    if (!m) continue;
    const source = m[1];          // AUTOMOBILETN / TAYARA / FACEBOOK ...
    const idPost = m[1] + '-' + m[2];
    // Carte = plus proche ancêtre "raisonnable" contenant le prix
    let carte = img.closest('a, article, li, div');
    for (let i = 0; i < 4 && carte && !/DT/.test(carte.innerText || ''); i++) {
      carte = carte.parentElement;
    }
    if (!carte) continue;
    const txt = (carte.innerText || '').replace(/\u00a0/g, ' ');
    const tel = carte.querySelector('a[href^="tel:"]');
    cartes.push({
      source, idPost,
      titre: img.getAttribute('alt') || null,
      texte: txt,
      tel: tel ? tel.getAttribute('href').replace('tel:', '') : null,
    });
  }
  return cartes;
}
"""


def _num(s):
    if not s:
        return None
    n = re.sub(r"[^\d]", "", str(s))
    return int(n) if n else None


def parser_carte(c):
    """Transforme le texte brut d'une carte en champs. Heuristique, basée sur le
    format observé : 'TITRE\\nYEAR Marque Modèle\\nNN km\\nNcv Carburant\\nBoite\\nPRIX DT ...'"""
    txt = c["texte"] or ""
    prix = None
    mprix = re.search(r"([\d\s]+)\s*DT", txt)
    if mprix:
        prix = _num(mprix.group(1))
    annee = None
    man = re.search(r"\b(19|20)\d{2}\b", txt)
    if man:
        annee = int(man.group(0))
    km = None
    mkm = re.search(r"([\d\s]+)\s*km", txt, re.IGNORECASE)
    if mkm:
        km = _num(mkm.group(1))
    cv = None
    mcv = re.search(r"(\d{1,2})\s*cv", txt, re.IGNORECASE)
    if mcv:
        cv = int(mcv.group(1))
    energie = next((e for e in ["Essence", "Diesel", "Hybrid", "Electrique", "Électrique"]
                    if re.search(e, txt, re.IGNORECASE)), None)
    boite = next((b for b in ["Automatique", "Manuelle"]
                  if re.search(b, txt, re.IGNORECASE)), None)
    # Ligne "YEAR Marque Modèle" -> marque/modèle
    marque = modele = None
    mmm = re.search(r"\b(?:19|20)\d{2}\s+([A-Za-zÀ-ÿ]+)\s+(.+)", txt)
    if mmm:
        marque = mmm.group(1).strip()
        modele = mmm.group(2).splitlines()[0].strip()
    # Localisation : dernière ligne courte sans chiffre (ex "Tunis", "Sfax")
    loc = None
    for ligne in reversed([l.strip() for l in txt.splitlines() if l.strip()]):
        if len(ligne) <= 20 and not re.search(r"\d|DT|km|cv|Appeler|il y a", ligne):
            loc = ligne
            break
    return {
        "Source": f"autocentral.tn ({c['source'].lower()})",
        "Titre": c["titre"],
        "Marque": marque, "Modèle": modele, "Année": annee,
        "Prix_DT": prix, "Kilométrage": km, "Energie": energie, "Boite": boite,
        "Localisation": loc, "Puissance_Fiscale": cv, "Etat_Vehicule": None,
        "Description": None,  # visible seulement sur la fiche détail (best-effort désactivé)
        "Annonce-Deposee": None,
        "Annonce-Detectee": datetime.now().strftime("%Y-%m-%d"),
        "Statut": "Active",
        # Lien stable et unique à partir de l'id de post agrégé (pas d'URL /annonce fiable)
        "Lien": f"{BASE}/#{c['idPost']}",
        "_id": c["idPost"],
    }


def liens_deja_vus(chemin):
    if not os.path.exists(chemin):
        return set()
    try:
        return set(pd.read_csv(chemin, sep=";", encoding="utf-8-sig")["Lien"].dropna())
    except Exception:
        return set()


def enregistrer_ligne(car, chemin=FICHIER):
    os.makedirs(os.path.dirname(chemin), exist_ok=True)
    ligne = {col: car.get(col) for col in COLONNES_FINALES}
    entete = not os.path.exists(chemin) or os.path.getsize(chemin) == 0
    pd.DataFrame([ligne])[COLONNES_FINALES].to_csv(
        chemin, mode="a", header=entete, index=False, sep=";", encoding="utf-8-sig")


def scraper():
    deja = liens_deja_vus(FICHIER)
    gardes = ignores = 0
    with sync_playwright() as p:
        nav = p.chromium.launch(headless=True)
        page = nav.new_context(locale="fr-FR").new_page()
        page.goto(BASE, wait_until="domcontentloaded", timeout=45000)
        try:
            page.wait_for_selector('img[src*="car-posts/"]', timeout=15000)
        except Exception:
            print("❌ Aucune annonce chargée (structure changée ?).")
            nav.close()
            return

        # Pagination : cliquer « Charger plus d'annonces » plusieurs fois.
        for _ in range(MAX_CHARGEMENTS):
            try:
                btn = page.get_by_text(re.compile("Charger plus", re.I))
                if btn.count() == 0:
                    break
                btn.first.click(timeout=5000)
                time.sleep(random.uniform(1.2, 2.2))
            except Exception:
                break

        cartes = page.evaluate(EXTRACTEUR_JS)
        vus_run = set()
        for c in cartes:
            # DÉDUP PAR SOURCE : on saute ce qu'on récupère déjà en direct.
            if c["source"] in SOURCES_DEJA_VUES:
                ignores += 1
                continue
            car = parser_carte(c)
            if car["_id"] in vus_run or car["Lien"] in deja:
                continue
            vus_run.add(car["_id"])
            if car["Titre"] and car["Prix_DT"]:
                enregistrer_ligne(car)
                gardes += 1
        nav.close()

    print(f"✅ Terminé : {gardes} annonce(s) NOUVELLE(s) gardée(s), "
          f"{ignores} ignorée(s) (déjà scrapées via leur source d'origine) -> {FICHIER}")


if __name__ == "__main__":
    scraper()
