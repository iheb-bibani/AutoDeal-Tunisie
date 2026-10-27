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
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from core.dates import aujourd_hui  # noqa: E402

import pandas as pd
from playwright.sync_api import sync_playwright

from core.regles_texte import MARQUES_VALIDES
from scrapers.liens_vus import enregistrer_liens_vus
from scrapers.parsing import parser_annee, parser_entier, parser_prix_tnd

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
    // Carte = plus proche ancêtre contenant le prix, SANS jamais remonter
    // jusqu'à un conteneur qui englobe plusieurs annonces (sinon on lisait le
    // texte de la grille entière : année/prix de la 1re carte pour toutes).
    const nbVignettes = (el) => el.querySelectorAll('img[src*="car-posts/"]').length;
    let carte = img.closest('a, article, li, div');
    for (let i = 0; i < 6 && carte && !/DT/.test(carte.innerText || ''); i++) {
      const parent = carte.parentElement;
      if (!parent || nbVignettes(parent) > 1) break;
      carte = parent;
    }
    if (!carte || nbVignettes(carte) > 1 || !/DT/.test(carte.innerText || '')) continue;
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


_ENERGIES = ["Essence", "Diesel", "Hybride", "Hybrid", "Electrique", "Électrique", "GPL"]
_BOITES = ["Automatique", "Manuelle"]
_MOTS_NON_LIEU = {e.lower() for e in _ENERGIES + _BOITES} | {"dsg", "bva", "bvm", "voir l'annonce", "voir l’annonce"}


def parser_carte(c):
    """Transforme le texte brut d'une carte en champs, LIGNE PAR LIGNE.

    Format observé : 'TITRE\nYEAR Marque Modèle\nNN km\nNcv Carburant\nBoite\nPRIX DT ...'
    Les anciennes regex utilisaient [\\d\\s]+ qui traversait les sauts de ligne
    ("2020\\n15000 km" -> 202015000 km) et prenaient n'importe quel mot après
    l'année comme marque ("DSG", "Diesel"…).
    """
    txt = (c.get("texte") or "").replace("\u00a0", " ")
    lignes = [l.strip() for l in txt.splitlines() if l.strip()]

    prix = next((parser_prix_tnd(l) for l in lignes if re.search(r"\bDT\b", l, re.I) and parser_prix_tnd(l)), None)
    km = next((parser_entier(l, maxi=1_000_000) for l in lignes if re.search(r"\d\s*km\b", l, re.I)), None)
    cv = None
    for l in lignes:
        mcv = re.search(r"\b(\d{1,2})\s*cv\b", l, re.I)
        if mcv:
            cv = int(mcv.group(1))
            break
    energie = next((e for e in _ENERGIES if re.search(rf"\b{e}\b", txt, re.I)), None)
    boite = next((b for b in _BOITES if re.search(rf"\b{b}\b", txt, re.I)), None)

    # Ligne "YEAR Marque Modèle" : la marque doit être une marque connue.
    annee = marque = modele = None
    for l in lignes:
        m = re.match(r"^((?:19|20)\d{2})\s+(\S+)(?:\s+(.+))?$", l)
        if m:
            annee = int(m.group(1))
            cand = MARQUES_VALIDES.get(m.group(2).lower())
            if cand:
                marque = cand
                modele = (m.group(3) or "").strip() or None
            break
    if annee is None:
        annee = parser_annee(c.get("titre") or "")

    # Localisation : dernière ligne courte, sans chiffre, qui n'est ni une
    # énergie, ni une boîte, ni un bouton.
    loc = None
    for ligne in reversed(lignes):
        bas = ligne.lower()
        if (len(ligne) <= 20 and not re.search(r"\d|DT|km|cv|appeler|il y a|→", ligne, re.I)
                and bas not in _MOTS_NON_LIEU):
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
        "Annonce-Detectee": aujourd_hui(),
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
    liens_vus: dict[str, object] = {}
    fin_naturelle = False
    with sync_playwright() as p:
        nav = p.chromium.launch(headless=True)
        try:
            page = nav.new_context(locale="fr-FR").new_page()
            page.goto(BASE, wait_until="domcontentloaded", timeout=45000)
            try:
                page.wait_for_selector('img[src*="car-posts/"]', timeout=15000)
            except Exception:
                print("❌ Aucune annonce chargée (structure changée ?).")
                sys.exit(1)

            # Pagination : cliquer « Charger plus d'annonces » plusieurs fois.
            for _ in range(MAX_CHARGEMENTS):
                try:
                    btn = page.get_by_text(re.compile("Charger plus", re.I))
                    if btn.count() == 0:
                        fin_naturelle = True
                        break
                    btn.first.click(timeout=5000)
                    time.sleep(random.uniform(1.2, 2.2))
                except Exception:
                    break

            cartes = page.evaluate(EXTRACTEUR_JS)
        finally:
            nav.close()

    vus_run = set()
    for c in cartes:
        # DÉDUP PAR SOURCE : on saute ce qu'on récupère déjà en direct.
        if c["source"] in SOURCES_DEJA_VUES:
            ignores += 1
            continue
        car = parser_carte(c)
        liens_vus[car["Lien"]] = car["Prix_DT"]
        if car["_id"] in vus_run or car["Lien"] in deja:
            continue
        vus_run.add(car["_id"])
        if car["Titre"] and car["Prix_DT"]:
            enregistrer_ligne(car)
            gardes += 1

    if not cartes:
        print("❌ Aucune carte d'annonce extraite (structure changée ?).")
        sys.exit(1)
    if liens_vus:
        enregistrer_liens_vus("autocentral", liens_vus, complet=fin_naturelle, nouvelles=gardes)
    print(f"ℹ️ {len(cartes)} carte(s) extraite(s), {len(liens_vus)} hors sources directes, "
          f"{sum(1 for c in cartes if c['source'] not in SOURCES_DEJA_VUES and not parser_carte(c)['Prix_DT'])} sans prix lisible.")

    print(f"✅ Terminé : {gardes} annonce(s) NOUVELLE(s) gardée(s), "
          f"{ignores} ignorée(s) (déjà scrapées via leur source d'origine) -> {FICHIER}")


if __name__ == "__main__":
    scraper()
