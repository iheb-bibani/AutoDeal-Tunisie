"""
Scraper sayyaratn.com — 4e source pour AutoDeal.

Le site est rendu côté client (React/Next.js) : les annonces sont injectées par
JavaScript, `requests` ne verrait qu'une coquille "Chargement en cours...".
On utilise donc Playwright (comme tayara/automobile/automax).

Flux :
  1. Parcourir /annonces?page=N, attendre le rendu, collecter les <a href="/annonce/{uuid}">.
  2. S'arrêter dès qu'une page ne renvoie aucune annonce (fin de pagination).
  3. Pour chaque annonce : ouvrir la page détail, extraire titre, prix, toutes
     les paires libellé/valeur (cartes + <dl> Spécifications + Motorisation),
     description et localisation, via un seul evaluate() JS résilient.
  4. Mapper vers le schéma commun et écrire au fil de l'eau (dédup par lien).

⚠️ Les sélecteurs de description/localisation viennent des classes fournies
   (`whitespace-pre-line`, `text-sm text-muted-foreground mt-0.5`). Si le site
   change son markup, ajuste EXTRACTEUR_JS ci-dessous.

Usage :
    python scrapers/scraper_sayyarat.py
"""
import os
import re
import sys
import time
import random
import unicodedata
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from playwright.sync_api import sync_playwright

BASE = "https://www.sayyaratn.com"
LISTE_URL = BASE + "/annonces?page={}"
FICHIER = "data/raw/sayyarat.csv"

# Sur le Cloud, mets 2-3 pour ne récupérer que les nouveautés (delta).
MAX_PAGES = 300

COLONNES_FINALES = [
    "Source", "Titre", "Marque", "Modèle", "Année", "Prix_DT",
    "Kilométrage", "Energie", "Boite", "Localisation",
    "Puissance_Fiscale", "Etat_Vehicule", "Description", "Annonce-Deposee",
    "Annonce-Detectee", "Statut", "Lien",
]

# Extracteur exécuté DANS la page (DOM déjà rendu). Renvoie un objet JS -> dict.
EXTRACTEUR_JS = r"""
() => {
  const t = el => el ? el.textContent.replace(/\s+/g, ' ').trim() : null;
  const specs = {};

  // 1) Paires <dt>/<dd> des <dl> (Spécifications + Motorisation)
  document.querySelectorAll('dl').forEach(dl => {
    const dts = dl.querySelectorAll('dt');
    const dds = dl.querySelectorAll('dd');
    const n = Math.min(dts.length, dds.length);
    for (let i = 0; i < n; i++) {
      const k = t(dts[i]); const v = t(dds[i]);
      if (k && v) specs[k] = v;
    }
  });

  // 2) Cartes de mise en avant (Kilométrage, Carburant...) : petit libellé
  //    en MAJUSCULES suivi d'une valeur en gras. On lit le libellé et le frère
  //    suivant.
  document.querySelectorAll('div').forEach(d => {
    const c = d.className || '';
    if (typeof c === 'string' && c.includes('uppercase') &&
        (c.includes('text-[11px]') || c.includes('text-xs'))) {
      const label = t(d);
      const val = d.nextElementSibling ? t(d.nextElementSibling) : null;
      if (label && val && !(label in specs)) specs[label] = val;
    }
  });

  // 3) Description : <p class="whitespace-pre-line ...">
  const descEl = document.querySelector('p.whitespace-pre-line') ||
    [...document.querySelectorAll('p')].find(p => (p.className || '').includes('whitespace-pre-line'));

  // 4) Localisation : <p class="text-sm text-muted-foreground mt-0.5">
  let loc = null;
  document.querySelectorAll('p').forEach(p => {
    const c = p.className || '';
    if (!loc && c.includes('text-sm') && c.includes('text-muted-foreground') && c.includes('mt-0.5')) {
      loc = t(p);
    }
  });

  // 5) Titre + prix
  const h1 = document.querySelector('h1');
  const body = document.body ? document.body.innerText : '';
  const m = body.match(/([\d][\d\s.,]*)\s*DT\b/);

  return {
    titre: t(h1),
    prix: m ? m[1] : null,
    specs: specs,
    description: t(descEl),
    localisation: loc,
  };
}
"""


def _na(s):
    return "".join(c for c in unicodedata.normalize("NFD", str(s)) if unicodedata.category(c) != "Mn").lower()


def _chiffres(s):
    """'123,333 DT' / '9,000 km' -> 123333 / 9000 (int) ou None."""
    if not s:
        return None
    n = re.sub(r"[^\d]", "", str(s))
    return int(n) if n else None


def _get(specs, *cles):
    """Récupère la 1re valeur dont le libellé (sans accents/casse) matche."""
    norm = {_na(k): v for k, v in specs.items()}
    for c in cles:
        for k, v in norm.items():
            if _na(c) in k:
                return v
    return None


def mapper(data, lien):
    specs = data.get("specs", {}) or {}
    titre = data.get("titre")
    annee = _get(specs, "annee", "année")
    if not annee and titre:
        m = re.search(r"\b(19|20)\d{2}\b", titre)
        annee = m.group(0) if m else None
    return {
        "Source": "sayyaratn.com",
        "Titre": titre,
        "Marque": _get(specs, "marque"),
        "Modèle": _get(specs, "modele", "modèle"),
        "Année": _chiffres(annee),
        "Prix_DT": _chiffres(data.get("prix")),
        "Kilométrage": _chiffres(_get(specs, "kilometrage", "kilométrage")),
        "Energie": _get(specs, "carburant", "energie", "énergie"),
        "Boite": _get(specs, "boite vitesse", "boîte", "boite"),
        "Localisation": data.get("localisation"),
        "Puissance_Fiscale": _chiffres(_get(specs, "puissance fiscale", "puissance")),
        "Etat_Vehicule": _get(specs, "etat", "état"),
        "Description": data.get("description"),
        "Annonce-Deposee": None,  # non exposé de façon fiable sur la fiche
        "Annonce-Detectee": datetime.now().strftime("%Y-%m-%d"),
        "Statut": "Active",
        "Lien": lien,
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
    ligne = {c: car.get(c) for c in COLONNES_FINALES}
    entete = not os.path.exists(chemin) or os.path.getsize(chemin) == 0
    pd.DataFrame([ligne])[COLONNES_FINALES].to_csv(
        chemin, mode="a", header=entete, index=False, sep=";", encoding="utf-8-sig"
    )


def collecter_liens(page, num_page):
    """Charge une page de liste et renvoie les liens /annonce/... présents."""
    page.goto(LISTE_URL.format(num_page), wait_until="domcontentloaded", timeout=45000)
    try:
        # Attendre le rendu client des cartes (le grid + au moins un lien annonce).
        page.wait_for_selector('#listings-grid a[href^="/annonce/"]', timeout=15000)
    except Exception:
        return []
    hrefs = page.eval_on_selector_all(
        '#listings-grid a[href^="/annonce/"]',
        "els => els.map(e => e.getAttribute('href'))",
    )
    liens, vus = [], set()
    for h in hrefs:
        if not h:
            continue
        url = h if h.startswith("http") else BASE + h
        if url not in vus:
            vus.add(url)
            liens.append(url)
    return liens


def scraper():
    deja = liens_deja_vus(FICHIER)
    total = 0
    with sync_playwright() as p:
        navigateur = p.chromium.launch(headless=True)
        contexte = navigateur.new_context(
            user_agent=("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"),
            locale="fr-FR",
        )
        page = contexte.new_page()

        for num in range(1, MAX_PAGES + 1):
            liens = collecter_liens(page, num)
            if not liens:
                print(f"Page {num} : aucune annonce -> fin de pagination.")
                break
            nouveaux = [l for l in liens if l not in deja]
            print(f"Page {num} : {len(liens)} annonces, {len(nouveaux)} nouvelles.")

            for lien in nouveaux:
                try:
                    page.goto(lien, wait_until="domcontentloaded", timeout=30000)
                    page.wait_for_selector("h1", timeout=12000)
                    data = page.evaluate(EXTRACTEUR_JS)
                    car = mapper(data, lien)
                    if car["Titre"] and car["Prix_DT"]:
                        enregistrer_ligne(car)
                        deja.add(lien)
                        total += 1
                except Exception as e:
                    print(f"  ⚠️ {lien} ignorée ({type(e).__name__}).")
                time.sleep(random.uniform(0.8, 1.8))

        navigateur.close()
    print(f"✅ Terminé : {total} nouvelle(s) annonce(s) -> {FICHIER}")


if __name__ == "__main__":
    scraper()
