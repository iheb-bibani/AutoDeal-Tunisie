import time
import random
import os
import re
from datetime import datetime, timedelta
import pandas as pd
from playwright.sync_api import sync_playwright

# ⚙️ CONFIGURATION DE L'AUTOMATISATION
# Sur le Cloud, mets 1 ou 2 pour ne scraper rapidement que les nouveautés et appliquer le Delta.
MAX_PAGES_A_SCRAPER = 303

# -------------------------------------------------
# HELPERS : DATE RELATIVE -> YYYY-MM-DD
# -------------------------------------------------

def parser_anciennete(texte, reference=None):
    """Convertit '20 hours ago' / 'a day ago' / 'an hour ago' en YYYY-MM-DD."""

    if not isinstance(texte, str):
        return None

    reference = reference or datetime.now()
    texte = texte.strip().lower()

    match = re.match(r"(a|an|\d+)\s+(minute|hour|day|week|month)s?\s+ago", texte)

    if not match:
        return None

    qty_str, unite = match.groups()
    qty = 1 if qty_str in ("a", "an") else int(qty_str)

    delta_map = {
        "minute": timedelta(minutes=qty),
        "hour": timedelta(hours=qty),
        "day": timedelta(days=qty),
        "week": timedelta(weeks=qty),
        "month": timedelta(days=qty * 30),  # approximation
    }

    return (reference - delta_map[unite]).strftime("%Y-%m-%d")


# -------------------------------------------------
# HELPERS : LABELS BRUTS DU SITE -> COLONNES STANDARD
# -------------------------------------------------

MAPPING_LABELS = {
    "marque": "Marque",
    "modèle": "Modèle",
    "modele": "Modèle",
    "année": "Année",
    "annee": "Année",
    "kilométrage": "Kilométrage",
    "kilometrage": "Kilométrage",
    "boite": "Boite",
    "boîte": "Boite",
    "carburant": "Energie",
    "énergie": "Energie",
    "energie": "Energie",
    "puissance fiscale": "Puissance_Fiscale",
    "état du véhicule": "Etat_Vehicule",
    "etat du vehicule": "Etat_Vehicule",
}

FICHIER_TAYARA = "data/raw/tayara.csv"

COLONNES_FINALES = [
    "Source", "Titre", "Marque", "Modèle", "Année", "Prix_DT",
    "Kilométrage", "Energie", "Boite", "Localisation",
    "Puissance_Fiscale", "Etat_Vehicule", "Annonce-Deposee",
    "Annonce-Detectee", "Statut", "Lien",
]


def enregistrer_ligne(car, chemin=FICHIER_TAYARA):
    """Ajoute une seule annonce au CSV, tout de suite (au fur et à mesure).
    Ecrit l'en-tête seulement si le fichier n'existe pas encore ou est vide."""

    os.makedirs(os.path.dirname(chemin), exist_ok=True)

    ligne = {col: car.get(col, pd.NA) for col in COLONNES_FINALES}
    ligne["Statut"] = car.get("Statut", pd.NA)

    fichier_existe = os.path.exists(chemin) and os.path.getsize(chemin) > 0

    pd.DataFrame([ligne])[COLONNES_FINALES].to_csv(
        chemin,
        mode="a",
        header=not fichier_existe,
        index=False,
        sep=";",
        encoding="utf-8-sig",
    )


def charger_liens_deja_scrapes(chemin=FICHIER_TAYARA):
    """Au démarrage, récupère les liens déjà présents dans tayara.csv pour ne
    pas re-scraper ce qui a déjà été fait (utile en cas de reprise après crash)."""

    if not (os.path.exists(chemin) and os.path.getsize(chemin) > 0):
        return set()

    try:
        df = pd.read_csv(chemin, sep=";")
        return set(df["Lien"].dropna())
    except Exception:
        return set()

def scrape_tayara():
    data_cars = []
    page_num = 1

    liens_deja_scrapes = charger_liens_deja_scrapes()
    if liens_deja_scrapes:
        print(f"↪️  Reprise : {len(liens_deja_scrapes)} annonces déjà présentes dans {FICHIER_TAYARA}, elles seront sautées.")

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            viewport={"width": 1280, "height": 800}
        )
        page = context.new_page()
        
        # RUSE OPTIONNELLE : Bloquer les images et les pubs pour charger 5x plus vite
        def block_aggressively(route):
            if route.request.resource_type in ["image", "media", "font"] or "google" in route.request.url:
                route.abort()
            else:
                route.continue_()
        page.route("**/*", block_aggressively)

        while page_num <= MAX_PAGES_A_SCRAPER:
            print(f"--- Chargement de la page {page_num} ---")
            url = f"https://www.tayara.tn/listing/k/voiture/?page={page_num}"
            
            try:
                page.goto(url, wait_until="domcontentloaded", timeout=45000)
                
                try:
                    page.wait_for_selector("article.mx-0", timeout=8000)
                except:
                    print(f"Plus d'annonces détectées ou chargement trop long. Fin à la page {page_num - 1}.")
                    break
                
                articles = page.locator("article.mx-0").all()
                
                for article in articles:
                    try:
                        link_element = article.locator("a").first
                        relative_href = link_element.get_attribute("href")
                        full_url = f"https://www.tayara.tn{relative_href}" if relative_href else None
                        
                        price_element = article.locator("data.font-bold").first
                        price = price_element.get_attribute("value") if price_element.count() > 0 else "N/A"
                        
                        title = article.locator("h2.card-title").text_content().strip()
                        
                        loc_time_text = article.locator("span.line-clamp-1").text_content().strip()
                        location = loc_time_text.split(",")[0].strip() if "," in loc_time_text else loc_time_text
                        time_ago = loc_time_text.split(",")[1].strip() if "," in loc_time_text else "N/A"
                        
                        car_info = {
                            "Source": "tayara.tn",
                            "Titre": title,
                            "Prix_DT": price,
                            "Localisation": location,
                            "Anciennete": time_ago,
                            "Annonce-Deposee": parser_anciennete(time_ago),
                            "Annonce-Detectee": datetime.now().strftime("%Y-%m-%d"),
                            "Lien": full_url
                        }

                        if full_url in liens_deja_scrapes:
                            continue

                        data_cars.append(car_info)
                        
                    except Exception:
                        continue
                
                time.sleep(random.uniform(1.5, 3.0))
                page_num += 1
                
            except Exception as e:
                print(f"Erreur de chargement à la page {page_num}: {e}")
                break
                
        # --- VISITE DES LIENS ---
        print(f"\nVisite de {len(data_cars)} annonces individuelles pour récupérer les critères...")
        for i, car in enumerate(data_cars):
            if car["Lien"]:
                try:
                    print(f"[{i+1}/{len(data_cars)}] Extraction : {car['Titre']}")
                    page.goto(car["Lien"], wait_until="domcontentloaded", timeout=20000)
                    
                    page.wait_for_selector("ul.grid-cols-12", timeout=5000)
                    
                    criterion_items = page.locator("ul.grid-cols-12 li").all()
                    for item in criterion_items:
                        label_el = item.locator("span.text-gray-600\\/80, span[class*='text-gray-600']").first
                        value_el = item.locator("span.text-gray-700\\/80, span[class*='text-gray-700']").first
                        
                        if label_el.count() > 0 and value_el.count() > 0:
                            label = label_el.text_content().strip()
                            value = value_el.text_content().strip()

                            # On ne garde que les labels qu'on sait mapper vers
                            # le schéma standard (mêmes 16 colonnes qu'AutoMax).
                            # Les autres (Couleur, Cylindrée, Type de carrosserie...)
                            # sont ignorés pour l'instant, à ajouter plus tard si besoin.
                            colonne_standard = MAPPING_LABELS.get(label.strip().lower())

                            if colonne_standard:
                                car[colonne_standard] = value

                    time.sleep(random.uniform(1.0, 2.0))

                except Exception:
                    pass  # même en cas d'erreur sur cette annonce, on l'enregistre avec ce qu'on a pu récupérer

                # ECRITURE IMMEDIATE : au fur et à mesure, pas seulement à la fin.
                # Si le script plante à l'annonce 1500/2171, les 1499 précédentes
                # sont déjà sur le disque, pas besoin de tout relancer.
                enregistrer_ligne(car)

        browser.close()

    print(f"\n✅ Terminé : {len(data_cars)} annonces traitées et enregistrées dans {FICHIER_TAYARA}")

    # Filet de sécurité : dédoublonnage léger sur 'Lien', au cas où une même
    # annonce serait apparue deux fois dans la pagination du run en cours.
    if os.path.exists(FICHIER_TAYARA) and os.path.getsize(FICHIER_TAYARA) > 0:
        try:
            df_final = pd.read_csv(FICHIER_TAYARA, sep=";")
            avant = len(df_final)
            df_final = df_final.drop_duplicates(subset=["Lien"], keep="last")
            df_final.to_csv(FICHIER_TAYARA, index=False, sep=";", encoding="utf-8-sig")
            if avant != len(df_final):
                print(f"🧹 {avant - len(df_final)} doublons supprimés. Total : {len(df_final)} lignes.")
        except Exception as e:
            print(f"⚠️ Dédoublonnage final ignoré (erreur : {e})")

if __name__ == "__main__":
    scrape_tayara()