import time
import random
import os
import re
from datetime import datetime
from patchright.sync_api import sync_playwright
from bs4 import BeautifulSoup
import pandas as pd

# -------------------------------------------------
# CONFIGURATION
# -------------------------------------------------

MAX_PAGES = 160
FICHIER_AUTOMOBILE = "data/raw/automobile.csv"
USER_DATA_DIR = "./user_data"

# Mêmes 16 colonnes qu'AutoMax et tayara.tn, pour rester alignés au moment
# de l'unification finale (clean_data.py).
COLONNES_FINALES = [
    "Source", "Titre", "Marque", "Modèle", "Année", "Prix_DT",
    "Kilométrage", "Energie", "Boite", "Localisation",
    "Puissance_Fiscale", "Etat_Vehicule", "Annonce-Deposee",
    "Annonce-Detectee", "Statut", "Lien",
    # Équipements réellement présents sur le véhicule (bloc "Équipements" de
    # la page détail, déjà chargée pour les spécifications -- coût nul).
    "Nb_Options", "Options",
]


# -------------------------------------------------
# HELPERS : DATE DE L'ANNONCE, SAUVEGARDE PROGRESSIVE
# -------------------------------------------------

def parser_date_dmy(texte):
    """Convertit '12.07.2026' (format du site) en '2026-07-12'.
    Retourne None si le texte est vide ou ne matche pas le format attendu."""

    if not texte or not isinstance(texte, str):
        return None

    match = re.match(r"(\d{1,2})[.\-/](\d{1,2})[.\-/](\d{4})", texte.strip())

    if not match:
        return None

    jour, mois, annee = match.groups()
    return f"{int(annee):04d}-{int(mois):02d}-{int(jour):02d}"


def migrer_colonnes(chemin=FICHIER_AUTOMOBILE):
    """Ajoute les colonnes manquantes à un CSV écrit par une version
    antérieure du scraper.

    Les lignes sont ajoutées en mode append : si l'en-tête du fichier compte
    16 colonnes et qu'on y ajoute des lignes de 18 valeurs, tout le fichier
    se décale silencieusement. On réécrit donc l'en-tête une bonne fois,
    les anciennes annonces gardant simplement des options vides.
    """
    if not os.path.exists(chemin) or os.path.getsize(chemin) == 0:
        return
    try:
        df = pd.read_csv(chemin, sep=";", encoding="utf-8-sig")
    except Exception:
        return
    manquantes = [c for c in COLONNES_FINALES if c not in df.columns]
    if not manquantes:
        return
    for col in manquantes:
        df[col] = pd.NA
    df[COLONNES_FINALES].to_csv(chemin, index=False, sep=";", encoding="utf-8-sig")
    print(f"Colonnes ajoutées à {chemin} : {', '.join(manquantes)}")


def enregistrer_ligne(car, chemin=FICHIER_AUTOMOBILE):
    """Ajoute une seule annonce au CSV, tout de suite (au fur et à mesure).
    Ecrit l'en-tête seulement si le fichier n'existe pas encore ou est vide."""

    os.makedirs(os.path.dirname(chemin), exist_ok=True)

    ligne = {col: car.get(col, pd.NA) for col in COLONNES_FINALES}

    fichier_existe = os.path.exists(chemin) and os.path.getsize(chemin) > 0

    pd.DataFrame([ligne])[COLONNES_FINALES].to_csv(
        chemin,
        mode="a",
        header=not fichier_existe,
        index=False,
        sep=";",
        encoding="utf-8-sig",
    )


def charger_liens_deja_scrapes(chemin=FICHIER_AUTOMOBILE):
    """Au démarrage, récupère les liens déjà présents dans automobile.csv
    pour ne pas re-scraper ce qui a déjà été fait (reprise après crash)."""

    if not (os.path.exists(chemin) and os.path.getsize(chemin) > 0):
        return set()

    try:
        df = pd.read_csv(chemin, sep=";")
        return set(df["Lien"].dropna())
    except Exception:
        return set()


# -------------------------------------------------
# EXTRACTION DES SPECS D'UNE ANNONCE
# -------------------------------------------------

def chercher_spec_approximative(specs: dict, *sous_chaines):
    """Cherche une clé de specs_trouvees qui CONTIENT une des sous-chaînes
    données, plutôt qu'une correspondance exacte. Utile quand le site combine
    plusieurs mots dans un même libellé (ex: 'Année-Modèle' au lieu d'
    'Année' seul) -- un .get('année') exact ne matche jamais dans ce cas,
    alors qu'une recherche par sous-chaîne le trouve.
    Retourne la première valeur trouvée, ou None si rien ne correspond."""
    for cle, valeur in specs.items():
        for sous_chaine in sous_chaines:
            if sous_chaine in cle:
                return valeur
    return None


def extraire_equipements(soup):
    """Liste les options RÉELLEMENT présentes sur le véhicule.

    Structure du bloc Équipements sur automobile.tn :

        <div class="equipments-wrapper">
          <div class="box">
            <div class="box-inner-title"> Sécurité </div>
            <div class="checked-specs">
              <ul>
                <li><span class="spec-value">ABS</span></li>                <- absente
                <li class="highlighted"><span class="spec-value">ESP</span></li>  <- PRÉSENTE
              </ul>

    Chaque <li> est une option *possible* pour ce modèle ; seules celles
    portant la classe "highlighted" équipent effectivement la voiture. Lire
    tous les <li> reviendrait à donner la liste du catalogue constructeur,
    identique pour deux voitures pourtant très différemment équipées.

    Aucune requête supplémentaire : la page détail est déjà chargée pour en
    extraire les spécifications.
    """
    wrapper = soup.select_one("div.equipments-wrapper")
    if not wrapper:
        return []
    options = []
    for li in wrapper.select("div.checked-specs li.highlighted"):
        span_valeur = li.select_one("span.spec-value")
        if span_valeur:
            texte = " ".join(span_valeur.get_text().split())
            if texte:
                options.append(texte)
    return options


def extraire_specs_dynamiques(html_page, url_annonce):
    soup = BeautifulSoup(html_page, 'html.parser')
    specs_trouvees = {}

    h1_el = soup.find('h1')
    titre = h1_el.get_text().strip() if h1_el else "N/A"
    titre = " ".join(titre.split())

    prix_el = soup.select_one('.price-box, .price, [class*="price"]')
    # BUG CORRIGE : si l'élément prix n'est pas trouvé (structure HTML différente
    # sur cette fiche, "Prix sur demande", etc.), le défaut était "0" -- un Prix_DT
    # à 0 ressemble à la meilleure affaire du siècle pour n'importe quel modèle de
    # scoring, et pouvait générer une fausse alerte "voiture à 0 DT". None est le
    # bon défaut : une valeur manquante ne doit jamais devenir un zéro fantôme.
    prix_dt = "".join(filter(str.isdigit, prix_el.get_text())) if prix_el else None

    # Paires clé/valeur génériques (Marque, Modèle, Gouvernorat, Date de
    # l'annonce, etc. utilisent toutes cette même structure spec-name/spec-value)
    spans_name = soup.find_all('span', class_='spec-name')
    for span_name in spans_name:
        parent_li = span_name.find_parent('li')
        if parent_li:
            span_value = parent_li.select_one('span[class^="spec-value"]')
            if span_value:
                cle = span_name.get_text().replace(":", "").strip().lower()
                valeur = span_value.get_text().strip()
                specs_trouvees[cle] = " ".join(valeur.split())

    infos_standard = {
        "Source": "automobile.tn",
        "Titre": titre,
        "Prix_DT": prix_dt,
        "Localisation": specs_trouvees.get("gouvernorat") or "Tunisie",
        "Lien": url_annonce,
        # BUG CORRIGE : .capitalize() ne majuscule que la toute première lettre
        # de la chaîne entière ("land rover" -> "Land rover"), alors que .title()
        # majuscule chaque mot ("land rover" -> "Land Rover") -- confirmé sur les
        # vraies données où "Land rover" et "Mercedes-benz" fragmentaient la
        # marque en deux catégories différentes pour le modèle.
        "Marque": (specs_trouvees.get("marque") or "").strip().title() or None,
        "Modèle": (specs_trouvees.get("modèle") or specs_trouvees.get("modele") or "").strip() or None,
        # BUG CORRIGE (piste la plus probable) : Année était introuvable pour
        # 100% des annonces automobile.tn avec un .get("année") exact. Les
        # sites d'annonces auto étiquettent très souvent ce champ "Année-Modèle"
        # (un seul libellé combiné) plutôt que "Année" seul -- une recherche par
        # sous-chaîne le retrouve, une correspondance exacte jamais.
        # A CONFIRMER avec un vrai screenshot du site si le taux de capture ne
        # remonte pas après ce correctif.
        # BUG CORRIGE (confirmé par screenshot du vrai site) : le champ ne
        # s'appelle ni "Année" ni "Année-Modèle" -- automobile.tn l'étiquette
        # "Mise en circulation", avec une valeur au format "MM.AAAA" (ex:
        # "02.2019"), pas juste une année seule. L'ancien code filtrait tous
        # les chiffres du texte, ce qui aurait transformé "02.2019" en
        # "022019" -- une fausse année à 6 chiffres. On extrait maintenant
        # spécifiquement 4 chiffres consécutifs (l'année), peu importe le
        # format autour (MM.AAAA, AAAA seul, JJ/MM/AAAA...).
        "Année": (lambda v: (re.search(r"(\d{4})", str(v)).group(1) if v and re.search(r"(\d{4})", str(v)) else None))(
            chercher_spec_approximative(specs_trouvees, "année", "annee", "circulation")
        ),
        # BUG CORRIGE : même problème que Prix_DT -- le défaut "0" transformait
        # un kilométrage manquant en "voiture à 0 km", ce qui fausse le modèle
        # (0 km = neuf, très différent de "on ne sait pas"). None est le bon défaut,
        # comme c'était déjà fait correctement pour Année juste au-dessus.
        "Kilométrage": ("".join(filter(str.isdigit, str(specs_trouvees.get("kilométrage", ""))))) or None,
        "Energie": specs_trouvees.get("énergie") or specs_trouvees.get("energie") or None,
        # BUG CORRIGE (confirmé sur données réelles) : le fallback .get("transmission")
        # capturait en fait le type de TRANSMISSION AUX ROUES (Traction/Intégrale/
        # Propulsion), pas le type de BOITE DE VITESSES (Manuelle/Automatique) --
        # deux concepts différents en français automobile que ce fallback confondait.
        # Retiré en attendant d'identifier le vrai libellé du champ boîte de
        # vitesses sur automobile.tn (nécessite un nouveau screenshot du site).
        "Boite": specs_trouvees.get("boite") or specs_trouvees.get("boîte") or None,
        "Puissance_Fiscale": specs_trouvees.get("puissance fiscale") or None,
        "Etat_Vehicule": specs_trouvees.get("état") or specs_trouvees.get("etat") or None,
        # Nouveau : date de dépôt de l'annonce (ex: "12.07.2026" -> "2026-07-12")
        "Annonce-Deposee": parser_date_dmy(specs_trouvees.get("date de l'annonce")),
        # Date à laquelle NOUS avons scrapé l'annonce (différent de la date de dépôt)
        "Annonce-Detectee": datetime.now().strftime("%Y-%m-%d"),
        "Statut": None,
    }

    options = extraire_equipements(soup)
    infos_standard["Nb_Options"] = len(options)
    infos_standard["Options"] = " | ".join(options) if options else None

    return infos_standard


def attendre_cloudflare(page, timeout_ms=120000):
    """
    Attend la disparition du challenge Cloudflare en se basant sur plusieurs
    signaux combinés (titre, présence de l'iframe Turnstile, contenu réel de
    la page), plutôt que sur un seul sélecteur qui peut ne jamais apparaître
    tant que le challenge est actif.
    """
    titre_actuel = page.title()
    if "vérification" not in titre_actuel.lower() and "security" not in titre_actuel.lower():
        return True  # pas de challenge, on continue directement

    print("🛑 Challenge Cloudflare détecté. Résolvez-le manuellement dans la fenêtre du navigateur...")
    print(f"   (attente jusqu'à {timeout_ms // 1000}s)")

    deadline = time.time() + (timeout_ms / 1000)
    while time.time() < deadline:
        try:
            titre = page.title()
        except Exception:
            titre = ""
        if "vérification" not in titre.lower() and "security" not in titre.lower():
            # Le titre a changé : la page réelle a chargé
            print("✅ Challenge résolu, reprise du scraping.")
            time.sleep(1.5)
            return True
        time.sleep(1)

    print("⚠️ Timeout : le challenge n'a pas été résolu à temps.")
    return False


# -------------------------------------------------
# SCRAPER PRINCIPAL
# -------------------------------------------------

def scrape_automobile_tn():
    migrer_colonnes()
    print("🛰️ Lancement du Scraper Automobile.tn (Anti-Bot activé)...")
    liens_annonces = []

    os.makedirs("data", exist_ok=True)
    os.makedirs(USER_DATA_DIR, exist_ok=True)

    liens_deja_scrapes = charger_liens_deja_scrapes()
    if liens_deja_scrapes:
        print(f"↪️  Reprise : {len(liens_deja_scrapes)} annonces déjà présentes dans {FICHIER_AUTOMOBILE}, elles seront sautées.")

    with sync_playwright() as p:
        # Patchright patche les fuites CDP à la source : pas besoin (et pas
        # recommandé) d'ajouter des scripts de masquage supplémentaires, cela
        # peut réintroduire des artefacts détectables. On utilise le vrai
        # Chrome installé (channel="chrome") plutôt que le Chromium fourni,
        # ce qui est le réglage le plus fiable contre Cloudflare/Turnstile.
        browser = p.chromium.launch_persistent_context(
            user_data_dir=USER_DATA_DIR,
            channel="chrome",
            headless=False,
            viewport={"width": 1366, "height": 900},
            locale="fr-FR",
            no_viewport=False,
        )
        page = browser.pages[0] if browser.pages else browser.new_page()

        for page_num in range(1, MAX_PAGES + 1):
            print(f"📖 Page {page_num}...")
            url_liste = (
                "https://www.automobile.tn/fr/occasion"
                if page_num == 1
                else f"https://www.automobile.tn/fr/occasion?page={page_num}"
            )

            try:
                page.goto(url_liste, timeout=60000, wait_until="domcontentloaded")

                # Gestion du challenge Cloudflare, robuste (attente active, pas un seul selector)
                ok = attendre_cloudflare(page, timeout_ms=120000)
                if not ok:
                    print(f"⏭️ Passage à la page suivante, échec sur la page {page_num}.")
                    continue

                # Laisse le JS du site finir de peupler le DOM après le challenge
                try:
                    page.wait_for_selector('.occasion-link-overlay', timeout=20000)
                except Exception:
                    print("⚠️ Aucune annonce détectée sur cette page (structure HTML peut-être différente).")

                soup_liste = BeautifulSoup(page.content(), 'html.parser')
                elements = soup_liste.find_all('a', class_='occasion-link-overlay')

                for el in elements:
                    href = el.get('href')
                    if href and "/fr/occasion/" in href:
                        full_url = (
                            f"https://www.automobile.tn{href}"
                            if not href.startswith("http")
                            else href
                        )
                        if full_url not in liens_annonces and full_url not in liens_deja_scrapes:
                            liens_annonces.append(full_url)

                print(f"   → {len(elements)} annonces trouvées sur cette page.")
                time.sleep(random.uniform(3, 6))

            except Exception as e:
                print(f"Erreur page {page_num}: {e}")
                continue

        # --- Visite de chaque annonce pour extraire les specs détaillées ---
        nb_extraites = 0
        for i, lien in enumerate(liens_annonces, start=1):
            print(f"🔍 Annonce {i}/{len(liens_annonces)}: {lien}")
            try:
                page.goto(lien, timeout=60000, wait_until="domcontentloaded")
                ok = attendre_cloudflare(page, timeout_ms=90000)
                if not ok:
                    continue
                infos = extraire_specs_dynamiques(page.content(), lien)

                # ECRITURE IMMEDIATE : au fur et à mesure, pas seulement à la fin.
                # Si le script plante à l'annonce 300/500, les 299 précédentes
                # sont déjà sur le disque, pas besoin de tout relancer.
                enregistrer_ligne(infos)
                nb_extraites += 1

                time.sleep(random.uniform(2, 4))
            except Exception as e:
                print(f"Erreur annonce {lien}: {e}")
                continue

        browser.close()

    print(f"✅ Collecte terminée. {len(liens_annonces)} véhicules trouvés, {nb_extraites} fiches extraites et enregistrées dans {FICHIER_AUTOMOBILE}.")

    # Filet de sécurité : dédoublonnage léger sur 'Lien', au cas où une même
    # annonce serait apparue deux fois dans la pagination du run en cours.
    if os.path.exists(FICHIER_AUTOMOBILE) and os.path.getsize(FICHIER_AUTOMOBILE) > 0:
        try:
            df_final = pd.read_csv(FICHIER_AUTOMOBILE, sep=";")
            avant = len(df_final)
            df_final = df_final.drop_duplicates(subset=["Lien"], keep="last")
            df_final.to_csv(FICHIER_AUTOMOBILE, index=False, sep=";", encoding="utf-8-sig")
            if avant != len(df_final):
                print(f"🧹 {avant - len(df_final)} doublons supprimés. Total : {len(df_final)} lignes.")
        except Exception as e:
            print(f"⚠️ Dédoublonnage final ignoré (erreur : {e})")

    return nb_extraites


if __name__ == "__main__":
    scrape_automobile_tn()
