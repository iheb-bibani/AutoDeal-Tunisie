import os
import re
import time
import random
import pandas as pd
import requests

from datetime import datetime
from bs4 import BeautifulSoup


# -------------------------------------------------
# CONFIG
# -------------------------------------------------

FICHIER_SORTIE = "data/raw/automax.csv"

BASE_URL = "https://www.automax.tn"

LISTE_URL = "https://www.automax.tn/voitures-occasion/"

AJAX_URL = "https://www.automax.tn/wp-admin/admin-ajax.php"


HEADERS = {

    "User-Agent":
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",

    "Accept":
        "application/json, text/javascript, */*; q=0.01",

    "X-Requested-With":
        "XMLHttpRequest"

}

# -------------------------------------------------
# HELPERS DATE / PUISSANCE FISCALE
# -------------------------------------------------

MOIS_FR = {
    "janvier": 1, "février": 2, "fevrier": 2, "mars": 3,
    "avril": 4, "mai": 5, "juin": 6, "juillet": 7,
    "août": 8, "aout": 8, "septembre": 9, "octobre": 10,
    "novembre": 11, "décembre": 12, "decembre": 12,
}


def parser_date_fr(texte):
    """Convertit '29 juin 2026' en '2026-06-29'. Retourne None si le format
    ne matche pas (site changé, texte vide, etc.)."""

    if not texte:
        return None

    morceaux = texte.strip().lower().split()

    if len(morceaux) != 3:
        return None

    jour_str, mois_str, annee_str = morceaux

    mois = MOIS_FR.get(mois_str)

    if not (jour_str.isdigit() and annee_str.isdigit() and mois):
        return None

    return f"{int(annee_str):04d}-{mois:02d}-{int(jour_str):02d}"


def extraire_puissance_fiscale(soup):
    """Cherche parmi les 'vehica-car-feature' celui qui contient 'X cv'
    (ex: Essence / 9 cv sont tous les deux des vehica-car-feature,
    on ne garde que celui qui matche le motif chevaux fiscaux)."""

    for tag in soup.find_all(class_="vehica-car-feature"):

        texte_tag = (tag.get("title") or tag.get_text(strip=True) or "")

        match = re.search(r"(\d+)\s*cv", texte_tag.lower())

        if match:

            return match.group(1)

    return None

# -------------------------------------------------
# SCRAPER DETAIL ANNONCE
# -------------------------------------------------

def scraper_details_annonce(url_annonce, session):

    for tentative in range(1, 4):

        try:

            r = session.get(
                url_annonce,
                headers=HEADERS,
                timeout=30
            )

            if r.status_code != 200:

                print(
                    "HTTP erreur",
                    r.status_code
                )

                time.sleep(3)

                continue

            soup = BeautifulSoup(
                r.text,
                "html.parser"
            )

            data = {

                "Titre": None,
                "Prix_DT": None,
                "Lien": url_annonce,
                "Source": "AutoMax",
                "Annonce-Detectee":
                    datetime.now().strftime("%Y-%m-%d"),
                "Annonce-Deposee": None,

                "Marque": None,
                "Modèle": None,
                "Année": None,
                "Kilométrage": None,
                "Boite": None,
                "Energie": None,
                "Localisation": None,
                "Puissance_Fiscale": None,
                "Etat_Vehicule": None,
                "Statut": None

            }

            # TITRE

            titre = soup.find("h1")

            if titre:

                data["Titre"] = (
                    titre.get_text(strip=True)
                )

            # PRIX

            prix = soup.find(
                class_="vehica-car-price"
            )

            if prix:

                data["Prix_DT"] = "".join(

                    filter(
                        str.isdigit,
                        prix.get_text()
                    )

                )

            # DATE DE DEPOT (ex: "29 juin 2026")

            date_depot = soup.find(
                class_="vehica-car-date"
            )

            if date_depot:

                data["Annonce-Deposee"] = parser_date_fr(
                    date_depot.get_text(strip=True)
                )

            # PUISSANCE FISCALE (ex: "9 cv" dans les vehica-car-feature)

            data["Puissance_Fiscale"] = extraire_puissance_fiscale(soup)

            # ATTRIBUTS

            specs = {}

            cles = soup.find_all(
                class_="vehica-car-attributes__name"
            )

            valeurs = soup.find_all(
                class_="vehica-car-attributes__values"
            )

            for c, v in zip(
                cles,
                valeurs
            ):

                cle = (
                    c.get_text(strip=True)
                    .replace(":", "")
                    .lower()
                )

                specs[cle] = (
                    v.get_text(strip=True)
                )

            data["Marque"] = specs.get(
                "marque"
            )

            data["Modèle"] = (

                specs.get("modèle")

                or

                specs.get("modele")

            )

            data["Année"] = "".join(

                filter(
                    str.isdigit,
                    specs.get("année", "")
                )

            )

            data["Kilométrage"] = "".join(

                filter(
                    str.isdigit,
                    specs.get("kilométrage", "")
                )

            )

            data["Boite"] = (

                specs.get("boite")

                or

                specs.get("transmission")

            )

            data["Energie"] = (

                specs.get("energie")

                or

                specs.get("carburant")

            )

            data["Localisation"] = (

                specs.get("gouvernorat")

                or

                specs.get("localisation")

            )

            return data

        except Exception as e:

            print(
                f"Tentative {tentative}/3 échouée :",
                url_annonce
            )

            time.sleep(
                random.uniform(3, 6)
            )

    print(
        "Abandon annonce :",
        url_annonce
    )

    return None

# -------------------------------------------------
# RECUPERATION PAGE AJAX
# -------------------------------------------------

def recuperer_annonces_page(offset, session):

    payload = [

        ("action", "vehica_car_results"),

        ("limit", "12"),

        ("offset", str(offset)),

        ("taxonomyTermsCountIds[]", "6659"),
        ("taxonomyTermsCountIds[]", "6660"),
        ("taxonomyTermsCountIds[]", "6663"),
        ("taxonomyTermsCountIds[]", "6662"),
        ("taxonomyTermsCountIds[]", "23548"),
        ("taxonomyTermsCountIds[]", "12770"),

        ("keyword", ""),

        ("markerContentFieldKey", ""),

        ("mapMode", "0"),

        ("baseUrl", LISTE_URL),

        ("trier-par", "recent"),

        ("base_url", LISTE_URL),

        ("cardConfig[type]", "vehica_card_v3"),

        ("cardConfig[showLabels]", "true")

    ]

    for tentative in range(1,4):

        try:

            r = session.post(

                AJAX_URL,

                data=payload,

                headers=HEADERS,

                timeout=30

            )

            data = r.json()

            html = data.get(
                "results",
                ""
            )

            if not html:

                return []

            soup = BeautifulSoup(

                html,

                "html.parser"

            )

            liens = []

            for a in soup.find_all(

                "a",

                class_="vehica-car-card-link",

                href=True

            ):

                href = a["href"]

                if not href.startswith("http"):

                    href = BASE_URL + href

                liens.append(href)

            return liens

        except Exception as e:

            print(
                f"Erreur AJAX tentative {tentative}/3 :",
                e
            )

            time.sleep(
                random.uniform(5,10)
            )

    return []

# -------------------------------------------------
# SAUVEGARDE
# -------------------------------------------------

def sauvegarder(annonces):

    if not annonces:

        return

    df_new = pd.DataFrame(
        annonces
    )

    os.makedirs(
        os.path.dirname(FICHIER_SORTIE),
        exist_ok=True
    )

    if os.path.exists(
        FICHIER_SORTIE
    ):

        df_old = pd.read_csv(
            FICHIER_SORTIE,
            sep=";"
        )

        df_final = pd.concat(
            [
                df_old,
                df_new
            ],
            ignore_index=True
        )

    else:

        df_final = df_new

    df_final.drop_duplicates(
        subset="Lien",
        inplace=True
    )

    df_final.to_csv(

        FICHIER_SORTIE,

        index=False,

        sep=";",

        encoding="utf-8-sig"

    )

    print(
        len(df_final),
        "annonces sauvegardées"
    )

# -------------------------------------------------
# SCRAPER PRINCIPAL
# -------------------------------------------------

def lancer_scraper():

    session = requests.Session()

    session.get(
        LISTE_URL,
        headers=HEADERS,
        timeout=30
    )

    urls_vues = set()

    # reprise CSV

    if os.path.exists(
        FICHIER_SORTIE
    ):

        df = pd.read_csv(
            FICHIER_SORTIE,
            sep=";"
        )

        if "Lien" in df.columns:

            urls_vues = set(
                df["Lien"].dropna()
            )

        print(
            len(urls_vues),
            "annonces déjà présentes"
        )

    offset = 0

    while True:

        print("="*60)

        print(
            "OFFSET :",
            offset
        )

        liens = recuperer_annonces_page(
            offset,
            session
        )

        print(
            len(liens),
            "annonces trouvées"
        )

        if len(liens) == 0:

            print(
                "Fin pagination"
            )

            break

        nouvelles = [

            x for x in liens

            if x not in urls_vues

        ]

        for url in nouvelles:

            print(
                "Scraping :",
                url
            )

            data = scraper_details_annonce(
                url,
                session
            )

            urls_vues.add(url)

            # BUG CORRIGE : la sauvegarde n'avait lieu que tous les 20
            # annonces -- en cas de coupure réseau/wifi entre deux
            # sauvegardes, jusqu'à 19 annonces déjà scrapées pouvaient être
            # perdues. On sauvegarde maintenant chaque annonce individuellement,
            # tout de suite, comme pour scraper_tayara.py et
            # scraper_automobile.py -- une coupure ne fait perdre au pire que
            # l'annonce en cours de traitement.
            if data:

                sauvegarder([data])

            time.sleep(
                random.uniform(2,4)
            )

        offset += 12

    print("=" * 60)

    print(
        "SCRAPING TERMINE"
    )

# -------------------------------------------------
# MAIN
# -------------------------------------------------

if __name__ == "__main__":

    lancer_scraper()