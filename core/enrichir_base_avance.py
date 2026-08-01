"""
enrichir_base_avance.py
Étape 3 du pipeline : ajoute les variables dérivées utilisées par le modèle
et par l'application, à partir de tunisia-cars-recent.csv.

Variables créées :
  - Segment_Vehicule    : 1 = marque de luxe (config.MARQUES_LUXE), 0 sinon.
  - Est_Presque_Neuve   : 1 = millésime de l'année en cours ou précédente.
                          (calculé dynamiquement -- pas d'année en dur qui
                          deviendrait fausse au 1er janvier suivant)
  - Zone_Economique     : 1 = Grand Tunis (config.GRAND_TUNIS), 0 = Province.
  - Age_Vehicule        : années écoulées depuis le millésime.
  - Log_Kilometrage     : log1p(km), robuste à la longue traîne du kilométrage.
  - Age_Carre           : âge², permet aux modèles simples de capter la non-linéarité.
  - Km_Par_An           : intensité d'usage, plus informative que km ou âge seuls.
  - Segment_Libelle / Zone_Libelle : versions lisibles conservées pour
    l'application (les codes 0/1 ne servent qu'au modèle -- l'app n'a plus
    besoin de "re-deviner" le sens des codes).

Les encodages sont FIXES (1 = Luxe, 1 = Grand_Tunis) et non dérivés de
cat.codes : cat.codes dépendait de l'ordre alphabétique des catégories
présentes dans les données du jour, donc le sens de 0/1 pouvait s'inverser
d'un scraping à l'autre.
"""

import re
import sys
import unicodedata
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from config import PROCESSED_FILES, MARQUES_LUXE, GRAND_TUNIS


def _sans_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")


# Marqueurs d'état présents dans le TEXTE des annonces (pas dans un champ
# structuré). Le prix les reflète, le modèle les ignore aujourd'hui. Motifs
# écrits sans accents car on normalise le texte avant de matcher.
MOTIFS_ETAT = {
    "Premiere_Main": r"1\s*ere?\s*main|premiere main|1ere main|1er main",
    "Non_Accidentee": r"jamais accident|non accident|pas accident|aucun accident|non accidentee",
    "Full_Options": r"full\s*option|toutes?\s*options?|ttes?\s*option|tte\s*option",
    "Etat_Origine": r"etat d.?origine|peinture d.?origine|tout.? d.?origine|etats? d.?origine",
}


def deriver_drapeaux_etat(texte: pd.Series) -> pd.DataFrame:
    """À partir d'un texte libre d'annonce, dérive des drapeaux binaires d'état
    (1ère main, non accidentée, full options, état d'origine). Insensible aux
    accents et à la casse. Renvoie un DataFrame de colonnes 0/1 indexé comme
    l'entrée."""
    t = texte.fillna("").astype(str).map(lambda s: _sans_accents(s.lower()))
    return pd.DataFrame(
        {col: t.str.contains(rx, regex=True, na=False).astype(int) for col, rx in MOTIFS_ETAT.items()},
        index=texte.index,
    )


def enrichir():
    df = pd.read_csv(PROCESSED_FILES["recent"], sep=";", encoding="utf-8-sig")

    annee_courante = datetime.now().year

    # Segment : 1 = Luxe, 0 = Standard (encodage fixe)
    df["Segment_Vehicule"] = df["Marque"].isin(MARQUES_LUXE).astype(int)
    df["Segment_Libelle"] = np.where(df["Segment_Vehicule"] == 1, "Luxe", "Standard")

    # Presque neuve : millésime de l'année en cours ou de l'année précédente
    df["Est_Presque_Neuve"] = (df["Année"] >= annee_courante - 1).fillna(False).astype(int)

    # Âge du véhicule (utile au modèle et aux analyses de dépréciation)
    df["Age_Vehicule"] = (annee_courante - df["Année"]).clip(lower=0)

    # Feature engineering robuste : le kilométrage a une longue traîne et son
    # sens dépend fortement de l'âge. On garde les variables brutes ET ces
    # transformations afin que les modèles linéaires comme les arbres puissent
    # exploiter la relation sous plusieurs formes.
    km = pd.to_numeric(df["Kilométrage"], errors="coerce").clip(lower=0)
    age = pd.to_numeric(df["Age_Vehicule"], errors="coerce").clip(lower=0)
    df["Log_Kilometrage"] = np.log1p(km)
    df["Age_Carre"] = age.pow(2)
    # âge=0 : on divise par 1 pour éviter l'infini ; clip anti-erreurs de saisie.
    df["Km_Par_An"] = (km / age.clip(lower=1)).clip(upper=100000)

    # Zone économique : 1 = Grand Tunis, 0 = Province (encodage fixe)
    df["Zone_Economique"] = df["Localisation"].isin(GRAND_TUNIS).astype(int)
    df["Zone_Libelle"] = np.where(df["Zone_Economique"] == 1, "Grand Tunis", "Province")

    # Puissance réelle (DIN) et cylindrée, extraites du titre.
    # automobile.tn décrit ses annonces de façon très normée
    # ("Volkswagen Golf 7 Smartline 1.2 TSI 16V S&S 110 cv Boîte auto") :
    # deux caractéristiques techniques qui expliquent une part du prix y sont
    # disponibles gratuitement, sans champ structuré ni scraping supplémentaire.
    # À ne pas confondre avec Puissance_Fiscale (base de la taxe, échelle
    # administrative) : ici c'est la puissance moteur réelle.
    titre = df["Titre"].astype(str)
    df["Puissance_DIN"] = pd.to_numeric(
        titre.str.extract(r"(\d{2,3})\s*cv", flags=re.IGNORECASE, expand=False), errors="coerce"
    )
    df.loc[~df["Puissance_DIN"].between(40, 700), "Puissance_DIN"] = np.nan
    df["Cylindree"] = pd.to_numeric(titre.str.extract(r"\b(\d\.\d)\b", expand=False), errors="coerce")
    df.loc[~df["Cylindree"].between(0.6, 8.0), "Cylindree"] = np.nan

    # Drapeaux d'état, dérivés du texte de l'annonce : "Description" si les
    # scrapers la remontent, sinon le "Titre" (court -> drapeaux surtout à 0).
    # INERTES tant que la description n'est pas scrapée : les colonnes existent
    # (0/1) et le modèle les utilise, mais elles n'ont de signal qu'une fois la
    # description capturée par les scrapers, puis re-scraping + réentraînement.
    source_texte = df["Description"] if "Description" in df.columns else df["Titre"]
    for col, valeurs in deriver_drapeaux_etat(source_texte).items():
        df[col] = valeurs

    df.to_csv(PROCESSED_FILES["enriched"], index=False, sep=";", encoding="utf-8-sig")
    print(f"Base enrichie : {len(df)} annonces -> {PROCESSED_FILES['enriched']}")


if __name__ == "__main__":
    enrichir()
