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
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from config import PROCESSED_FILES, MARQUES_LUXE, GRAND_TUNIS


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

    df.to_csv(PROCESSED_FILES["enriched"], index=False, sep=";", encoding="utf-8-sig")
    print(f"Base enrichie : {len(df)} annonces -> {PROCESSED_FILES['enriched']}")


if __name__ == "__main__":
    enrichir()
