"""
nettoyer_base.py
Étape 2 du pipeline : nettoie tunisia-cars.csv et produit tunisia-cars-recent.csv.

Rôles :
  1. Corriger les valeurs invalides héritées du scraping (Boite_Vitesse
     contenant un type de transmission, prix/km/CV/années aberrants...).
  2. Filtrer le bruit de tayara.tn (immobilier, pièces détachées, locations,
     annonces sans contenu).
  3. Deviner Marque et Modèle depuis le TITRE quand le champ structuré est
     vide ou "Autres" -- trois niveaux :
       a) marques/modèles déjà connus dans la base,
       b) référentiel du marché tunisien (config.MARQUES_ALIAS,
          MODELES_PAR_MARQUE_REFERENTIEL),
       c) modèle qui implique la marque ("Golf 7" -> Volkswagen Golf).
     Règles spécifiques pour les allemandes premium : codes moteur BMW
     (320d -> Série 3), classes Mercedes (C180 -> Classe C), nomenclature
     Audi (A3, Q5, TT).
  4. Ne garder que les annonces <= MAX_DAYS_OLD jours.
"""

import re
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from config import (
    PROCESSED_FILES, MAX_DAYS_OLD,
    MARQUES_ALIAS, MODELE_IMPLIQUE_MARQUE, MODELES_PAR_MARQUE_REFERENTIEL,
)


# ---------------------------------------------------------------------------
# Bruit : annonces qui ne sont pas des ventes de voitures particulières
# ---------------------------------------------------------------------------

MOTS_BRUIT = [
    # Immobilier (vraies annonces immo glissées dans la catégorie voiture de tayara)
    "terrain", "maison", "appartement", "villa", "duplex", "studio", "triplex",
    "hammam", "immeuble", "local commercial", "m²", "m2 ",
    # Pièces détachées et accessoires
    "chaise", "pièce", "piece de rechange", "rechange", "accessoire",
    "alternateur", "demarreur", "démarreur", "compresseur", "climatiseur",
    "kit embrayage", "embrayage", "amortisseur", "pare choc", "pare-choc",
    "jantes", "boujie", "bougie", "radiateur", "turbo à vendre",
    "autoradio", "antenne", "jouet", "enfant",
    # Pas des ventes de voitures particulières
    "location", "louer", "camion ", "remorque", "moto ", "scooter",
    # Pages non-annonces capturées par erreur par le scraper automobile.tn
    "présentation de la cote",
]

MOTS_A_IGNORER = {
    "a", "à", "de", "du", "des", "occasion", "vendre", "vente", "en", "pour",
    "avec", "sans", "et", "ou", "la", "le", "les", "un", "une", "tres", "très",
    "bon", "bonne", "état", "etat", "toutes", "options", "importée", "importee",
}

# Marques premium allemandes : le mot après la marque est presque toujours une
# motorisation/finition, pas un modèle -- récupération libre interdite,
# seules les règles dédiées ci-dessous s'appliquent.
MARQUES_PREMIUM = {"Audi", "Bmw", "Mercedes-Benz"}


def normaliser_pour_comparaison(texte: str) -> str:
    """Retire espaces et tirets : 'Golf 7', 'Golf-7' et 'Golf7' deviennent
    identiques."""
    return re.sub(r"[\s\-]+", "", str(texte)).lower()


# ---------------------------------------------------------------------------
# Inférence de la Marque depuis le titre
# ---------------------------------------------------------------------------

def construire_devineur_marque(marques_connues):
    """Retourne une fonction titre -> marque (ou None).

    Ordre de priorité :
      1. Marque déjà présente dans la base (mot entier dans le titre).
      2. Alias du référentiel (couvre fautes d'orthographe et marques
         absentes de la base : Cupra, Haval, Dongfeng, Lada, Wallys...).
      3. Modèle qui implique la marque ("clio" -> Renault).
    """
    marques_triees = sorted(set(marques_connues), key=len, reverse=True)
    motif_connues = (
        re.compile(r"(?i)\b(" + "|".join(re.escape(m) for m in marques_triees) + r")\b")
        if marques_triees else None
    )
    casse_connues = {m.lower(): m for m in marques_triees}

    def deviner(titre: str):
        titre = str(titre)
        if motif_connues:
            match = motif_connues.search(titre)
            if match:
                return casse_connues[match.group(1).lower()]
        mots = re.findall(r"[a-zà-ÿA-ZÀ-Ÿ0-9\-]+", titre.lower())
        for mot in mots:
            if mot in MARQUES_ALIAS:
                return MARQUES_ALIAS[mot]
        for mot in mots:
            if mot in MODELE_IMPLIQUE_MARQUE:
                return MODELE_IMPLIQUE_MARQUE[mot][0]
        return None

    return deviner


# ---------------------------------------------------------------------------
# Inférence du Modèle depuis le titre
# ---------------------------------------------------------------------------

def construire_devineur_modele(modeles_connus_par_marque):
    """Retourne une fonction (marque, titre) -> modèle (ou None).

    Le catalogue de modèles candidats par marque = modèles déjà vus dans la
    base ∪ référentiel tunisien -- toujours essayés du plus long au plus
    court ('Golf 7' avant 'Golf', 'Grand i10' avant 'i10').
    """
    catalogue = {}
    marques = set(modeles_connus_par_marque) | set(MODELES_PAR_MARQUE_REFERENTIEL)
    for marque in marques:
        vus = [str(m) for m in modeles_connus_par_marque.get(marque, [])]
        ref = MODELES_PAR_MARQUE_REFERENTIEL.get(marque, [])
        # Le nom déjà utilisé dans la base garde la priorité sur le
        # référentiel à normalisation égale (cohérence des libellés).
        fusion = {}
        for m in ref:
            fusion[normaliser_pour_comparaison(m)] = m
        for m in vus:
            fusion[normaliser_pour_comparaison(m)] = m
        catalogue[marque] = sorted(fusion.values(), key=len, reverse=True)

    def deviner(marque, titre: str):
        if pd.isna(marque):
            return None
        titre = str(titre)

        # Règles dédiées aux premium allemandes (codes moteur / nomenclature)
        if marque == "Bmw":
            match = re.search(r"\b([1-8])\d{2}(?:d|i|e)\b", titre, re.IGNORECASE)
            if match:
                return f"Série {match.group(1)}"
        elif marque == "Mercedes-Benz":
            # C180, E250, A200, B160, S500... -> Classe C/E/A/B/S
            match = re.search(r"\b([ABCES])[\s\-]?(\d{3})\b", titre, re.IGNORECASE)
            if match:
                return f"Classe {match.group(1).upper()}"

        # Niveau 1 : modèle du catalogue présent dans le titre (mot entier)
        candidats = catalogue.get(marque, [])
        for modele in candidats:
            if re.search(rf"(?i)(?<![a-z0-9]){re.escape(modele)}(?![a-z0-9])", titre):
                return modele

        # Niveau 1b : repli insensible aux espaces/tirets ("Golf7", "grand-i10")
        titre_norm = normaliser_pour_comparaison(titre)
        for modele in candidats:
            modele_norm = normaliser_pour_comparaison(modele)
            if modele_norm and modele_norm in titre_norm:
                return modele

        # Niveau 2 : mot juste après la marque -- interdit pour les premium
        # (le mot suivant y désigne une motorisation, pas un modèle)
        if marque in MARQUES_PREMIUM:
            return None
        match = re.search(rf"(?i)\b{re.escape(marque)}\s+([A-Za-zÀ-ÿ0-9\-]+)", titre)
        if match:
            candidat = match.group(1)
            if candidat.lower() in MOTS_A_IGNORER:
                return None
            # Un nombre après la marque n'est un modèle que s'il ressemble à
            # une appellation (208, 3008, 500...), pas à une année ou un prix.
            if candidat.isdigit() and (len(candidat) not in (3, 4) or candidat.startswith(("19", "20"))):
                return None
            return candidat.capitalize() if candidat.isalpha() else candidat
        return None

    return deviner


# ---------------------------------------------------------------------------
# Nettoyage principal
# ---------------------------------------------------------------------------

def est_valeur_chiffre_repete(valeur) -> bool:
    """Détecte les valeurs de remplissage type 11111, 999999 (>= 4 chiffres,
    tous identiques) -- jamais un vrai prix ou kilométrage."""
    if pd.isna(valeur):
        return False
    chiffres = str(int(valeur))
    return len(chiffres) >= 4 and len(set(chiffres)) == 1


def process_data():
    print("Chargement du fichier...")
    df = pd.read_csv(PROCESSED_FILES["merged"], sep=";", encoding="utf-8-sig")

    # -- Boite_Vitesse : un ancien fallback du scraper automobile.tn capturait
    # le type de TRANSMISSION AUX ROUES (Traction/Intégrale/Propulsion) au
    # lieu du type de boîte. Le scraper est corrigé, mais on nettoie aussi ici
    # pour ne pas dépendre d'un nouveau scraping.
    # -- Boite_Vitesse : sur automobile.tn, le scraper capture le type de
    # TRANSMISSION AUX ROUES (Traction/Intégrale/Propulsion) au lieu du type
    # de boîte -- 100% des annonces de cette source. Cette valeur n'est pas
    # fausse en soi, elle est simplement dans la mauvaise colonne : on la
    # déplace vers Transmission (vraie information, exploitable) au lieu de
    # la jeter, puis on vide Boite_Vitesse.
    valeurs_transmission = ["traction", "intégrale", "integrale", "propulsion", "4x4", "4x2"]
    mask = df["Boite_Vitesse"].astype(str).str.lower().isin(valeurs_transmission)
    df["Transmission"] = df["Boite_Vitesse"].where(mask)
    if mask.sum():
        print(f"↔️  {mask.sum()} valeurs de Boite_Vitesse déplacées vers Transmission (type de transmission, pas de boîte).")
    df.loc[mask, "Boite_Vitesse"] = pd.NA

    # -- Récupération de la boîte depuis le titre pour les annonces ainsi
    # vidées. automobile.tn ne mentionne QUE les automatiques dans ses titres
    # ("... 110 cv Boîte auto") -- jamais les manuelles. On ne récupère donc
    # que ce qui est explicite : supposer que l'absence de mention vaut
    # "Manuelle" serait une invention, pas une déduction.
    motif_auto = r"bo[iî]te auto|automatique|dsg\d?|tiptronic|s-tronic|\bedc\b|\bcvt\b|g-tronic|eat\d"
    besoin_boite = df["Boite_Vitesse"].isna()
    est_auto = df["Titre"].astype(str).str.lower().str.contains(motif_auto, regex=True, na=False)
    recuperees_boite = (besoin_boite & est_auto).sum()
    df.loc[besoin_boite & est_auto, "Boite_Vitesse"] = "Automatique"
    if recuperees_boite:
        print(f"🔧 {recuperees_boite} boîtes 'Automatique' récupérées depuis le titre.")

    # -- Puissance_Fiscale : automobile.tn la stocke en TEXTE ("5cv", "12cv").
    # Une conversion numérique directe renvoyait NaN sur la totalité de cette
    # source -- plus de 2 000 valeurs parfaitement exploitables perdues
    # silencieusement. On extrait le nombre avant toute conversion.
    df["Puissance_Fiscale"] = (
        df["Puissance_Fiscale"].astype(str).str.extract(r"(\d+(?:[.,]\d+)?)", expand=False).str.replace(",", ".")
    )

    # -- "Autres" = champ non rempli chez tayara, pas une vraie valeur
    for col in ["Marque", "Modèle"]:
        mask_autres = df[col].astype(str).str.strip().str.lower().isin(["autres", "autre"])
        df.loc[mask_autres, col] = pd.NA

    # -- Bruit : immobilier, pièces détachées, locations, pages non-annonces
    mask_bruit = df["Titre"].astype(str).str.contains(
        "|".join(MOTS_BRUIT), case=False, na=False
    )
    if mask_bruit.sum():
        print(f"🧹 {mask_bruit.sum()} annonces filtrées (immobilier, pièces, locations, pages parasites).")
    df = df[~mask_bruit]

    # -- Titres vides ET aucune marque : lignes inexploitables
    mask_vide = df["Titre"].isna() & df["Marque"].isna()
    df = df[~mask_vide]

    # -- Inférence Marque depuis le titre
    marques_connues = df["Marque"].dropna().unique()
    deviner_marque = construire_devineur_marque(marques_connues)
    avant = df["Marque"].isna().sum()
    besoin = df["Marque"].isna()
    df.loc[besoin, "Marque"] = df.loc[besoin, "Titre"].apply(deviner_marque)
    recuperees = avant - df["Marque"].isna().sum()
    if recuperees:
        print(f"🔧 {recuperees} marques récupérées depuis le titre.")

    # -- Inférence Modèle depuis le titre
    modeles_connus = (
        df.dropna(subset=["Marque", "Modèle"]).groupby("Marque")["Modèle"].unique().to_dict()
    )
    deviner_modele = construire_devineur_modele(modeles_connus)
    avant = df["Modèle"].isna().sum()
    besoin = df["Modèle"].isna() & df["Marque"].notna()
    df.loc[besoin, "Modèle"] = df.loc[besoin].apply(
        lambda r: deviner_modele(r["Marque"], r["Titre"]), axis=1
    )
    recuperes = avant - df["Modèle"].isna().sum()
    if recuperes:
        print(f"🔧 {recuperes} modèles récupérés depuis le titre.")

    # -- Corrections ciblées d'erreurs de saisie côté vendeurs (tayara laisse
    # le vendeur choisir marque/modèle dans des listes -- certains se trompent)
    titre_bas = df["Titre"].astype(str).str.lower()
    # "Great Wall M4" (souvent étiqueté Haval/Autres par les vendeurs)
    mask_gw = titre_bas.str.contains(r"great\s*(?:wall|falls)", regex=True, na=False)
    if mask_gw.sum():
        df.loc[mask_gw, "Marque"] = "Gwm"
        df.loc[mask_gw & titre_bas.str.contains(r"\bm4\b", regex=True), "Modèle"] = "M4"
    # "Gol" choisi à la place de "Golf" alors que le titre dit golf
    mask_gol = (df["Marque"] == "Volkswagen") & (df["Modèle"] == "Gol") & titre_bas.str.contains("golf", na=False)
    df.loc[mask_gol, "Modèle"] = "Golf"

    # -- Casse canonique des modèles : "jolion", "Jolion" et "JOLION" doivent
    # être une seule catégorie. Pour chaque (marque, modèle normalisé), on
    # retient la graphie la plus fréquente -- le référentiel a priorité.
    canon = {}
    for marque, modeles in MODELES_PAR_MARQUE_REFERENTIEL.items():
        for m in modeles:
            canon[(marque, normaliser_pour_comparaison(m))] = m
    freq = (
        df.dropna(subset=["Marque", "Modèle"])
        .assign(_cle=lambda d: d["Modèle"].map(normaliser_pour_comparaison))
        .groupby(["Marque", "_cle"])["Modèle"]
        .agg(lambda s: s.value_counts().idxmax())
    )
    for (marque, cle), graphie in freq.items():
        canon.setdefault((marque, cle), graphie)
    a_modele = df["Modèle"].notna() & df["Marque"].notna()
    df.loc[a_modele, "Modèle"] = df.loc[a_modele].apply(
        lambda r: canon.get((r["Marque"], normaliser_pour_comparaison(r["Modèle"])), r["Modèle"]),
        axis=1,
    )

    # -- Une annonce sans Marque, même après inférence, n'est pas exploitable
    avant = len(df)
    df = df.dropna(subset=["Marque"])
    if avant != len(df):
        print(f"🧹 {avant - len(df)} annonces sans Marque supprimées (fiches trop incomplètes).")

    # -- Normalisation des numériques
    for col in ["Prix", "Année", "Kilométrage", "Puissance_Fiscale"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # -- Prix hors plage plausible pour ce marché
    df = df[(df["Prix"] >= 3000) & (df["Prix"] <= 500000)]

    # -- Valeurs de remplissage (11111, 999999...) : Prix -> ligne écartée,
    # Kilométrage -> mis à vide (le reste de l'annonce reste exploitable)
    mask_prix = df["Prix"].apply(est_valeur_chiffre_repete)
    if mask_prix.sum():
        print(f"🧹 {mask_prix.sum()} prix 'chiffre répété' (ex: 111111) écartés.")
    df = df[~mask_prix]

    mask_km = df["Kilométrage"].apply(est_valeur_chiffre_repete)
    df.loc[mask_km, "Kilométrage"] = pd.NA

    # -- Kilométrage implausible (> 500 000 km ou négatif) -> vide
    hors_km = (df["Kilométrage"] < 0) | (df["Kilométrage"] > 500000)
    if hors_km.sum():
        print(f"🧹 {hors_km.sum()} kilométrages implausibles mis à vide.")
    df.loc[hors_km, "Kilométrage"] = pd.NA

    # -- Puissance fiscale implausible (numéros de téléphone, ID, négatifs)
    hors_cv = (df["Puissance_Fiscale"] <= 0) | (df["Puissance_Fiscale"] > 30)
    if hors_cv.sum():
        print(f"🧹 {hors_cv.sum()} puissances fiscales implausibles mises à vide.")
    df.loc[hors_cv, "Puissance_Fiscale"] = pd.NA

    # -- Année implausible (< 1980 ou dans le futur)
    annee_max = datetime.now().year + 1
    hors_annee = (df["Année"] < 1980) | (df["Année"] > annee_max)
    if hors_annee.sum():
        print(f"🧹 {hors_annee.sum()} années implausibles mises à vide.")
    df.loc[hors_annee, "Année"] = pd.NA

    # -- Dates : nos scrapers écrivent TOUJOURS en ISO (AAAA-MM-JJ) --
    # surtout pas de dayfirst=True, qui inverserait jour et mois.
    df["Annonce-Deposee"] = pd.to_datetime(df["Annonce-Deposee"], errors="coerce")
    aujourd_hui = pd.Timestamp.now().normalize()
    df["Age_Annonce_Jours"] = (aujourd_hui - df["Annonce-Deposee"]).dt.days.clip(lower=0)

    # -- Fraîcheur : base "recent" limitée à MAX_DAYS_OLD jours
    df_recent = df[df["Age_Annonce_Jours"] <= MAX_DAYS_OLD].copy()

    df_recent.to_csv(PROCESSED_FILES["recent"], index=False, sep=";", encoding="utf-8-sig")

    print("-" * 30)
    print(f"Annonces totales après nettoyage : {len(df)}")
    print(f"Annonces gardées (<= {MAX_DAYS_OLD} jours) : {len(df_recent)}")
    print(f"Fichier généré : {PROCESSED_FILES['recent']}")
    print("-" * 30)


if __name__ == "__main__":
    process_data()
