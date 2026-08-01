"""
merging_files.py
Étape 1 du pipeline : fusionne les 3 sources (automax, automobile, tayara)
avec normalisation complète et filtrage du bruit détecté en données réelles
"""

import sys
from pathlib import Path

# Ajouter le répertoire parent au path pour les imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from logger import get_logger
from config import PROCESSED_FILES, SCRAPERS, CORRECTIONS_MARQUE, MOTS_BRUIT
from utils.common import charger_dataframe_securise, sauvegarder_dataframe

logger = get_logger(__name__)


def normaliser_marque(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Marque"] = df["Marque"].astype(str).str.strip().str.title()
    cle_comparaison = df["Marque"].str.lower()
    corrections = cle_comparaison.map(CORRECTIONS_MARQUE)
    # CORRECTIONS_MARQUE peut mapper vers None ("autres" = champ non rempli
    # chez tayara, pas une marque) : on distingue "clé absente du mapping"
    # (garder la valeur d'origine) de "clé mappée vers None" (mettre à vide).
    a_une_correction = cle_comparaison.isin(CORRECTIONS_MARQUE.keys())
    df["Marque"] = corrections.where(a_une_correction, df["Marque"])
    df.loc[df["Marque"].isin(["Nan", "None", ""]) | df["Marque"].isna(), "Marque"] = pd.NA
    # Même logique pour Modèle : "Autres" n'est pas un modèle
    if "Modèle" in df.columns:
        mask_autres = df["Modèle"].astype(str).str.strip().str.lower().isin(["autres", "autre"])
        df.loc[mask_autres, "Modèle"] = pd.NA
    return df


def normaliser_energie(df: pd.DataFrame) -> pd.DataFrame:
    """BUG CORRIGE (confirmé sur données réelles) : 'Hybride Diesel' et
    'Hybride diesel' existaient comme deux catégories séparées -- même
    fragmentation que pour Marque, mais l'inverse s'applique ici : .title()
    aurait donné 'Hybride Diesel' partout (faux, casse chaque mot en
    majuscule), alors que la convention déjà utilisée partout ailleurs est la
    casse "phrase" ('Hybride léger diesel', 'Hybride rechargeable essence') --
    .capitalize() est la bonne normalisation ici, pas .title()."""
    df = df.copy()
    if "Energie" in df.columns:
        df["Energie"] = df["Energie"].astype(str).str.strip().str.capitalize()
        df.loc[df["Energie"].isin(["Nan", "None", ""]), "Energie"] = pd.NA
    return df


def normaliser_localisation(df: pd.DataFrame) -> pd.DataFrame:
    """BUG CORRIGE (confirmé sur données réelles) : 'Tunis' (842 annonces) et
    'tunis' (61 annonces) étaient deux catégories séparées -- même famille de
    bug que Marque. Les gouvernorats sont des noms propres multi-mots
    ('La Manouba', 'Ben Arous', 'Sidi Bouzid') -- .title() est ici la bonne
    normalisation, comme pour Marque."""
    df = df.copy()
    if "Localisation" in df.columns:
        df["Localisation"] = df["Localisation"].astype(str).str.strip().str.title()
        df.loc[df["Localisation"].isin(["Nan", "None", ""]), "Localisation"] = pd.NA
    return df


def fusionner_sources():
    """Charge les 3 fichiers bruts, normalise et fusionne"""
    
    logger.info("🛰️ Début de la fusion des sources...")
    
    # Chargement des données brutes
    df_automax = charger_dataframe_securise(SCRAPERS["automax"])
    df_auto = charger_dataframe_securise(SCRAPERS["automobile"])
    df_tayara = charger_dataframe_securise(SCRAPERS["tayara"])
    # sayyaratn : source additionnelle, OPTIONNELLE (peut être absente au premier
    # run ou si le scraper a échoué) -> on ne bloque pas la fusion si elle manque.
    df_sayyarat = charger_dataframe_securise(SCRAPERS.get("sayyarat", ""))
    df_autocentral = charger_dataframe_securise(SCRAPERS.get("autocentral", ""))
    
    if df_automax is None or df_auto is None or df_tayara is None:
        logger.error("❌ Une ou plusieurs sources principales sont manquantes")
        return None
    
    # Standardisation des noms de colonnes
    standard_cols = {
        'Prix_DT': 'Prix',
        'Boite': 'Boite_Vitesse'
    }
    
    sources = [df_automax, df_auto, df_tayara]
    if df_sayyarat is not None:
        sources.append(df_sayyarat)
        logger.info(f"➕ Source sayyaratn incluse ({len(df_sayyarat)} annonces)")
    if df_autocentral is not None:
        sources.append(df_autocentral)
        logger.info(f"➕ Source autocentral (non-doublons) incluse ({len(df_autocentral)} annonces)")

    for df in sources:
        df.rename(columns=standard_cols, inplace=True)
    
    # Fusion
    df_final = pd.concat(sources, ignore_index=True)
    logger.info(f"✅ Fusion brute: {len(df_final)} annonces")

    # Déduplication par Lien : si une annonce a été scrapée plusieurs fois
    # (relances, appends successifs), on garde la version la plus récente.
    avant_dedup = len(df_final)
    if "Annonce-Detectee" in df_final.columns:
        df_final = df_final.sort_values("Annonce-Detectee")
    df_final = df_final.drop_duplicates(subset=["Lien"], keep="last")
    nb_doublons = avant_dedup - len(df_final)
    if nb_doublons:
        logger.info(f"🧹 {nb_doublons} doublons (même Lien) supprimés")
    
    # Filtrage du bruit -- MOTS_BRUIT partagé avec nettoyer_base.py (config)
    mask_bruit = df_final['Titre'].astype(str).str.contains('|'.join(MOTS_BRUIT), case=False, na=False)
    nb_bruit = mask_bruit.sum()
    if nb_bruit:
        logger.info(f"🧹 {nb_bruit} annonces filtrées (bruit: pièces, immobilier, jouets...)")
    df_final = df_final[~mask_bruit]
    
    # AMELIORATION : Filtrage de tayara.tn par catégorie URL
    # (plus fiable que le filtrage par titre)
    est_tayara = df_final["Source"].astype(str).str.lower() == "tayara.tn"
    categorie_url = df_final["Lien"].astype(str).str.extract(r"tayara\.tn/item/([^/]+)/", expand=False)
    mask_mauvaise_categorie = est_tayara & (categorie_url != "voitures")
    nb_mauvaise_categorie = mask_mauvaise_categorie.sum()
    if nb_mauvaise_categorie:
        logger.info(f"🧹 {nb_mauvaise_categorie} annonces tayara hors catégorie 'voitures'")
    df_final = df_final[~mask_mauvaise_categorie]
    
    # Normalisation
    df_final = normaliser_marque(df_final)
    df_final = normaliser_energie(df_final)
    df_final = normaliser_localisation(df_final)
    
    # Sauvegarde
    sortie = PROCESSED_FILES["merged"]
    if sauvegarder_dataframe(df_final, sortie):
        logger.info(f"✅ {len(df_final)} annonces fusionnées sauvegardées")
        return df_final
    else:
        logger.error("❌ Erreur lors de la sauvegarde")
        return None


if __name__ == "__main__":
    fusionner_sources()
