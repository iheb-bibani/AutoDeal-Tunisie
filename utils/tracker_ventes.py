"""
tracker_ventes.py
Détecte les annonces disparues (vendues) en effectuant des requêtes HEAD
"""

import os
import time
import random
import requests
from datetime import datetime
import pandas as pd
from logger import get_logger
from config import PROCESSED_FILES, HEADERS_DEFAULT

logger = get_logger(__name__)

FICHIER_COMPLET = PROCESSED_FILES["merged"]


def checker_annonces_vendues():
    """Détecte les annonces vendues via requêtes HEAD"""
    
    if not os.path.exists(FICHIER_COMPLET):
        logger.error(f"❌ Fichier introuvable: {FICHIER_COMPLET}")
        return

    logger.info("🧠 Vérification de l'état des annonces en cours...")
    
    try:
        df = pd.read_csv(FICHIER_COMPLET, sep=";", encoding="utf-8-sig")
    except Exception as e:
        logger.error(f"❌ Erreur lors du chargement: {e}")
        return

    # Initialiser les colonnes de suivi si nécessaire
    if "Statut" not in df.columns:
        df["Statut"] = "Actif"
    else:
        df["Statut"] = df["Statut"].fillna("Actif")

    if "Date_Disparition" not in df.columns:
        df["Date_Disparition"] = pd.NA

    total_verifies = 0
    nouvelles_ventes = 0
    erreurs_reseau = 0

    # Vérifier que les colonnes critiques existent
    if "Lien" not in df.columns or "Marque" not in df.columns:
        logger.error("❌ Colonnes 'Lien' ou 'Marque' manquantes")
        return

    # Parcourir les annonces actives
    for idx, row in df.iterrows():
        if row["Statut"] != "Actif":
            continue

        url = str(row.get("Lien", "")).strip()
        if not url or url == "nan":
            continue

        total_verifies += 1

        try:
            # HEAD request très rapide (sans charger le HTML complet)
            response = requests.head(url, headers=HEADERS_DEFAULT, timeout=10)

            if response.status_code == 404:
                df.at[idx, "Statut"] = "Vendu"
                df.at[idx, "Date_Disparition"] = datetime.now().strftime("%Y-%m-%d")
                nouvelles_ventes += 1
                marque = row.get("Marque", "Inconnue")
                modele = row.get("Modèle", "")
                logger.info(f"🔥 [VENDU] {marque} {modele}")
            elif response.status_code == 403:
                # Cloudflare ou protection - impossible de vérifier
                logger.debug(f"⚠️ Code 403 (Cloudflare?) pour {url[:50]}...")
                erreurs_reseau += 1

        except requests.exceptions.Timeout:
            logger.debug(f"⏱️ Timeout pour {url[:50]}...")
            erreurs_reseau += 1
        except Exception as e:
            logger.debug(f"⚠️ Erreur réseau: {str(e)[:50]}")
            erreurs_reseau += 1

        # Éviter de surcharger les serveurs
        time.sleep(random.uniform(0.5, 1.5))

    # Sauvegarde des modifications
    try:
        df.to_csv(FICHIER_COMPLET, index=False, sep=";", encoding="utf-8-sig")
        logger.info(f"✅ Bilan: {total_verifies} vérifiées, {nouvelles_ventes} vendues, {erreurs_reseau} erreurs réseau")
    except Exception as e:
        logger.error(f"❌ Erreur lors de la sauvegarde: {e}")


if __name__ == "__main__":
    checker_annonces_vendues()

