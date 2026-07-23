"""
utils/common.py
Fonctions communes réutilisables à travers le projet
"""

import os
import time
import random
import pandas as pd
from pathlib import Path
from typing import Optional, Set
from logger import get_logger

logger = get_logger(__name__)


def charger_liens_deja_scrapes(chemin: str) -> Set[str]:
    """Récupère les liens déjà scrapés pour éviter les doublons"""
    if not (os.path.exists(chemin) and os.path.getsize(chemin) > 0):
        return set()
    
    try:
        df = pd.read_csv(chemin, sep=";", encoding="utf-8-sig")
        liens = set(df["Lien"].dropna().astype(str).unique())
        logger.info(f"✅ {len(liens)} liens déjà scrapés chargés depuis {chemin}")
        return liens
    except Exception as e:
        logger.warning(f"⚠️ Erreur lors du chargement des liens: {e}")
        return set()


def enregistrer_ligne(car: dict, chemin: str, colonnes: list):
    """
    Ajoute une seule annonce au CSV, au fur et à mesure (append mode)
    
    Args:
        car: Dictionnaire avec les données de l'annonce
        chemin: Chemin du fichier CSV
        colonnes: Liste des colonnes attendues
    """
    try:
        os.makedirs(os.path.dirname(chemin), exist_ok=True)
        
        ligne = {col: car.get(col, pd.NA) for col in colonnes}
        fichier_existe = os.path.exists(chemin) and os.path.getsize(chemin) > 0
        
        pd.DataFrame([ligne])[colonnes].to_csv(
            chemin,
            mode="a",
            header=not fichier_existe,
            index=False,
            sep=";",
            encoding="utf-8-sig",
        )
    except Exception as e:
        logger.error(f"❌ Erreur lors de l'enregistrement de la ligne: {e}")


def attendre_aleatoire(min_sec: float = 0.5, max_sec: float = 2.0) -> None:
    """Pause aléatoire pour éviter de surcharger les serveurs"""
    delai = random.uniform(min_sec, max_sec)
    time.sleep(delai)


def normaliser_texte(texte: Optional[str]) -> Optional[str]:
    """Normalise un texte: strip, None si vide"""
    if texte is None:
        return None
    texte = str(texte).strip()
    if texte.lower() in ["nan", "none", ""]:
        return None
    return texte


def normaliser_nombre(valeur) -> Optional[float]:
    """Convertit une valeur en nombre, retourne None si impossible"""
    try:
        if pd.isna(valeur) or valeur is None:
            return None
        return float(valeur)
    except (ValueError, TypeError):
        return None


def fichier_existe_et_non_vide(chemin: str) -> bool:
    """Vérifie qu'un fichier existe et n'est pas vide"""
    try:
        return os.path.exists(chemin) and os.path.getsize(chemin) > 0
    except OSError:
        return False


def charger_dataframe_securise(chemin: str, sep: str = ";") -> Optional[pd.DataFrame]:
    """Charge un CSV avec gestion d'erreurs"""
    try:
        if not fichier_existe_et_non_vide(chemin):
            logger.warning(f"⚠️ Fichier introuvable ou vide: {chemin}")
            return None
        
        df = pd.read_csv(chemin, sep=sep, encoding="utf-8-sig")
        logger.info(f"✅ Fichier chargé: {chemin} ({len(df)} lignes)")
        return df
    except Exception as e:
        logger.error(f"❌ Erreur lors du chargement de {chemin}: {e}")
        return None


def sauvegarder_dataframe(df: pd.DataFrame, chemin: str, sep: str = ";") -> bool:
    """Sauvegarde un DataFrame avec gestion d'erreurs"""
    try:
        os.makedirs(os.path.dirname(chemin) or ".", exist_ok=True)
        df.to_csv(chemin, index=False, sep=sep, encoding="utf-8-sig")
        logger.info(f"✅ Fichier sauvegardé: {chemin} ({len(df)} lignes)")
        return True
    except Exception as e:
        logger.error(f"❌ Erreur lors de la sauvegarde de {chemin}: {e}")
        return False


def verifier_colonnes_requises(df: pd.DataFrame, colonnes_requises: list) -> bool:
    """Vérifie que le DataFrame contient les colonnes requises"""
    manquantes = set(colonnes_requises) - set(df.columns)
    if manquantes:
        logger.error(f"❌ Colonnes manquantes: {manquantes}")
        return False
    return True
