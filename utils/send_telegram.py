import sys
from pathlib import Path

# Lancé en tant que script (`python utils/send_telegram.py`), Python place le
# dossier du SCRIPT en tête de sys.path -- ici `utils/` -- et non la racine du
# projet. Les imports `config` et `logger`, qui vivent à la racine, échouent
# alors avec ModuleNotFoundError. Les scripts de core/ font déjà cet ajout ;
# ceux d'utils/ l'avaient oublié, ce qui cassait les alertes Telegram en CI.
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import requests
import os
import time
from dotenv import load_dotenv
from logger import get_logger
from config import PROCESSED_FILES, TELEGRAM_TOKEN, TELEGRAM_CHAT_ID

load_dotenv()
logger = get_logger(__name__)


FICHIER_ALERTES = PROCESSED_FILES["deals"]
FICHIER_DEJA_ENVOYEES = PROCESSED_FILES["sent_alerts"]


def charger_liens_envoyes():
    """Charge les liens déjà envoyés pour éviter les doublons"""
    if not os.path.exists(FICHIER_DEJA_ENVOYEES):
        return set()
    try:
        with open(FICHIER_DEJA_ENVOYEES, "r", encoding="utf-8") as f:
            liens = set(line.strip() for line in f if line.strip())
            logger.info(f"✅ {len(liens)} liens déjà envoyés chargés")
            return liens
    except Exception as e:
        logger.error(f"❌ Erreur lors du chargement des liens: {e}")
        return set()


def sauvegarder_lien_envoye(lien):
    """Enregistre un lien comme envoyé"""
    try:
        os.makedirs(os.path.dirname(FICHIER_DEJA_ENVOYEES), exist_ok=True)
        with open(FICHIER_DEJA_ENVOYEES, "a", encoding="utf-8") as f:
            f.write(f"{lien}\n")
    except Exception as e:
        logger.error(f"❌ Erreur lors de la sauvegarde du lien: {e}")


def envoyer_message_telegram(texte):
    """Envoie un message sur Telegram"""
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        logger.warning("⚠️ TELEGRAM_TOKEN / TELEGRAM_CHAT_ID manquants (variables d'environnement)")
        return False
    
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": texte, "parse_mode": "Markdown"}
    
    try:
        response = requests.post(url, json=payload, timeout=10)
        if response.status_code == 200:
            logger.info("✅ Message Telegram envoyé")
            return True
        else:
            logger.error(f"❌ Erreur Telegram: {response.status_code}")
            return False
    except Exception as e:
        logger.error(f"❌ Erreur réseau Telegram: {e}")
        return False


def envoyer_alertes_telegram():
    """Envoie les alertes de bonnes affaires sur Telegram"""
    
    if not os.path.exists(FICHIER_ALERTES):
        logger.warning(f"⚠️ Fichier introuvable: {FICHIER_ALERTES}")
        return

    try:
        df_alertes = pd.read_csv(FICHIER_ALERTES, sep=";", encoding="utf-8-sig")
    except Exception as e:
        logger.error(f"❌ Erreur lors du chargement des alertes: {e}")
        return

    # N'alerter que sur les opportunités appuyées sur assez d'annonces
    # comparables : une "affaire" calculée à partir de 2 annonces d'un modèle
    # rare est presque toujours une erreur d'estimation. La colonne est posée
    # par core/detect_deals.py ; si elle est absente (ancien fichier), on
    # envoie tout, comme avant.
    if "Alerte_Telegram" in df_alertes.columns:
        total = len(df_alertes)
        df_alertes = df_alertes[df_alertes["Alerte_Telegram"] == True]  # noqa: E712
        ignorees = total - len(df_alertes)
        if ignorees:
            logger.info(f"🔇 {ignorees} opportunité(s) ignorée(s) : trop peu d'annonces comparables pour être fiables.")

    liens_deja_envoyes = charger_liens_envoyes()
    compteur_envois = 0

    logger.info(f"📬 Analyse de {len(df_alertes)} opportunités filtrées...")

    for _, row in df_alertes.iterrows():
        lien_annonce = str(row.get("Lien", "")).strip()

        if not lien_annonce or lien_annonce == "nan" or lien_annonce in liens_deja_envoyes:
            continue

        # Prix et gain
        prix_theorique = row.get('Prix_Theorique', 0)
        prix_reel = row.get('Prix', 0)
        
        try:
            prix_theorique = float(prix_theorique) if prix_theorique else 0
            prix_reel = float(prix_reel) if prix_reel else 0
        except (ValueError, TypeError):
            logger.warning(f"⚠️ Prix invalide: {prix_theorique} / {prix_reel}")
            continue
        
        gain = int(prix_theorique - prix_reel)

        # Construction du message
        message = (
            f"🔥 *BONNE AFFAIRE AUTO DÉTECTÉE* 🔥\n\n"
            f"🚘 *Véhicule :* {row.get('Titre', 'N/A')}\n"
            f"🏷️ *Marque :* {row.get('Marque', 'N/A')}\n"
            f"💰 *Prix :* {prix_reel:,} DT *(Sous l'argus de ~{gain:,} DT !)*\n"
            f"📅 *Année :* {row.get('Année', 'N/A')} | 🛣️ *KM :* {row.get('Kilométrage', '0'):,}\n"
            f"⚙️ *Énergie :* {row.get('Energie', 'N/A')} | 🕹️ *Boîte :* {row.get('Boite_Vitesse', 'N/A')}\n"
            f"🐎 *Puissance Fiscale :* {row.get('Puissance_Fiscale', 'N/A')} CV\n"
            f"📍 *Région :* {row.get('Localisation', 'N/A')}\n\n"
            f"🔗 [Ouvrir l'Annonce Directement]({lien_annonce})"
        )

        if envoyer_message_telegram(message):
            sauvegarder_lien_envoye(lien_annonce)
            compteur_envois += 1
            time.sleep(1)  # Éviter les trop de requêtes rapides

    logger.info(f"✅ {compteur_envois} alerte(s) envoyée(s)")


if __name__ == "__main__":
    envoyer_alertes_telegram()
