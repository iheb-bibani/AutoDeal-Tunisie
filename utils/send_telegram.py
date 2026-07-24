# isort: skip_file
"""Envoi des alertes Telegram pour les nouvelles opportunités détectées.

Lancé par `python utils/send_telegram.py`, Python place `utils/` en tête du
chemin de recherche, jamais la racine du projet : les imports `config` et
`logger` échouent sans le bootstrap ci-dessous. Ce bootstrap DOIT rester
au-dessus de tous les imports du projet — le marqueur `# isort: skip_file`
empêche un formatage automatique de le contourner en remontant les imports.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import os  # noqa: E402
import time  # noqa: E402

import pandas as pd  # noqa: E402
import requests  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

from config import PROCESSED_FILES, TELEGRAM_CHAT_ID, TELEGRAM_TOKEN  # noqa: E402
from logger import get_logger  # noqa: E402

load_dotenv()
logger = get_logger(__name__)


FICHIER_ALERTES = PROCESSED_FILES["deals"]
FICHIER_DEJA_ENVOYEES = PROCESSED_FILES["sent_alerts"]


def _entier(valeur, defaut=0):
    """Convertit une valeur de cellule en entier pour l'affichage.

    Tolère les vides, les NaN et les valeurs textuelles : un formatage
    `f"{valeur:,}"` appliqué directement à une chaîne lève une ValueError et
    interrompt tout l'envoi.
    """
    try:
        nombre = float(valeur)
    except (TypeError, ValueError):
        return defaut
    if pd.isna(nombre):
        return defaut
    return int(nombre)


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
        logger.warning(
            "⚠️ TELEGRAM_TOKEN / TELEGRAM_CHAT_ID manquants (variables d'environnement)"
        )
        return False

    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": texte, "parse_mode": "Markdown"}

    try:
        response = requests.post(url, json=payload, timeout=10)
        if response.status_code == 200:
            logger.info("✅ Message Telegram envoyé")
            return True
        # Le corps de la réponse contient la raison exacte du refus
        # (parse_mode invalide, chat introuvable, message trop long...).
        logger.error(f"❌ Erreur Telegram {response.status_code}: {response.text[:300]}")
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
    #
    # La comparaison passe par le texte : après un aller-retour CSV, la colonne
    # peut revenir en chaînes "True"/"False", et `== True` ne filtrerait alors
    # plus rien du tout, sans erreur visible.
    if "Alerte_Telegram" in df_alertes.columns:
        total = len(df_alertes)
        masque = (
            df_alertes["Alerte_Telegram"]
            .astype(str)
            .str.strip()
            .str.lower()
            .isin({"true", "1", "vrai"})
        )
        df_alertes = df_alertes[masque]
        ignorees = total - len(df_alertes)
        if ignorees:
            logger.info(
                f"🔇 {ignorees} opportunité(s) ignorée(s) : "
                "trop peu d'annonces comparables pour être fiables."
            )

    liens_deja_envoyes = charger_liens_envoyes()
    compteur_envois = 0

    logger.info(f"📬 Analyse de {len(df_alertes)} opportunités filtrées...")

    for _, row in df_alertes.iterrows():
        lien_annonce = str(row.get("Lien", "")).strip()

        if (
            not lien_annonce
            or lien_annonce == "nan"
            or lien_annonce in liens_deja_envoyes
        ):
            continue

        prix_theorique = _entier(row.get("Prix_Theorique"))
        prix_reel = _entier(row.get("Prix"))

        if prix_theorique <= 0 or prix_reel <= 0:
            logger.warning(
                f"⚠️ Prix invalide, annonce ignorée : "
                f"{row.get('Prix_Theorique')} / {row.get('Prix')} — {lien_annonce}"
            )
            continue

        gain = prix_theorique - prix_reel
        kilometrage = _entier(row.get("Kilométrage"))

        message = (
            f"🔥 *BONNE AFFAIRE AUTO DÉTECTÉE* 🔥\n\n"
            f"🚘 *Véhicule :* {row.get('Titre', 'N/A')}\n"
            f"🏷️ *Marque :* {row.get('Marque', 'N/A')}\n"
            f"💰 *Prix :* {prix_reel:,} DT *(Sous l'argus de ~{gain:,} DT !)*\n"
            f"📅 *Année :* {row.get('Année', 'N/A')} | 🛣️ *KM :* {kilometrage:,}\n"
            f"⚙️ *Énergie :* {row.get('Energie', 'N/A')} | "
            f"🕹️ *Boîte :* {row.get('Boite_Vitesse', 'N/A')}\n"
            f"🐎 *Puissance Fiscale :* {row.get('Puissance_Fiscale', 'N/A')} CV\n"
            f"📍 *Région :* {row.get('Localisation', 'N/A')}\n\n"
            f"🔗 [Ouvrir l'Annonce Directement]({lien_annonce})"
        )

        if envoyer_message_telegram(message):
            sauvegarder_lien_envoye(lien_annonce)
            compteur_envois += 1
            time.sleep(1)  # Éviter trop de requêtes rapprochées

    logger.info(f"✅ {compteur_envois} alerte(s) envoyée(s)")


if __name__ == "__main__":
    envoyer_alertes_telegram()
