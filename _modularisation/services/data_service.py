"""Accès aux données : chemins, lecture CSV/JSON (locale ou GitHub), chargements cachés."""
import json
import pandas as pd
import numpy as np
from services.analytics_service import MARQUES_LUXE, GRAND_TUNIS

SCORED_PATH = "data/processed/tunisia-cars-scored.csv"
DEALS_PATH = "data/processed/alertes_bonnes_affaires.csv"
MODELE_PATH = "data/models/modele_prix.pkl"
DEPOT_GITHUB = "iheb-bibani/AutoDeal-Tunisie"
BRANCHE_GITHUB = "main"
BASE_RAW = f"https://raw.githubusercontent.com/{DEPOT_GITHUB}/{BRANCHE_GITHUB}/"
LIRE_DEPUIS_GITHUB = True
DUREE_CACHE = 3600  # secondes
DIAG_PATH = "data/processed/diagnostics_modele.json"
CALIB_PATH = "data/processed/calibration_fenetre.json"
SHAP_PATH = "data/processed/shap_importance.json"
HISTO_PATH = "data/processed/historique_performance.json"
def _url(chemin_relatif):
    return BASE_RAW + chemin_relatif
def lire_csv(chemin):
    """Lit un CSV depuis GitHub si possible, sinon depuis le disque local."""
    if LIRE_DEPUIS_GITHUB:
        try:
            return pd.read_csv(_url(chemin), sep=";", encoding="utf-8-sig")
        except Exception:
            pass
    return pd.read_csv(chemin, sep=";", encoding="utf-8-sig")
def lire_json_distant(chemin):
    if LIRE_DEPUIS_GITHUB:
        try:
            import urllib.request
            with urllib.request.urlopen(_url(chemin), timeout=15) as r:
                return json.loads(r.read().decode("utf-8"))
        except Exception:
            pass
    try:
        with open(chemin, encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None
def charger_scored():
    try:
        df = lire_csv(SCORED_PATH)
    except Exception:
        return None
    # Colonnes lisibles si absentes (compatibilité avec d'anciens fichiers)
    if "Segment_Libelle" not in df.columns:
        df["Segment_Libelle"] = np.where(df["Marque"].isin(MARQUES_LUXE), "Luxe", "Standard")
    if "Zone_Libelle" not in df.columns:
        df["Zone_Libelle"] = np.where(df["Localisation"].isin(GRAND_TUNIS), "Grand Tunis", "Province")
    if "Age_Vehicule" not in df.columns and "Année" in df.columns:
        df["Age_Vehicule"] = (pd.Timestamp.now().year - df["Année"]).clip(lower=0)
    return df
def charger_deals():
    try:
        df = lire_csv(DEALS_PATH)
        return df if not df.empty else None
    except Exception:
        return None
def charger_json(chemin):
    return lire_json_distant(chemin)
