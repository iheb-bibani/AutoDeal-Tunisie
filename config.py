"""
config.py
Configuration centralisée du projet AutoDeal Tunisie
Élimine la duplication et améliore la maintenabilité
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Charge les variables d'environnement
load_dotenv()

# ============================================================================
# CHEMINS
# ============================================================================

PROJECT_ROOT = Path(__file__).parent.absolute()
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = DATA_DIR / "models"

# Créer les répertoires s'ils n'existent pas
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# FICHIERS DE DONNÉES
# ============================================================================

# Scrapers outputs
SCRAPERS = {
    "tayara": str(RAW_DATA_DIR / "tayara.csv"),
    "automax": str(RAW_DATA_DIR / "automax.csv"),
    "automobile": str(RAW_DATA_DIR / "automobile.csv"),
}

# Pipeline processing
PROCESSED_FILES = {
    "merged": str(PROCESSED_DATA_DIR / "tunisia-cars.csv"),
    "recent": str(PROCESSED_DATA_DIR / "tunisia-cars-recent.csv"),
    "enriched": str(PROCESSED_DATA_DIR / "tunisia-cars-final-features.csv"),
    "scored": str(PROCESSED_DATA_DIR / "tunisia-cars-scored.csv"),
    "deals": str(PROCESSED_DATA_DIR / "alertes_bonnes_affaires.csv"),
    "sent_alerts": str(PROCESSED_DATA_DIR / "alertes_envoyees.txt"),
}

# Models
MODEL_PATH = str(MODELS_DIR / "modele_prix.pkl")

# ============================================================================
# PARAMÈTRES DE TRAITEMENT
# ============================================================================

# Limites de jours (nettoyer_base.py)
MAX_DAYS_OLD = 60  # Fenêtre de fraîcheur. Testée empiriquement de 30 à 365 jours :
                   # 30 j est nettement le pire (10,83 % d'erreur), 60 j gagne ~0,5 point
                   # en apportant 560 annonces de plus. Au-delà, plus aucun gain mesurable
                   # (toutes les fenêtres de 60 à 365 j se valent, à l'intérieur du bruit).

# Seuils de marques/modèles rares
SEUIL_MARQUE_RARE = 10   # Regrouper marques avec < N annonces
SEUIL_MODELE_RARE = 5    # Regrouper modèles avec < N annonces

# Seuils de deals (detect_deals.py)
SEUIL_DEAL_MIN = 0.15    # Au moins 15% sous prix théorique
SEUIL_DEAL_MAX = 0.55    # Au-delà de 55%, c'est une erreur

# Lissage bayésien (pour les graphiques région/énergie)
K_LISSAGE = 15

# ============================================================================
# MARQUES ET ZONES
# ============================================================================

MARQUES_LUXE = [
    "Porsche", "Land Rover", "Mercedes-Benz", "Jaguar",
    "Audi", "Bmw", "Tesla"
]

GRAND_TUNIS = [
    "Tunis", "Ariana", "Ben Arous", "Manouba"
]

# Corrections de marques (normalisations)
CORRECTIONS_MARQUE = {
    "citroen": "Citroën",
    "skoda": "Škoda",
    "mercedes": "Mercedes-Benz",
    "great-wall": "Gwm",
    "great wall": "Gwm",
    "land-rover": "Land Rover",
    "alfa-romeo": "Alfa Romeo",
    "alfa romeo": "Alfa Romeo",
    "autres": None,   # "Autres" chez tayara = champ non rempli, pas une marque
    "autre": None,
}

# ============================================================================
# RÉFÉRENTIEL MARQUES / MODÈLES (inférence depuis les titres)
# ============================================================================

# Mot-clé (insensible à la casse, mot entier) -> marque canonique.
# Couvre les marques présentes en Tunisie qui n'apparaissent pas toujours
# dans le champ structuré des annonces (tayara surtout).
MARQUES_ALIAS = {
    "vw": "Volkswagen", "volkswagen": "Volkswagen", "volswagen": "Volkswagen",
    "mercedes": "Mercedes-Benz", "mercédès": "Mercedes-Benz", "benz": "Mercedes-Benz",
    "bmw": "Bmw", "audi": "Audi", "peugeot": "Peugeot", "pigeot": "Peugeot",
    "renault": "Renault", "citroen": "Citroën", "citroën": "Citroën",
    "fiat": "Fiat", "ford": "Ford", "kia": "Kia", "hyundai": "Hyundai",
    "toyota": "Toyota", "seat": "Seat", "skoda": "Škoda", "škoda": "Škoda",
    "suzuki": "Suzuki", "nissan": "Nissan", "mazda": "Mazda", "honda": "Honda",
    "opel": "Opel", "dacia": "Dacia", "chevrolet": "Chevrolet", "jeep": "Jeep",
    "mitsubishi": "Mitsubishi", "porsche": "Porsche", "mahindra": "Mahindra",
    "chery": "Chery", "geely": "Geely", "ssangyong": "Ssangyong", "mg": "Mg",
    "cupra": "Cupra", "haval": "Haval", "dongfeng": "Dongfeng", "dfsk": "Dfsk",
    "byd": "Byd", "lada": "Lada", "wallys": "Wallys", "wallyscar": "Wallys",
    "isuzu": "Isuzu", "volvo": "Volvo", "mini": "Mini", "jaguar": "Jaguar",
    "lexus": "Lexus", "subaru": "Subaru", "daihatsu": "Daihatsu", "tata": "Tata",
    "foton": "Foton", "jac": "Jac", "baic": "Baic", "chana": "Chana",
    "gwm": "Gwm", "landrover": "Land Rover", "tesla": "Tesla", "alfa": "Alfa Romeo",
}

# Modèle -> (marque, modèle canonique) : quand le titre ne contient QUE le
# nom du modèle ("Golf 7 toutes options", "vente clio 4"), le modèle suffit
# à identifier la marque sans ambiguïté sur le marché tunisien.
MODELE_IMPLIQUE_MARQUE = {
    "golf": ("Volkswagen", "Golf"), "polo": ("Volkswagen", "Polo"),
    "passat": ("Volkswagen", "Passat"), "caddy": ("Volkswagen", "Caddy"),
    "tiguan": ("Volkswagen", "Tiguan"), "touareg": ("Volkswagen", "Touareg"),
    "jetta": ("Volkswagen", "Jetta"), "coccinelle": ("Volkswagen", "Coccinelle"),
    "clio": ("Renault", "Clio"), "megane": ("Renault", "Megane"),
    "megan": ("Renault", "Megane"), "mégane": ("Renault", "Megane"),
    "symbol": ("Renault", "Symbol"), "kangoo": ("Renault", "Kangoo"),
    "twingo": ("Renault", "Twingo"), "captur": ("Renault", "Captur"),
    "kwid": ("Renault", "Kwid"), "fluence": ("Renault", "Fluence"),
    "focus": ("Ford", "Focus"), "fiesta": ("Ford", "Fiesta"),
    "kuga": ("Ford", "Kuga"), "ranger": ("Ford", "Ranger"),
    "punto": ("Fiat", "Punto"), "panda": ("Fiat", "Panda"),
    "tipo": ("Fiat", "Tipo"), "fiorino": ("Fiat", "Fiorino"),
    "doblo": ("Fiat", "Doblo"), "uno": ("Fiat", "Uno"),
    "berlingo": ("Citroën", "Berlingo"), "b9": ("Citroën", "Berlingo"),
    "jumpy": ("Citroën", "Jumpy"), "jumper": ("Citroën", "Jumper"),
    "partner": ("Peugeot", "Partner"), "kamsa": ("Peugeot", "404"),
    "ibiza": ("Seat", "Ibiza"), "leon": ("Seat", "Leon"),
    "ateca": ("Seat", "Ateca"), "arona": ("Seat", "Arona"),
    "picanto": ("Kia", "Picanto"), "sportage": ("Kia", "Sportage"),
    "sorento": ("Kia", "Sorento"), "cerato": ("Kia", "Cerato"),
    "seltos": ("Kia", "Seltos"), "sonet": ("Kia", "Sonet"),
    "i10": ("Hyundai", "Grand i10"), "i20": ("Hyundai", "i20"),
    "i30": ("Hyundai", "i30"), "tucson": ("Hyundai", "Tucson"),
    "accent": ("Hyundai", "Accent"), "creta": ("Hyundai", "Creta"),
    "yaris": ("Toyota", "Yaris"), "corolla": ("Toyota", "Corolla"),
    "hilux": ("Toyota", "Hilux"), "agya": ("Toyota", "Agya"),
    "qashqai": ("Nissan", "Qashqai"), "juke": ("Nissan", "Juke"),
    "micra": ("Nissan", "Micra"), "navara": ("Nissan", "Navara"),
    "duster": ("Dacia", "Duster"), "sandero": ("Dacia", "Sandero"),
    "logan": ("Dacia", "Logan"), "dokker": ("Dacia", "Dokker"),
    "corsa": ("Opel", "Corsa"), "astra": ("Opel", "Astra"),
    "insignia": ("Opel", "Insignia"), "grandland": ("Opel", "Grandland"),
    "swift": ("Suzuki", "Swift"), "celerio": ("Suzuki", "Celerio"),
    "jimny": ("Suzuki", "Jimny"), "vitara": ("Suzuki", "Vitara"),
    "fabia": ("Škoda", "Fabia"), "octavia": ("Škoda", "Octavia"),
    "kamiq": ("Škoda", "Kamiq"), "kodiaq": ("Škoda", "Kodiaq"),
    "tiggo": ("Chery", "Tiggo"), "jolion": ("Haval", "Jolion"),
    "formentor": ("Cupra", "Formentor"), "tivoli": ("Ssangyong", "Tivoli"),
    "attrage": ("Mitsubishi", "Attrage"), "mirage": ("Mitsubishi", "Mirage"),
    "pajero": ("Mitsubishi", "Pajero"), "l200": ("Mitsubishi", "L200"),
    "kuv100": ("Mahindra", "KUV100"), "xuv300": ("Mahindra", "XUV300"),
    "sx3": ("Dongfeng", "SX3"), "niva": ("Lada", "Niva"),
    "santafe": ("Hyundai", "Santa Fe"),
}

# Modèles usuels par marque (complète les modèles déjà présents dans la base
# pour la récupération "niveau 1" -- utile quand un modèle courant en Tunisie
# n'a encore aucune annonce correctement étiquetée dans la base).
MODELES_PAR_MARQUE_REFERENTIEL = {
    "Peugeot": ["104", "106", "107", "108", "204", "205", "206", "207", "208",
                "301", "304", "305", "306", "307", "308", "404", "405", "406",
                "407", "508", "2008", "3008", "5008", "Partner", "Expert",
                "Rifter", "Landtrek", "RCZ"],
    "Renault": ["Clio", "Megane", "Symbol", "Kangoo", "Twingo", "Captur",
                "Kwid", "Fluence", "Laguna", "Scenic", "Talisman", "Kadjar",
                "Koleos", "Express", "4L", "R4", "R5", "R9", "R12", "Trafic",
                "Master"],
    "Volkswagen": ["Golf", "Polo", "Passat", "Caddy", "Tiguan", "Touareg",
                   "Jetta", "Touran", "Amarok", "T-Roc", "T-Cross", "Up",
                   "Coccinelle", "Transporter"],
    "Citroën": ["C1", "C2", "C3", "C4", "C5", "C15", "C-Elysée", "Berlingo",
                "Jumpy", "Jumper", "Xsara", "Saxo", "DS3", "DS4"],
    "Fiat": ["Punto", "Panda", "Tipo", "Fiorino", "Doblo", "Uno", "500",
             "500X", "Palio", "Linea", "Bravo", "Fullback"],
    "Ford": ["Focus", "Fiesta", "Kuga", "Ranger", "EcoSport", "Mondeo",
             "C-Max", "Transit", "Escort", "Puma"],
    "Toyota": ["Yaris", "Corolla", "Hilux", "Agya", "RAV4", "Land Cruiser",
               "Prado", "Avanza", "C-HR", "Auris"],
    "Kia": ["Picanto", "Sportage", "Sorento", "Cerato", "Rio", "Seltos",
            "Sonet", "Carens", "K2700", "Soul", "Stonic", "Pegas"],
    "Hyundai": ["Grand i10", "i10", "i20", "i30", "Tucson", "Accent", "Creta",
                "Santa Fe", "Atos", "Getz", "Elantra", "Venue", "H1", "Kona"],
    "Mercedes-Benz": ["Classe A", "Classe B", "Classe C", "Classe E",
                      "Classe S", "CLA", "CLS", "GLA", "GLB", "GLC", "GLE",
                      "GLK", "ML", "Vito", "Sprinter", "190", "Classe G"],
    "Bmw": ["Série 1", "Série 2", "Série 3", "Série 4", "Série 5", "Série 7",
            "X1", "X2", "X3", "X4", "X5", "X6", "i3"],
    "Audi": ["A1", "A3", "A4", "A5", "A6", "A7", "A8", "Q2", "Q3", "Q5",
             "Q7", "Q8", "TT", "e-tron"],
    "Seat": ["Ibiza", "Leon", "Ateca", "Arona", "Toledo", "Cordoba", "Altea"],
    "Škoda": ["Fabia", "Octavia", "Kamiq", "Kodiaq", "Superb", "Rapid",
              "Scala", "Roomster"],
    "Nissan": ["Qashqai", "Juke", "Micra", "Navara", "X-Trail", "Sunny",
               "Pathfinder", "Patrol"],
    "Dacia": ["Duster", "Sandero", "Logan", "Dokker", "Lodgy"],
    "Opel": ["Corsa", "Astra", "Insignia", "Grandland", "Crossland",
             "Mokka", "Vectra", "Combo"],
    "Suzuki": ["Swift", "Celerio", "Jimny", "Vitara", "Baleno", "Alto",
               "Ertiga", "Dzire", "S-Presso"],
    "Chery": ["Tiggo", "Tiggo 2", "Tiggo 3", "Tiggo 4", "Tiggo 7", "Tiggo 8",
              "Arrizo", "E3", "QQ"],
    "Mitsubishi": ["Attrage", "Mirage", "Pajero", "L200", "Outlander", "ASX"],
    "Mahindra": ["KUV100", "XUV300", "Scorpio", "Bolero", "Pik Up"],
    "Haval": ["Jolion", "H6", "H2"],
    "Gwm": ["M4", "Wingle", "Poer", "Voleex"],
    "Cupra": ["Formentor", "Leon", "Ateca"],
    "Dongfeng": ["SX3", "Glory", "Rich"],
    "Lada": ["Niva", "4x4", "Granta"],
    "Wallys": ["Iris", "619", "Annibal"],
    "Ssangyong": ["Tivoli", "Korando", "Rexton", "Musso"],
    "Geely": ["GX3", "Emgrand", "Coolray", "Azkarra"],
    "Mg": ["ZS", "MG3", "MG5", "HS", "RX5"],
    "Land Rover": ["Range Rover", "Evoque", "Discovery", "Defender",
                   "Freelander", "Velar"],
}

# ============================================================================
# MODÈLE MACHINE LEARNING
# ============================================================================

FEATURES_NUMERIQUES = [
    "Kilométrage", "Année", "Age_Vehicule", "Puissance_Fiscale",
    "Segment_Vehicule", "Est_Presque_Neuve", "Zone_Economique"
]

FEATURES_CATEGORIELLES = [
    "Marque", "Modèle", "Boite_Vitesse", "Energie"
]

# ============================================================================
# TELEGRAM
# ============================================================================

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

# ============================================================================
# SCRAPING
# ============================================================================

SCRAPING_CONFIG = {
    "tayara": {
        "max_pages": 303,
        "base_url": "https://www.tayara.tn",
    },
    "automax": {
        "max_pages": None,  # Pas de limite (pagination AJAX)
        "base_url": "https://www.automax.tn",
    },
    "automobile": {
        "max_pages": 160,
        "base_url": "https://www.automobile.tn",
    },
}

HEADERS_DEFAULT = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

REQUEST_TIMEOUT = 30  # secondes
REQUEST_RETRY_COUNT = 3
REQUEST_RETRY_DELAY = 2  # secondes

SCRAPING_DELAY_MIN = 0.5  # secondes
SCRAPING_DELAY_MAX = 2.0  # secondes

# ============================================================================
# COLONNES STANDARD DES CSVS
# ============================================================================

COLONNES_FINALES = [
    "Source", "Titre", "Marque", "Modèle", "Année", "Prix_DT",
    "Kilométrage", "Energie", "Boite", "Localisation",
    "Puissance_Fiscale", "Etat_Vehicule", "Annonce-Deposee",
    "Annonce-Detectee", "Statut", "Lien",
]

# ============================================================================
# LOGGING
# ============================================================================

LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "format": "[%(asctime)s] %(levelname)s - %(name)s - %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        },
        "detailed": {
            "format": "[%(asctime)s] %(levelname)s - %(name)s:%(lineno)d - %(funcName)s() - %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": "INFO",
            "formatter": "default",
            "stream": "ext://sys.stdout",
        },
        "file": {
            "class": "logging.FileHandler",
            "level": "DEBUG",
            "formatter": "detailed",
            "filename": str(PROJECT_ROOT / "logs" / "autodeal.log"),
        },
    },
    "loggers": {
        "": {
            "level": "DEBUG",
            "handlers": ["console", "file"],
        },
    },
}

# Créer le répertoire logs
LOGS_DIR = PROJECT_ROOT / "logs"
LOGS_DIR.mkdir(exist_ok=True)


# ============================================================================
# VALIDATION
# ============================================================================

def valider_config():
    """Vérifie que tous les répertoires nécessaires existent"""
    for path in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, LOGS_DIR]:
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)


valider_config()
