"""
app.py
AutoDeal Tunisie -- application d'intelligence du marché de l'occasion.

Navigation par RÔLE :

  CONCESSIONNAIRE (comprendre & valoriser le parc)
    🏢 Concessionnaire : structure du marché, dépréciation par segment,
       stratégie d'achat, prix par région et par énergie.
    💰 Calculateur     : estimation d'une voiture précise par le modèle ML,
       fourchette de confiance, décomposition SHAP, comparables réels.

  SAMSAR (achat-revente)
    🤝 Samsar          : opportunités chiffrées en dinars, rotation, fenêtre
       d'achat (liquidité x décote), arbitrage géographique, SHAP par annonce.

  EXPLORER (outils transverses)
    🔎 Recherche       : filtres instantanés sur tout le marché scoré.
    🗺️ Carte           : marché par gouvernorat (prix médian / densité d'affaires).
    🤖 Assistant       : questions en langage naturel, adossées au modèle.

  ADMIN (technique)
    🛠️ Admin           : sélection du modèle, fiabilité, importance globale des
       variables (SHAP), validation des opportunités, santé du scraping.

Comptes, favoris et alertes personnalisées peuvent être activés via Supabase.
Le mode public de consultation reste disponible sans compte.
"""

import json
import re
import unicodedata
import numpy as np
import joblib
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from config import MAX_DAYS_OLD
from core.market_valuation import market_valuation, compare_ml_to_market, evaluate_market_ml_coverage
from core.product_insights import valuation_confidence
from core.access_control import access_badge, can_open_page, visible_pro_pages
from services.supabase_service import current_access_context, is_configured as supabase_is_configured

from product_views import (
    page_accueil, page_acheter, page_detail, page_comparateur,
    page_historique, page_sante_marche, page_alertes, page_compte, page_tarifs,
)

st.set_page_config(
    page_title="AutoDeal Tunisie",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

SCORED_PATH = "data/processed/tunisia-cars-scored.csv"
DEALS_PATH = "data/processed/alertes_bonnes_affaires.csv"
MODELE_PATH = "data/models/modele_prix.pkl"

# ---------------------------------------------------------------------------
# Source des données
# ---------------------------------------------------------------------------
# Le rafraîchissement automatique de Streamlit Cloud après un push GitHub n'est
# pas fiable : l'app conserve fréquemment l'ancien contenu du dépôt jusqu'à un
# redémarrage manuel. Comme les données sont réécrites chaque nuit par GitHub
# Actions, s'appuyer dessus reviendrait à afficher des annonces périmées sans
# aucun signal visible.
#
# On lit donc les fichiers directement depuis GitHub (raw), avec un cache d'une
# heure : l'app se met à jour d'elle-même, redémarrage ou pas. En cas d'échec
# réseau, on retombe silencieusement sur les fichiers locaux -- ce qui est aussi
# le mode utilisé en développement.
DEPOT_GITHUB = "iheb-bibani/AutoDeal-Tunisie"
BRANCHE_GITHUB = "main"
BASE_RAW = f"https://raw.githubusercontent.com/{DEPOT_GITHUB}/{BRANCHE_GITHUB}/"

# Mettre à False pour forcer la lecture locale (développement hors ligne).
LIRE_DEPUIS_GITHUB = True
DUREE_CACHE = 3600  # secondes


def _url(chemin_relatif):
    return BASE_RAW + chemin_relatif


@st.cache_data(ttl=DUREE_CACHE, show_spinner=False)
def lire_csv(chemin):
    """Lit un CSV depuis GitHub si possible, sinon depuis le disque local."""
    if LIRE_DEPUIS_GITHUB:
        try:
            return pd.read_csv(_url(chemin), sep=";", encoding="utf-8-sig")
        except Exception:
            pass
    return pd.read_csv(chemin, sep=";", encoding="utf-8-sig")


@st.cache_data(ttl=DUREE_CACHE, show_spinner=False)
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

MARQUES_LUXE = ["Porsche", "Land Rover", "Mercedes-Benz", "Jaguar", "Audi", "Bmw", "Tesla"]
GRAND_TUNIS = ["Tunis", "Ariana", "Ben Arous", "Manouba"]
K_LISSAGE = 15

# ---------------------------------------------------------------------------
# Identité visuelle : palette "asphalte & dinar"
# ---------------------------------------------------------------------------

C_ENCRE = "#15232E"      # texte, barres neutres
C_ASPHALTE = "#2C3E50"   # barres principales
C_GAIN = "#0E9F6E"       # vert "gain" -- opportunités, hausses
C_ALERTE = "#D9480F"     # orange brûlé -- points d'attention
C_SABLE = "#C9A227"      # accent secondaire (or/sable)
C_GRIS = "#8A97A3"

SEQ_CATEGORIELLE = [C_ASPHALTE, C_GAIN, C_SABLE, C_ALERTE, "#5C7A99", "#A3B2BF"]

st.markdown(
    """
    <style>
    /* Cartes KPI */
    div[data-testid="stMetric"] {
        background: linear-gradient(180deg, #FFFFFF 0%, #F7F9FA 100%);
        border: 1px solid #E4E9ED;
        border-left: 4px solid #0E9F6E;
        border-radius: 10px;
        padding: 14px 18px 10px 18px;
        box-shadow: 0 1px 3px rgba(21, 35, 46, .06);
    }
    div[data-testid="stMetric"] label { color: #5C6B78 !important; }

    /* Barre latérale sombre */
    section[data-testid="stSidebar"] {
        background: #15232E;
    }
    section[data-testid="stSidebar"] * { color: #E8EDF1 !important; }
    section[data-testid="stSidebar"] hr { border-color: #2C3E50; }

    /* Titres */
    h1, h2, h3 { color: #15232E; }
    h1 { letter-spacing: -0.5px; }

    /* Onglets */
    button[data-baseweb="tab"] { font-weight: 600; }

    /* Liens dans les tableaux */
    a { color: #0E9F6E; }

    /* Masquer la barre d'outils + logo Plotly (le petit bloc au coin des graphes) */
    .modebar { display: none !important; }
    .js-plotly-plot .plotly .modebar-container { display: none !important; }

    /* Boutons de navigation (sidebar) : texte foncé sur fond blanc (inactifs) */
    section[data-testid="stSidebar"] .stButton button,
    section[data-testid="stSidebar"] .stButton button p {
        color: #15232E !important;
    }
    /* Bouton actif (primary, vert) : texte blanc */
    section[data-testid="stSidebar"] .stButton button[kind="primary"],
    section[data-testid="stSidebar"] .stButton button[kind="primary"] p {
        color: #FFFFFF !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def style_figure(fig, hauteur=380):
    """Applique le gabarit visuel commun à toutes les figures Plotly."""
    fig.update_layout(
        height=hauteur,
        margin=dict(l=10, r=10, t=48, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Source Sans Pro, sans-serif", color=C_ENCRE, size=13),
        title_font=dict(size=15, color=C_ENCRE),
        colorway=SEQ_CATEGORIELLE,
        hoverlabel=dict(bgcolor="#FFFFFF", font_color=C_ENCRE, bordercolor="#E4E9ED"),
    )
    fig.update_xaxes(gridcolor="#EDF1F4", zerolinecolor="#E4E9ED")
    fig.update_yaxes(gridcolor="#EDF1F4", zerolinecolor="#E4E9ED")
    return fig


# ---------------------------------------------------------------------------
# Fiabilité par tranche de prix + interprétation SHAP (partagés entre pages)
# ---------------------------------------------------------------------------

BORNES_PRIX = [0, 15000, 25000, 35000, 50000, 75000, 100000, float("inf")]
LIBELLES_PRIX = ["< 15k", "15–25k", "25–35k", "35–50k", "50–75k", "75–100k", "100k +"]

# Libellés lisibles des features techniques (sections Interprétation / SHAP)
LABELS_FEATURES = {
    "Est_Presque_Neuve": "Millésime récent (≤1 an)", "Zone_Economique": "Zone (Grand Tunis)",
    "Nb_Options": "Nb d'options", "Cylindree": "Cylindrée",
    "Age_Vehicule": "Âge du véhicule", "Puissance_Fiscale": "Puissance fiscale (CV)",
    "Kilométrage": "Kilométrage", "Marque": "Marque", "Segment_Vehicule": "Segment (luxe)",
    "Transmission": "Transmission", "Puissance_DIN": "Puissance réelle (ch)",
    "Année": "Année", "Boite_Vitesse": "Boîte de vitesses", "Modèle": "Modèle",
    "Energie": "Énergie",
    "Premiere_Main": "1ère main", "Non_Accidentee": "Jamais accidentée",
    "Full_Options": "Toutes options", "Etat_Origine": "État d'origine",
}


# Noms courts des modèles (l'entraînement compare Ridge / RandomForest / HGB,
# mais le pipeline stocke le nom de classe scikit-learn). Affichage homogène.
NOMS_MODELE_COURT = {
    "HistGradientBoostingRegressor": "HGB",
    "RandomForestRegressor": "RandomForest",
    "Ridge": "Ridge",
}


def nom_modele_court(nom):
    return NOMS_MODELE_COURT.get(str(nom), str(nom))


def table_fiabilite_prix(df):
    """Erreur relative médiane du modèle (|prédit - réel| / réel) par tranche
    de prix. Sert à la courbe de fiabilité (Admin) ET à l'intervalle de
    confiance du calculateur. Renvoie un DataFrame vide si données insuffisantes."""
    if "Prix_Theorique" not in df.columns:
        return pd.DataFrame()
    d = df.dropna(subset=["Prix", "Prix_Theorique"]).copy()
    d = d[(d["Prix"] > 0) & (d["Prix_Theorique"] > 0)]
    if len(d) < 50:
        return pd.DataFrame()
    d["_err"] = (d["Prix_Theorique"] - d["Prix"]).abs() / d["Prix"]
    d["_tranche"] = pd.cut(d["Prix"], bins=BORNES_PRIX, labels=LIBELLES_PRIX)
    gr = (d.groupby("_tranche", observed=True)
          .agg(n=("_err", "size"), mdape=("_err", lambda x: 100 * x.median()))
          .reset_index())
    return gr[gr["n"] >= 10]


def erreur_pour_prix(gr, prix):
    """MdAPE (%) de la tranche contenant `prix`, ou l'erreur médiane globale."""
    if gr is None or len(gr) == 0:
        return None
    tranche = pd.cut([prix], bins=BORNES_PRIX, labels=LIBELLES_PRIX)[0]
    ligne = gr[gr["_tranche"] == tranche]
    if len(ligne):
        return float(ligne["mdape"].iloc[0])
    return float(gr["mdape"].median())


@st.cache_data(ttl=3600, show_spinner=False)
def diagnostic_market_ml(df):
    """Contrôle leave-one-out entre estimation ML et fourchette des comparables."""
    return evaluate_market_ml_coverage(df, min_n=5)


def explication(titre, corps):
    """Conteneur d'explication standard sous un graphe : montre → calcule → décide."""
    with st.expander(f"ℹ️ {titre}"):
        st.markdown(corps)


def analyse_rendement_capital(deals):
    """Par tranche de PRIX D'ACHAT : gain médian (DT), ROI médian (%), et
    'régularité' = ROI médian ÷ dispersion du ROI (un Sharpe-*like*).
    ⚠️ Ce n'est PAS un Sharpe financier : la dispersion mêle la variété réelle des
    affaires ET l'erreur du modèle (~10,7 %), et n'inclut pas le risque de revente
    (qui viendra du suivi des disparitions). À lire : 'consistance de l'écart estimé'."""
    d = deals.dropna(subset=["Prix", "Prix_Theorique"]).copy()
    d["gain"] = d["Prix_Theorique"] - d["Prix"]
    d["roi"] = d["gain"] / d["Prix"] * 100
    bornes = [0, 25000, 40000, 60000, 90000, float("inf")]
    libelles = ["< 25k", "25-40k", "40-60k", "60-90k", "90k +"]
    d["_tr"] = pd.cut(d["Prix"], bornes, labels=libelles)
    rows = []
    for tr, g in d.groupby("_tr", observed=True):
        if len(g) < 8:
            continue
        roi_med, roi_sd = g["roi"].median(), g["roi"].std()
        rows.append({
            "tranche": str(tr), "n": len(g), "gain": g["gain"].median(),
            "roi": roi_med, "regularite": roi_med / roi_sd if roi_sd else np.nan,
        })
    return pd.DataFrame(rows)


def analyse_decote_segment(df, seg_label):
    """Décote annuelle par phase (falaise 2→4 ans, plateau 4→7, longue traîne
    7→10) et âges les plus liquides, pour un segment. La falaise démarre à
    l'âge 2 (et non 1) car l'âge 1 est rare et dominé par des quasi-neuves
    haut de gamme : partir de là surestimerait la falaise (la 'prime du neuf'
    qui s'évapore n'est pas de la dépréciation). Basé sur des phases pour
    rester robuste au bruit. Renvoie None si données insuffisantes."""
    if "Age_Vehicule" not in df.columns or "Segment_Vehicule" not in df.columns:
        return None
    seg_val = 1 if "luxe" in seg_label.lower() else 0
    d = df.dropna(subset=["Age_Vehicule", "Prix"])
    d = d[d["Segment_Vehicule"] == seg_val]
    g = (d[(d["Age_Vehicule"] >= 0) & (d["Age_Vehicule"] <= 18)]
         .groupby("Age_Vehicule").agg(prix=("Prix", "median"), n=("Prix", "size")))
    g = g[g["n"] >= 8]
    if len(g) < 4:
        return None
    ages, prix = g.index.values.astype(float), g["prix"].values.astype(float)

    def px(a):
        return float(prix[int(np.argmin(np.abs(ages - a)))])

    def cagr(a, b):
        p0, p1 = px(a), px(b)
        return (1 - (p1 / p0) ** (1 / (b - a))) * 100 if p0 > 0 and b > a else float("nan")

    liq = g[g.index >= 3].sort_values("n", ascending=False)
    ages_liquides = sorted(int(x) for x in liq.head(3).index) if len(liq) else []
    return {
        "falaise": cagr(2, 4), "plateau": cagr(4, 7), "traine": cagr(7, 10),
        "ages_liquides": ages_liquides,
        "n_liquide": int(liq["n"].head(3).sum()) if len(liq) else 0,
    }


@st.cache_resource(ttl=DUREE_CACHE)
def _shap_explainer(_bundle):
    """Explainer SHAP du modèle (mis en cache : construit une seule fois).
    shap est optionnel -> None si absent."""
    try:
        import shap
    except ImportError:
        return None
    try:
        return shap.TreeExplainer(_bundle["pipeline"].named_steps["model"])
    except Exception:
        return None


def _ligne_modele(bundle, saisie):
    """Construit un DataFrame 1 ligne aux bons dtypes pour le modèle : features
    numériques en float, catégorielles en str. Indispensable car une catégorielle
    à NaN sur une seule ligne devient float64 et casse l'OrdinalEncoder
    (`isnan` sur des catégories textuelles). NaN catégoriel -> "nan" -> inconnu (-1)."""
    noms = bundle["features_numeriques"] + bundle["features_categorielles"]
    X1 = pd.DataFrame([{c: saisie.get(c) for c in noms}])
    for c in bundle["features_numeriques"]:
        X1[c] = pd.to_numeric(X1[c], errors="coerce")
    for c in bundle["features_categorielles"]:
        X1[c] = X1[c].astype(str)
    return X1[noms]


def expliquer_prix(bundle, saisie, df=None):
    """Décompose une prédiction : contribution de chaque variable pour une voiture
    donnée. Utilise **SHAP** si disponible ; sinon un **repli par perturbation**
    (effet marginal = prédiction réelle − prédiction avec la variable ramenée à sa
    médiane/mode), qui ne dépend d'aucun package. Renvoie un DataFrame trié par
    impact (colonne 'shap'), avec .attrs['methode'] ∈ {'SHAP','perturbation'}, ou
    None si rien n'est calculable."""
    noms = bundle["features_numeriques"] + bundle["features_categorielles"]

    # 1) Voie SHAP (préférée)
    explainer = _shap_explainer(bundle)
    if explainer is not None:
        try:
            Xt = bundle["pipeline"].named_steps["prep"].transform(_ligne_modele(bundle, saisie))
            sv = explainer.shap_values(Xt)[0]
            if len(sv) == len(noms):
                contrib = pd.DataFrame({"feature": noms, "shap": sv})
                contrib["label"] = contrib["feature"].map(LABELS_FEATURES).fillna(contrib["feature"])
                contrib["abs"] = contrib["shap"].abs()
                out = contrib[contrib["abs"] > 1e-6].sort_values("abs", ascending=False)
                out.attrs["methode"] = "SHAP"
                return out
        except Exception:
            pass  # bascule sur le repli

    # 2) Repli sans shap : effet marginal par perturbation (nécessite des données)
    if df is None:
        return None
    try:
        pipe = bundle["pipeline"]
        base = float(pipe.predict(_ligne_modele(bundle, saisie))[0])
        rows = []
        for f in noms:
            if f not in df.columns or df[f].dropna().empty:
                continue
            baseline = (df[f].median() if f in bundle["features_numeriques"]
                        else df[f].mode().iloc[0])
            alt = dict(saisie)
            alt[f] = baseline
            rows.append({"feature": f, "shap": base - float(pipe.predict(_ligne_modele(bundle, alt))[0])})
        contrib = pd.DataFrame(rows)
        if contrib.empty:
            return None
        contrib["label"] = contrib["feature"].map(LABELS_FEATURES).fillna(contrib["feature"])
        contrib["abs"] = contrib["shap"].abs()
        out = contrib[contrib["abs"] > 1e-6].sort_values("abs", ascending=False)
        out.attrs["methode"] = "perturbation"
        return out
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Chargement des données
# ---------------------------------------------------------------------------

@st.cache_data(ttl=DUREE_CACHE)
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


@st.cache_data(ttl=DUREE_CACHE)
def charger_deals():
    try:
        df = lire_csv(DEALS_PATH)
        return df if not df.empty else None
    except Exception:
        return None


def _tenter_charger_modele():
    """Tente de charger le modèle et RENVOIE (bundle, message_erreur) sans
    rien avaler : sert au diagnostic quand le calculateur n'a pas de modèle.
    Essaie d’abord le disque local (le repo est déjà cloné par Streamlit), puis GitHub en secours."""
    erreurs = []
    try:
        bundle = joblib.load(MODELE_PATH)
        if isinstance(bundle, dict) and "pipeline" in bundle:
            return bundle, None
        erreurs.append("local : format de modèle inattendu (pas de clé 'pipeline').")
    except Exception as e:
        erreurs.append(f"local : {type(e).__name__} — {str(e)[:200]}")
    if LIRE_DEPUIS_GITHUB:
        try:
            import io, urllib.request
            with urllib.request.urlopen(_url(MODELE_PATH), timeout=30) as r:
                bundle = joblib.load(io.BytesIO(r.read()))
            if isinstance(bundle, dict) and "pipeline" in bundle:
                return bundle, None
            erreurs.append("GitHub : format de modèle inattendu (pas de clé 'pipeline').")
        except Exception as e:
            erreurs.append(f"GitHub : {type(e).__name__} — {str(e)[:200]}")
    return None, " | ".join(erreurs)


@st.cache_resource(ttl=DUREE_CACHE)
def charger_modele():
    """Le modèle est réentraîné chaque nuit : on le recharge depuis GitHub avec
    la même durée de cache que les données, pour ne pas scorer des annonces
    fraîches avec un modèle de la semaine dernière."""
    bundle, _ = _tenter_charger_modele()
    return bundle


def calculer_prix_ajuste(df, colonne, k=K_LISSAGE):
    """Lissage bayésien : ramène la médiane d'une catégorie peu représentée
    vers la médiane nationale, pour ne pas se faire piéger par le hasard d'un
    petit échantillon."""
    mediane_nat = df["Prix"].median()
    stats = df.groupby(colonne)["Prix"].agg(["median", "count"])
    stats["prix_ajuste"] = (stats["count"] * stats["median"] + k * mediane_nat) / (stats["count"] + k)
    return stats.sort_values("prix_ajuste", ascending=False)


def fmt_dt(v):
    return f"{v:,.0f}".replace(",", " ") + " DT"


# ---------------------------------------------------------------------------
# Analyses agrégées : décote annuelle et prime professionnelle
# ---------------------------------------------------------------------------

MIN_ANNONCES_DECOTE = 15      # points nécessaires pour ajuster une pente crédible
MIN_AGES_DISTINCTS = 4        # sans plusieurs âges différents, aucune pente n'a de sens
MIN_PAR_COTE = 3              # annonces minimum de chaque type de vendeur
MIN_RECOUVREMENT_ANS = 2      # recouvrement d'âge minimum entre pros et particuliers


@st.cache_data
def calculer_niveau_regional(df, n_min=30):
    """Niveau de prix par région, à véhicule comparable.

    Comparer les prix médians régionaux revient à comparer des paniers de
    véhicules différents : la corrélation entre le prix médian d'une région
    et sa part de marques premium est de +0,70 sur les données réelles.

    On estime donc log(prix) ~ modèle + âge + kilométrage + région. Les
    indicatrices de modèle absorbent la composition ; le coefficient de
    région mesure ce qui reste, c'est-à-dire le niveau de prix local pour un
    véhicule identique.
    """
    d = df.dropna(subset=["Prix", "Age_Vehicule", "Kilométrage", "Modèle", "Localisation"])
    d = d[(d["Prix"] > 0) & d["Age_Vehicule"].between(0, 25)].copy()
    vc = d["Localisation"].value_counts()
    d = d[d["Localisation"].isin(vc[vc >= n_min].index)]
    vm = d["Modèle"].value_counts()
    d = d[d["Modèle"].isin(vm[vm >= 5].index)]
    if d["Localisation"].nunique() < 3 or len(d) < 200:
        return pd.DataFrame()

    X = pd.get_dummies(d[["Modèle", "Localisation"]], drop_first=True).astype(float)
    X["age"] = d["Age_Vehicule"].values
    X["km"] = d["Kilométrage"].values / 10000
    X.insert(0, "const", 1.0)
    try:
        coef, *_ = np.linalg.lstsq(X.values, np.log(d["Prix"].values), rcond=None)
    except np.linalg.LinAlgError:
        return pd.DataFrame()

    cols = list(X.columns)
    lignes = []
    for region in sorted(d["Localisation"].unique()):
        nom = f"Localisation_{region}"
        effet = coef[cols.index(nom)] if nom in cols else 0.0
        lignes.append({
            "Localisation": region,
            "prime_pct": (np.exp(effet) - 1) * 100,
            "n": int((d["Localisation"] == region).sum()),
        })
    return pd.DataFrame(lignes)


@st.cache_data
def calculer_decote_annuelle(df):
    """Décote annuelle moyenne par modèle.

    On ajuste log(prix) ~ âge : la pente donne directement un taux de décote
    constant en pourcentage par an (exp(pente) - 1), ce qui correspond à la
    façon dont une voiture se déprécie réellement -- un pourcentage du prix
    restant, pas un montant fixe en dinars.

    Une simple différence de prix médian entre deux âges serait beaucoup plus
    fragile : elle ne repose que sur deux points, alors que la pente utilise
    toutes les annonces du modèle.
    """
    d = df.dropna(subset=["Prix", "Age_Vehicule", "Marque", "Modèle"])
    d = d[d["Age_Vehicule"].between(0, 20) & (d["Prix"] > 0)]

    lignes = []
    for (marque, modele), g in d.groupby(["Marque", "Modèle"]):
        if len(g) < MIN_ANNONCES_DECOTE or g["Age_Vehicule"].nunique() < MIN_AGES_DISTINCTS:
            continue
        pente = np.polyfit(g["Age_Vehicule"], np.log(g["Prix"]), 1)[0]
        decote = (np.exp(pente) - 1) * 100
        if not np.isfinite(decote) or decote > 0:   # une décote positive = données incohérentes
            continue
        lignes.append({
            "libelle": f"{marque} {modele}",
            "Marque": marque, "Modèle": modele,
            "decote_pct_an": decote,
            "n": len(g),
            "prix_median": g["Prix"].median(),
            "age_median": g["Age_Vehicule"].median(),
        })
    return pd.DataFrame(lignes)


@st.cache_data
def calculer_prime_pro(df):
    """Écart de prix pro vs particulier pour un même modèle, à âge égal.

    Pour chaque modèle : log(prix) ~ âge + kilométrage + vendeur_pro. Le
    coefficient du vendeur donne l'écart une fois l'âge ET le kilométrage
    neutralisés.

    Le kilométrage est indispensable ici : à âge égal, pros et particuliers ne
    vendent pas des véhicules au même compteur. Sans ce contrôle, 5 modèles
    sur 35 affichaient un écart de signe opposé à la réalité -- un pro vendant
    des exemplaires plus roulés paraissait « vendre moins cher » alors qu'à
    kilométrage comparable il vend plus cher.

    Garde-fou décisif : les deux populations doivent se recouvrir en âge sur
    au moins MIN_RECOUVREMENT_ANS années, et la régression n'est faite que sur
    cette zone de recouvrement. Sans cela, "corriger de l'âge" reviendrait à
    extrapoler hors des données -- c'est exactement ce qui produisait des
    écarts absurdes (+169 %) dans la version précédente, où les pros vendaient
    des véhicules récents et les particuliers des véhicules deux fois plus
    vieux du même modèle.
    """
    d = df.dropna(subset=["Prix", "Age_Vehicule", "Kilométrage", "Marque", "Modèle", "Source"])
    d = d[d["Age_Vehicule"].between(0, 25) & (d["Prix"] > 0)].copy()
    d["est_pro"] = d["Source"].str.lower().str.contains("automobile", na=False).astype(int)

    lignes = []
    for (marque, modele), g in d.groupby(["Marque", "Modèle"]):
        pro, part = g[g["est_pro"] == 1], g[g["est_pro"] == 0]
        if len(pro) < MIN_PAR_COTE or len(part) < MIN_PAR_COTE:
            continue
        age_min = max(pro["Age_Vehicule"].min(), part["Age_Vehicule"].min())
        age_max = min(pro["Age_Vehicule"].max(), part["Age_Vehicule"].max())
        if age_max - age_min < MIN_RECOUVREMENT_ANS:
            continue
        zone = g[g["Age_Vehicule"].between(age_min, age_max)]
        n_pro = int((zone["est_pro"] == 1).sum())
        n_part = int((zone["est_pro"] == 0).sum())
        if n_pro < MIN_PAR_COTE or n_part < MIN_PAR_COTE:
            continue
        X = np.column_stack([
            np.ones(len(zone)),
            zone["Age_Vehicule"],
            zone["Kilométrage"] / 10000,
            zone["est_pro"],
        ])
        try:
            coef, *_ = np.linalg.lstsq(X, np.log(zone["Prix"]), rcond=None)
        except np.linalg.LinAlgError:
            continue
        prime = (np.exp(coef[3]) - 1) * 100
        if not np.isfinite(prime):
            continue
        lignes.append({
            "libelle": f"{marque} {modele}",
            "Marque": marque, "Modèle": modele,
            "prime_pct": prime,
            "n_pro": n_pro, "n_particulier": n_part,
            "age_min": age_min, "age_max": age_max,
            "km_median_pro": float(zone[zone["est_pro"] == 1]["Kilométrage"].median()),
            "km_median_particulier": float(zone[zone["est_pro"] == 0]["Kilométrage"].median()),
        })
    return pd.DataFrame(lignes)


# ===========================================================================
# 1. VUE MARCHÉ (concessionnaire)
# ===========================================================================

@st.cache_data
def calculer_indice_depreciation(df, marques, age_max=20, n_min_par_age=8):
    """Profil de dépréciation par marque, corrigé de la composition.

    Une courbe de prix médian par âge est trompeuse : le panier de modèles
    change avec l'âge. Sur les données réelles, la courbe Peugeot *monte*
    entre 3 et 5 ans -- non parce qu'une Peugeot prend de la valeur, mais
    parce qu'à 3 ans l'échantillon est dominé par des 301 (berline
    économique) et à 5 ans par des 3008 (SUV). C'est un changement de
    composition, pas de la dépréciation.

    Ici : log(prix) ~ effets fixes MODÈLE + effets fixes ÂGE. Les indicatrices
    de modèle absorbent la composition ; le profil d'âge restant est celui
    d'un même modèle qui vieillit. Il est exprimé en indice base 100 à l'âge
    le plus bas observé, et l'âge reste catégoriel pour conserver la forme
    réelle de la courbe (chute forte les premières années) plutôt que de
    l'aplatir en droite.
    """
    d = df.dropna(subset=["Prix", "Age_Vehicule", "Marque", "Modèle"])
    d = d[d["Age_Vehicule"].between(0, age_max) & (d["Prix"] > 0)]

    sorties = []
    for marque in marques:
        sub = d[d["Marque"] == marque].copy()
        # Modèles trop rares : ils ne peuvent pas servir de référence stable
        vc = sub["Modèle"].value_counts()
        sub = sub[sub["Modèle"].isin(vc[vc >= 5].index)]
        # Âges trop peu représentés : une médiane sur 2 annonces n'est pas un point
        va = sub["Age_Vehicule"].value_counts()
        sub = sub[sub["Age_Vehicule"].isin(va[va >= n_min_par_age].index)]
        if len(sub) < 40 or sub["Modèle"].nunique() < 2 or sub["Age_Vehicule"].nunique() < 4:
            continue

        # L'âge est traité en catégoriel, mais la référence doit être l'âge le
        # PLUS JEUNE. Encoder l'âge en texte et laisser drop_first choisir
        # prendrait la première valeur par ordre alphabétique -- soit "10"
        # avant "2" -- et l'indice serait rapporté à une base absurde.
        ages = sorted(sub["Age_Vehicule"].astype(int).unique())
        for age in ages[1:]:
            sub[f"_age_{age}"] = (sub["Age_Vehicule"].astype(int) == age).astype(float)
        colonnes_age = [f"_age_{a}" for a in ages[1:]]

        X = pd.get_dummies(sub[["Modèle"]], drop_first=True).astype(float)
        for col in colonnes_age:
            X[col] = sub[col].values
        X.insert(0, "const", 1.0)
        try:
            coef, *_ = np.linalg.lstsq(X.values, np.log(sub["Prix"].values), rcond=None)
        except np.linalg.LinAlgError:
            continue

        cols = list(X.columns)
        # Profil brut d'âge (en log), puis régression isotone décroissante.
        # Contrairement à l'ancienne logique, on ne SUPPRIME aucun âge parce
        # qu'il remonte : tous les points restent dans l'estimation et leurs
        # effectifs servent de poids. L'isotonic regression donne la meilleure
        # approximation monotone au sens des moindres carrés pondérés.
        profil = []
        for age in ages:
            nom = f"_age_{age}"
            effet = coef[cols.index(nom)] if nom in cols else 0.0
            profil.append({"age": age, "effet": float(effet),
                           "n": int((sub["Age_Vehicule"] == age).sum())})
        if len(profil) < 2:
            continue
        try:
            from sklearn.isotonic import IsotonicRegression
            ir = IsotonicRegression(increasing=False, out_of_bounds="clip")
            effets_lisses = ir.fit_transform(
                [p["age"] for p in profil],
                [p["effet"] for p in profil],
                sample_weight=[p["n"] for p in profil],
            )
        except Exception:
            effets_lisses = [p["effet"] for p in profil]
        base = float(effets_lisses[0])
        for p, effet_lisse in zip(profil, effets_lisses):
            sorties.append({
                "Marque": marque, "age": p["age"],
                "indice": float(np.exp(float(effet_lisse) - base) * 100),
                "indice_brut": float(np.exp(p["effet"] - profil[0]["effet"]) * 100),
                "n": p["n"], "age_ref": profil[0]["age"],
            })
    return pd.DataFrame(sorties)


def calculer_indice_modele_representatif(df, marques, age_max=20, n_min_par_age=5):
    """Décote d'UN modèle représentatif par marque : le plus fréquent qui a une
    couverture d'âge suffisante. Vue la plus littérale (le même modèle qui
    vieillit, sans régression) — au prix de moins de données. Indexé base 100
    au plus jeune âge observé de ce modèle."""
    d = df.dropna(subset=["Prix", "Age_Vehicule", "Marque", "Modèle"])
    d = d[d["Age_Vehicule"].between(0, age_max) & (d["Prix"] > 0)]
    sorties = []
    for marque in marques:
        sub = d[d["Marque"] == marque]
        meilleur = None
        for modele, g in sub.groupby("Modèle"):
            va = g["Age_Vehicule"].astype(int).value_counts()
            ages_ok = sorted(int(a) for a in va[va >= n_min_par_age].index)
            if len(ages_ok) >= 4 and (meilleur is None or len(g) > meilleur[2]):
                meilleur = (modele, ages_ok, len(g))
        if not meilleur:
            continue
        modele, ages, _ = meilleur
        gm = sub[(sub["Modèle"] == modele) & (sub["Age_Vehicule"].astype(int).isin(ages))]
        med = gm.groupby(gm["Age_Vehicule"].astype(int))["Prix"].median()
        base = med.loc[min(ages)]
        for a in sorted(med.index):
            sorties.append({
                "Marque": f"{marque} — {modele}", "age": a,
                "indice": float(med.loc[a] / base * 100) if base else np.nan,
                "n": int((gm["Age_Vehicule"].astype(int) == a).sum()),
            })
    return pd.DataFrame(sorties)


def composition_par_age(df, seg_val=None):
    """Modèle le plus présent par tranche d'âge — illustre le biais de composition.
    seg_val : None = marché global, 0 = généraliste, 1 = segment luxe."""
    d = df.dropna(subset=["Age_Vehicule", "Prix", "Marque", "Modèle"]).copy()
    if seg_val is not None and "Segment_Vehicule" in d.columns:
        d = d[d["Segment_Vehicule"] == seg_val]
    if d.empty:
        return pd.DataFrame()
    d["_mm"] = d["Marque"].astype(str) + " " + d["Modèle"].astype(str)
    bornes = [0, 2, 4, 6, 9, 13, 18, 99]
    libs = ["0-2 ans", "3-4 ans", "5-6 ans", "7-9 ans", "10-13 ans", "14-18 ans", "18+ ans"]
    d["_tr"] = pd.cut(d["Age_Vehicule"], bornes, labels=libs, include_lowest=True)
    rows = []
    for tr, g in d.groupby("_tr", observed=True):
        if len(g) < 10:
            continue
        top = g["_mm"].value_counts().head(3)
        cell = [f"{m} ({c / len(g) * 100:.0f} %)" for m, c in top.items()]
        while len(cell) < 3:
            cell.append("—")
        rows.append({
            "Tranche d'âge": str(tr), "Annonces": len(g),
            "Prix médian": fmt_dt(g["Prix"].median()),
            "Modèle dominant": cell[0], "2ᵉ": cell[1], "3ᵉ": cell[2],
        })
    return pd.DataFrame(rows)


def page_marche(df):
    st.title("🏢 Concessionnaire")
    st.caption("Comprendre et valoriser le parc : structure et niveaux de prix, "
               "que vaut le marché, où se vend quoi, à quel prix.")

    if df is None:
        st.error(f"`{SCORED_PATH}` introuvable — lance `python main.py` d'abord.")
        return

    # ---- KPIs ------------------------------------------------------------
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(f"Annonces récentes (≤ {MAX_DAYS_OLD} j)", f"{len(df):,}".replace(",", " "))
    c2.metric("Prix médian du marché", fmt_dt(df["Prix"].median()))
    age_median = df["Age_Vehicule"].median()
    c3.metric("Âge médian du parc", f"{age_median:.0f} ans" if pd.notna(age_median) else "—")
    km_median = df["Kilométrage"].median()
    c4.metric("Kilométrage médian", f"{km_median:,.0f} km".replace(",", " ") if pd.notna(km_median) else "—")

    st.divider()

    # ---- Parts de marché et niveaux de prix ------------------------------
    col_a, col_b = st.columns(2)
    with col_a:
        parts = df["Marque"].value_counts().head(15).sort_values()
        fig = px.bar(
            x=parts.values, y=parts.index, orientation="h",
            title="Part des annonces observées par marque (top 15)",
            labels={"x": "Annonces", "y": ""},
        )
        fig.update_traces(marker_color=C_ASPHALTE)
        st.plotly_chart(style_figure(fig, 430), width="stretch")

    with col_b:
        top_m = df.groupby("Marque").agg(prix=("Prix", "median"), n=("Prix", "count"))
        top_m = top_m[top_m["n"] >= 10].sort_values("prix").tail(15)
        fig = px.bar(
            x=top_m["prix"], y=top_m.index, orientation="h",
            title="Prix médian par marque (≥ 10 annonces, top 15)",
            labels={"x": "Prix médian (DT)", "y": ""},
        )
        fig.update_traces(
            marker_color=[C_SABLE if m in MARQUES_LUXE else C_ASPHALTE for m in top_m.index],
            customdata=top_m["n"], hovertemplate="%{y} : %{x:,.0f} DT (n=%{customdata})<extra></extra>",
        )
        st.plotly_chart(style_figure(fig, 430), width="stretch")

    # ---- Dépréciation ----------------------------------------------------
    st.subheader("Courbe de dépréciation")
    st.caption("Perte de valeur selon l'âge, **à modèle constant** — la composition du parc "
               "change avec l'âge, et la corriger est indispensable pour lire une vraie "
               "dépréciation.")

    marques_dispo = df["Marque"].value_counts()
    marques_courbe = st.multiselect(
        "Marques à comparer (≥ 15 annonces)",
        options=list(marques_dispo[marques_dispo >= 15].index),
        default=[m for m in ["Volkswagen", "Peugeot", "Kia", "Mercedes-Benz"] if marques_dispo.get(m, 0) >= 15][:4],
    )

    vue_depr = st.radio(
        "Méthode",
        ["Profil marque (composition corrigée par régression)", "Modèle représentatif par marque"],
        horizontal=True,
        help="La régression garde toutes les données et corrige le mix de modèles. "
             "Le modèle représentatif est plus littéral (un seul modèle qui vieillit) "
             "mais repose sur moins d'annonces.",
    )
    if not marques_courbe:
        indice = pd.DataFrame()
    elif vue_depr.startswith("Modèle"):
        indice = calculer_indice_modele_representatif(df, marques_courbe)
    else:
        indice = calculer_indice_depreciation(df, marques_courbe)
    if len(indice):
        fig = px.line(
            indice, x="age", y="indice", color="Marque", markers=True,
            title="Indice de valeur selon l'âge (base 100 au plus jeune âge observé)",
            labels={"age": "Âge (années)", "indice": "Indice de valeur"},
            custom_data=["n"],
        )
        fig.update_traces(hovertemplate="%{y:.0f} (base 100)<br>%{x} ans — "
                                        "%{customdata[0]} annonces<extra></extra>")
        st.plotly_chart(style_figure(fig), width="stretch")

        with st.expander("ℹ️ Pourquoi un indice, et pas le prix médian par âge"):
            st.write(
                """
Tracer le prix médian par âge donne une courbe fausse, parce que le panier de
modèles change avec l'âge. Sur les données réelles, la courbe Peugeot
**montait** entre 3 et 5 ans :

| Âge | Prix médian | Modèles dominants |
|---|---|---|
| 3 ans | 34 700 DT | **301** ×14, 208 ×6 |
| 5 ans | 62 500 DT | **3008** ×5, 208 ×5 |

Une Peugeot ne prend pas de la valeur en vieillissant : à 3 ans l'échantillon
est dominé par des 301 (berline économique), à 5 ans par des 3008 (SUV). C'est
un changement de composition, pas de la dépréciation.

L'indice est calculé par `log(prix) ~ effets fixes modèle + effets fixes âge`.
Les indicatrices de modèle absorbent la composition ; le profil d'âge restant
est celui d'un même modèle qui vieillit. L'âge reste catégoriel pour conserver
la forme réelle de la courbe — la chute des premières années — au lieu de
l'aplatir en droite. Les âges représentés par moins de 5 annonces sont exclus :
une médiane sur 2 annonces n'est pas un point de courbe.

**Limite.** La mesure reste transversale : on compare des véhicules d'âges
différents à un instant donné, on ne suit pas un véhicule dans le temps. Si les
millésimes récents sont mieux équipés que les anciens, une part de l'écart vient
de l'équipement et non de l'âge.
"""
            )
    elif marques_courbe:
        st.info("Pas assez d'annonces pour ces marques (il faut au moins 40 annonces, "
                "2 modèles distincts et 4 âges différents avec 5 annonces chacun).")

    # ---- Décote annuelle par modèle --------------------------------------
    st.subheader("Décote annuelle par modèle")
    st.caption("Combien un modèle perd, en pourcentage de sa valeur, chaque année. "
               "C'est le chiffre qui sert à fixer une reprise ou à choisir un véhicule "
               "qui tiendra sa valeur.")

    decote = calculer_decote_annuelle(df)
    if len(decote):
        # Les deux panneaux ne doivent JAMAIS montrer les mêmes modèles.
        # Avec un head(15) fixe, dès que la fonction retourne moins de 30
        # modèles les deux listes se recouvrent, et à 15 elles deviennent
        # identiques : on affiche alors deux fois la même chose, l'une
        # étiquetée "perdent le plus vite" et l'autre "tiennent le mieux".
        # En prenant au plus la moitié de chaque côté, le recouvrement est
        # structurellement impossible.
        k = min(15, len(decote) // 2)

        if k < 3:
            # Trop peu de modèles pour opposer deux groupes : un seul
            # graphique, honnête, avec tout ce qui est disponible.
            tout = decote.sort_values("decote_pct_an")
            fig = px.bar(
                tout, x="decote_pct_an", y="libelle", orientation="h",
                title=f"Décote annuelle — les {len(tout)} modèles disponibles",
                labels={"decote_pct_an": "Décote (% par an)", "libelle": ""},
                custom_data=["n", "prix_median"],
            )
            fig.update_traces(
                marker_color=C_ASPHALTE,
                hovertemplate="%{y} : %{x:.1f} %/an<br>%{customdata[0]} annonces — "
                              "prix médian %{customdata[1]:,.0f} DT<extra></extra>",
            )
            st.plotly_chart(style_figure(fig, max(280, 30 * len(tout))), width="stretch")
            st.warning(f"Seuls {len(decote)} modèles ont assez d'annonces pour estimer une "
                       "décote. Trop peu pour opposer « perdent le plus vite » et « tiennent "
                       "le mieux » sans afficher deux fois les mêmes véhicules.")
        else:
            col_g, col_h = st.columns(2)
            with col_g:
                top_d = decote.sort_values("decote_pct_an").head(k).sort_values("decote_pct_an", ascending=False)
                fig = px.bar(
                    top_d, x="decote_pct_an", y="libelle", orientation="h",
                    title=f"Perdent le plus vite leur valeur (top {k})",
                    labels={"decote_pct_an": "Décote (% par an)", "libelle": ""},
                    custom_data=["n", "prix_median"],
                )
                fig.update_traces(
                    marker_color=C_ALERTE,
                    hovertemplate="%{y} : %{x:.1f} %/an<br>%{customdata[0]} annonces — "
                                  "prix médian %{customdata[1]:,.0f} DT<extra></extra>",
                )
                st.plotly_chart(style_figure(fig, 460), width="stretch")
            with col_h:
                garde = decote.sort_values("decote_pct_an", ascending=False).head(k)
                fig = px.bar(
                    garde, x="decote_pct_an", y="libelle", orientation="h",
                    title=f"Tiennent le mieux leur valeur (top {k})",
                    labels={"decote_pct_an": "Décote (% par an)", "libelle": ""},
                    custom_data=["n", "prix_median"],
                )
                fig.update_traces(
                    marker_color=C_GAIN,
                    hovertemplate="%{y} : %{x:.1f} %/an<br>%{customdata[0]} annonces — "
                                  "prix médian %{customdata[1]:,.0f} DT<extra></extra>",
                )
                st.plotly_chart(style_figure(fig, 460), width="stretch")

        st.download_button(
            "⬇️ Télécharger le tableau des décotes (CSV)",
            decote.sort_values("decote_pct_an")[
                ["Marque", "Modèle", "decote_pct_an", "n", "prix_median", "age_median"]
            ].round(2).to_csv(index=False, sep=";").encode("utf-8-sig"),
            file_name="decote_annuelle_par_modele.csv",
            mime="text/csv",
        )
        st.caption(f"{len(decote)} modèles ont assez d'annonces (≥ {MIN_ANNONCES_DECOTE}, "
                   f"réparties sur ≥ {MIN_AGES_DISTINCTS} âges différents) pour une décote fiable.")
    else:
        st.info("Pas encore assez d'annonces par modèle pour estimer une décote annuelle.")
    st.subheader("Prime professionnelle, à âge égal")
    st.caption("automobile.tn est surtout alimenté par les professionnels, tayara.tn par les "
               "particuliers. La question utile : pour un même modèle **et un même âge**, "
               "de combien un pro affiche-t-il au-dessus d'un particulier ?")

    prime = calculer_prime_pro(df)
    if len(prime):
        top = prime.reindex(prime["prime_pct"].abs().sort_values(ascending=False).index).head(15)
        top = top.sort_values("prime_pct")
        fig = px.bar(
            top, x="prime_pct", y="libelle", orientation="h",
            title="Écart de prix pro vs particulier, à âge comparable",
            labels={"prime_pct": "Prime professionnelle (%)", "libelle": ""},
            custom_data=["n_pro", "n_particulier", "age_min", "age_max",
                         "km_median_pro", "km_median_particulier"],
        )
        fig.update_traces(
            marker_color=[C_GAIN if v > 0 else C_ALERTE for v in top["prime_pct"]],
            hovertemplate="%{y} : %{x:+.0f} %<br>%{customdata[0]} annonces pro / "
                          "%{customdata[1]} particulier<br>Âges comparés : "
                          "%{customdata[2]:.0f}–%{customdata[3]:.0f} ans"
                          "<br>Km médian : %{customdata[4]:,.0f} (pro) / "
                          "%{customdata[5]:,.0f} (particulier)<extra></extra>",
        )
        fig.add_vline(x=0, line_color=C_GRIS, line_width=1)
        st.plotly_chart(style_figure(fig, 430), width="stretch")

        mediane = prime["prime_pct"].median()
        st.markdown(f"**Prime pro médiane à âge égal : {mediane:+.0f} %** "
                    f"sur {len(prime)} modèles comparables.")

        with st.expander("ℹ️ Pourquoi « à âge égal » change tout"):
            st.write(
                """
Comparer directement le prix médian des pros à celui des particuliers donne un
écart médian de **+15 %**, avec des valeurs absurdes (+169 % sur une Škoda
Octavia, +129 % sur une Mercedes Classe C). Ce n'est pas une marge : c'est un
**biais de composition**. Les pros vendent des Classe C de 3 ans, les
particuliers des Classe C de 15 ans. On compare des voitures différentes et on
appelle ça un écart de prix.

Ici, pour chaque modèle, le prix est expliqué par l'âge **et** par le type de
vendeur (`log(prix) ~ âge + vendeur_pro`). L'écart affiché est ce qui reste une
fois l'âge neutralisé. Deux garde-fous : au moins 3 annonces de chaque côté, et
surtout un **recouvrement d'au moins 2 ans** entre les âges des deux
populations — sans recouvrement, corriger de l'âge reviendrait à extrapoler
hors des données observées.

Résultat : la prime médiane tombe à **+2 %**. L'arbitrage « acheter au
particulier, revendre au pro » est donc bien plus étroit que ne le suggérait le
graphique brut, et ne tient réellement que sur les quelques modèles en tête de
liste.
"""
            )
    else:
        st.info("Pas encore assez de modèles où pros et particuliers vendent des véhicules "
                "d'âges comparables (minimum 3 annonces de chaque côté et 2 ans de recouvrement).")

    # ---- Structure par gamme de prix ------------------------------------
    col_c, col_d = st.columns(2)
    with col_c:
        tranches = pd.cut(
            df["Prix"],
            bins=[0, 20000, 35000, 50000, 80000, 120000, 200000, np.inf],
            labels=["< 20k", "20–35k", "35–50k", "50–80k", "80–120k", "120–200k", "> 200k"],
        )
        repartition = tranches.value_counts().sort_index()
        fig = px.bar(
            x=repartition.index.astype(str), y=repartition.values,
            title="Structure du marché par gamme de prix (DT)",
            labels={"x": "Gamme de prix", "y": "Annonces"},
        )
        fig.update_traces(marker_color=C_ASPHALTE)
        st.plotly_chart(style_figure(fig), width="stretch")

    with col_d:
        if "Energie" in df.columns:
            stats_e = calculer_prix_ajuste(df, "Energie")
            fig = px.bar(
                x=stats_e.index, y=stats_e["prix_ajuste"],
                title="Prix médian par énergie (ajusté petits échantillons)",
                labels={"x": "", "y": "Prix ajusté (DT)"},
            )
            fig.update_traces(
                marker_color=C_ASPHALTE, customdata=stats_e["count"],
                hovertemplate="%{x} : %{y:,.0f} DT (n=%{customdata})<extra></extra>",
            )
            st.plotly_chart(style_figure(fig), width="stretch")

    # ---- Prix par région, à véhicule comparable --------------------------
    if "Localisation" in df.columns:
        niveau = calculer_niveau_regional(df)
        if len(niveau):
            fig = px.bar(
                niveau.sort_values("prime_pct"), x="prime_pct", y="Localisation",
                orientation="h",
                title="Niveau de prix par région, à véhicule comparable",
                labels={"prime_pct": "Écart vs référence (%)", "Localisation": ""},
                custom_data=["n"],
            )
            fig.update_traces(
                marker_color=[C_GAIN if v < 0 else C_ALERTE
                              for v in niveau.sort_values("prime_pct")["prime_pct"]],
                hovertemplate="%{y} : %{x:+.1f} %<br>%{customdata[0]} annonces<extra></extra>",
            )
            fig.add_vline(x=0, line_color=C_GRIS, line_width=1)
            st.plotly_chart(style_figure(fig, 430), width="stretch")
            with st.expander("ℹ️ Pourquoi pas simplement le prix médian par région"):
                st.write(
                    """
Le prix médian brut par région ne mesure pas le niveau de prix local, mais la
composition du parc qui y est vendu. Sur les données réelles, la corrélation
entre le prix médian d'une région et sa **part de marques premium** est de
**+0,70** : Tunis n'est pas « chère », 34 % de ses annonces sont des Mercedes,
BMW ou Audi, contre 10 % à Médenine.

Le graphique ci-dessus vient de `log(prix) ~ modèle + âge + kilométrage +
région`. Le coefficient de région indique combien **le même véhicule** se
négocie plus ou moins cher selon l'endroit. Le vert signale les régions où
acheter, l'orange celles où revendre.
"""
                )

    # ---- Tendance temporelle --------------------------------------------
    # On trace par SOURCE : une médiane globale par jour serait biaisée si la
    # composition des scrapers change (ex. plus d'automobile.tn un jour donné).
    if "Annonce-Detectee" in df.columns and df["Annonce-Detectee"].nunique() >= 2:
        if "Source" in df.columns and df["Source"].nunique() >= 2:
            tendance = (df.groupby(["Annonce-Detectee", "Source"])
                          .agg(prix=("Prix", "median"), n=("Prix", "count"))
                          .reset_index())
            tendance = tendance[tendance["n"] >= 3]
            if not tendance.empty:
                fig = px.line(
                    tendance, x="Annonce-Detectee", y="prix", color="Source", markers=True,
                    title="Prix médian des annonces collectées, par source",
                    labels={"Annonce-Detectee": "", "prix": "Prix médian (DT)"},
                    hover_data={"n": True},
                )
                st.plotly_chart(style_figure(fig, 340), width="stretch")
                st.caption("Lecture : évolution du prix médian au sein de chaque source. "
                           "Ce graphique n'est pas un indice de prix du marché tunisien.")
        else:
            tendance = df.groupby("Annonce-Detectee").agg(prix=("Prix", "median"), n=("Prix", "count")).reset_index()
            fig = px.line(tendance, x="Annonce-Detectee", y="prix", markers=True,
                          title="Prix médian des annonces collectées par jour",
                          labels={"Annonce-Detectee": "", "prix": "Prix médian (DT)"})
            st.plotly_chart(style_figure(fig, 320), width="stretch")
            st.caption("Médiane de l'échantillon collecté, pas un indice officiel du marché.")

    st.divider()

    # ---- Courbe de décote : comment le prix baisse avec l'âge ------------
    st.subheader("Courbe de décote — comment le prix baisse avec l'âge")
    st.caption("Prix médian selon l'âge du véhicule, séparé par segment. La pente donne "
               "la dépréciation ; les toutes premières années reflètent surtout la prime du neuf.")
    if "Age_Vehicule" in df.columns and "Segment_Vehicule" in df.columns:
        d = df.dropna(subset=["Age_Vehicule", "Prix"]).copy()
        d = d[(d["Age_Vehicule"] >= 0) & (d["Age_Vehicule"] <= 25)]
        d["_seg"] = d["Segment_Vehicule"].map({1: "Segment luxe", 0: "Généraliste"}).fillna("Généraliste")
        courbe = (d.groupby(["_seg", "Age_Vehicule"], observed=True)
                  .agg(prix=("Prix", "median"), n=("Prix", "size")).reset_index())
        courbe = courbe[courbe["n"] >= 10]  # plancher de bruit minimal

        # Une voiture d'occasion ne prend PAS de valeur en vieillissant : toute
        # montée en tête de courbe est un artefact (échantillon quasi-neuf, rare
        # et biaisé vers l'entrée de gamme). On coupe ces premiers âges tant que
        # le prix médian est inférieur à l'âge suivant — robuste, sans seuil magique.
        morceaux = []
        for seg, g in courbe.groupby("_seg", observed=True):
            g = g.sort_values("Age_Vehicule").reset_index(drop=True)
            i = 0
            while i + 1 < len(g) and g.loc[i, "prix"] < g.loc[i + 1, "prix"]:
                i += 1
            morceaux.append(g.iloc[i:])
        courbe = pd.concat(morceaux, ignore_index=True) if morceaux else courbe

        if len(courbe):
            fig = px.line(courbe, x="Age_Vehicule", y="prix", color="_seg", markers=True,
                          title="Prix médian selon l'âge, par segment",
                          labels={"Age_Vehicule": "Âge (années)", "prix": "Prix médian (DT)", "_seg": "Segment"})
            st.plotly_chart(style_figure(fig, 380), width="stretch")
            gen = (courbe[(courbe["_seg"] == "Généraliste") & courbe["Age_Vehicule"].between(2, 8)]
                   .sort_values("Age_Vehicule"))
            if len(gen) >= 3:
                p0, p1 = gen["prix"].iloc[0], gen["prix"].iloc[-1]
                ans = gen["Age_Vehicule"].iloc[-1] - gen["Age_Vehicule"].iloc[0]
                if p0 > 0 and ans > 0:
                    taux = (1 - (p1 / p0) ** (1 / ans)) * 100
                    st.caption(f"Décote annuelle moyenne (généraliste, **2–8 ans**) : **≈ {taux:.0f} % par an**. "
                               "La chute plus marquée avant 2 ans est surtout la « prime du neuf » "
                               "qui s'évapore, pas de la dépréciation d'occasion.")
        else:
            st.info("Pas encore assez de données par âge pour tracer la courbe.")
    else:
        st.info("Colonnes d'âge/segment absentes des données scorées.")

    # ---- Qui compose le marché à chaque âge ? ----------------------------
    st.divider()
    st.subheader("Qui compose le marché à chaque âge ?")
    st.caption("Le modèle le plus présent dans les annonces, par tranche d'âge. C'est la "
               "**preuve du biais de composition** : le prix médian baisse en partie parce "
               "qu'on ne regarde plus les mêmes voitures en vieillissant.")
    vue_comp = st.radio("Vue", ["Marché global", "Généraliste", "Segment luxe"],
                        horizontal=True, key="vue_composition")
    seg_map = {"Marché global": None, "Généraliste": 0, "Segment luxe": 1}
    comp = composition_par_age(df, seg_map[vue_comp])
    if len(comp):
        st.dataframe(comp, hide_index=True, width="stretch")
        explication(
            "Comment lire ce tableau",
            "**Ce que ça montre.** Le modèle le plus fréquent dans les annonces, par tranche "
            "d'âge, avec le prix médian de la tranche.\n\n"
            "**Comment le lire.** Suis le modèle dominant en descendant : il passe de récents et "
            "chers (SUV, berlines allemandes) à anciens et économiques (Fiesta, Polo, 206). Le prix "
            "médian chute **en partie à cause de ce changement**, pas seulement de la dépréciation — "
            "à 0-2 ans tu vois des Mercedes, à 18+ ans des Fiesta : ce ne sont pas les mêmes "
            "voitures qui ont vieilli.\n\n"
            "**Pourquoi c'est important.** C'est exactement le biais que la courbe de dépréciation "
            "corrige par régression (comparer une Golf à une Golf). Note que les parts sont faibles "
            "(4-7 %) : le marché est fragmenté, aucun modèle ne domine — c'est le *type* de voiture "
            "qui se déplace."
        )
    else:
        st.info("Pas assez de données pour cette vue.")

    # ---- Décote par segment + stratégie propriétaire ---------------------
    st.divider()
    st.subheader("Décote par segment & stratégie d'achat propriétaire")
    infos = {s: analyse_decote_segment(df, s) for s in ["Généraliste", "Segment luxe"]}
    infos = {s: v for s, v in infos.items() if v}
    if infos:
        st.caption("Décote annuelle réelle par tranche d'âge (mesurée à partir de 2 ans : "
                   "l'âge 0-1, quasi-neuves rares et chères, relève de la « prime du neuf » "
                   "et non de la dépréciation d'occasion).")
        tab = pd.DataFrame([{
            "Segment": seg,
            "2→4 ans": f"−{v['falaise']:.0f} %/an",
            "4→7 ans": f"−{v['plateau']:.0f} %/an",
            "7→10 ans": f"−{v['traine']:.0f} %/an",
        } for seg, v in infos.items()])
        st.dataframe(tab, hide_index=True, width="stretch")

        gen = infos.get("Généraliste")
        lux = infos.get("Segment luxe") or infos.get("Luxe")
        st.markdown("**🧍 Pour un propriétaire — ce que disent vraiment les chiffres**")
        if gen:
            st.markdown(
                f"- **Le généraliste tient sa valeur au début, puis décroche.** De 2 à 4 ans, "
                f"il ne perd que **~{gen['falaise']:.0f} %/an** ; la décote **accélère ensuite** "
                f"(~{gen['plateau']:.0f} %/an de 4 à 7 ans, ~{gen['traine']:.0f} %/an après). "
                f"Il n'y a donc pas de « falaise » d'occasion — la seule vraie chute, c'est la "
                f"**prime du neuf** qui s'évapore la 1re année, que tu évites simplement en "
                f"achetant d'occasion.\n"
                f"- **Point d'entrée optimal : 2 à 4 ans** — perte de détention la plus faible. "
                f"Si tu comptes revendre, fais-le **avant que la décote accélère** (~6-7 ans).")
        if lux:
            st.markdown(
                f"- **Le luxe, lui, décroche plus tôt** (~{lux['falaise']:.0f} %/an dès 2-4 ans) : "
                f"la vieille règle « laisser le premier propriétaire payer la chute » s'y applique "
                f"vraiment. Achète-le plutôt vers 4-5 ans, une fois la grosse perte encaissée.")
    else:
        st.info("Pas assez de données par segment pour l'analyse de décote.")


# ===========================================================================
# 2. VUE SAMSAR (achat-revente)
# ===========================================================================

def page_samsar(df_scored, df_deals):
    st.title("🤝 Samsar")
    st.caption("Achat-revente : les affaires chiffrées en dinars, ce qui tourne vite, "
               "et où acheter moins cher.")

    if df_scored is None:
        st.error(f"`{SCORED_PATH}` introuvable — lance `python main.py` d'abord.")
        return

    # ---- Opportunités ----------------------------------------------------
    if df_deals is None:
        st.info("Aucune opportunité détectée sur le dernier scraping — les KPIs et la matrice "
                "ci-dessous apparaîtront dès qu'il y en aura.")
    else:
        deals = df_deals.copy()
        deals["Gain_DT"] = (deals["Prix_Theorique"] - deals["Prix"]).clip(lower=0)

        # ---- Filtres (au-dessus des cartes : elles s'adaptent à la sélection) ----
        colf1, colf2, colf3 = st.columns([2, 1, 1])
        with colf1:
            budget = st.slider(
                "Budget d'achat maximum (DT)",
                min_value=5000, max_value=int(deals["Prix"].max()) + 5000,
                value=min(80000, int(deals["Prix"].max()) + 5000), step=5000,
            )
        with colf2:
            marques_f = st.multiselect("Marques", sorted(deals["Marque"].dropna().unique()))
        with colf3:
            zones_f = st.multiselect(
                "Zone", sorted(deals["Zone_Libelle"].dropna().unique())
                if "Zone_Libelle" in deals.columns else [],
            )

        fiables_seulement = st.checkbox(
            "Estimations solides uniquement (≥ 8 annonces comparables)", value=True,
            help="Le prix théorique d'un modèle presque absent du marché est peu fiable : "
                 "une décote apparente y est le plus souvent une erreur d'estimation, pas une affaire.",
        )

        sel = deals[deals["Prix"] <= budget]
        if fiables_seulement and "Nb_Comparables" in sel.columns:
            sel = sel[sel["Nb_Comparables"] >= 8]
        if marques_f:
            sel = sel[sel["Marque"].isin(marques_f)]
        if zones_f and "Zone_Libelle" in sel.columns:
            sel = sel[sel["Zone_Libelle"].isin(zones_f)]

        # ---- Cartes : reflètent LA SÉLECTION (marché global gardé en contexte) ----
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Opportunités (sélection)", len(sel))
        c2.metric("Gain médian", fmt_dt(sel["Gain_DT"].median()) if len(sel) else "—")
        c3.metric("Gain cumulé", fmt_dt(sel["Gain_DT"].sum()) if len(sel) else "—")
        c4.metric("Décote médiane vs argus",
                  f"{sel['Score_Opportunite'].median():.0%}" if len(sel) else "—")
        if "Nb_Comparables" in deals.columns:
            solides_g = int((deals["Nb_Comparables"] >= 8).sum())
            ctx = f"**{len(deals)}** opportunités détectées, dont **{solides_g}** sur ≥ 8 comparables"
        else:
            ctx = f"**{len(deals)}** opportunités détectées"
        st.caption(f"Les cartes s'adaptent à tes filtres ci-dessus. Marché global : {ctx}.")

        st.divider()

        # ---- Où chasser selon ton capital --------------------------------
        st.subheader("Où chasser selon ton capital")
        st.caption("Le gain absolu monte avec le prix d'achat, mais le rendement (%) et la "
                   "régularité, non. À toi de choisir la tranche selon ton capital et ton appétit.")
        rc = analyse_rendement_capital(deals)
        if len(rc) >= 2:
            fig_rc = go.Figure()
            fig_rc.add_bar(x=rc["tranche"], y=rc["gain"], name="Gain médian (DT)",
                           marker_color=C_ASPHALTE)
            fig_rc.add_scatter(x=rc["tranche"], y=rc["roi"], name="ROI médian (%)", yaxis="y2",
                               mode="lines+markers", line=dict(color=C_GAIN, width=3))
            fig_rc.update_layout(
                title_text="",
                yaxis=dict(title="Gain médian (DT)"),
                yaxis2=dict(title="ROI médian (%)", overlaying="y", side="right", showgrid=False),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
            )
            st.plotly_chart(style_figure(fig_rc, 380), width="stretch")

            tab_rc = pd.DataFrame({
                "Tranche d'achat": rc["tranche"],
                "Gain médian": rc["gain"].map(fmt_dt),
                "ROI médian": rc["roi"].map(lambda x: f"{x:.0f} %"),
                "Régularité": rc["regularite"].map(lambda x: f"{x:.2f}"),
                "Nb affaires": rc["n"],
            })
            st.dataframe(tab_rc, hide_index=True, width="stretch")

            best_roi = rc.loc[rc["roi"].idxmax()]
            best_reg = rc.loc[rc["regularite"].idxmax()]
            worst = rc.loc[rc["roi"].idxmin()]
            st.markdown(
                f"- **Petit capital → maximise le rendement.** La tranche **{best_roi['tranche']}** "
                f"donne le meilleur ROI (**{best_roi['roi']:.0f} %**) : tu fais tourner ton argent vite.\n"
                f"- **Gros capital → maximise l'absolu.** Les tranches hautes rapportent le plus gros "
                f"gain par flip (jusqu'à **{fmt_dt(rc['gain'].max())}**), et la meilleure régularité est "
                f"en **{best_reg['tranche']}** (mais peu d'affaires : {int(best_reg['n'])}).\n"
                f"- **Évite {worst['tranche']}** : le ROI le plus bas (**{worst['roi']:.0f} %**) sans "
                f"régularité qui compense — le piège du milieu de gamme."
            )
            explication(
                "Comment lire « gain », « ROI » et « régularité »",
                "**Ce que ça montre.** Pour chaque tranche de prix d'achat : le gain médian en "
                "dinars, le rendement médian (%), et la *régularité* du rendement.\n\n"
                "**Comment c'est calculé.** Gain = prix théorique − prix affiché ; ROI = gain / prix. "
                "**Régularité = ROI médian ÷ dispersion du ROI** (un Sharpe-*like*).\n\n"
                "**Comment décider.** Gros ROI = capital qui tourne vite (petit budget) ; gros gain "
                "absolu = grosse marge par flip (gros budget) ; forte régularité = affaires "
                "consistantes.\n\n"
                "⚠️ **Ce n'est pas un Sharpe financier.** La dispersion mêle la variété réelle des "
                "affaires et l'erreur du modèle (~10,7 %), et n'intègre pas le risque de revente — "
                "celui-ci viendra quand le suivi des disparitions aura accumulé de l'historique."
            )

        st.divider()

        # ---- Matrice gain × profondeur de marché ------------------------------------
        if len(sel):
            fig = px.scatter(
                sel, x="Score_Opportunite", y="Score_Liquidite", size="Gain_DT",
                color="Score_Liquidite", color_continuous_scale=["#C9CFD6", C_GAIN],
                hover_data={"Titre": True, "Marque": True, "Modèle": True,
                            "Prix": ":,.0f", "Gain_DT": ":,.0f",
                            "Score_Opportunite": ":.0%", "Score_Liquidite": ":.2f"},
                title="Matrice des affaires — en haut à droite : gros gain ET revente facile",
                labels={"Score_Opportunite": "Décote vs prix théorique",
                        "Score_Liquidite": "Profondeur du marché (proxy)"},
            )
            fig.update_layout(coloraxis_showscale=False)
            fig.update_xaxes(tickformat=".0%")
            st.plotly_chart(style_figure(fig, 430), width="stretch")
            explication(
                "Comment lire la matrice des affaires",
                "**Ce que ça montre.** Chaque point est une opportunité. En abscisse, sa décote "
                "(à quel point elle est sous le prix théorique) ; en ordonnée, sa profondeur de marché "
                "(facilité de revente) ; la taille du point = le gain en dinars.\n\n"
                "**Comment c'est calculé.** Décote = 1 − prix/prix_théorique ; profondeur de marché = proxy "
                "basé sur le volume d'annonces du modèle (à affiner avec la vraie vitesse de revente).\n\n"
                "**Comment décider.** Vise **en haut à droite** : grosse décote *et* revente facile. "
                "Les gros points en haut à droite sont les affaires idéales ; un gros point en bas "
                "(gros gain mais peu liquide) peut rester longtemps sur les bras."
            )

            # ---- Tableau -------------------------------------------------
            colonnes = ["Titre", "Marque", "Modèle", "Année", "Kilométrage", "Prix",
                        "Prix_Theorique", "Gain_DT", "Score_Opportunite",
                        "Fiabilite_Estimation", "Nb_Comparables", "Localisation", "Lien"]
            colonnes = [c for c in colonnes if c in sel.columns]
            st.dataframe(
                sel[colonnes].sort_values("Gain_DT", ascending=False),
                width="stretch", hide_index=True,
                column_config={
                    "Prix": st.column_config.NumberColumn("Prix affiché", format="%d DT"),
                    "Prix_Theorique": st.column_config.NumberColumn("Prix théorique", format="%d DT"),
                    "Gain_DT": st.column_config.NumberColumn("Gain potentiel", format="%d DT"),
                    "Score_Opportunite": st.column_config.NumberColumn("Décote", format="percent"),
                    "Kilométrage": st.column_config.NumberColumn("Km", format="%d"),
                    "Année": st.column_config.NumberColumn("Année", format="%d"),
                    "Fiabilite_Estimation": st.column_config.TextColumn("Fiabilité"),
                    "Nb_Comparables": st.column_config.NumberColumn("Comparables", format="%d"),
                    "Lien": st.column_config.LinkColumn("Annonce", display_text="Ouvrir ↗"),
                },
            )

            # ---- Pourquoi cette affaire ? (SHAP local) -------------------
            sel_shap = sel.sort_values("Gain_DT", ascending=False).reset_index(drop=True)

            def _label_deal(r):
                an = "" if pd.isna(r.get("Année")) else f" {int(r['Année'])}"
                return f"{r['Marque']} {r['Modèle']}{an} — {r['Prix']:,.0f} DT".replace(",", " ")

            sel_shap["_label"] = sel_shap.apply(_label_deal, axis=1)
            choix = st.selectbox("🔍 Analyser une affaire — pourquoi le modèle l'estime plus chère",
                                 sel_shap["_label"].tolist())
            ligne = sel_shap[sel_shap["_label"] == choix].iloc[0]

            bundle_s = charger_modele()
            contrib = expliquer_prix(bundle_s, ligne.to_dict(), df_scored) if bundle_s is not None else None
            if contrib is not None and len(contrib):
                decote = ligne.get("Score_Opportunite")
                st.caption(
                    f"Estimée à **{ligne['Prix_Theorique']:,.0f} DT**, affichée à "
                    f"**{ligne['Prix']:,.0f} DT** → décote de **{decote:.0%}**. "
                    "Voici ce qui, selon le modèle, justifie sa valeur estimée :".replace(",", " ")
                )
                top = contrib.head(6).iloc[::-1]
                fig = go.Figure(go.Bar(
                    x=top["shap"], y=top["label"], orientation="h",
                    marker_color=[C_GAIN if v > 0 else C_ALERTE for v in top["shap"]],
                    hovertemplate="%{y}<extra></extra>", showlegend=False))
                fig.update_layout(title="Ce qui valorise (vert) ou dévalorise (rouge) cette voiture")
                fig.update_xaxes(title="contribution (échelle log-prix)")
                st.plotly_chart(style_figure(fig, 300), width="stretch")
            elif bundle_s is None:
                st.caption("Modèle indisponible — l'explication par annonce nécessite le modèle chargé.")
            else:
                st.caption("Explication indisponible pour cette annonce.")

    st.divider()

    # ---- Rotation : qu'est-ce qui tourne vite ? --------------------------
    st.subheader("Rotation du marché — qu'est-ce qui part vite ?")
    st.caption("Volume d'annonces = demande et facilité de revente. L'âge des annonces encore "
               "en ligne sert de proxy d'écoulement — mais uniquement sur tayara.tn, pour la "
               "raison expliquée sous le second graphique.")

    # L'âge des annonces n'est comparable QU'À SOURCE ÉGALE. automobile.tn ne
    # conserve pas d'annonces anciennes : la corrélation entre "part de pros"
    # d'un modèle et l'âge médian de ses annonces est de -0,43. Mélanger les
    # sources faisait apparaître Audi A5, BMW Série 5 ou Range Rover Sport
    # comme les modèles "les plus rapides" -- tous à 100 % de pros. On ne
    # mesurait pas la vitesse d'écoulement mais le site d'origine.
    base_rotation = df_scored[
        df_scored["Source"].astype(str).str.lower().str.contains("tayara", na=False)
    ]

    rotation = (
        base_rotation.dropna(subset=["Marque", "Modèle"])
        .groupby(["Marque", "Modèle"])
        .agg(volume=("Prix", "count"),
             age_annonce_median=("Age_Annonce_Jours", "median"),
             prix_median=("Prix", "median"))
        .reset_index()
    )
    rotation = rotation[rotation["volume"] >= 8]
    rotation["libelle"] = rotation["Marque"] + " " + rotation["Modèle"].astype(str)

    if len(rotation):
        col_r1, col_r2 = st.columns(2)
        with col_r1:
            top_vol = rotation.sort_values("volume").tail(12)
            fig = px.bar(
                top_vol, x="volume", y="libelle", orientation="h",
                title="Modèles les plus présents sur le marché",
                labels={"volume": "Annonces actives", "libelle": ""},
                custom_data=["prix_median"],
            )
            fig.update_traces(
                marker_color=C_ASPHALTE,
                hovertemplate="%{y} : %{x} annonces<br>Prix médian : %{customdata[0]:,.0f} DT<extra></extra>",
            )
            st.plotly_chart(style_figure(fig, 420), width="stretch")
        with col_r2:
            rapides = rotation.sort_values("age_annonce_median").head(12).sort_values(
                "age_annonce_median", ascending=False)
            fig = px.bar(
                rapides, x="age_annonce_median", y="libelle", orientation="h",
                title="Écoulement le plus rapide (tayara.tn uniquement)",
                labels={"age_annonce_median": "Âge médian des annonces (jours)", "libelle": ""},
                custom_data=["volume", "prix_median"],
            )
            fig.update_traces(
                marker_color=C_GAIN,
                hovertemplate="%{y} : %{x:.0f} j (n=%{customdata[0]})"
                              "<br>Prix médian : %{customdata[1]:,.0f} DT<extra></extra>",
            )
            st.plotly_chart(style_figure(fig, 420), width="stretch")

        with st.expander("ℹ️ Pourquoi seulement tayara.tn ici"):
            st.write(
                """
L'âge des annonces encore en ligne n'est comparable qu'à **source égale**.
automobile.tn ne conserve pas d'annonces anciennes : sur les données réelles,
la corrélation entre la part de professionnels d'un modèle et l'âge médian de
ses annonces est de **−0,43**. En mélangeant les sources, les modèles
« les plus rapides » étaient l'Audi A5 Sportback, la BMW Série 5, le X3 et le
Range Rover Sport — tous à **100 % de pros**. Ce n'était pas de la vitesse
d'écoulement, c'était le site d'origine.

Restreindre à tayara.tn rend la comparaison honnête, au prix d'un périmètre
plus étroit. Cette mesure indirecte disparaîtra dès que le suivi des annonces
(`core/suivi_annonces.py`) aura accumulé quelques semaines : il donnera la
durée réelle entre publication et disparition, sans proxy.
"""
            )

    # ---- Fenêtre d'achat samsar : cheap mais revendable ------------------
    st.divider()
    st.subheader("Fenêtre d'achat samsar — cheap, mais qui se revend")
    st.caption("Le compromis achat-revente : assez vieux pour acheter bas (falaise passée), "
               "assez demandé pour ne pas rester en stock.")
    infos_s = {s: analyse_decote_segment(df_scored, s) for s in ["Généraliste", "Segment luxe"]}
    infos_s = {s: v for s, v in infos_s.items() if v}
    if infos_s:
        ref = infos_s.get("Généraliste") or next(iter(infos_s.values()))
        liq = ref["ages_liquides"]
        if liq:
            a_min, a_max = min(liq), max(liq)
            st.markdown(
                f"- **Vise la zone {a_min}-{a_max} ans (généraliste).** C'est là que le marché a le "
                f"plus d'annonces (**{ref['n_liquide']} sur ces âges**) : le plus d'acheteurs, donc "
                f"**revente rapide et risque d'invendu minimal**.\n"
                f"- **La prime du neuf est déjà encaissée** : la grosse perte de la 1re année a "
                f"été payée par le vendeur précédent — tu achètes bas, sans la subir.\n"
                f"- **Faible perte pendant la détention** (~{ref['plateau']:.0f} %/an) : ta marge n'est "
                f"pas grignotée si la voiture reste quelques semaines en vitrine.\n"
                f"- **⚠️ Le piège du très vieux (> 12 ans)** : encore moins cher, mais beaucoup moins "
                f"d'acheteurs. Une voiture cheap qui ne se revend pas immobilise ton cash — ça coûte "
                f"plus qu'une marge plus fine qui tourne vite."
            )
        else:
            st.info("Pas assez d'annonces par âge pour cerner la fenêtre liquide.")
    else:
        st.info("Pas assez de données par segment pour la stratégie samsar.")

    # ---- Arbitrage géographique -----------------------------------------
    st.subheader("Arbitrage géographique — où acheter, où revendre")
    st.caption("Dans quelle région le prix est-il le plus bas / le plus haut, **à âge et "
               "kilométrage comparables** ? Deux angles : **par modèle** (précis mais peu "
               "d'annonces) ou **par segment** (robuste statistiquement, composition plus grossière).")

    @st.cache_data
    def calculer_arbitrage_geo(df_):
        """Régression log(prix) ~ âge + km + région par modèle.
        Le coefficient région donne l'écart une fois l'âge et le km neutralisés.
        Garde-fous : ≥ 3 annonces par région, ≥ 12 annonces au total,
        recouvrement d'âge ≥ 2 ans entre toutes les paires de régions."""
        rows = []
        base = df_.dropna(subset=["Prix", "Age_Vehicule", "Kilométrage", "Localisation"])
        for (m, mo), g in base.groupby(["Marque", "Modèle"]):
            regs_ok = g.groupby("Localisation").filter(lambda x: len(x) >= 3)["Localisation"].value_counts()
            regs_ok = regs_ok[regs_ok >= 3].index.tolist()
            g2 = g[g["Localisation"].isin(regs_ok)]
            if g2["Localisation"].nunique() < 2 or len(g2) < 12:
                continue
            ar = g2.groupby("Localisation")["Age_Vehicule"].agg(["min", "max"])
            ok = all(
                min(r1["max"], r2["max"]) - max(r1["min"], r2["min"]) >= 2
                for i, r1 in ar.iterrows() for j, r2 in ar.iterrows() if i < j
            )
            if not ok:
                continue
            X = pd.get_dummies(g2[["Localisation"]], drop_first=True).astype(float)
            X["age"] = g2["Age_Vehicule"]
            X["km"] = g2["Kilométrage"] / 10000
            X.insert(0, "const", 1.0)
            try:
                c, *_ = np.linalg.lstsq(X.values, np.log(g2["Prix"].values), rcond=None)
            except np.linalg.LinAlgError:
                continue
            coefs = {col.replace("Localisation_", ""): (np.exp(c[list(X.columns).index(col)]) - 1) * 100
                     for col in X.columns if col.startswith("Localisation_")}
            ref = [r for r in regs_ok if r not in coefs]
            if ref:
                coefs[ref[0]] = 0.0
            if len(coefs) < 2:
                continue
            best = max(coefs, key=coefs.get)
            worst = min(coefs, key=coefs.get)
            ecart = coefs[best] - coefs[worst]
            prix_base = float(np.exp(
                c[0]
                + c[list(X.columns).index("age")] * g2["Age_Vehicule"].median()
                + c[list(X.columns).index("km")] * g2["Kilométrage"].median() / 10000
            ))
            n_ach = int((g2["Localisation"] == worst).sum())
            n_rev = int((g2["Localisation"] == best).sum())
            rows.append({
                "libelle": f"{m} {mo}", "Marque": m, "Modèle": str(mo),
                "n": len(g2), "n_regs": g2["Localisation"].nunique(),
                "acheter": worst, "revendre": best,
                "n_acheter": n_ach, "n_revendre": n_rev,
                "ecart_pct": round(ecart, 1),
                "ecart_dt": int(prix_base * ecart / 100),
                "prix_achat_est": int(prix_base),
                "coefs": coefs,
            })
        return pd.DataFrame(rows).sort_values("ecart_pct", ascending=False) if rows else pd.DataFrame()

    @st.cache_data
    def calculer_arbitrage_geo_segment(df_, seg_val):
        """Même régression log(prix) ~ âge + km + gouvernorat, mais poolée sur TOUT
        un segment (au lieu d'un seul modèle). Beaucoup plus d'annonces par région
        -> plus robuste ; en contrepartie la composition modèle n'est pas contrôlée
        (âge et km le sont). Localisation agrégée au gouvernorat."""
        base = df_.dropna(subset=["Prix", "Age_Vehicule", "Kilométrage", "Localisation"]).copy()
        base = base[base["Segment_Vehicule"] == seg_val]
        base["gouv"] = base["Localisation"].map(_match_gouvernorat)
        base = base.dropna(subset=["gouv"])
        vc = base["gouv"].value_counts()
        regs = vc[vc >= 15].index.tolist()
        base = base[base["gouv"].isin(regs)]
        if base["gouv"].nunique() < 2 or len(base) < 40:
            return None
        X = pd.get_dummies(base[["gouv"]], drop_first=True).astype(float)
        X["age"] = base["Age_Vehicule"]
        X["km"] = base["Kilométrage"] / 10000
        X.insert(0, "const", 1.0)
        try:
            c, *_ = np.linalg.lstsq(X.values, np.log(base["Prix"].values), rcond=None)
        except np.linalg.LinAlgError:
            return None
        coefs = {col.replace("gouv_", ""): (np.exp(c[list(X.columns).index(col)]) - 1) * 100
                 for col in X.columns if col.startswith("gouv_")}
        ref = [r for r in regs if r not in coefs]
        if ref:
            coefs[ref[0]] = 0.0
        best, worst = max(coefs, key=coefs.get), min(coefs, key=coefs.get)
        prix_base = float(np.exp(
            c[0] + c[list(X.columns).index("age")] * base["Age_Vehicule"].median()
            + c[list(X.columns).index("km")] * base["Kilométrage"].median() / 10000))
        ecart = coefs[best] - coefs[worst]
        return {
            "coefs": coefs, "acheter": worst, "revendre": best, "n": len(base),
            "n_acheter": int((base["gouv"] == worst).sum()),
            "n_revendre": int((base["gouv"] == best).sum()), "n_regs": len(regs),
            "ecart_pct": round(ecart, 1), "ecart_dt": int(prix_base * ecart / 100),
        }

    def _rendre_arbitrage(titre, coefs, acheter, revendre, n_ach, n_rev, n_tot, ecart_pct, ecart_dt):
        """Carte d'action à confiance graduée + graphe, commune aux deux vues."""
        min_reg = min(n_ach, n_rev)
        if min_reg >= 8 and n_tot >= 30 and ecart_pct >= 15:
            emoji, niveau, action = "🟢", "Forte", (
                f"**Play net.** Achète à **{acheter}**, revends à **{revendre}**. L'écart est "
                f"solide et repose sur assez d'annonces des deux côtés — il dépasse largement le "
                f"bruit. Il te reste à couvrir transport + mutation avec la marge.")
        elif min_reg >= 5 and n_tot >= 20 and ecart_pct >= 8:
            emoji, niveau, action = "🟡", "Moyenne", (
                f"**Écart réel, à confirmer.** Achète à **{acheter}**, revends à **{revendre}** — "
                f"mais la marge est plus fine : vérifie l'état et chiffre les frais (mutation, "
                f"transport) avant de t'engager.")
        else:
            emoji, niveau, action = "🔴", "Faible", (
                f"**Signal indicatif seulement** ({n_ach} annonce(s) à l'achat, {n_rev} à la "
                f"revente). Traite-le comme une piste à confirmer à la main, pas une certitude.")
        m1, m2, m3 = st.columns(3)
        m1.metric("Profit potentiel (brut)", f"{ecart_dt:,} DT".replace(",", " "))
        m2.metric("Profit %", f"+{ecart_pct:.0f} %")
        m3.metric("Confiance", f"{emoji} {niveau}")
        st.info(action)
        st.caption(f"À âge et km comparables, sur {n_tot} annonces ({n_ach} à {acheter}, "
                   f"{n_rev} à {revendre}). **Profit brut, avant frais** de transport et de "
                   "mutation — un plafond théorique, pas un net dans ta poche.")
        coefs_df = pd.DataFrame({"region": list(coefs.keys()),
                                 "prime_pct": list(coefs.values())}).sort_values("prime_pct")
        fig = px.bar(coefs_df, x="prime_pct", y="region", orientation="h",
                     title=f"{titre} — prime de prix par région, à âge et km comparables",
                     labels={"prime_pct": "Prime vs référence (%)", "region": ""})
        couleurs = [C_GAIN if i == 0 else (C_ALERTE if i == len(coefs_df) - 1 else C_ASPHALTE)
                    for i in range(len(coefs_df))]
        fig.update_traces(marker_color=couleurs, hovertemplate="%{y} : %{x:+.1f} %<extra></extra>")
        fig.add_vline(x=0, line_color=C_GRIS, line_width=1)
        st.plotly_chart(style_figure(fig, 340), width="stretch")

    vue_arb = st.radio("Vue", ["Par modèle (précis)", "Par segment (robuste)"], horizontal=True)
    if vue_arb.startswith("Par modèle"):
        arb = calculer_arbitrage_geo(df_scored)
        if len(arb):
            modele_choisi = st.selectbox("Modèle à analyser", arb["libelle"].tolist())
            l = arb[arb["libelle"] == modele_choisi].iloc[0]
            _rendre_arbitrage(modele_choisi, l["coefs"], l["acheter"], l["revendre"],
                              l["n_acheter"], l["n_revendre"], l["n"], l["ecart_pct"], l["ecart_dt"])
        else:
            st.info("Pas assez de données par modèle pour un arbitrage fiable.")
    else:
        seg_label = st.selectbox("Segment", ["Généraliste", "Segment luxe"])
        res = calculer_arbitrage_geo_segment(df_scored, 1 if "luxe" in seg_label.lower() else 0)
        if res:
            _rendre_arbitrage(seg_label, res["coefs"], res["acheter"], res["revendre"],
                              res["n_acheter"], res["n_revendre"], res["n"], res["ecart_pct"], res["ecart_dt"])
        else:
            st.info("Pas assez de données pour ce segment.")

    with st.expander("ℹ️ Pourquoi « à âge et km comparables » change tout"):
        st.write(
            """
Comparer les prix médians *bruts* par région est trompeur : une région peut
sembler moins chère simplement parce que ses voitures y sont plus vieilles ou
plus kilométrées. Exemple observé : sur l'Audi A3 Sportback, un écart Sfax/Nabeul
venait surtout de l'âge (2 ans en moyenne vs 4 ans), pas de la région.

Ici, une régression `log(prix) ~ âge + kilométrage + région` isole l'effet région
une fois l'âge et le km neutralisés. La barre à zéro est la région de référence ;
les autres se lisent comme des écarts par rapport à elle.

- **Vue par modèle** — garde-fous : ≥ 3 annonces/région, ≥ 12 au total, recouvrement
  d'âge ≥ 2 ans entre régions. Précis (même modèle), mais souvent peu d'annonces →
  confiance basse.
- **Vue par segment** — même régression poolée sur tout un segment (≥ 15 annonces
  par gouvernorat). Beaucoup plus robuste statistiquement, au prix d'un contrôle de
  composition plus grossier (les modèles d'un même segment sont mélangés).
"""
        )


# ===========================================================================
# 3. CALCULATEUR DE JUSTE PRIX
# ===========================================================================

def page_calculateur(df, bundle):
    st.title("💰 Calculateur de Juste Prix")
    st.caption(f"Estimation par le modèle entraîné sur les annonces récentes (≤ {MAX_DAYS_OLD} jours), "
               "confrontée à des annonces réellement proches (âge, km, énergie).")

    if df is None:
        st.error(
            f"`{SCORED_PATH}` introuvable ou illisible. Lance `python main.py`, "
            "puis pousse `data/` sur GitHub (l'app lit raw.githubusercontent en premier)."
        )
        return
    if bundle is None:
        _, raison = _tenter_charger_modele()
        st.warning("Le modèle n'a pas pu être chargé.")
        if raison:
            st.code(raison, language="text")
        st.caption(
            "Causes fréquentes : (1) version de scikit-learn différente entre "
            "l'entraînement du `.pkl` et l'app (voir `requirements.txt`) — "
            "réentraîne avec `python core/modele_prediction.py` dans le même "
            "environnement, puis `git push` ; (2) modèle pas encore poussé sur GitHub."
        )
        return

    col1, col2, col3 = st.columns(3)
    with col1:
        marques = sorted(df["Marque"].dropna().unique())
        marque = st.selectbox("Marque", marques, index=marques.index("Volkswagen") if "Volkswagen" in marques else 0)
        modeles_m = sorted(df[df["Marque"] == marque]["Modèle"].dropna().astype(str).unique())
        modele = st.selectbox("Modèle", modeles_m) if modeles_m else st.text_input("Modèle")
        annee = st.number_input("Année", min_value=1990, max_value=pd.Timestamp.now().year + 1, value=2020)
    with col2:
        km = st.number_input("Kilométrage", min_value=0, max_value=500000, value=80000, step=5000)
        cv = st.number_input("Puissance fiscale (CV)", min_value=2, max_value=30, value=7)
        zone = st.selectbox("Zone", ["Grand Tunis", "Province"])
    with col3:
        energies = sorted(df["Energie"].dropna().unique())
        energie = st.selectbox("Énergie", energies)
        boites = sorted(df["Boite_Vitesse"].dropna().astype(str).unique()) if "Boite_Vitesse" in df else []
        boite = st.selectbox("Boîte de vitesses", boites) if boites else None
        # Cylindrée optionnelle : renseignée -> utilisée par le modèle ;
        # laissée à 0 -> transmise en NaN. Réduit l'écart de précision entre ce
        # calculateur et le scoring automatique du pipeline.
        cyl = st.number_input("Cylindrée (L) — optionnel", min_value=0.0, max_value=8.0, value=0.0, step=0.1,
                              help="0 = inconnu. Ex : 1.6. Améliore l'estimation si renseigné.")

    segment = "Luxe" if marque in MARQUES_LUXE else "Standard"
    annee_courante = pd.Timestamp.now().year
    presque_neuve = annee >= annee_courante - 1
    st.caption(f"Segment déduit : **{segment}** — Presque neuve : **{'Oui' if presque_neuve else 'Non'}**")

    if st.button("Calculer le prix théorique", type="primary"):
        colonnes = bundle.get("features_numeriques", []) + bundle.get("features_categorielles", [])
        saisie = {
            "Kilométrage": km, "Log_Kilometrage": float(np.log1p(km)),
            "Age_Vehicule": max(annee_courante - annee, 0),
            "Age_Carre": float(max(annee_courante - annee, 0) ** 2),
            "Km_Par_An": float(min(km / max(max(annee_courante - annee, 0), 1), 100000)),
            "Puissance_Fiscale": cv,
            "Segment_Vehicule": int(segment == "Luxe"),
            "Zone_Economique": int(zone == "Grand Tunis"),
            "Marque": marque, "Modèle": modele, "Energie": energie,
            # Cylindrée (optionnelle) : NaN si laissée à 0, sinon la valeur saisie.
            "Cylindree": np.nan if cyl == 0 else float(cyl),
            # Boîte de vitesses harmonisée entre les cinq sources.
            "Boite_Vitesse": boite,
        }
        X = _ligne_modele(bundle, saisie)
        try:
            prix_log = bundle["pipeline"].predict(X)[0]
            prix_theorique = float(np.expm1(prix_log))
        except Exception as e:
            st.error(f"Erreur lors de la prédiction : {str(e)[:120]}")
            st.info("Relance `python core/modele_prediction.py` pour réentraîner le modèle "
                    "avec ta version de scikit-learn.")
            return

        if np.isnan(prix_theorique) or prix_theorique <= 0:
            st.error("Le modèle n'a pas pu produire une estimation cohérente pour cette combinaison.")
            return

        # ---- Double valorisation indépendante ---------------------------
        # 1) ML Valuation : prédiction du modèle + intervalle issu de son erreur historique.
        # 2) Market Comparable Valuation : médiane + Q25/Q75 de véhicules comparables,
        #    calculées SANS utiliser la prédiction ML.
        target_market = {
            "Marque": marque, "Modèle": modele, "Année": annee,
            "Age_Vehicule": max(annee_courante - annee, 0),
            "Kilométrage": km, "Energie": energie,
            "Boite_Vitesse": boite,
        }
        market, comparables = market_valuation(df, target_market, min_n=5)

        gr_fia = table_fiabilite_prix(df)
        err_pct = erreur_pour_prix(gr_fia, prix_theorique)
        ml_bas = ml_haut = None
        if err_pct is not None:
            ml_bas = prix_theorique * (1 - err_pct / 100)
            ml_haut = prix_theorique * (1 + err_pct / 100)

        st.markdown("### Deux avis indépendants sur la valeur")
        r1, r2, r3 = st.columns(3)
        r1.metric("🤖 Estimation ML", fmt_dt(prix_theorique))
        if ml_bas is not None:
            r1.caption(f"Intervalle modèle : {fmt_dt(ml_bas)} – {fmt_dt(ml_haut)} (±{err_pct:.0f} %)")
        else:
            r1.caption("Intervalle modèle indisponible")

        if market.n_comparables >= 5:
            r2.metric("📊 Médiane marché comparable", fmt_dt(market.median_price))
            r2.caption(f"Q25–Q75 : {fmt_dt(market.q25)} – {fmt_dt(market.q75)}")
            r3.metric("Comparables", market.n_comparables)
            width_pct = 100 * market.relative_width if market.relative_width is not None else np.nan
            r3.caption(f"Voisinage {market.selection_level} · homogénéité {market.homogeneity} · largeur {width_pct:.1f} %")

            agreement = compare_ml_to_market(prix_theorique, market)
            gap_pct = 100 * agreement["gap_vs_market_median_pct"]
            if agreement["inside_market_range"]:
                st.success(f"✅ L'estimation ML est **dans la fourchette centrale du marché**. "
                           f"Écart à la médiane : {gap_pct:+.1f} %.")
            elif agreement.get("inside_market_range_p10_p90"):
                st.warning(f"🟡 L'estimation ML sort du cœur Q25–Q75 mais reste dans le marché élargi P10–P90. "
                           f"Écart à la médiane : {gap_pct:+.1f} %.")
            else:
                st.error(f"⚠️ L'estimation ML sort aussi du marché élargi P10–P90. "
                         f"Écart à la médiane : {gap_pct:+.1f} %. Prudence sur cette combinaison.")

            # Score de confiance distinct du potentiel de bonne affaire.
            pseudo_row = dict(target_market)
            pseudo_row["Erreur_Relative_Modele"] = (err_pct / 100) if err_pct is not None else np.nan
            conf_score, conf_label, conf_parts = valuation_confidence(pseudo_row, market, agreement)
            st.markdown("#### Confiance de la valorisation")
            st.progress(conf_score / 100.0, text=f"{conf_score}/100 — {conf_label}")
            cc1, cc2, cc3, cc4 = st.columns(4)
            cc1.metric("Historique ML", f"{conf_parts['Historique ML']}/100")
            cc2.metric("Comparables", f"{conf_parts['Comparables']}/100")
            cc3.metric("Homogénéité", f"{conf_parts['Homogénéité']}/100")
            cc4.metric("Accord ML ↔ marché", f"{conf_parts['Accord ML ↔ marché']}/100")

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=[market.q25, market.q75], y=[1, 1], mode="lines+markers",
                name="Marché Q25–Q75", line=dict(width=12),
                hovertemplate="%{x:,.0f} DT<extra>Fourchette marché</extra>",
            ))
            fig.add_trace(go.Scatter(
                x=[market.median_price], y=[1], mode="markers", name="Médiane marché",
                marker=dict(size=15, symbol="diamond"),
                hovertemplate="%{x:,.0f} DT<extra>Médiane marché</extra>",
            ))
            fig.add_trace(go.Scatter(
                x=[prix_theorique], y=[1.08], mode="markers", name="Estimation ML",
                marker=dict(size=17, symbol="triangle-down"),
                hovertemplate="%{x:,.0f} DT<extra>Estimation ML</extra>",
            ))
            fig.update_yaxes(visible=False, range=[0.78, 1.22])
            fig.update_xaxes(title="Prix (DT)")
            fig.update_layout(title=f"ML vs marché — {market.n_comparables} comparables réels",
                              height=270, legend=dict(orientation="h"))
            st.plotly_chart(style_figure(fig, 270), width="stretch")

            with st.expander(f"Voir les {market.n_comparables} comparables utilisés"):
                cols_comp = [c for c in ["Marque", "Modèle", "Année", "Kilométrage", "Energie",
                                                 "Boite_Vitesse", "Prix", "Localisation", "Lien"]
                             if c in comparables.columns]
                st.dataframe(comparables[cols_comp], hide_index=True, width="stretch",
                             column_config={"Prix": st.column_config.NumberColumn(format="%d DT"),
                                            "Lien": st.column_config.LinkColumn("Annonce", display_text="ouvrir")})
                st.caption("La voiture saisie n'entre pas dans le calcul. La fourchette est empirique : Q25–Q75 des prix observés.")
        else:
            r2.metric("📊 Marché comparable", "—")
            r3.metric("Comparables", market.n_comparables)
            st.info("Moins de 5 annonces suffisamment comparables : AutoDeal affiche l'estimation ML, "
                    "mais ne fabrique pas artificiellement une fourchette de marché.")

        # ---- Pourquoi ce prix ? (décomposition SHAP ou repli) ------------
        contrib = expliquer_prix(bundle, saisie, df)
        if contrib is not None and len(contrib):
            methode = contrib.attrs.get("methode", "SHAP")
            st.markdown("**Pourquoi ce prix ?**")
            top = contrib.head(6).iloc[::-1]  # plus fort en haut du graphe horizontal
            fig = go.Figure(go.Bar(
                x=top["shap"], y=top["label"], orientation="h",
                marker_color=[C_GAIN if v > 0 else C_ALERTE for v in top["shap"]],
                hovertemplate="%{y}<extra></extra>", showlegend=False,
            ))
            fig.update_layout(title="Ce qui pousse le prix vers le haut (vert) ou le bas (rouge)")
            fig.update_xaxes(title="contribution (échelle log-prix)")
            st.plotly_chart(style_figure(fig, 300), width="stretch")
            methode_txt = ("Décomposition SHAP de cette estimation" if methode == "SHAP"
                           else "Effet marginal de chaque variable (approximation, SHAP indisponible)")
            st.caption(f"{methode_txt} : vert = tire le prix vers le haut, rouge = vers le bas. "
                       "Échelle logarithmique (le modèle prédit le log du prix).")

        st.caption("💡 Une annonce réelle nettement sous ce prix ? Regarde la page Samsar : "
                   "elle y est probablement déjà signalée.")


# ===========================================================================
# 4. ADMIN — diagnostics du modèle et de la donnée
# ===========================================================================

DIAG_PATH = "data/processed/diagnostics_modele.json"
CALIB_PATH = "data/processed/calibration_fenetre.json"
SHAP_PATH = "data/processed/shap_importance.json"
HISTO_PATH = "data/processed/historique_performance.json"


def charger_json(chemin):
    return lire_json_distant(chemin)


def page_admin(df, df_deals):
    st.title("🛠️ Admin")
    st.caption("Diagnostics du modèle et de la donnée. Rien ici n'est destiné à un utilisateur "
               "final : c'est de quoi juger si les chiffres affichés ailleurs méritent confiance.")

    if df is None:
        st.error(f"`{SCORED_PATH}` introuvable — lance `python main.py` d'abord.")
        return

    diag = charger_json(DIAG_PATH)

    # ---- Modèles comparés ------------------------------------------------
    st.subheader("Comparaison des modèles")
    if diag:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Modèle retenu", diag["modele_retenu"].split(" (")[0])
        c2.metric("Annonces d'entraînement", f"{diag['n_annonces']:,}".replace(",", " "))
        c3.metric("Fenêtre de fraîcheur", f"{diag['fenetre_jours']} j")
        c4.metric("Validation croisée", f"{diag['n_folds']} folds")

        cand = pd.DataFrame(diag["candidats"]).sort_values("mdape_pct")
        cand["retenu"] = cand["nom"] == diag["modele_retenu"]

        fig = px.bar(
            cand.sort_values("mdape_pct", ascending=False),
            x="mdape_pct", y="nom", orientation="h",
            title="Erreur relative médiane par modèle candidat (out-of-fold)",
            labels={"mdape_pct": "Erreur relative médiane (%)", "nom": ""},
            custom_data=["mdae_dt", "mae_dt"],
        )
        fig.update_traces(
            marker_color=[C_GAIN if r else C_GRIS
                          for r in cand.sort_values("mdape_pct", ascending=False)["retenu"]],
            hovertemplate="%{y}<br>Erreur relative médiane : %{x:.2f} %"
                          "<br>Erreur absolue médiane : %{customdata[0]:,.0f} DT"
                          "<br>MAE : %{customdata[1]:,.0f} DT<extra></extra>",
        )
        st.plotly_chart(style_figure(fig, 280), width="stretch")

        st.dataframe(
            cand[["nom", "mdape_pct", "mdae_dt", "mae_dt"]].rename(columns={
                "nom": "Modèle", "mdape_pct": "Erreur relative médiane (%)",
                "mdae_dt": "Erreur absolue médiane (DT)", "mae_dt": "MAE (DT)"}),
            width="stretch", hide_index=True,
        )
        st.caption(f"Entraînement du {diag['date_entrainement'].replace('T', ' à ')}. "
                   "La sélection se fait sur l'erreur relative médiane : `Score_Opportunite` "
                   "étant un écart relatif, la métrique de sélection doit l'être aussi. "
                   "Le MAE est affiché pour information — il est dominé par le haut de gamme.")

        robuste = diag.get("validation_robuste", {})
        if robuste:
            st.markdown("**Stress-tests de généralisation**")
            lignes = []
            g = robuste.get("groupkfold_marque_modele")
            if g:
                lignes.append({"Test": "GroupKFold marque-modèle", "MdAPE (%)": round(g["mdape"] * 100, 2), "Niveau": "Familles jamais vues"})
            t = robuste.get("time_holdout_20pct")
            if t:
                lignes.append({"Test": "Holdout temporel 20%", "MdAPE (%)": round(t["mdape"] * 100, 2), "Niveau": "Annonces les plus récentes"})
            for src in robuste.get("source_holdout", []):
                lignes.append({"Test": f"Source holdout — {src['source']}", "MdAPE (%)": round(src["mdape"] * 100, 2), "Niveau": f"n={src['n_test']}"})
            if lignes:
                st.dataframe(pd.DataFrame(lignes), width="stretch", hide_index=True)
            z = diag.get("diagnostic_zone_economique")
            if z:
                st.caption(
                    f"Biais géographique : sans Zone_Economique = {z['production_sans_zone_mdape_pct']:.2f}% ; "
                    f"avec zone = {z['avec_zone_mdape_pct']:.2f}% (gain {z['gain_zone_points']:.2f} pt). "
                    "La zone reste exclue du modèle tant que ce gain n'est pas stable hors-source."
                )
    else:
        st.warning(f"`{DIAG_PATH}` introuvable — relance `python core/modele_prediction.py` "
                   "pour générer les diagnostics.")

    st.divider()

    # ---- Erreur par gamme de prix ---------------------------------------
    st.subheader("Où le modèle se trompe")
    df_e = df.dropna(subset=["Prix", "Prix_Theorique"]).copy()
    df_e["err"] = (df_e["Prix"] - df_e["Prix_Theorique"]).abs()
    df_e["err_rel"] = df_e["err"] / df_e["Prix"]
    df_e["gamme"] = pd.cut(
        df_e["Prix"], bins=[0, 30000, 50000, 80000, 120000, 200000, np.inf],
        labels=["< 30k", "30–50k", "50–80k", "80–120k", "120–200k", "> 200k"],
    )
    par_gamme = df_e.groupby("gamme", observed=True).agg(
        n=("err", "size"), err_rel=("err_rel", "median"), err_dt=("err", "median"),
        mae=("err", "mean")).reset_index()

    col_a, col_b = st.columns(2)
    with col_a:
        fig = px.bar(par_gamme, x="gamme", y="err_rel",
                     title="Erreur relative médiane par gamme de prix",
                     labels={"gamme": "", "err_rel": "Erreur relative médiane"},
                     custom_data=["n", "err_dt"])
        fig.update_traces(marker_color=C_ASPHALTE,
                          hovertemplate="%{x} : %{y:.1%}<br>n=%{customdata[0]} — "
                                        "médiane %{customdata[1]:,.0f} DT<extra></extra>")
        fig.update_yaxes(tickformat=".0%")
        st.plotly_chart(style_figure(fig, 340), width="stretch")
    with col_b:
        fig = px.bar(par_gamme, x="gamme", y="mae",
                     title="Erreur absolue moyenne (MAE) par gamme",
                     labels={"gamme": "", "mae": "MAE (DT)"}, custom_data=["n"])
        fig.update_traces(marker_color=C_ALERTE,
                          hovertemplate="%{x} : %{y:,.0f} DT (n=%{customdata[0]})<extra></extra>")
        st.plotly_chart(style_figure(fig, 340), width="stretch")

    st.caption("Le contraste entre ces deux graphiques explique pourquoi la sélection ne se fait "
               "pas sur le MAE : en relatif l'erreur est à peu près stable, en dinars elle explose "
               "sur le haut de gamme. Optimiser le MAE reviendrait à optimiser pour les voitures "
               "de luxe, qui sont une petite minorité des annonces.")

    # ---- Pires modèles ---------------------------------------------------
    pires = (df_e.groupby(["Marque", "Modèle"])
             .agg(n=("err_rel", "size"), err_rel=("err_rel", "median"))
             .reset_index())
    pires = pires[pires["n"] >= 10].sort_values("err_rel", ascending=False).head(12)
    if len(pires):
        pires["libelle"] = pires["Marque"] + " " + pires["Modèle"].astype(str)
        fig = px.bar(pires.sort_values("err_rel"), x="err_rel", y="libelle", orientation="h",
                     title="Modèles les moins bien estimés (≥ 10 annonces)",
                     labels={"err_rel": "Erreur relative médiane", "libelle": ""},
                     custom_data=["n"])
        fig.update_traces(marker_color=C_ALERTE,
                          hovertemplate="%{y} : %{x:.1%} (n=%{customdata[0]})<extra></extra>")
        fig.update_xaxes(tickformat=".0%")
        st.plotly_chart(style_figure(fig, 400), width="stretch")

    st.divider()

    # ---- Calibration de la fenêtre --------------------------------------
    st.subheader("Calibration de la fenêtre de fraîcheur")
    calib = charger_json(CALIB_PATH)
    if calib:
        c = pd.DataFrame(calib["resultats"])
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=c["fenetre"], y=c["err_rel_moy"] + c["ecart_type"],
            line=dict(width=0), showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(
            x=c["fenetre"], y=c["err_rel_moy"] - c["ecart_type"],
            fill="tonexty", fillcolor="rgba(138,151,163,0.20)", line=dict(width=0),
            showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(
            x=c["fenetre"], y=c["err_rel_moy"], mode="lines+markers",
            line=dict(color=C_GAIN, width=2), marker=dict(size=9, color=C_GAIN),
            customdata=c[["n_train", "ecart_type"]],
            hovertemplate="%{x} : %{y:.2f} % ± %{customdata[1]:.2f}"
                          "<br>%{customdata[0]:,} annonces d'entraînement<extra></extra>",
            showlegend=False))
        fig.update_layout(title="Erreur selon la quantité de données d'entraînement")
        fig.update_yaxes(title="Erreur relative médiane (%)")
        st.plotly_chart(style_figure(fig, 360), width="stretch")

        st.dataframe(
            c[["fenetre", "n_train", "err_rel_moy", "ecart_type"]].rename(columns={
                "fenetre": "Fenêtre", "n_train": "Annonces d'entraînement",
                "err_rel_moy": "Erreur relative médiane (%)", "ecart_type": "Écart-type (5 graines)"}),
            width="stretch", hide_index=True,
        )
        with st.expander("ℹ️ Protocole et lecture"):
            st.write(f"**Protocole.** {calib['protocole']}\n\n**Conclusion.** {calib['conclusion']}")
    else:
        st.info(f"`{CALIB_PATH}` introuvable — la calibration est une mesure ponctuelle, "
                "pas une étape du pipeline.")

    st.divider()

    # ---- Fiabilité de l'estimation par tranche de prix -------------------
    st.subheader("Fiabilité de l'estimation par tranche de prix")
    st.caption("Erreur du modèle (prédiction hors-échantillon vs prix réel) selon la "
               "gamme de prix. Plus la courbe est basse, plus l'estimation est fiable "
               "dans cette tranche — à lire avant de se fier à un prix estimé.")

    if "Prix_Theorique" in df.columns:
        d_fia = df.dropna(subset=["Prix", "Prix_Theorique"]).copy()
        d_fia = d_fia[(d_fia["Prix"] > 0) & (d_fia["Prix_Theorique"] > 0)]
    else:
        d_fia = pd.DataFrame()

    if len(d_fia) >= 50:
        d_fia["_err"] = (d_fia["Prix_Theorique"] - d_fia["Prix"]).abs() / d_fia["Prix"]
        bornes = [0, 15000, 25000, 35000, 50000, 75000, 100000, float("inf")]
        libelles = ["< 15k", "15–25k", "25–35k", "35–50k", "50–75k", "75–100k", "100k +"]
        d_fia["_tranche"] = pd.cut(d_fia["Prix"], bins=bornes, labels=libelles)
        gr = (d_fia.groupby("_tranche", observed=True)
              .agg(n=("_err", "size"), mdape=("_err", lambda x: 100 * x.median()))
              .reset_index())
        gr = gr[gr["n"] >= 10]  # une tranche trop peu peuplée donne une mesure instable

        mdape_global = 100 * d_fia["_err"].median()

        fig = go.Figure()
        fig.add_hline(y=mdape_global, line=dict(color="rgba(138,151,163,0.6)", dash="dot"),
                      annotation_text=f"erreur globale {mdape_global:.0f}%",
                      annotation_position="top left")
        fig.add_trace(go.Scatter(
            x=gr["_tranche"], y=gr["mdape"], mode="lines+markers",
            line=dict(color=C_GAIN, width=2), marker=dict(size=10, color=C_GAIN),
            customdata=gr[["n"]],
            hovertemplate="%{x} DT : erreur médiane %{y:.1f} %"
                          "<br>%{customdata[0]:,} annonces<extra></extra>",
            showlegend=False))
        fig.update_layout(title="Erreur relative médiane par tranche de prix (DT)")
        fig.update_yaxes(title="Erreur relative médiane (%)")
        fig.update_xaxes(title="Tranche de prix")
        st.plotly_chart(style_figure(fig, 360), width="stretch")

        best = gr.loc[gr["mdape"].idxmin()]
        pire = gr.loc[gr["mdape"].idxmax()]
        st.caption(
            f"Zone la plus fiable : **{best['_tranche']} DT** ({best['mdape']:.0f} % d'erreur). "
            f"La moins fiable : **{pire['_tranche']} DT** ({pire['mdape']:.0f} %) — une estimation "
            "dans cette gamme est à prendre avec prudence."
        )
    else:
        st.info("Pas assez de données scorées (avec `Prix_Theorique`) pour mesurer la "
                "fiabilité par tranche.")

    st.divider()

    # ---- Performance & drift dans le temps -------------------------------
    st.subheader("Performance du modèle dans le temps")
    st.caption("MdAPE et composition du marché à chaque run nocturne. "
               "La vraie preuve de fiabilité, ce n'est pas un snapshot : c'est la stabilité "
               "dans la durée. Alimenté par `core/suivi_performance.py`.")
    histo = charger_json(HISTO_PATH)
    if not histo or len(histo) < 2:
        n = 0 if not histo else len(histo)
        st.info(f"Pas encore assez d'historique ({n} run(s)) — la courbe apparaîtra après "
                "quelques exécutions nocturnes. `suivi_performance.py` est branché au pipeline "
                "et commencera à accumuler dès le prochain run.")
    else:
        hp = pd.DataFrame([{
            "date": pd.to_datetime(h.get("date")),
            "MdAPE (%)": h.get("mdape_global"),
            "Volume": h.get("instantane", {}).get("n"),
            "Prix médian": h.get("instantane", {}).get("prix_median"),
        } for h in histo]).sort_values("date")

        fig = go.Figure()
        fig.add_scatter(x=hp["date"], y=hp["MdAPE (%)"], mode="lines+markers",
                        name="MdAPE (%)", line=dict(color=C_GAIN, width=3))
        fig.add_scatter(x=hp["date"], y=hp["Volume"], mode="lines", name="Volume",
                        yaxis="y2", line=dict(color=C_GRIS, width=1, dash="dot"))
        fig.update_layout(
            yaxis=dict(title="MdAPE (%)"),
            yaxis2=dict(title="Volume", overlaying="y", side="right", showgrid=False),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0), title_text="",
        )
        st.plotly_chart(style_figure(fig, 340), width="stretch")

        c1, c2, c3 = st.columns(3)
        derniere = hp.iloc[-1]
        c1.metric("MdAPE (dernier run)", f"{derniere['MdAPE (%)']:.1f} %"
                  if pd.notna(derniere["MdAPE (%)"]) else "—")
        if len(hp) >= 2 and pd.notna(hp.iloc[-2]["MdAPE (%)"]) and pd.notna(derniere["MdAPE (%)"]):
            delta = derniere["MdAPE (%)"] - hp.iloc[-2]["MdAPE (%)"]
            c2.metric("Évolution vs run précédent", f"{delta:+.1f} pts",
                      delta=f"{delta:+.1f}", delta_color="inverse")
        c3.metric("Runs enregistrés", len(hp))

        drift = histo[-1].get("drift", [])
        if drift:
            st.warning("⚠️ Drift détecté au dernier run : " + " · ".join(drift))
        else:
            st.caption("✓ Aucun drift détecté au dernier run.")

        # ---- Prix médian glissant (à travers les runs, non biaisé) --------
        hp["glissant"] = hp["Prix médian"].rolling(7, min_periods=2, center=True).median()
        figp = go.Figure()
        figp.add_scatter(x=hp["date"], y=hp["Prix médian"], mode="markers",
                         name="Prix médian (par run)", marker=dict(color=C_GRIS, size=5))
        figp.add_scatter(x=hp["date"], y=hp["glissant"], mode="lines",
                         name="Médiane glissante (7 runs)", line=dict(color=C_ASPHALTE, width=3))
        figp.update_layout(yaxis=dict(title="Prix médian marché (DT)"), title_text="",
                           legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0))
        st.plotly_chart(style_figure(figp, 300), width="stretch")
        st.caption("Prix médian du marché à chaque run (points) et sa **médiane glissante** (ligne). "
                   "Chaque run mesure le marché *actuellement en ligne* → pas de biais de survie, "
                   "contrairement à une médiane par date de dépôt (où les vieilles annonces encore "
                   "présentes sont les invendues, souvent surcotées).")

        explication(
            "Comment lire ce suivi",
            "**Ce que ça montre.** L'erreur du modèle (MdAPE) et le volume de données à chaque "
            "run nocturne.\n\n"
            "**Comment c'est calculé.** Chaque nuit, `suivi_performance.py` enregistre la MdAPE "
            "hors-échantillon et un instantané du marché (prix médian, mix des sources, "
            "complétude), puis compare à la médiane des 7 derniers runs pour détecter un drift.\n\n"
            "**Comment décider.** Une **MdAPE plate** = modèle fiable dans la durée. Une **hausse "
            "progressive** ou un **drift** signalé = le marché a bougé → réentraîner ou investiguer "
            "(nouvelle source, changement de composition)."
        )

    st.divider()

    # ---- Interprétation du modèle (SHAP) ---------------------------------
    st.subheader("Interprétation — qu'est-ce qui fait le prix ?")
    st.caption("Importance globale de chaque variable dans les prédictions "
               "(valeurs SHAP : moyenne des contributions absolues au prix estimé). "
               "Plus la barre est longue, plus la variable pèse.")
    shap_data = charger_json(SHAP_PATH)
    if shap_data and shap_data.get("importances"):
        imp = pd.DataFrame(shap_data["importances"])
        imp["label"] = imp["feature"].map(LABELS_FEATURES).fillna(imp["feature"])
        imp = imp.sort_values("pct")  # ascendant -> plus important en haut du barh
        fig = go.Figure(go.Bar(
            x=imp["pct"], y=imp["label"], orientation="h",
            marker=dict(color=C_GAIN),
            hovertemplate="%{y} : %{x:.1f} %<extra></extra>", showlegend=False))
        fig.update_layout(title="Poids de chaque variable dans le prix estimé")
        fig.update_xaxes(title="Part de l'importance totale (%)")
        st.plotly_chart(style_figure(fig, 420), width="stretch")

        top3 = imp.sort_values("pct", ascending=False).head(3)["label"].tolist()
        st.caption(f"Le prix est surtout piloté par : **{', '.join(top3)}**. "
                   "Marque et modèle pèsent étonnamment peu — le modèle s'appuie "
                   "davantage sur le millésime, l'équipement et la motorisation que sur le badge. "
                   "(Le champ d'état réel `Etat_Vehicule` n'est pas rempli par les scrapers, donc non utilisé.)")
        with st.expander("ℹ️ Comment lire ces valeurs"):
            st.write(
                f"Modèle : `{nom_modele_court(shap_data.get('modele', '?'))}`, échantillon de "
                f"{shap_data.get('n_lignes', '?')} annonces. Échelle : "
                f"{shap_data.get('echelle', '')}. Les valeurs SHAP mesurent la "
                "contribution moyenne (en valeur absolue) de chaque variable à "
                "l'écart de prix prédit, normalisées ici en % du total."
            )
    else:
        st.info(f"`{SHAP_PATH}` introuvable — relance `python core/modele_prediction.py` "
                "(avec `shap` installé) pour générer l'interprétation.")

    st.divider()

    # ---- Validation par les disparitions réelles -------------------------
    st.subheader("Validation — les opportunités partent-elles plus vite ?")
    st.caption("Le seul endroit du projet où l'on confronte les prédictions à un fait observé. "
               "Une annonce qui disparaît n'est pas forcément vendue — elle peut avoir été "
               "retirée ou avoir expiré — mais c'est le meilleur proxy disponible.")

    suivi = None
    try:
        suivi = lire_csv("data/processed/suivi_annonces.csv")
    except Exception:
        pass

    if suivi is None or "Jours_En_Ligne" not in suivi.columns:
        st.info("Le suivi des annonces n'a pas encore tourné. Il démarre au prochain "
                "`python main.py`.")
    else:
        jours = pd.to_numeric(suivi["Jours_En_Ligne"], errors="coerce")
        mesurees = suivi[jours.notna()].copy()
        mesurees["jours"] = jours[jours.notna()]

        c1, c2, c3 = st.columns(3)
        c1.metric("Annonces suivies", f"{len(suivi):,}".replace(",", " "))
        c2.metric("Disparues (mesurables)", len(mesurees))
        c3.metric("Signalées comme opportunité",
                  int(suivi["Etait_Opportunite"].sum()) if "Etait_Opportunite" in suivi.columns else 0)

        if len(mesurees) < 30:
            st.info(
                f"**{len(mesurees)} annonces disparues** — il en faut plusieurs dizaines pour "
                "comparer quoi que ce soit. Le suivi s'enrichit à chaque exécution nocturne : "
                "compte quelques semaines avant que cette section devienne lisible.\n\n"
                "Ce qui apparaîtra ici : la durée en ligne des annonces signalées comme "
                "opportunités face à celle des autres. Si les deux sont identiques, le "
                "détecteur ne détecte rien d'utile — et il vaudra mieux le savoir."
            )
        else:
            deals = mesurees[mesurees["Etait_Opportunite"] == True]      # noqa: E712
            autres = mesurees[mesurees["Etait_Opportunite"] != True]     # noqa: E712
            if len(deals) >= 10 and len(autres) >= 10:
                ca, cb = st.columns(2)
                ca.metric("Durée médiane — opportunités", f"{deals['jours'].median():.0f} j")
                cb.metric("Durée médiane — autres annonces", f"{autres['jours'].median():.0f} j")

                fig = go.Figure()
                fig.add_trace(go.Box(x=autres["jours"], name="Autres annonces",
                                     marker_color=C_GRIS, boxmean=True))
                fig.add_trace(go.Box(x=deals["jours"], name="Signalées opportunité",
                                     marker_color=C_GAIN, boxmean=True))
                fig.update_layout(title="Durée en ligne avant disparition")
                fig.update_xaxes(title="Jours en ligne")
                st.plotly_chart(style_figure(fig, 320), width="stretch")

                ecart = autres["jours"].median() - deals["jours"].median()
                if ecart > 1:
                    st.success(f"Les annonces signalées disparaissent environ **{ecart:.0f} jours "
                               "plus vite** que les autres — le détecteur capte bien quelque chose.")
                elif ecart < -1:
                    st.error("Les annonces signalées restent **plus longtemps** en ligne que les "
                             "autres. Le détecteur sélectionne probablement des véhicules peu "
                             "demandés plutôt que des bonnes affaires — seuils à revoir.")
                else:
                    st.warning("Aucune différence nette entre les deux groupes. En l'état, le "
                               "détecteur n'apporte pas de signal mesurable sur la vitesse "
                               "d'écoulement.")
            else:
                st.info("Pas encore assez d'annonces disparues dans chacun des deux groupes "
                        "(minimum 10 de chaque côté).")

            # Liquidité réelle par modèle
            par_modele = (mesurees.groupby(["Marque", "Modèle"])["jours"]
                          .agg(["size", "median"]).reset_index())
            par_modele = par_modele[par_modele["size"] >= 5].sort_values("median")
            if len(par_modele):
                par_modele["libelle"] = par_modele["Marque"] + " " + par_modele["Modèle"].astype(str)
                fig = px.bar(par_modele.head(15).sort_values("median", ascending=False),
                             x="median", y="libelle", orientation="h",
                             title="Modèles qui partent le plus vite (durée réelle en ligne)",
                             labels={"median": "Jours en ligne (médiane)", "libelle": ""},
                             custom_data=["size"])
                fig.update_traces(marker_color=C_GAIN,
                                  hovertemplate="%{y} : %{x:.0f} j (n=%{customdata[0]})<extra></extra>")
                st.plotly_chart(style_figure(fig, 420), width="stretch")
                st.caption("Cette mesure remplacera à terme le `Score_Liquidite`, qui n'est "
                           "aujourd'hui qu'un proxy fondé sur le volume d'annonces.")

    st.divider()

    # ---- Validation indépendante Marché ↔ ML ----------------------------
    st.subheader("Validation croisée — Marché comparable ↔ modèle ML")
    st.caption("Contrôle leave-one-out : pour chaque annonce éligible, AutoDeal reconstruit une "
               "fourchette Q25–Q75 à partir d'au moins 5 véhicules comparables, en excluant "
               "l'annonce elle-même, puis vérifie où tombe la prédiction ML.")
    with st.spinner("Calcul du contrôle marché ↔ ML (mis en cache ensuite)…"):
        diag_mm = diagnostic_market_ml(df)

    if diag_mm is None or diag_mm.empty:
        st.info("Pas assez de comparables pour calculer ce diagnostic.")
    else:
        coverage = 100 * diag_mm["ML_In_Market_Range"].mean()
        coverage_wide = 100 * diag_mm["ML_In_Market_Range_P10_P90"].mean()
        median_width = 100 * diag_mm["Market_Relative_Width"].median()
        median_gap = 100 * diag_mm["ML_Market_Gap_Pct"].abs().median()
        eligible_rate = 100 * len(diag_mm) / max(len(df), 1)
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Dans Q25–Q75", f"{coverage:.1f} %")
        m2.metric("Dans P10–P90", f"{coverage_wide:.1f} %")
        m3.metric("Annonces évaluables", f"{len(diag_mm):,}".replace(",", " "), delta=f"{eligible_rate:.1f} % du dataset")
        m4.metric("Largeur médiane Q25–Q75", f"{median_width:.1f} %")
        m5.metric("Écart médian ML ↔ marché", f"{median_gap:.1f} %")
        st.caption("Q25–Q75 contient la moitié centrale des prix comparables. P10–P90 est un contrôle élargi : "
                   "si le ML sort souvent de P10–P90, l'écart au marché devient beaucoup plus préoccupant.")

        # 1) Homogénéité des intervalles + direction de l'écart ML.
        hom = (diag_mm["Market_Homogeneity"].value_counts().rename_axis("Homogénéité").reset_index(name="Annonces"))
        order_h = ["élevée", "modérée", "faible"]
        hom["Homogénéité"] = pd.Categorical(hom["Homogénéité"], categories=order_h, ordered=True)
        hom = hom.sort_values("Homogénéité")

        cmm1, cmm2 = st.columns(2)
        with cmm1:
            fig = px.bar(hom, x="Homogénéité", y="Annonces", title="Homogénéité des fourchettes de marché")
            fig.update_traces(texttemplate="%{y}", textposition="outside")
            st.plotly_chart(style_figure(fig, 320), width="stretch")
            st.caption("Élevée : largeur Q25–Q75 ≤ 15 % de la médiane · Modérée : 15–30 % · Faible : > 30 %.")
        with cmm2:
            order_pos = ["Sous Q25", "Dans Q25–Q75", "Au-dessus Q75"]
            rep = (diag_mm["ML_Market_Position"].value_counts().reindex(order_pos, fill_value=0)
                   .rename_axis("Position ML").reset_index(name="Annonces"))
            rep["Part (%)"] = 100 * rep["Annonces"] / max(rep["Annonces"].sum(), 1)
            fig = px.bar(rep, x="Position ML", y="Annonces", title="Position du ML face au cœur du marché",
                         custom_data=["Part (%)"])
            fig.update_traces(texttemplate="%{customdata[0]:.1f}%", textposition="outside",
                              hovertemplate="%{x}<br>%{y} annonces<br>%{customdata[0]:.1f}%<extra></extra>")
            st.plotly_chart(style_figure(fig, 320), width="stretch")

        # 2) Le KPI le plus important : l'accord conditionnel à la qualité du signal marché.
        by_hom = (diag_mm.groupby("Market_Homogeneity", observed=True)
                  .agg(n=("ML_In_Market_Range", "size"),
                       couverture_q25_q75=("ML_In_Market_Range", "mean"),
                       couverture_p10_p90=("ML_In_Market_Range_P10_P90", "mean"),
                       ecart_abs=("ML_Market_Gap_Pct", lambda x: x.abs().median()))
                  .reset_index().rename(columns={"Market_Homogeneity": "Homogénéité"}))
        by_hom["Homogénéité"] = pd.Categorical(by_hom["Homogénéité"], categories=order_h, ordered=True)
        by_hom = by_hom.sort_values("Homogénéité")
        for c in ["couverture_q25_q75", "couverture_p10_p90", "ecart_abs"]:
            by_hom[c] = 100 * by_hom[c]

        st.markdown("#### Le ML est-il surtout cohérent quand le marché est homogène ?")
        fig = px.bar(by_hom, x="Homogénéité", y=["couverture_q25_q75", "couverture_p10_p90"],
                     barmode="group", title="Couverture ML selon l'homogénéité du marché",
                     labels={"value": "Couverture (%)", "variable": "Intervalle"})
        fig.for_each_trace(lambda t: t.update(name="Q25–Q75" if t.name == "couverture_q25_q75" else "P10–P90"))
        st.plotly_chart(style_figure(fig, 350), width="stretch")
        st.dataframe(by_hom.rename(columns={
            "n":"N", "couverture_q25_q75":"Dans Q25–Q75 %",
            "couverture_p10_p90":"Dans P10–P90 %", "ecart_abs":"Écart médian absolu %"
        }), hide_index=True, width="stretch")

        def _coverage_table(data, group_cols, min_obs):
            g = (data.groupby(group_cols, observed=True)
                 .agg(n=("ML_In_Market_Range", "size"),
                      couverture=("ML_In_Market_Range", "mean"),
                      couverture_p10_p90=("ML_In_Market_Range_P10_P90", "mean"),
                      sous_q25=("ML_Market_Position", lambda x: (x == "Sous Q25").mean()),
                      au_dessus_q75=("ML_Market_Position", lambda x: (x == "Au-dessus Q75").mean()),
                      largeur=("Market_Relative_Width", "median"),
                      ecart_ml_marche=("ML_Market_Gap_Pct", lambda x: x.abs().median()),
                      biais_median=("ML_Market_Gap_Pct", "median"))
                 .reset_index())
            g = g[g["n"] >= min_obs].copy()
            for col in ["couverture", "couverture_p10_p90", "sous_q25", "au_dessus_q75", "largeur", "ecart_ml_marche", "biais_median"]:
                g[col] = 100 * g[col]
            return g

        t_brand, t_model, t_age, t_km, t_price = st.tabs(["Par marque", "Par modèle", "Par âge", "Par km", "Par prix"])
        with t_brand:
            g = _coverage_table(diag_mm, ["Marque"], 20).sort_values("couverture")
            if len(g):
                fig = px.bar(g, x="couverture", y="Marque", orientation="h",
                             title="Prédictions ML dans Q25–Q75 du marché — par marque",
                             labels={"couverture": "Dans Q25–Q75 (%)", "Marque": ""},
                             hover_data=["n", "couverture_p10_p90", "sous_q25", "au_dessus_q75", "largeur", "ecart_ml_marche", "biais_median"])
                st.plotly_chart(style_figure(fig, max(320, 25 * len(g))), width="stretch")
        with t_model:
            mm = diag_mm.copy()
            mm["Véhicule"] = mm["Marque"].astype(str) + " " + mm["Modèle"].astype(str)
            g = _coverage_table(mm, ["Véhicule"], 15).sort_values("couverture")
            if len(g):
                st.dataframe(g.rename(columns={"n":"N", "couverture":"Dans Q25–Q75 %", "couverture_p10_p90":"Dans P10–P90 %", "sous_q25":"Sous Q25 %", "au_dessus_q75":"Au-dessus Q75 %", "largeur":"Largeur marché %", "ecart_ml_marche":"Écart abs. ML-marché %", "biais_median":"Biais médian ML-marché %"}),
                             hide_index=True, width="stretch")
        with t_age:
            aa = diag_mm.copy()
            aa["Tranche âge"] = pd.cut(aa["Age_Vehicule"], bins=[-1, 2, 5, 8, 12, 20, np.inf],
                                        labels=["0–2 ans", "3–5 ans", "6–8 ans", "9–12 ans", "13–20 ans", "20+ ans"])
            g = _coverage_table(aa, ["Tranche âge"], 15)
            st.dataframe(g.rename(columns={"n":"N", "couverture":"Dans Q25–Q75 %", "couverture_p10_p90":"Dans P10–P90 %", "sous_q25":"Sous Q25 %", "au_dessus_q75":"Au-dessus Q75 %", "largeur":"Largeur marché %", "ecart_ml_marche":"Écart abs. ML-marché %", "biais_median":"Biais médian ML-marché %"}),
                         hide_index=True, width="stretch")
        with t_km:
            kk = diag_mm.copy()
            kk["Tranche km"] = pd.cut(pd.to_numeric(kk["Kilométrage"], errors="coerce"),
                                      bins=[-1, 30_000, 60_000, 100_000, 150_000, 200_000, np.inf],
                                      labels=["≤30k", "30–60k", "60–100k", "100–150k", "150–200k", "200k+"])
            g = _coverage_table(kk, ["Tranche km"], 15)
            st.dataframe(g.rename(columns={"n":"N", "couverture":"Dans Q25–Q75 %", "couverture_p10_p90":"Dans P10–P90 %", "sous_q25":"Sous Q25 %", "au_dessus_q75":"Au-dessus Q75 %", "largeur":"Largeur marché %", "ecart_ml_marche":"Écart abs. ML-marché %", "biais_median":"Biais médian ML-marché %"}),
                         hide_index=True, width="stretch")
        with t_price:
            pp = diag_mm.copy()
            pp["Tranche prix"] = pd.cut(pd.to_numeric(pp["Prix"], errors="coerce"),
                                        bins=[0, 25_000, 50_000, 80_000, 120_000, 200_000, np.inf],
                                        labels=["≤25k", "25–50k", "50–80k", "80–120k", "120–200k", "200k+"])
            g = _coverage_table(pp, ["Tranche prix"], 15)
            st.dataframe(g.rename(columns={"n":"N", "couverture":"Dans Q25–Q75 %", "couverture_p10_p90":"Dans P10–P90 %", "sous_q25":"Sous Q25 %", "au_dessus_q75":"Au-dessus Q75 %", "largeur":"Largeur marché %", "ecart_ml_marche":"Écart abs. ML-marché %", "biais_median":"Biais médian ML-marché %"}),
                         hide_index=True, width="stretch")

    st.divider()

    # ---- Qualité de la donnée -------------------------------------------
    st.subheader("Qualité de la donnée")
    taux = (df.notna().mean() * 100).round(0).sort_values()
    taux = taux[taux < 100]
    if len(taux):
        fig = px.bar(x=taux.values, y=taux.index, orientation="h",
                     title="Taux de remplissage des colonnes incomplètes (%)",
                     labels={"x": "Rempli (%)", "y": ""})
        fig.update_traces(marker_color=[C_ALERTE if v < 50 else C_ASPHALTE for v in taux.values],
                          hovertemplate="%{y} : %{x:.0f} %<extra></extra>")
        st.plotly_chart(style_figure(fig, max(280, 26 * len(taux))), width="stretch")

    col_c, col_d = st.columns(2)
    with col_c:
        if "Fiabilite_Estimation" in df.columns:
            rep = df["Fiabilite_Estimation"].value_counts()
            fig = px.bar(x=rep.index.astype(str), y=rep.values,
                         title="Fiabilité des estimations (nb de comparables)",
                         labels={"x": "", "y": "Annonces"})
            fig.update_traces(marker_color=C_ASPHALTE)
            st.plotly_chart(style_figure(fig, 300), width="stretch")
    with col_d:
        if df_deals is not None and "Nb_Comparables" in df_deals.columns:
            solides = int((df_deals["Nb_Comparables"] >= 8).sum())
            st.metric("Opportunités détectées", len(df_deals))
            st.metric("Dont estimations solides", solides)
            st.metric("Écartées faute de comparables", len(df_deals) - solides)
            st.caption("Seules les opportunités solides déclenchent une alerte Telegram.")

    st.divider()

    # ---- Santé du scraping : volume par source (et par jour) -------------
    st.subheader("Santé du scraping — volume par source")
    st.caption("Annonces détectées par source. Un effondrement soudain sur une source = "
               "scraper probablement cassé (changement de structure du site, anti-bot…).")
    try:
        suivi_sc = lire_csv("data/processed/suivi_annonces.csv")
    except Exception:
        suivi_sc = None
    if suivi_sc is not None and {"Source", "Premiere_Vue"}.issubset(suivi_sc.columns):
        s = suivi_sc.copy()
        s["_jour"] = pd.to_datetime(s["Premiere_Vue"], errors="coerce").dt.date
        s = s.dropna(subset=["_jour"])
        if s["_jour"].nunique() >= 2:
            par_jour = s.groupby(["_jour", "Source"], observed=True).size().reset_index(name="n")
            fig = px.line(par_jour, x="_jour", y="n", color="Source", markers=True,
                          title="Annonces détectées par source et par jour",
                          labels={"_jour": "", "n": "Annonces détectées", "Source": "Source"})
            st.plotly_chart(style_figure(fig, 320), width="stretch")
        else:
            par_src = (s.groupby("Source", observed=True).size()
                       .reset_index(name="n").sort_values("n", ascending=False))
            fig = px.bar(par_src, x="Source", y="n", title="Annonces par source (collecte du jour)",
                         labels={"Source": "", "n": "Annonces"})
            fig.update_traces(marker_color=C_GAIN)
            st.plotly_chart(style_figure(fig, 300), width="stretch")
            st.caption(f"Une seule journée d'historique — ce graphe deviendra une courbe temporelle "
                       f"par source dès plusieurs jours de collecte. Total : {len(s)} annonces sur "
                       f"{s['Source'].nunique()} sources.")
    else:
        st.info("`suivi_annonces.csv` indisponible ou sans colonnes Source / Premiere_Vue.")

    if diag:
        with st.expander("Variables utilisées par le modèle"):
            st.write("**Numériques** : " + ", ".join(f"`{f}`" for f in diag["features_numeriques"]))
            st.write("**Catégorielles** : " + ", ".join(f"`{f}`" for f in diag["features_categorielles"]))


# ===========================================================================
# Navigation
# ===========================================================================

def page_recherche(df):
    st.title("🔎 Recherche")
    st.caption("Filtre tout le marché scoré selon tes critères. Chaque résultat montre "
               "son prix affiché, l'estimation du modèle et l'écart (= l'opportunité).")
    if df is None or df.empty:
        st.info("Données indisponibles.")
        return

    d = df.copy()
    c1, c2, c3 = st.columns(3)
    with c1:
        sel_marques = st.multiselect("Marque", sorted(d["Marque"].dropna().unique()))
        sel_energie = st.multiselect(
            "Énergie", sorted(d["Energie"].dropna().unique()) if "Energie" in d.columns else [])
    with c2:
        sel_lieux = st.multiselect(
            "Région", sorted(d["Localisation"].dropna().unique()) if "Localisation" in d.columns else [])
        sel_boite = st.multiselect(
            "Boîte", sorted(d["Boite_Vitesse"].dropna().unique()) if "Boite_Vitesse" in d.columns else [])
    with c3:
        seulement_deals = st.checkbox("Opportunités uniquement (25–55 % sous le prix)")
        tri = st.selectbox("Trier par",
                           ["Meilleure affaire", "Prix croissant", "Prix décroissant", "Plus récent"])

    s1, s2, s3 = st.columns(3)
    prix_ok = d["Prix"].dropna()
    with s1:
        pmin = int(prix_ok.min()) if len(prix_ok) else 0
        pmax = int(prix_ok.quantile(0.99)) if len(prix_ok) else 500000
        budget = st.slider("Budget (DT)", pmin, max(pmax, pmin + 1000), (pmin, pmax), step=1000)
    with s2:
        an_ok = d["Année"].dropna()
        amin = int(an_ok.min()) if len(an_ok) else 1990
        amax = int(an_ok.max()) if len(an_ok) else pd.Timestamp.now().year
        annee = st.slider("Année", amin, max(amax, amin + 1), (amin, amax))
    with s3:
        km_ok = d["Kilométrage"].dropna()
        kmax = int(km_ok.quantile(0.99)) if len(km_ok) else 500000
        km_max = st.slider("Kilométrage max", 0, max(kmax, 10000), max(kmax, 10000), step=10000)

    if sel_marques:
        d = d[d["Marque"].isin(sel_marques)]
    if sel_energie:
        d = d[d["Energie"].isin(sel_energie)]
    if sel_lieux:
        d = d[d["Localisation"].isin(sel_lieux)]
    if sel_boite:
        d = d[d["Boite_Vitesse"].isin(sel_boite)]
    d = d[d["Prix"].between(budget[0], budget[1])]
    d = d[d["Année"].between(annee[0], annee[1])]
    d = d[d["Kilométrage"].fillna(0) <= km_max]
    if seulement_deals:
        d = d[d["Score_Opportunite"].between(0.25, 0.55)]

    if tri == "Meilleure affaire":
        d = d.sort_values("Score_Opportunite", ascending=False)
    elif tri == "Prix croissant":
        d = d.sort_values("Prix")
    elif tri == "Prix décroissant":
        d = d.sort_values("Prix", ascending=False)
    elif tri == "Plus récent" and "Annonce-Detectee" in d.columns:
        d = d.sort_values("Annonce-Detectee", ascending=False)

    st.markdown(f"**{len(d)} annonces** correspondent" +
                (f" · {min(len(d), 200)} affichées" if len(d) > 200 else "") + ".")

    aff = d.head(200).copy()
    aff["Écart"] = (aff["Score_Opportunite"] * 100).round(0)
    cols = [c for c in ["Marque", "Modèle", "Année", "Kilométrage", "Prix",
                        "Prix_Theorique", "Écart", "Localisation", "Lien"] if c in aff.columns]
    st.dataframe(
        aff[cols], hide_index=True, width="stretch",
        column_config={
            "Prix": st.column_config.NumberColumn("Prix", format="%d DT"),
            "Prix_Theorique": st.column_config.NumberColumn("Estimé", format="%d DT"),
            "Kilométrage": st.column_config.NumberColumn("Km", format="%d"),
            "Écart": st.column_config.NumberColumn("Écart", format="%d %%",
                                                   help="% sous le prix estimé par le modèle"),
            "Lien": st.column_config.LinkColumn("Annonce", display_text="ouvrir"),
        },
    )


# Centroïdes des 24 gouvernorats (lat, lon) — pour la carte régionale.
GOUVERNORATS_COORD = {
    "Tunis": (36.80, 10.18), "Ariana": (36.86, 10.19), "Ben Arous": (36.75, 10.23),
    "Manouba": (36.81, 9.84), "Nabeul": (36.45, 10.74), "Zaghouan": (36.40, 10.14),
    "Bizerte": (37.27, 9.87), "Béja": (36.73, 9.18), "Jendouba": (36.50, 8.78),
    "Le Kef": (36.17, 8.70), "Siliana": (36.08, 9.37), "Kairouan": (35.68, 10.10),
    "Kasserine": (35.17, 8.83), "Sidi Bouzid": (35.04, 9.48), "Sousse": (35.83, 10.63),
    "Monastir": (35.77, 10.83), "Mahdia": (35.50, 11.06), "Sfax": (34.74, 10.76),
    "Gafsa": (34.42, 8.78), "Tozeur": (33.92, 8.13), "Kébili": (33.70, 8.97),
    "Gabès": (33.88, 10.10), "Médenine": (33.35, 10.50), "Tataouine": (32.93, 10.45),
}
_GOUV_NORM = {"".join(c for c in unicodedata.normalize("NFD", g.lower())
                      if unicodedata.category(c) != "Mn"): g for g in GOUVERNORATS_COORD}


def _match_gouvernorat(localisation):
    """Rattache une localisation libre (ex 'Sfax Ville') à un gouvernorat."""
    s = "".join(c for c in unicodedata.normalize("NFD", str(localisation).lower())
                if unicodedata.category(c) != "Mn")
    for norm, disp in _GOUV_NORM.items():
        if norm in s:
            return disp
    return None


def page_carte(df):
    st.title("🗺️ Carte du marché")
    st.caption("Le marché par gouvernorat : où les prix sont hauts, où se concentrent "
               "les opportunités. Croise avec l'arbitrage géographique de la page Samsar.")
    if df is None or df.empty or "Localisation" not in df.columns:
        st.info("Données de localisation indisponibles.")
        return

    metrique = st.radio("Colorer par", ["Prix médian", "Densité d'opportunités"],
                        horizontal=True)

    d = df.dropna(subset=["Localisation", "Prix"]).copy()
    d["_gouv"] = d["Localisation"].map(_match_gouvernorat)
    d = d.dropna(subset=["_gouv"])
    if d.empty:
        st.info("Aucune localisation n'a pu être rattachée à un gouvernorat.")
        return

    agg = d.groupby("_gouv").agg(
        n=("Prix", "size"),
        prix_median=("Prix", "median"),
        n_deals=("Score_Opportunite", lambda s: int(s.between(0.25, 0.55).sum())),
    ).reset_index()
    agg = agg[agg["n"] >= 5]
    agg["taux_deals"] = (100 * agg["n_deals"] / agg["n"]).round(1)
    agg["lat"] = agg["_gouv"].map(lambda g: GOUVERNORATS_COORD[g][0])
    agg["lon"] = agg["_gouv"].map(lambda g: GOUVERNORATS_COORD[g][1])

    couleur = "prix_median" if metrique.startswith("Prix") else "taux_deals"
    labels = {"prix_median": "Prix médian (DT)", "taux_deals": "% d'opportunités",
              "n": "Annonces", "_gouv": "Gouvernorat"}
    fig = px.scatter_mapbox(
        agg, lat="lat", lon="lon", size="n", color=couleur,
        hover_name="_gouv",
        hover_data={"n": True, "prix_median": ":,.0f", "taux_deals": True,
                    "lat": False, "lon": False},
        color_continuous_scale="Turbo", size_max=45, zoom=5.1,
        center={"lat": 34.5, "lon": 9.6}, labels=labels, height=560,
    )
    fig.update_layout(mapbox_style="open-street-map",
                      margin={"l": 0, "r": 0, "t": 10, "b": 0})
    st.plotly_chart(fig, width="stretch")

    if metrique.startswith("Prix"):
        cher = agg.loc[agg["prix_median"].idxmax()]
        pas_cher = agg.loc[agg["prix_median"].idxmin()]
        st.caption(f"Le plus cher : **{cher['_gouv']}** ({cher['prix_median']:,.0f} DT médian). "
                   f"Le moins cher : **{pas_cher['_gouv']}** ({pas_cher['prix_median']:,.0f} DT). "
                   "L'écart, c'est le potentiel d'arbitrage géographique.".replace(",", " "))
    else:
        top = agg.loc[agg["taux_deals"].idxmax()]
        st.caption(f"Plus forte densité d'opportunités : **{top['_gouv']}** "
                   f"({top['taux_deals']:.0f} % des annonces sous le prix estimé). "
                   "C'est là qu'il y a le plus à chasser.")

    explication(
        "Comment lire la carte",
        "**Ce que ça montre.** Le marché par gouvernorat. Chaque bulle = un gouvernorat ; sa "
        "**taille** = le nombre d'annonces, sa **couleur** = la métrique choisie (prix médian ou "
        "densité d'opportunités).\n\n"
        "**Comment c'est calculé.** Les localisations libres sont rattachées à leur gouvernorat, "
        "puis agrégées (prix médian, % d'annonces sous le prix estimé). Seuls les gouvernorats à "
        "≥ 5 annonces sont affichés.\n\n"
        "**Comment décider.** En mode *prix médian*, l'écart entre zones chères (Grand Tunis) et "
        "bon marché (intérieur) est le potentiel d'arbitrage géographique. En mode *densité "
        "d'opportunités*, c'est là où chasser. C'est une vue à l'échelle du gouvernorat (pas de "
        "coordonnées plus fines disponibles)."
    )


def page_assistant(df, bundle):
    st.title("🤖 Assistant")
    st.caption("Pose ta question en langage naturel. L'assistant s'appuie sur TON modèle "
               "de prix et TES données scorées — pas un chatbot générique.")
    if df is None or df.empty:
        st.info("Données indisponibles.")
        return

    exemples = ("Exemples : « prix d'une Volkswagen Golf 2018 à 120000 km » · "
                "« affaires Peugeot sous 30000 DT à Sfax » · « quel gouvernorat est le moins cher »")
    st.caption(exemples)
    q = st.text_input("Ta question", placeholder="prix d'une Golf 2018 120000 km …")
    if not q:
        return

    qn = "".join(c for c in unicodedata.normalize("NFD", q.lower())
                 if unicodedata.category(c) != "Mn")

    # Extraction d'entités
    marques = sorted(df["Marque"].dropna().unique(), key=len, reverse=True)
    marque = next((m for m in marques
                   if "".join(c for c in unicodedata.normalize("NFD", str(m).lower())
                              if unicodedata.category(c) != "Mn") in qn), None)
    gouv = _match_gouvernorat(qn)
    annee = next((int(x) for x in re.findall(r"\b(19\d{2}|20\d{2})\b", qn)), None)
    nombres = [int(n.replace(" ", "")) for n in re.findall(r"\d[\d\s]{2,}", qn)]
    budget = next((n for n in sorted(nombres, reverse=True) if n >= 3000), None)
    km = next((n for n in nombres if 1000 <= n <= 500000 and n != budget), None)
    veut_affaires = any(w in qn for w in ["affaire", "opportunit", "deal", "bon plan", "moins cher que"])
    veut_prix = any(w in qn for w in ["prix", "combien", "estim", "vaut", "cote"])

    # Intention 1 : estimer un prix (modèle)
    if veut_prix and marque and not veut_affaires and bundle is not None:
        sous = df[df["Marque"] == marque]
        modele = None
        for md in sorted(sous["Modèle"].dropna().unique(), key=len, reverse=True):
            if "".join(c for c in unicodedata.normalize("NFD", str(md).lower())
                       if unicodedata.category(c) != "Mn") in qn:
                modele = md
                break
        annee = annee or int(sous["Année"].median()) if sous["Année"].notna().any() else 2018
        saisie = {
            "Kilométrage": km or 100000, "Age_Vehicule": max(pd.Timestamp.now().year - annee, 0),
            "Puissance_Fiscale": sous["Puissance_Fiscale"].median() if "Puissance_Fiscale" in sous else 6,
            "Segment_Vehicule": int(sous["Segment_Vehicule"].median()) if "Segment_Vehicule" in sous else 0,
            "Zone_Economique": 1 if gouv in ("Tunis", "Ariana", "Ben Arous", "Manouba") else 0,
            "Cylindree": sous["Cylindree"].median() if "Cylindree" in sous else np.nan,
            "Marque": marque, "Modèle": modele or "Autre",
            "Energie": (sous["Energie"].mode().iloc[0] if sous["Energie"].notna().any() else "Essence"),
            "Transmission": np.nan,
        }
        try:
            prix = float(np.expm1(bundle["pipeline"].predict(_ligne_modele(bundle, saisie))[0]))
            gr = table_fiabilite_prix(df)
            err = erreur_pour_prix(gr, prix) or 12
            st.success(f"**{marque} {modele or ''} {annee}**, {km or 100000:,} km".replace(",", " "))
            st.metric("Prix estimé par le modèle", f"{prix:,.0f} DT".replace(",", " "))
            st.caption(f"Fourchette ± {err:.0f} % : {prix*(1-err/100):,.0f} – {prix*(1+err/100):,.0f} DT."
                       .replace(",", " "))
            st.caption("Pour la décomposition « pourquoi ce prix » (SHAP) et les annonces "
                       "comparables réelles, ouvre le Calculateur :")
            if st.button("💰 Ouvrir dans le Calculateur", key="assist_to_calc"):
                st.session_state.page = "💰 Calculateur"
                st.rerun()
        except Exception:
            st.warning("Je n'ai pas pu estimer ce véhicule précis. Essaie avec marque + modèle + année.")
        return

    # Intention 2 : trouver des affaires
    if veut_affaires or budget or gouv or marque:
        res = df.copy()
        if marque:
            res = res[res["Marque"] == marque]
        if gouv and "Localisation" in res.columns:
            res = res[res["Localisation"].map(_match_gouvernorat) == gouv]
        if budget:
            res = res[res["Prix"] <= budget]
        if veut_affaires:
            res = res[res["Score_Opportunite"].between(0.25, 0.55)]
        res = res.sort_values("Score_Opportunite", ascending=False).head(20)
        crit = ", ".join(filter(None, [marque, f"≤ {budget:,} DT".replace(",", " ") if budget else None,
                                       gouv, "opportunités" if veut_affaires else None])) or "tout le marché"
        st.markdown(f"**{len(res)} résultat(s)** — {crit}")
        if len(res):
            aff = res.copy()
            aff["Écart"] = (aff["Score_Opportunite"] * 100).round(0)
            cols = [c for c in ["Marque", "Modèle", "Année", "Kilométrage", "Prix",
                                "Prix_Theorique", "Écart", "Localisation", "Lien"] if c in aff.columns]
            st.dataframe(aff[cols], hide_index=True, width="stretch",
                         column_config={
                             "Prix": st.column_config.NumberColumn("Prix", format="%d DT"),
                             "Prix_Theorique": st.column_config.NumberColumn("Estimé", format="%d DT"),
                             "Écart": st.column_config.NumberColumn("Écart", format="%d %%"),
                             "Lien": st.column_config.LinkColumn("Annonce", display_text="ouvrir"),
                         })
        else:
            st.info("Aucune annonce ne correspond. Élargis le budget ou la région.")
        return

    st.info("Je n'ai pas saisi la demande. Précise une marque, un budget, une région, "
            "ou demande un prix (« prix d'une Clio 2019 »).")


def main():
    # Le rôle (particulier / samsar / concessionnaire / admin) et le plan sont
    # distincts. Les menus professionnels n'apparaissent que pour le bon rôle
    # et les pages premium vérifient également les droits avant rendu.
    if supabase_is_configured():
        try:
            access = current_access_context(st.session_state)
        except Exception:
            access = {"role": "guest", "plan": "free", "subscription_status": "inactive"}
    else:
        access = {"role": "guest", "plan": "free", "subscription_status": "inactive"}

    nav = {
        "ACHETER": ["🏠 Accueil", "🛒 Acheter", "⚖️ Comparateur", "📊 Santé du marché", "📈 Historique", "🔔 Alertes", "👤 Mon compte", "💳 Tarifs"],
        "ESTIMER": ["💰 Calculateur", "🤖 Assistant"],
    }
    pro_pages = visible_pro_pages(access)
    if pro_pages:
        nav["PRO"] = pro_pages
    if access.get("role") == "admin":
        nav["ADMIN"] = ["🛠️ Admin"]

    if "page" not in st.session_state:
        st.session_state.page = "🏠 Accueil"
    if not can_open_page(st.session_state.page, access):
        st.session_state.page = "💳 Tarifs" if access.get("role") in {"samsar", "dealer"} else "🏠 Accueil"

    with st.sidebar:
        st.markdown("## 🚗 AutoDeal Tunisie")
        st.caption("Trouvez une voiture au bon prix — estimation, comparables et opportunités du marché tunisien.")
        if st.session_state.get("sb_user_email"):
            account_label = f"👤 {st.session_state.get('sb_user_email').split('@')[0]}"
            st.success(f"{access_badge(access)}")
        else:
            account_label = "👤 Se connecter"
        if st.button(account_label, key="sidebar_account_shortcut", use_container_width=True):
            st.session_state.page = "👤 Mon compte"
            st.rerun()

        for section, pages in nav.items():
            st.markdown(
                f"<div style='font-size:0.68rem;letter-spacing:0.09em;color:#8A97A3;"
                f"font-weight:700;margin:0.9rem 0 0.15rem'>{section}</div>",
                unsafe_allow_html=True,
            )
            for p in pages:
                actif = st.session_state.page == p
                if st.button(p, key=f"nav_{p}", use_container_width=True,
                             type="primary" if actif else "secondary"):
                    st.session_state.page = p
                    st.rerun()

        if st.session_state.page == "🚘 Détail annonce":
            st.markdown("<div style='font-size:0.68rem;letter-spacing:0.09em;color:#8A97A3;font-weight:700;margin:0.9rem 0 0.15rem'>ANNONCE</div>", unsafe_allow_html=True)
            st.button("🚘 Détail annonce", use_container_width=True, type="primary", disabled=True)

        page = st.session_state.page
        st.divider()
        df_temp = charger_scored()
        if df_temp is not None and "Annonce-Detectee" in df_temp.columns:
            derniere = df_temp["Annonce-Detectee"].max()
            st.caption(f"Données mises à jour : **{derniere}**")
            st.caption(f"Base : **{len(df_temp):,} annonces**".replace(",", " "))
            try:
                retard = (pd.Timestamp.now().normalize() - pd.to_datetime(derniere)).days
                if retard > 2:
                    st.warning(f"Données vieilles de {retard} jours — le scraping nocturne a peut-être échoué.")
            except Exception:
                pass
        if st.button("🔄 Rafraîchir les données"):
            st.cache_data.clear()
            st.cache_resource.clear()
            st.rerun()
        st.caption(f"Source : {'GitHub' if LIRE_DEPUIS_GITHUB else 'fichiers locaux'} · cache {DUREE_CACHE // 60} min")

    df_scored = charger_scored()
    df_deals = charger_deals()
    bundle = charger_modele()

    # Défense en profondeur : le menu caché ne suffit pas.
    if not can_open_page(page, access):
        st.error("Accès non autorisé pour votre profil ou votre abonnement.")
        if st.button("Voir les formules", type="primary"):
            st.session_state.page = "💳 Tarifs"
            st.rerun()
        return

    if page == "🏠 Accueil":
        page_accueil(df_scored)
    elif page == "🛒 Acheter":
        page_acheter(df_scored)
    elif page == "🚘 Détail annonce":
        page_detail(df_scored, bundle)
    elif page == "⚖️ Comparateur":
        page_comparateur(df_scored)
    elif page == "📊 Santé du marché":
        page_sante_marche(df_scored)
    elif page == "📈 Historique":
        page_historique(df_scored)
    elif page == "🔔 Alertes":
        page_alertes(df_scored, df_deals)
    elif page == "👤 Mon compte":
        page_compte(df_scored)
    elif page == "💳 Tarifs":
        page_tarifs(df_scored)
    elif page == "🏢 Concessionnaire":
        page_marche(df_scored)
    elif page == "🔎 Recherche avancée":
        page_recherche(df_scored)
    elif page == "🗺️ Carte":
        page_carte(df_scored)
    elif page == "🤝 Samsar":
        page_samsar(df_scored, df_deals)
    elif page == "💰 Calculateur":
        page_calculateur(df_scored, bundle)
    elif page == "🤖 Assistant":
        page_assistant(df_scored, bundle)
    elif page == "🛠️ Admin":
        page_admin(df_scored, df_deals)
    else:
        st.session_state.page = "🏠 Accueil"
        st.rerun()


if __name__ == "__main__":
    main()
