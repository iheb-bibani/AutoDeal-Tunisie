"""Modèle de prix : chargement, fiabilité par tranche, explication (SHAP ou repli)."""
import joblib
import pandas as pd
from services.data_service import MODELE_PATH, _url, LIRE_DEPUIS_GITHUB

BORNES_PRIX = [0, 15000, 25000, 35000, 50000, 75000, 100000, float("inf")]
LIBELLES_PRIX = ["< 15k", "15–25k", "25–35k", "35–50k", "50–75k", "75–100k", "100k +"]
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
def _tenter_charger_modele():
    """Tente de charger le modèle et RENVOIE (bundle, message_erreur) sans
    rien avaler : sert au diagnostic quand le calculateur n'a pas de modèle.
    Essaie GitHub puis le disque local."""
    erreurs = []
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
    try:
        bundle = joblib.load(MODELE_PATH)
        if isinstance(bundle, dict) and "pipeline" in bundle:
            return bundle, None
        erreurs.append("local : format de modèle inattendu (pas de clé 'pipeline').")
    except Exception as e:
        erreurs.append(f"local : {type(e).__name__} — {str(e)[:200]}")
    return None, " | ".join(erreurs)
def charger_modele():
    """Le modèle est réentraîné chaque nuit : on le recharge depuis GitHub avec
    la même durée de cache que les données, pour ne pas scorer des annonces
    fraîches avec un modèle de la semaine dernière."""
    bundle, _ = _tenter_charger_modele()
    return bundle
