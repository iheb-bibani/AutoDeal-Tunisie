"""Modélisation du juste prix AutoDeal — V2 Data Science.

Principes :
- cible log1p(Prix), sélection sur MdAPE (erreur relative médiane) ;
- benchmark Ridge / RF / HGB / CatBoost / LightGBM / XGBoost ;
- prédictions publiées strictement out-of-fold ;
- diagnostics de généralisation : KFold, GroupKFold Marque-Modèle,
  holdout temporel et holdout par source ;
- diagnostic explicite du biais Zone_Economique ;
- comparables locaux (même véhicule, âge/km proches) plutôt qu'un simple
  compteur Marque-Modèle.
"""

import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.doublons import cle_doublon  # noqa: E402
from core.inference import regrouper_categories, shap_par_feature  # noqa: E402
from core.io_atomique import dump_joblib_atomique, ecrire_csv_atomique, ecrire_json_atomique  # noqa: E402
from config import (  # noqa: E402
    FEATURES_CATEGORIELLES,
    FEATURES_NUMERIQUES,
    MAX_DAYS_OLD,
    SEUIL_MARQUE_RARE,
    SEUIL_MODELE_RARE,
)

IN_FICHIER = "data/processed/tunisia-cars-final-features.csv"
OUT_FICHIER = "data/processed/tunisia-cars-scored.csv"
OUT_MODELE = "data/models/modele_prix.pkl"
OUT_DIAGNOSTICS = "data/processed/diagnostics_modele.json"
OUT_SHAP = "data/processed/shap_importance.json"
CIBLE = "Prix"


def construire_features_entrainement(
    df: pd.DataFrame,
    features_num=None,
    features_cat=None,
) -> pd.DataFrame:
    """Matrice X avec regroupement des catégories réellement rares.

    Les colonnes exportées ne sont jamais écrasées : le regroupement n'existe
    que dans X. Les listes par défaut sont les objets de config.py (source unique).
    """
    features_num = FEATURES_NUMERIQUES if features_num is None else features_num
    features_cat = FEATURES_CATEGORIELLES if features_cat is None else features_cat
    X = df[list(features_num) + list(features_cat)].copy()
    marques, modeles = categories_frequentes(df)
    return regrouper_categories(X, marques, modeles)


def categories_frequentes(df: pd.DataFrame) -> tuple[set | None, set | None]:
    """Marques / modèles assez fréquents pour garder leur propre catégorie.

    Ces ensembles sont sauvegardés dans le bundle : l'inférence (calculateur,
    fiche annonce) applique ainsi exactement le même regroupement.
    """
    marques = modeles = None
    if "Marque" in df:
        counts = df["Marque"].value_counts()
        marques = set(counts[counts >= SEUIL_MARQUE_RARE].index)
    if "Modèle" in df:
        counts = df["Modèle"].fillna("Inconnu").value_counts()
        modeles = set(counts[counts >= SEUIL_MODELE_RARE].index)
    return marques, modeles


def cv_groupee(n_splits: int, seed: int = 42) -> GroupKFold:
    """KFold « par véhicule » : les doublons inter-sources (même voiture publiée
    sur deux sites, ou republiée) restent dans le même fold. Un KFold aléatoire
    les répartissait entre train et test, ce qui faisait presque mémoriser leur
    prix et sous-estimait l'erreur publiée."""
    return GroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)


def _sans_doublons_du_train(groupes: pd.Series, train_idx, test_idx):
    """Retire du test les annonces dont un doublon est dans le train."""
    vus = set(groupes.loc[train_idx])
    return pd.Index([i for i in test_idx if groupes.loc[i] not in vus])


def _split_temporel(dates: pd.Series, frac_test: float):
    """Holdout temporel coupé sur une FRONTIÈRE de date : une même journée
    n'est jamais à la fois en train et en test."""
    valides = dates.dropna()
    if valides.empty:
        return None, None
    coupure = valides.quantile(1 - frac_test)
    train_idx = dates.index[dates < coupure]
    test_idx = dates.index[dates >= coupure]
    return train_idx, test_idx


def construire_pipeline_onehot(modele, features_num=None, features_cat=None) -> Pipeline:
    features_num = FEATURES_NUMERIQUES if features_num is None else features_num
    features_cat = FEATURES_CATEGORIELLES if features_cat is None else features_cat
    preprocesseur = ColumnTransformer([
        ("num", Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]), list(features_num)),
        ("cat", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore")),
        ]), list(features_cat)),
    ])
    return Pipeline([("prep", preprocesseur), ("model", modele)])


def construire_pipeline_hgb(features_num=None, features_cat=None) -> Pipeline:
    features_num = FEATURES_NUMERIQUES if features_num is None else features_num
    features_cat = FEATURES_CATEGORIELLES if features_cat is None else features_cat
    preprocesseur = ColumnTransformer([
        ("num", "passthrough", list(features_num)),
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), list(features_cat)),
    ])
    indices_cat = list(range(len(features_num), len(features_num) + len(features_cat)))
    modele = HistGradientBoostingRegressor(
        categorical_features=indices_cat,
        learning_rate=0.07,
        max_iter=300,
        l2_regularization=1.0,
        random_state=42,
    )
    return Pipeline([("prep", preprocesseur), ("model", modele)])


def _candidats(features_num=None, features_cat=None):
    """Construit le benchmark. Les boosters externes sont optionnels à l'import
    pour garder les tests/imports utilisables même avant `pip install -r`."""
    features_num = FEATURES_NUMERIQUES if features_num is None else features_num
    features_cat = FEATURES_CATEGORIELLES if features_cat is None else features_cat
    candidats = {
        "Ridge": construire_pipeline_onehot(Ridge(alpha=1.0), features_num, features_cat),
        "RandomForest": construire_pipeline_onehot(
            RandomForestRegressor(n_estimators=350, max_depth=14, min_samples_leaf=2,
                                  random_state=42, n_jobs=-1),
            features_num, features_cat,
        ),
        "HistGradientBoosting": construire_pipeline_hgb(features_num, features_cat),
    }
    try:
        import catboost  # noqa: F401
        from core.model_wrappers import SklearnCatBoostRegressor
        candidats["CatBoost"] = construire_pipeline_onehot(
            SklearnCatBoostRegressor(iterations=500, depth=7, learning_rate=0.05,
                                     loss_function="RMSE", random_seed=42),
            features_num, features_cat,
        )
    except ImportError:
        pass
    try:
        from lightgbm import LGBMRegressor
        candidats["LightGBM"] = construire_pipeline_onehot(
            LGBMRegressor(n_estimators=500, learning_rate=0.04, num_leaves=31,
                          max_depth=-1, subsample=0.9, colsample_bytree=0.9,
                          reg_lambda=1.0, random_state=42, n_jobs=-1, verbosity=-1),
            features_num, features_cat,
        )
    except ImportError:
        pass
    try:
        from xgboost import XGBRegressor
        candidats["XGBoost"] = construire_pipeline_onehot(
            XGBRegressor(n_estimators=500, learning_rate=0.04, max_depth=7,
                         min_child_weight=3, subsample=0.9, colsample_bytree=0.9,
                         reg_lambda=1.0, objective="reg:squarederror",
                         random_state=42, n_jobs=-1),
            features_num, features_cat,
        )
    except ImportError:
        pass
    return candidats


def _metriques(y_brut, pred_log):
    pred = np.expm1(np.asarray(pred_log))
    yb = np.asarray(y_brut, dtype=float)
    erreur = np.abs(yb - pred)
    denom = np.maximum(yb, 1.0)
    return {
        "mae": float(np.mean(erreur)),
        "mdae": float(np.median(erreur)),
        "mdape": float(np.median(erreur / denom)),
    }


def evaluer_modeles(X: pd.DataFrame, y: pd.Series, y_brut: pd.Series, cv, features_num=None, features_cat=None,
                    groups=None) -> dict:
    resultats = {}
    for nom, pipeline in _candidats(features_num, features_cat).items():
        predictions_log = cross_val_predict(pipeline, X, y, cv=cv, groups=groups)
        m = _metriques(y_brut, predictions_log)
        resultats[nom] = {"pipeline": pipeline, "predictions_log": predictions_log, **m}
        print(f"{nom:22s} MdAPE={m['mdape']:5.1%} | médiane={m['mdae']:7.0f} DT | MAE={m['mae']:7.0f} DT")
    return resultats


def _eval_holdout(pipeline, X_train, y_train, X_test, y_test_brut):
    p = clone(pipeline)
    p.fit(X_train, y_train)
    return _metriques(y_test_brut, p.predict(X_test))


def diagnostics_generalisation(df, X, y, meilleur_pipeline, n_splits=5, doublons=None):
    """Mesure ce que le KFold aléatoire ne voit pas."""
    sortie = {}
    doublons = cle_doublon(df) if doublons is None else doublons

    # GroupKFold par famille Marque-Modèle : aucune famille du fold de test n'a
    # été vue à l'entraînement. C'est volontairement un stress-test sévère.
    groupes = (df["Marque"].fillna("?").astype(str) + "|" + df["Modèle"].fillna("?").astype(str))
    n_groupes = groupes.nunique()
    if n_groupes >= 3:
        gcv = GroupKFold(n_splits=min(n_splits, n_groupes))
        pred = cross_val_predict(clone(meilleur_pipeline), X, y, cv=gcv, groups=groupes)
        sortie["groupkfold_marque_modele"] = _metriques(df[CIBLE], pred)

    # Holdout temporel : ~20% des annonces les plus récentes, coupé sur une date.
    dates = pd.to_datetime(df.get("Annonce-Detectee"), errors="coerce")
    if dates.notna().sum() >= max(50, int(len(df) * 0.5)):
        train_idx, test_idx = _split_temporel(dates, 0.20)
        if test_idx is not None:
            test_idx = _sans_doublons_du_train(doublons, train_idx, test_idx)
        if test_idx is not None and len(test_idx) >= 20 and len(train_idx) >= 50:
            sortie["time_holdout_20pct"] = _eval_holdout(
                meilleur_pipeline, X.loc[train_idx], y.loc[train_idx],
                X.loc[test_idx], df.loc[test_idx, CIBLE],
            )

    # Leave-one-source-out : détecte les proxys de provenance et le dataset shift.
    par_source = []
    if "Source" in df.columns:
        for source, idx in df.groupby("Source").groups.items():
            idx = pd.Index(idx)
            if len(idx) < 30 or len(df) - len(idx) < 100:
                continue
            train_idx = df.index.difference(idx)
            # Les doublons inter-sources (même voiture sur deux sites) fausseraient
            # le test : ils sont retirés du jeu de test.
            idx = _sans_doublons_du_train(doublons, train_idx, idx)
            if len(idx) < 30:
                continue
            m = _eval_holdout(meilleur_pipeline, X.loc[train_idx], y.loc[train_idx], X.loc[idx], df.loc[idx, CIBLE])
            par_source.append({"source": str(source), "n_test": int(len(idx)), **m})
    sortie["source_holdout"] = par_source
    return sortie


def diagnostic_biais_zone(df, meilleur_nom, cv, groups=None):
    """Compare exactement le même algorithme avec et sans Zone_Economique.
    La production est sans zone ; ce test indique si la variable mérite un jour
    d'être réintroduite après contrôle du biais de source."""
    if "Zone_Economique" not in df.columns:
        return None
    num_sans = list(FEATURES_NUMERIQUES)
    num_avec = num_sans + ["Zone_Economique"]
    X_sans = construire_features_entrainement(df, num_sans, FEATURES_CATEGORIELLES)
    X_avec = construire_features_entrainement(df, num_avec, FEATURES_CATEGORIELLES)
    c_sans = _candidats(num_sans, FEATURES_CATEGORIELLES).get(meilleur_nom)
    c_avec = _candidats(num_avec, FEATURES_CATEGORIELLES).get(meilleur_nom)
    if c_sans is None or c_avec is None:
        return None
    ms = _metriques(df[CIBLE], cross_val_predict(c_sans, X_sans, np.log1p(df[CIBLE]), cv=cv, groups=groups))
    ma = _metriques(df[CIBLE], cross_val_predict(c_avec, X_avec, np.log1p(df[CIBLE]), cv=cv, groups=groups))

    # Mesure simple de confounding : part Grand Tunis et prix médian par source.
    sources = []
    if "Source" in df.columns:
        for src, g in df.groupby("Source"):
            sources.append({
                "source": str(src), "n": int(len(g)),
                "part_grand_tunis_pct": round(float(pd.to_numeric(g["Zone_Economique"], errors="coerce").mean() * 100), 1),
                "prix_median_dt": int(round(g[CIBLE].median())),
            })
    return {
        "production_sans_zone_mdape_pct": round(ms["mdape"] * 100, 2),
        "avec_zone_mdape_pct": round(ma["mdape"] * 100, 2),
        "gain_zone_points": round((ms["mdape"] - ma["mdape"]) * 100, 2),
        "decision": "Zone exclue de la production tant que le gain n'est pas stable en source-holdout.",
        "par_source": sources,
    }


def calculer_nb_comparables_locaux(df: pd.DataFrame) -> pd.Series:
    """Nombre de vrais voisins comparables par annonce.

    Règles : même marque/modèle, âge ±2 ans, kilométrage proche (± max 30k,
    35% du km). L'énergie est imposée lorsqu'elle est connue. La boîte de
    vitesses (``Boite_Vitesse``) est imposée seulement si elle laisse au moins
    3 candidats, pour ne pas tuer les petits échantillons. La ligne elle-même
    ET ses doublons (même voiture publiée ailleurs) sont exclus : ce ne sont
    pas des comparables indépendants.
    """
    result = pd.Series(0, index=df.index, dtype="int64")
    doublons = cle_doublon(df)
    col_boite = "Boite_Vitesse" if "Boite_Vitesse" in df else ("Transmission" if "Transmission" in df else None)
    age_all = pd.to_numeric(df.get("Age_Vehicule"), errors="coerce")
    km_all = pd.to_numeric(df.get("Kilométrage"), errors="coerce")

    for _, idx in df.groupby(["Marque", "Modèle"], dropna=False).groups.items():
        idx = pd.Index(idx)
        if len(idx) <= 1:
            continue
        g = df.loc[idx]
        ages = age_all.loc[idx]
        kms = km_all.loc[idx]
        for i in idx:
            mask = pd.Series(True, index=idx)
            if pd.notna(age_all.loc[i]):
                mask &= (ages - age_all.loc[i]).abs() <= 2
            if pd.notna(km_all.loc[i]):
                tol = max(30000.0, 0.35 * max(float(km_all.loc[i]), 40000.0))
                mask &= (kms - km_all.loc[i]).abs() <= tol
            energie = g.at[i, "Energie"] if "Energie" in g else None
            if pd.notna(energie):
                mask &= g["Energie"].eq(energie) | g["Energie"].isna()
            boite = g.at[i, col_boite] if col_boite else None
            if col_boite and pd.notna(boite):
                boite_mask = mask & (g[col_boite].eq(boite) | g[col_boite].isna())
                if int(boite_mask.sum()) - int(bool(boite_mask.get(i, False))) >= 3:
                    mask = boite_mask
            mask &= doublons.loc[idx] != doublons.loc[i]
            result.loc[i] = int(mask.sum())
    return result


def calculer_importance_shap(pipeline, X, chemin=OUT_SHAP, n_echantillon=120):
    """Importance globale SHAP par feature d'origine (one-hot réagrégé).

    En cas d'échec, le fichier est réécrit avec ``status: failed`` : un ancien
    JSON ne doit jamais rester affiché comme s'il décrivait le modèle actuel.
    """
    model = pipeline.named_steps["model"]
    noms = list(X.columns)
    try:
        Xs = X.sample(min(n_echantillon, len(X)), random_state=42)
        imp = np.abs(shap_par_feature(pipeline, Xs, noms)).mean(axis=0)
        total = float(np.sum(imp)) or 1.0
        importances = sorted(
            ({"feature": n, "pct": round(100 * float(v) / total, 1)} for n, v in zip(noms, imp, strict=True)),
            key=lambda d: d["pct"], reverse=True,
        )
        payload = {
            "modele": type(model).__name__, "status": "ok", "n_lignes": int(len(Xs)),
            "echelle": "log-prix", "importances": importances,
            "date_calcul": datetime.now().isoformat(timespec="seconds"),
        }
        print(f"Interprétation SHAP sauvegardée -> {chemin}")
    except Exception as e:
        print(f"SHAP ignoré ({type(e).__name__}: {str(e)[:140]}).")
        payload = {"modele": type(model).__name__, "status": "failed", "importances": [],
                   "erreur": type(e).__name__}
    ecrire_json_atomique(chemin, payload)


def diagnostic_stabilite(df, X, y, pipelines, seeds=(7, 21, 42, 84, 123), groups=None):
    """Répète le benchmark des finalistes pour mesurer le bruit de sélection.

    - 5 KFold avec seeds différentes ;
    - 3 holdouts temporels (15/20/25% les plus récents).
    Une différence inférieure à la variabilité observée est traitée comme un
    quasi-ex-aequo plutôt que comme une victoire artificielle au centième.
    """
    out = {}
    groups = cle_doublon(df) if groups is None else groups
    dates = pd.to_datetime(df.get("Annonce-Detectee"), errors="coerce")
    dates_ok = dates.notna().sum() >= max(50, int(len(df) * 0.5))
    for nom, pipeline in pipelines.items():
        k_scores=[]
        for seed in seeds:
            cv = cv_groupee(5, seed)
            pred=cross_val_predict(clone(pipeline), X, y, cv=cv, groups=groups)
            k_scores.append(_metriques(df[CIBLE], pred)["mdape"])
        t_scores=[]
        for frac in ((0.15, 0.20, 0.25) if dates_ok else ()):
            tr, te = _split_temporel(dates, frac)
            if te is None:
                continue
            te = _sans_doublons_du_train(groups, tr, te)
            if len(te) >= 20 and len(tr) >= 50:
                t_scores.append(_eval_holdout(pipeline,X.loc[tr],y.loc[tr],X.loc[te],df.loc[te,CIBLE])["mdape"])
        vals=k_scores+t_scores
        out[nom]={
            "kfold_seeds_pct":[round(x*100,2) for x in k_scores],
            "time_holdouts_pct":[round(x*100,2) for x in t_scores],
            "moyenne_pct":round(float(np.mean(vals))*100,2),
            "ecart_type_pct":round(float(np.std(vals,ddof=1))*100,2) if len(vals)>1 else 0.0,
            "min_pct":round(float(np.min(vals))*100,2),
            "max_pct":round(float(np.max(vals))*100,2),
        }
    return out

def score_robuste(generalisation: dict, stabilite: dict) -> float:
    """Erreur « robuste » d'un finaliste (plus bas = meilleur).

    Moyenne de trois familles de tests, à poids égal :
    - stabilité : 5 CV groupées (seeds) + 3 holdouts temporels ;
    - GroupKFold marque-modèle (familles jamais vues) ;
    - leave-one-source-out, moyenne PONDÉRÉE par le nombre d'annonces testées.

    Avant : la CV et le holdout temporel n'étaient comptés qu'une fois chacun,
    et la moyenne des sources n'était pas pondérée — la petite source
    autocentral (bruitée, déjà en quarantaine) pesait autant que tayara ou
    automobile.tn et décidait seule du modèle retenu.
    """
    valeurs = [stabilite["moyenne_pct"] / 100]
    if generalisation.get("groupkfold_marque_modele"):
        valeurs.append(generalisation["groupkfold_marque_modele"]["mdape"])
    src = generalisation.get("source_holdout") or []
    poids = [max(int(x.get("n_test") or 0), 0) for x in src]
    if src and sum(poids):
        valeurs.append(float(np.average([x["mdape"] for x in src], weights=poids)))
    return float(np.mean(valeurs))


def choisir_modele(scores: dict[str, float], stabilite: dict[str, dict]) -> tuple[str, str]:
    """Retient le meilleur score robuste, sauf quasi ex-aequo.

    Si l'écart de score robuste est inférieur au bruit mesuré (écart-type de
    stabilité du finaliste le plus instable), la différence n'est pas
    significative : on retient le modèle le plus précis ET stable en
    validation répétée (moyenne de stabilité la plus basse).
    """
    tries = sorted(scores, key=scores.get)
    if len(tries) == 1:
        return tries[0], "seul candidat"
    a, b = tries[0], tries[1]
    bruit = max(stabilite[a]["ecart_type_pct"], stabilite[b]["ecart_type_pct"]) / 100
    if scores[b] - scores[a] < bruit:
        gagnant = min((a, b), key=lambda k: stabilite[k]["moyenne_pct"])
        return gagnant, (
            f"quasi ex-aequo (écart {abs(scores[b] - scores[a]):.2%} < bruit {bruit:.2%}) : "
            f"meilleure stabilité ({stabilite[gagnant]['moyenne_pct']:.2f} %)"
        )
    return a, f"meilleur score robuste ({scores[a]:.2%} vs {scores[b]:.2%})"


def diagnostic_boite(df, candidats_noms, n_splits=5, groups=None):
    """Mesure l'apport de Boite_Vitesse sans la confondre avec la transmission aux roues."""
    if "Boite_Vitesse" not in df.columns: return {}
    groups = cle_doublon(df) if groups is None else groups
    cv = cv_groupee(n_splits)
    cats_avec=list(FEATURES_CATEGORIELLES)
    cats_sans=[c for c in cats_avec if c!="Boite_Vitesse"]
    y=np.log1p(df[CIBLE])
    result={"completude_par_source":{}}
    for src,g in (df.groupby("Source") if "Source" in df else []):
        result["completude_par_source"][str(src)]={
            "n":int(len(g)),"pct_renseigne":round(float(g["Boite_Vitesse"].notna().mean()*100),1),
            "manuelle":int(g["Boite_Vitesse"].astype(str).str.lower().eq("manuelle").sum()),
            "automatique":int(g["Boite_Vitesse"].astype(str).str.lower().eq("automatique").sum())}
    result["modeles"]={}
    for nom in candidats_noms:
        vals={}
        for label,cats in [("avec_boite",cats_avec),("sans_boite",cats_sans)]:
            X=construire_features_entrainement(df,FEATURES_NUMERIQUES,cats)
            pipe=_candidats(FEATURES_NUMERIQUES,cats).get(nom)
            if pipe is None: continue
            pred=cross_val_predict(pipe,X,y,cv=cv,groups=groups)
            vals[label]=round(_metriques(df[CIBLE],pred)["mdape"]*100,2)
        result["modeles"][nom]=vals
    return result

def main():
    df = pd.read_csv(IN_FICHIER, sep=";", encoding="utf-8-sig")
    df = df.dropna(subset=[CIBLE, "Marque"]).copy()
    df.index = pd.RangeIndex(len(df))

    X = construire_features_entrainement(df)
    marques_connues, modeles_connus = categories_frequentes(df)
    y = np.log1p(df[CIBLE])
    groupes = cle_doublon(df)
    n_groupes = int(groupes.nunique())
    n_splits = 5 if len(df) >= 100 else max(2, min(5, len(df) // 10))
    n_splits = max(2, min(n_splits, n_groupes))
    cv = cv_groupee(n_splits)

    print(f"Entraînement sur {len(df)} annonces / {n_groupes} véhicules distincts ({n_splits} folds groupés).\n")
    resultats = evaluer_modeles(X, y, df[CIBLE], cv, groups=groupes)

    # Le KFold aléatoire ne décide pas seul : si deux modèles sont très proches,
    # on compare les 2 meilleurs sur des scénarios plus réalistes. Le score
    # robuste est la moyenne de : KFold, GroupKFold, holdout temporel et moyenne
    # des source-holdouts. Cela évite de choisir un modèle pour 0,05 point de
    # MdAPE s'il généralise moins bien dans le temps ou entre sources.
    shortlist = sorted(resultats, key=lambda k: resultats[k]["mdape"])[:2]
    robustes = {}
    print("\nDiagnostics de généralisation des 2 meilleurs candidats...")
    for nom in shortlist:
        robustes[nom] = diagnostics_generalisation(
            df, X, y, resultats[nom]["pipeline"], n_splits=n_splits, doublons=groupes
        )

    stabilite = diagnostic_stabilite(df, X, y, {nom: resultats[nom]["pipeline"] for nom in shortlist}, groups=groupes)
    print("\nStabilité des finalistes (seeds + holdouts temporels) :")
    for nom, st in stabilite.items():
        print(f"  {nom:22s} moyenne={st['moyenne_pct']:.2f}% ± {st['ecart_type_pct']:.2f} pt "
              f"[{st['min_pct']:.2f}; {st['max_pct']:.2f}]")

    scores_robustes = {nom: score_robuste(robustes[nom], stabilite[nom]) for nom in shortlist}
    for nom in shortlist:
        print(f"  {nom:22s} score robuste={scores_robustes[nom]:.2%}")
    meilleur_nom, raison_choix = choisir_modele(scores_robustes, stabilite)
    print(f"  Choix : {raison_choix}")
    meilleur_pipeline = resultats[meilleur_nom]["pipeline"]
    erreur_relative_typique = resultats[meilleur_nom]["mdape"]
    generalisation = robustes[meilleur_nom]
    print(f"\n-> Modèle retenu : {meilleur_nom} (KFold {erreur_relative_typique:.2%}, "
          f"score robuste {scores_robustes[meilleur_nom]:.2%})")
    zone = diagnostic_biais_zone(df, meilleur_nom, cv, groups=groupes)

    date_entrainement = datetime.now().isoformat(timespec="seconds")
    diagnostics = {
        "date_entrainement": date_entrainement,
        "n_annonces": int(len(df)), "n_vehicules_distincts": n_groupes, "n_folds": int(n_splits),
        "validation_croisee": "GroupKFold par véhicule (doublons inter-sources regroupés)",
        "fenetre_jours": MAX_DAYS_OLD, "modele_retenu": meilleur_nom,
        "features_numeriques": FEATURES_NUMERIQUES,
        "features_categorielles": FEATURES_CATEGORIELLES,
        "candidats": [
            {"nom": nom, "mdape_pct": round(r["mdape"] * 100, 2),
             "mdae_dt": int(round(r["mdae"])), "mae_dt": int(round(r["mae"]))}
            for nom, r in resultats.items()
        ],
        "selection_robuste": {
            "shortlist": [
                {"nom": nom, "score_robuste_pct": round(scores_robustes[nom] * 100, 2)}
                for nom in shortlist
            ],
            "regle": "moyenne : stabilité (5 CV groupées + 3 holdouts temporels) + GroupKFold marque-modèle "
                     "+ source-holdouts pondérés par volume ; quasi ex-aequo départagé par la stabilité",
            "raison": raison_choix,
        },
        "validation_robuste": generalisation,
        "validation_finalistes": robustes,
        "stabilite_finalistes": stabilite,
        "diagnostic_boite_vitesses": diagnostic_boite(df, shortlist, n_splits, groups=groupes),
        "diagnostic_zone_economique": zone,
    }
    # OOF : scores de marché honnêtes.
    df["Prix_Theorique"] = np.clip(np.expm1(resultats[meilleur_nom]["predictions_log"]).round(), 1, None).astype(int)
    df["Score_Opportunite"] = (df["Prix_Theorique"] - df[CIBLE]) / df["Prix_Theorique"]
    df["Erreur_Absolue"] = (df[CIBLE] - df["Prix_Theorique"]).abs()

    # Profondeur de marché != liquidité réelle. On garde l'ancien alias pour
    # compatibilité UI, mais la nouvelle colonne porte le bon nom métier.
    volumes = df.groupby(["Marque", "Modèle"], dropna=False)[CIBLE].transform("count").fillna(0)
    vmin, vmax = float(volumes.min()), float(volumes.max())
    profondeur = (volumes - vmin) / (vmax - vmin) if vmax > vmin else pd.Series(0.0, index=df.index)
    df["Score_Profondeur_Marche"] = profondeur.round(2)
    df["Score_Liquidite"] = df["Score_Profondeur_Marche"]  # alias historique

    df["Nb_Comparables"] = calculer_nb_comparables_locaux(df)
    df["Fiabilite_Estimation"] = pd.cut(
        df["Nb_Comparables"], bins=[-1, 2, 7, np.inf], labels=["Faible", "Moyenne", "Élevée"]
    )
    df["Erreur_Relative_Modele"] = round(float(erreur_relative_typique), 4)

    df_final = df.sort_values("Score_Opportunite", ascending=False)

    # Modèle calculateur : entraîné sur 100% uniquement pour les nouvelles saisies.
    meilleur_pipeline.fit(X, y)
    bundle = {
        "pipeline": meilleur_pipeline,
        "features_numeriques": FEATURES_NUMERIQUES,
        "features_categorielles": FEATURES_CATEGORIELLES,
        "marques_connues": sorted(marques_connues or []),
        "modeles_connus": sorted(modeles_connus or []),
        "modele_nom": meilleur_nom,
        "mdape_oof": float(erreur_relative_typique),
        "date_entrainement": date_entrainement,
    }

    # Écritures atomiques groupées en fin de calcul : un crash ou un timeout
    # ne laisse jamais des scores OOF neufs à côté d'un ancien modèle, ni un
    # pickle tronqué. Le modèle est écrit en dernier.
    ecrire_json_atomique(OUT_DIAGNOSTICS, diagnostics)
    ecrire_csv_atomique(df_final, OUT_FICHIER)
    dump_joblib_atomique(bundle, OUT_MODELE)

    calculer_importance_shap(meilleur_pipeline, X, chemin=OUT_SHAP)
    print(f"\nRésultats -> {OUT_FICHIER}\nModèle -> {OUT_MODELE}\nDiagnostics -> {OUT_DIAGNOSTICS}")
    print("\nTop 5 opportunités :")
    print(df_final[["Titre", "Marque", "Modèle", CIBLE, "Prix_Theorique", "Score_Opportunite", "Nb_Comparables"]].head().to_string(index=False))


if __name__ == "__main__":
    main()
