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

import json
import os
import sys
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold, KFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

sys.path.insert(0, str(Path(__file__).parent.parent))

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

    if "Marque" in X:
        counts = X["Marque"].value_counts()
        rares = counts[counts < SEUIL_MARQUE_RARE].index
        X["Marque"] = X["Marque"].where(~X["Marque"].isin(rares), "Autre")

    if "Modèle" in X:
        X["Modèle"] = X["Modèle"].fillna("Inconnu")
        counts = X["Modèle"].value_counts()
        rares = counts[counts < SEUIL_MODELE_RARE].index
        X["Modèle"] = X["Modèle"].where(~X["Modèle"].isin(rares), "Autre_modele")
    return X


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


def evaluer_modeles(X: pd.DataFrame, y: pd.Series, y_brut: pd.Series, cv, features_num=None, features_cat=None) -> dict:
    resultats = {}
    for nom, pipeline in _candidats(features_num, features_cat).items():
        predictions_log = cross_val_predict(pipeline, X, y, cv=cv)
        m = _metriques(y_brut, predictions_log)
        resultats[nom] = {"pipeline": pipeline, "predictions_log": predictions_log, **m}
        print(f"{nom:22s} MdAPE={m['mdape']:5.1%} | médiane={m['mdae']:7.0f} DT | MAE={m['mae']:7.0f} DT")
    return resultats


def _eval_holdout(pipeline, X_train, y_train, X_test, y_test_brut):
    p = clone(pipeline)
    p.fit(X_train, y_train)
    return _metriques(y_test_brut, p.predict(X_test))


def diagnostics_generalisation(df, X, y, meilleur_pipeline, n_splits=5):
    """Mesure ce que le KFold aléatoire ne voit pas."""
    sortie = {}

    # GroupKFold par famille Marque-Modèle : aucune famille du fold de test n'a
    # été vue à l'entraînement. C'est volontairement un stress-test sévère.
    groupes = (df["Marque"].fillna("?").astype(str) + "|" + df["Modèle"].fillna("?").astype(str))
    n_groupes = groupes.nunique()
    if n_groupes >= 3:
        gcv = GroupKFold(n_splits=min(n_splits, n_groupes))
        pred = cross_val_predict(clone(meilleur_pipeline), X, y, cv=gcv, groups=groupes)
        sortie["groupkfold_marque_modele"] = _metriques(df[CIBLE], pred)

    # Holdout temporel : 20% des annonces les plus récentes, sans shuffle.
    dates = pd.to_datetime(df.get("Annonce-Detectee"), errors="coerce")
    if dates.notna().sum() >= max(50, int(len(df) * 0.5)):
        ordre = dates.fillna(pd.Timestamp.min).sort_values().index
        cut = max(1, int(len(ordre) * 0.8))
        train_idx, test_idx = ordre[:cut], ordre[cut:]
        if len(test_idx) >= 20:
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
            m = _eval_holdout(meilleur_pipeline, X.loc[train_idx], y.loc[train_idx], X.loc[idx], df.loc[idx, CIBLE])
            par_source.append({"source": str(source), "n_test": int(len(idx)), **m})
    sortie["source_holdout"] = par_source
    return sortie


def diagnostic_biais_zone(df, meilleur_nom, cv):
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
    ms = _metriques(df[CIBLE], cross_val_predict(c_sans, X_sans, np.log1p(df[CIBLE]), cv=cv))
    ma = _metriques(df[CIBLE], cross_val_predict(c_avec, X_avec, np.log1p(df[CIBLE]), cv=cv))

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
    35% du km). L'énergie est imposée lorsqu'elle est connue. La transmission
    est imposée seulement si elle laisse au moins 3 candidats, pour ne pas
    tuer les petits échantillons. La ligne elle-même est exclue.
    """
    result = pd.Series(0, index=df.index, dtype="int64")
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
            trans = g.at[i, "Transmission"] if "Transmission" in g else None
            if pd.notna(trans) and "Transmission" in g:
                trans_mask = mask & (g["Transmission"].eq(trans) | g["Transmission"].isna())
                if int(trans_mask.sum()) - int(bool(trans_mask.get(i, False))) >= 3:
                    mask = trans_mask
            mask.loc[i] = False
            result.loc[i] = int(mask.sum())
    return result


def calculer_importance_shap(pipeline, X, chemin=OUT_SHAP, n_echantillon=120):
    try:
        import shap
    except ImportError:
        print("shap non installé -> interprétation ignorée.")
        return
    try:
        Xs = X.sample(min(n_echantillon, len(X)), random_state=42)
        prep, model = pipeline.named_steps["prep"], pipeline.named_steps["model"]
        Xt = prep.transform(Xs)
        noms = list(Xs.columns)
        if getattr(Xt, "shape", (0, 0))[1] == len(noms):
            try:
                imp = np.abs(shap.TreeExplainer(model).shap_values(Xt)).mean(axis=0)
            except Exception:
                imp = np.abs(shap.Explainer(pipeline.predict, Xs)(Xs).values).mean(axis=0)
        else:
            # Explainer modèle-agnostique sur les features originales.
            imp = np.abs(shap.Explainer(pipeline.predict, Xs)(Xs).values).mean(axis=0)
        total = float(np.sum(imp)) or 1.0
        importances = sorted(
            ({"feature": n, "pct": round(100 * float(v) / total, 1)} for n, v in zip(noms, imp)),
            key=lambda d: d["pct"], reverse=True,
        )
        payload = {
            "modele": type(model).__name__, "n_lignes": int(len(Xs)),
            "echelle": "log-prix", "importances": importances,
        }
        with open(chemin, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"Interprétation SHAP sauvegardée -> {chemin}")
    except Exception as e:
        print(f"SHAP ignoré ({type(e).__name__}: {str(e)[:140]}).")


def diagnostic_stabilite(df, X, y, pipelines, seeds=(7, 21, 42, 84, 123)):
    """Répète le benchmark des finalistes pour mesurer le bruit de sélection.

    - 5 KFold avec seeds différentes ;
    - 3 holdouts temporels (15/20/25% les plus récents).
    Une différence inférieure à la variabilité observée est traitée comme un
    quasi-ex-aequo plutôt que comme une victoire artificielle au centième.
    """
    out = {}
    dates = pd.to_datetime(df.get("Annonce-Detectee"), errors="coerce")
    ordre = dates.fillna(pd.Timestamp.min).sort_values().index
    for nom, pipeline in pipelines.items():
        k_scores=[]
        for seed in seeds:
            cv=KFold(n_splits=5, shuffle=True, random_state=seed)
            pred=cross_val_predict(clone(pipeline), X, y, cv=cv)
            k_scores.append(_metriques(df[CIBLE], pred)["mdape"])
        t_scores=[]
        for frac in (0.15,0.20,0.25):
            cut=max(1,int(len(ordre)*(1-frac)))
            tr,te=ordre[:cut],ordre[cut:]
            if len(te)>=20:
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

def diagnostic_boite(df, candidats_noms, n_splits=5):
    """Mesure l'apport de Boite_Vitesse sans la confondre avec la transmission aux roues."""
    if "Boite_Vitesse" not in df.columns: return {}
    cv=KFold(n_splits=n_splits,shuffle=True,random_state=42)
    cats_avec=list(FEATURES_CATEGORIELLES)
    cats_sans=[c for c in cats_avec if c!="Boite_Vitesse"]
    y=np.log1p(df[CIBLE])
    result={"completude_par_source":{}}
    for src,g in df.groupby("Source"):
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
            pred=cross_val_predict(pipe,X,y,cv=cv)
            vals[label]=round(_metriques(df[CIBLE],pred)["mdape"]*100,2)
        result["modeles"][nom]=vals
    return result

def main():
    df = pd.read_csv(IN_FICHIER, sep=";", encoding="utf-8-sig")
    df = df.dropna(subset=[CIBLE, "Marque"]).copy()
    df.index = pd.RangeIndex(len(df))

    X = construire_features_entrainement(df)
    y = np.log1p(df[CIBLE])
    n_splits = 5 if len(df) >= 100 else max(2, min(5, len(df) // 10))
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    print(f"Entraînement sur {len(df)} annonces ({n_splits} folds).\n")
    resultats = evaluer_modeles(X, y, df[CIBLE], cv)

    # Le KFold aléatoire ne décide pas seul : si deux modèles sont très proches,
    # on compare les 2 meilleurs sur des scénarios plus réalistes. Le score
    # robuste est la moyenne de : KFold, GroupKFold, holdout temporel et moyenne
    # des source-holdouts. Cela évite de choisir un modèle pour 0,05 point de
    # MdAPE s'il généralise moins bien dans le temps ou entre sources.
    shortlist = sorted(resultats, key=lambda k: resultats[k]["mdape"])[:2]
    robustes = {}
    scores_robustes = {}
    print("\nDiagnostics de généralisation des 2 meilleurs candidats...")
    for nom in shortlist:
        d = diagnostics_generalisation(df, X, y, resultats[nom]["pipeline"], n_splits=n_splits)
        robustes[nom] = d
        valeurs = [resultats[nom]["mdape"]]
        if d.get("groupkfold_marque_modele"):
            valeurs.append(d["groupkfold_marque_modele"]["mdape"])
        if d.get("time_holdout_20pct"):
            valeurs.append(d["time_holdout_20pct"]["mdape"])
        src = [x["mdape"] for x in d.get("source_holdout", [])]
        if src:
            valeurs.append(float(np.mean(src)))
        scores_robustes[nom] = float(np.mean(valeurs))
        print(f"  {nom:22s} score robuste={scores_robustes[nom]:.2%}")

    stabilite = diagnostic_stabilite(df, X, y, {nom: resultats[nom]["pipeline"] for nom in shortlist})
    print("\nStabilité des finalistes (seeds + holdouts temporels) :")
    for nom, st in stabilite.items():
        print(f"  {nom:22s} moyenne={st['moyenne_pct']:.2f}% ± {st['ecart_type_pct']:.2f} pt "
              f"[{st['min_pct']:.2f}; {st['max_pct']:.2f}]")

    meilleur_nom = min(shortlist, key=lambda k: scores_robustes[k])
    meilleur_pipeline = resultats[meilleur_nom]["pipeline"]
    erreur_relative_typique = resultats[meilleur_nom]["mdape"]
    generalisation = robustes[meilleur_nom]
    print(f"\n-> Modèle retenu : {meilleur_nom} (KFold {erreur_relative_typique:.2%}, "
          f"score robuste {scores_robustes[meilleur_nom]:.2%})")
    zone = diagnostic_biais_zone(df, meilleur_nom, cv)

    diagnostics = {
        "date_entrainement": datetime.now().isoformat(timespec="seconds"),
        "n_annonces": int(len(df)), "n_folds": int(n_splits),
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
            "regle": "moyenne KFold + GroupKFold + holdout temporel + moyenne source-holdouts",
        },
        "validation_robuste": generalisation,
        "stabilite_finalistes": stabilite,
        "diagnostic_boite_vitesses": diagnostic_boite(df, shortlist, n_splits),
        "diagnostic_zone_economique": zone,
    }
    os.makedirs(os.path.dirname(OUT_DIAGNOSTICS), exist_ok=True)
    with open(OUT_DIAGNOSTICS, "w", encoding="utf-8") as f:
        json.dump(diagnostics, f, ensure_ascii=False, indent=2)

    # OOF : scores de marché honnêtes.
    df["Prix_Theorique"] = np.clip(np.expm1(resultats[meilleur_nom]["predictions_log"]).round(), 1, None).astype(int)
    df["Score_Opportunite"] = (df["Prix_Theorique"] - df[CIBLE]) / df["Prix_Theorique"]
    df["Erreur_Absolue"] = (df[CIBLE] - df["Prix_Theorique"]).abs()

    # Profondeur de marché != liquidité réelle. On garde l'ancien alias pour
    # compatibilité UI, mais la nouvelle colonne porte le bon nom métier.
    volumes = df.groupby(["Marque", "Modèle"])[CIBLE].transform("count")
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
    df_final.to_csv(OUT_FICHIER, index=False, sep=";", encoding="utf-8-sig")

    # Modèle calculateur : entraîné sur 100% uniquement pour les nouvelles saisies.
    meilleur_pipeline.fit(X, y)
    os.makedirs(os.path.dirname(OUT_MODELE), exist_ok=True)
    joblib.dump({
        "pipeline": meilleur_pipeline,
        "features_numeriques": FEATURES_NUMERIQUES,
        "features_categorielles": FEATURES_CATEGORIELLES,
        "modele_nom": meilleur_nom,
        "mdape_oof": float(erreur_relative_typique),
    }, OUT_MODELE)

    calculer_importance_shap(meilleur_pipeline, X)
    print(f"\nRésultats -> {OUT_FICHIER}\nModèle -> {OUT_MODELE}\nDiagnostics -> {OUT_DIAGNOSTICS}")
    print("\nTop 5 opportunités :")
    print(df_final[["Titre", "Marque", "Modèle", CIBLE, "Prix_Theorique", "Score_Opportunite", "Nb_Comparables"]].head().to_string(index=False))


if __name__ == "__main__":
    main()
