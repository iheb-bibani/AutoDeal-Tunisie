"""
modele_prediction.py
Script UNIQUE de modélisation du "Juste Prix", qui remplace les 6 variantes
qui coexistaient (modele_prediction.py, modele_prediction_seg.py,
validation_modele.py, validation_rigoureuse.py, validation_globale.py,
analyse_finale_robuste.py) -- toutes issues d'itérations successives sur le
même problème, jamais consolidées.

Pourquoi une seule version, et pourquoi celle-ci :

1. PAS de segmentation manuelle par Marque+Segment. Les tentatives précédentes
   (ex: groupe = Marque + "_" + Segment_Vehicule, entraîné dès que len > 5 ou
   > 10 lignes) réentraînaient un modèle par groupe minuscule -> surapprentissage
   quasi garanti, R2 instable, et un bug de seuil (`Segment_Vehicule > 2` pour
   "Luxe" alors que Segment_Vehicule ne prend que les valeurs 0/1) qui rendait
   le segment Luxe inatteignable dans validation_globale.py et
   analyse_finale_robuste.py.
   A la place : UN SEUL modèle sur toute la base, avec Marque en one-hot
   (marques rares regroupées) et Segment_Vehicule/Zone_Economique comme
   features -- le modèle apprend lui-même les interactions, sans qu'on ait à
   deviner des seuils de regroupement.

2. Validation croisée (KFold=5) pour choisir le modèle, PAS un simple .fit()
   sur toutes les données -- certaines variantes (modele_prediction_seg.py,
   validation_modele.py) n'avaient aucune validation du tout.

3. Prix_Theorique et Score_Opportunite calculés en OUT-OF-FOLD
   (cross_val_predict). C'est le point que même validation_rigoureuse.py et
   validation_globale.py n'appliquaient pas jusqu'au bout : elles affichaient
   un R2 honnête en CV à l'écran, mais le fichier exporté utilisait quand même
   des prédictions in-sample (le modèle note les voitures qu'il vient
   d'apprendre par coeur) -- ce qui rend Score_Opportunite artificiellement
   optimiste. Ici, la donnée exportée est aussi honnête que le score affiché.

Entrée  : data/tunisia-cars-final-features.csv
Sortie  : data/tunisia-cars-scored.csv (toutes les annonces + Prix_Theorique
          + Score_Opportunite), qui remplace les 6 fichiers resultats_*.csv
"""

import json
import os
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer

IN_FICHIER = "data/processed/tunisia-cars-final-features.csv"
OUT_FICHIER = "data/processed/tunisia-cars-scored.csv"
OUT_MODELE = "data/models/modele_prix.pkl"
OUT_DIAGNOSTICS = "data/processed/diagnostics_modele.json"

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import MAX_DAYS_OLD

FEATURES_NUMERIQUES = ["Kilométrage", "Année", "Age_Vehicule", "Puissance_Fiscale", "Puissance_DIN", "Cylindree", "Segment_Vehicule", "Est_Presque_Neuve", "Zone_Economique"]
FEATURES_CATEGORIELLES = ["Marque", "Modèle", "Boite_Vitesse", "Energie", "Transmission"]
CIBLE = "Prix"

SEUIL_MARQUE_RARE = 10   # marques avec moins de N annonces -> regroupées en "Autre"
SEUIL_MODELE_RARE = 5    # modèles avec moins de N annonces -> regroupés en "Autre_modele"
                         # (seuil plus bas que pour Marque : il y a naturellement
                         # beaucoup plus de modèles distincts que de marques)


def construire_features_entrainement(df: pd.DataFrame) -> pd.DataFrame:
    """Construit la matrice X en regroupant marques et modèles rares.

    Le regroupement en "Autre" / "Autre_modele" n'existe QUE dans cette
    matrice d'entraînement : les colonnes Marque et Modèle du DataFrame
    exporté conservent les vrais noms. (Auparavant, le regroupement écrasait
    les colonnes elles-mêmes -- l'application affichait "Autre_modele" pour
    des centaines d'annonces dont le modèle était parfaitement connu.)
    """
    X = df[FEATURES_NUMERIQUES + FEATURES_CATEGORIELLES].copy()

    counts = X["Marque"].value_counts()
    rares = counts[counts < SEUIL_MARQUE_RARE].index
    X["Marque"] = X["Marque"].where(~X["Marque"].isin(rares), "Autre")

    X["Modèle"] = X["Modèle"].fillna("Inconnu")
    counts = X["Modèle"].value_counts()
    rares = counts[counts < SEUIL_MODELE_RARE].index
    X["Modèle"] = X["Modèle"].where(~X["Modèle"].isin(rares), "Autre_modele")

    return X


def construire_pipeline_onehot(modele) -> Pipeline:
    """Pour Ridge et RandomForest : imputation (médiane/valeur la plus
    fréquente) + one-hot encoding classique."""
    preprocesseur = ColumnTransformer([
        ("num", Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]), FEATURES_NUMERIQUES),
        ("cat", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore")),
        ]), FEATURES_CATEGORIELLES),
    ])
    return Pipeline([("prep", preprocesseur), ("model", modele)])


def construire_pipeline_hgb() -> Pipeline:
    """NOUVEAU : HistGradientBoostingRegressor gère nativement les valeurs
    manquantes et les catégories -- pas d'imputation devinée (médiane/valeur
    la plus fréquente), le modèle apprend directement à partir des vrais
    trous plutôt que de les combler par une supposition. Encodage ordinal
    (pas one-hot) : HGB découpe lui-même les catégories, l'ordre numérique
    n'a pas de sens particulier mais ne biaise pas le modèle (contrairement à
    Ridge, qui interpréterait un ordre arbitraire comme une vraie échelle)."""
    preprocesseur = ColumnTransformer([
        ("num", "passthrough", FEATURES_NUMERIQUES),
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), FEATURES_CATEGORIELLES),
    ])
    indices_categorielles = list(range(len(FEATURES_NUMERIQUES), len(FEATURES_NUMERIQUES) + len(FEATURES_CATEGORIELLES)))
    modele = HistGradientBoostingRegressor(categorical_features=indices_categorielles, random_state=42)
    return Pipeline([("prep", preprocesseur), ("model", modele)])


def evaluer_modeles(X: pd.DataFrame, y: pd.Series, y_brut: pd.Series, cv) -> dict:
    """Compare Ridge, RandomForest et HistGradientBoosting.

    La sélection se fait sur l'ERREUR RELATIVE MÉDIANE (MdAPE), pas sur le MAE
    ni sur le R2 :

    - le R2 mesure la variance expliquée sur l'échelle log-prix, très loin de
      ce que l'utilisateur constate ;
    - le MAE en dinars est écrasé par le haut de gamme. Sur les données
      réelles, le MAE global de 15 000 DT vient presque entièrement des
      voitures à plus de 200 000 DT (MAE 75 000 DT sur ce segment, contre
      6 700 DT sous 30 000 DT) : sélectionner sur le MAE revient à choisir le
      modèle qui estime le mieux les Porsche, pas celui qui estime le mieux
      les voitures que la plupart des gens achètent ;
    - Score_Opportunite est un écart RELATIF ((théorique - prix) / théorique).
      La métrique de sélection doit donc être relative elle aussi, sinon on
      optimise autre chose que ce qu'on publie.

    Le MAE et l'erreur absolue médiane restent affichés pour information.
    """
    candidats = {
        "Ridge (baseline linéaire)": construire_pipeline_onehot(Ridge(alpha=1.0)),
        "RandomForest": construire_pipeline_onehot(RandomForestRegressor(n_estimators=300, max_depth=12, random_state=42, n_jobs=-1)),
        "HistGradientBoosting": construire_pipeline_hgb(),
    }
    resultats = {}
    for nom, pipeline in candidats.items():
        predictions_log = cross_val_predict(pipeline, X, y, cv=cv)
        prix_predit = np.expm1(predictions_log)
        erreur = (y_brut - prix_predit).abs()
        mae = erreur.mean()
        mdae = erreur.median()
        mdape = (erreur / y_brut).median()
        resultats[nom] = {
            "pipeline": pipeline, "mae": mae, "mdae": mdae, "mdape": mdape,
            "predictions_log": predictions_log,
        }
        print(f"{nom:25s} erreur relative médiane = {mdape:5.1%}   "
              f"(médiane {mdae:6.0f} DT, moyenne {mae:6.0f} DT)")
    return resultats


if __name__ == "__main__":
    df = pd.read_csv(IN_FICHIER, sep=";", encoding="utf-8-sig")

    # Etat_Vehicule est vide à 100% dans les données actuelles (bug d'extraction
    # côté scrapers, déjà signalé) -- on le retire des features, il n'apporte rien.
    df = df.dropna(subset=[CIBLE])

    # Garde-fou : une annonce sans Marque n'est pas exploitable (souvent une
    # annonce immobilière mal cataloguée, voir nettoyer_base.py). Le filtre en
    # amont devrait déjà les avoir retirées, mais on ne prend pas de risque ici.
    avant = len(df)
    df = df.dropna(subset=["Marque"])
    if avant != len(df):
        print(f"🧹 {avant - len(df)} annonces sans Marque écartées avant l'entraînement.")

    n_min_cv = 5
    if len(df) < n_min_cv * 2:
        print(f"⚠️ Seulement {len(df)} annonces disponibles -- pas assez pour une validation croisée fiable (minimum recommandé : ~50-100).")
        print("   Relance ce script avec un volume plus important avant de faire confiance aux scores ci-dessous.")

    X = construire_features_entrainement(df)
    y = np.log1p(df[CIBLE])  # log1p stabilise la distribution des prix (pratique déjà utilisée avant, conservée)

    n_splits = min(5, max(2, len(df) // 5)) if len(df) >= 10 else 2
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    print(f"Entraînement sur {len(df)} annonces ({n_splits} folds).\n")
    resultats = evaluer_modeles(X, y, df[CIBLE], cv)

    meilleur_nom = min(resultats, key=lambda k: resultats[k]["mdape"])
    meilleur_pipeline = resultats[meilleur_nom]["pipeline"]
    erreur_relative_typique = resultats[meilleur_nom]["mdape"]
    print(f"\n-> Modèle retenu : {meilleur_nom} (meilleure erreur relative médiane)")

    # Diagnostics persistés pour l'onglet Admin de l'application : sans ça,
    # la comparaison des modèles n'existe que dans la sortie console d'un
    # entraînement déjà terminé, et n'est plus consultable nulle part.
    diagnostics = {
        "date_entrainement": datetime.now().isoformat(timespec="seconds"),
        "n_annonces": int(len(df)),
        "n_folds": int(n_splits),
        "fenetre_jours": MAX_DAYS_OLD,
        "modele_retenu": meilleur_nom,
        "features_numeriques": FEATURES_NUMERIQUES,
        "features_categorielles": FEATURES_CATEGORIELLES,
        "candidats": [
            {"nom": nom, "mdape_pct": round(r["mdape"] * 100, 2),
             "mdae_dt": int(round(r["mdae"])), "mae_dt": int(round(r["mae"]))}
            for nom, r in resultats.items()
        ],
    }
    os.makedirs(os.path.dirname(OUT_DIAGNOSTICS), exist_ok=True)
    with open(OUT_DIAGNOSTICS, "w", encoding="utf-8") as f:
        json.dump(diagnostics, f, ensure_ascii=False, indent=2)
    print(f"Diagnostics sauvegardés -> {OUT_DIAGNOSTICS}")

    # Prix_Theorique HONNETE : prédictions out-of-fold déjà calculées dans
    # evaluer_modeles() -- pas besoin de refaire un cross_val_predict, on
    # réutilise directement (même règle qu'avant : jamais un modèle qui a
    # déjà vu la ligne pendant qu'il calcule sa prédiction pour cette ligne).
    predictions_log = resultats[meilleur_nom]["predictions_log"]
    df["Prix_Theorique"] = np.expm1(predictions_log).round().astype(int)

    df["Score_Opportunite"] = (df["Prix_Theorique"] - df[CIBLE]) / df["Prix_Theorique"]
    df["Erreur_Absolue"] = (df[CIBLE] - df["Prix_Theorique"]).abs()

    # NOUVEAU : Score_Liquidite (mentionné dans nos échanges précédents mais
    # jamais implémenté). Proxy de popularité par volume d'annonces pour le
    # couple Marque+Modèle -- pas une vraie durée de vie (incompatible avec
    # le filtre à 30 jours, voir la discussion complète à ce sujet), mais un
    # signal utile : un modèle qui apparaît souvent est plus facile à revendre.
    volumes = df.groupby(["Marque", "Modèle"])[CIBLE].transform("count")
    v_min, v_max = volumes.min(), volumes.max()
    df["Score_Liquidite"] = ((volumes - v_min) / (v_max - v_min)).round(2) if v_max > v_min else 0.0

    # NOUVEAU : fiabilité de l'estimation. Un Prix_Theorique appuyé sur 60
    # Golf 7 et un Prix_Theorique appuyé sur 2 annonces d'un modèle rare
    # s'affichaient jusqu'ici exactement de la même façon, alors qu'ils n'ont
    # pas du tout la même valeur -- une "affaire" sur un modèle quasi absent
    # du marché est le plus souvent une erreur d'estimation, pas une affaire.
    df["Nb_Comparables"] = volumes
    df["Fiabilite_Estimation"] = pd.cut(
        volumes, bins=[-1, 7, 19, np.inf], labels=["Faible", "Moyenne", "Élevée"]
    )

    # Erreur relative typique du modèle : sert de garde-fou au seuil de
    # détection des deals (voir detect_deals.py). Stockée pour que l'aval
    # n'ait pas à la redeviner.
    df["Erreur_Relative_Modele"] = round(float(erreur_relative_typique), 4)

    print(f"\nErreur absolue moyenne (MAE, out-of-fold) : {df['Erreur_Absolue'].mean():.0f} DT")

    df_final = df.sort_values("Score_Opportunite", ascending=False)
    df_final.to_csv(OUT_FICHIER, index=False, sep=";", encoding="utf-8-sig")

    # Ré-entraînement final sur 100% des données (les prédictions out-of-fold
    # ci-dessus restent honnêtes pour Score_Opportunite ; ce modèle-ci, entraîné
    # sur tout, sert uniquement au calculateur "Juste Prix" de l'app -- une
    # nouvelle voiture qu'on saisit à la main n'a jamais été vue par personne).
    meilleur_pipeline.fit(X, y)
    os.makedirs(os.path.dirname(OUT_MODELE), exist_ok=True)
    joblib.dump({
        "pipeline": meilleur_pipeline,
        "features_numeriques": FEATURES_NUMERIQUES,
        "features_categorielles": FEATURES_CATEGORIELLES,
    }, OUT_MODELE)

    print(f"\nRésultats sauvegardés -> {OUT_FICHIER}")
    print(f"Modèle sauvegardé -> {OUT_MODELE}")
    print("\nTop 5 opportunités :")
    print(df_final[["Titre", "Marque", "Modèle", CIBLE, "Prix_Theorique", "Score_Opportunite"]].head(5).to_string(index=False))
