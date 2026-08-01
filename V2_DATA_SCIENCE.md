# AutoDeal Tunisie — V2 Data Science

Cette version renforce la partie Data Science avant toute nouvelle amélioration UI.

## 1. Benchmark modèles

Le pipeline compare désormais : Ridge, Random Forest, HistGradientBoosting, CatBoost, LightGBM et XGBoost.
La sélection ne repose plus uniquement sur le KFold aléatoire : les deux meilleurs candidats sont stress-testés avec GroupKFold Marque-Modèle, holdout temporel et leave-one-source-out.

Sur le dataset livré (3 331 annonces récentes après nettoyage) :
- Ridge : MdAPE 11,29 %
- Random Forest : 12,25 %
- HistGradientBoosting : 10,49 %
- CatBoost : 11,77 %
- LightGBM : 12,08 %
- XGBoost : 10,44 %

XGBoost gagne de très peu le KFold standard, mais HistGradientBoosting gagne le score robuste de généralisation : 12,71 % contre 13,02 %. HGB est donc conservé en production.

## 2. Validation robuste

`data/processed/diagnostics_modele.json` contient maintenant :
- KFold out-of-fold ;
- GroupKFold par Marque-Modèle ;
- holdout temporel sur les 20 % les plus récents ;
- holdout complet de chaque source ;
- diagnostic séparé de `Zone_Economique`.

## 3. Biais Zone_Economique

`Zone_Economique` reste dans la base pour les analyses géographiques, mais est exclue des features du modèle de prix. Le diagnostic montre que son petit gain KFold n'est pas suffisamment fiable face au fort confounding source / gamme de prix.

## 4. Feature engineering

Ajouts :
- `Log_Kilometrage = log1p(Kilométrage)` ;
- `Age_Carre = Age_Vehicule²` ;
- `Km_Par_An = Kilométrage / max(Age_Vehicule, 1)` avec garde-fou.

Les scrapers Tayara, AutoMax et automobile.tn tentent maintenant aussi de récupérer `Description`, `Options` et `Nb_Options` lorsqu'ils sont disponibles. Ces données nécessitent un nouveau scraping pour être remplies dans l'historique.

## 5. Nettoyage renforcé

Les véhicules non standards (épaves, sans carte grise, pour pièces, moteur HS, sinistrés / accidentés explicitement) sont exclus du modèle de marché afin de ne pas créer de faux deals.

## 6. Vrais comparables

`Nb_Comparables` n'est plus le volume total Marque-Modèle. Un comparable doit désormais être :
- même marque et modèle ;
- âge à ±2 ans ;
- kilométrage dans une tolérance locale ;
- même énergie lorsqu'elle est connue ;
- même transmission lorsque l'échantillon reste suffisant.

Le compteur devient donc une vraie mesure locale de support de l'estimation.

## 7. Visualisations corrigées

- « Parts de marché » -> « Part des annonces observées » ;
- KPI 30 jours -> fenêtre réelle `MAX_DAYS_OLD` (60 jours) ;
- tendance du prix médian tracée par source pour limiter le biais de composition ;
- courbe de dépréciation : plus aucune suppression opportuniste des premiers âges ; utilisation d'une régression isotone décroissante pondérée ;
- `Score_Liquidite` est présenté comme profondeur de marché / proxy, pas comme vitesse de revente observée.

## Validation

72 tests pytest passent sur cette version.
