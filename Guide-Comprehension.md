# AutoDeal Tunisie — Guide de compréhension

**But de ce document :** que tu maîtrises *vraiment* tout ce qu'on a construit — pas juste que « ça marche ». Pour chaque bloc : le **problème**, le **pourquoi**, l'**outil/méthode**, l'**interprétation**, et une phrase **« comment en parler en entretien »**.

Lis-le lentement. Chaque concept ici (biais de composition, biais de survie, out-of-fold, régularité, drift, CI…) est une carte que tu peux jouer face à un recruteur.

---

## Table des matières

1. [Le fil rouge du projet](#1-le-fil-rouge-du-projet)
2. [Rigueur analytique : les 4 biais qu'on a corrigés](#2-rigueur-analytique--les-4-biais-quon-a-corrigés)
3. [Le raisonnement quant (page Samsar)](#3-le-raisonnement-quant-page-samsar)
4. [Le modèle et son interprétabilité](#4-le-modèle-et-son-interprétabilité)
5. [Industrialisation (les recommandations du review)](#5-industrialisation-les-recommandations-du-review)
6. [Git : l'incident du force-push](#6-git--lincident-du-force-push)
7. [Le bug Mermaid](#7-le-bug-mermaid)
8. [Glossaire express](#8-glossaire-express)
9. [Ce qui reste à faire](#9-ce-qui-reste-à-faire)

---

## 1. Le fil rouge du projet

AutoDeal répond à une question simple : **« cette voiture d'occasion est-elle une bonne affaire ? »**

La chaîne complète :

> 5 sources scrapées → nettoyage → feature engineering → **estimation du juste prix** (ML) → **scoring hors-échantillon** → détection d'opportunités → explication (SHAP) → suivi dans le temps → application métier (dashboard).

Ce qui fait la valeur du projet, ce n'est **pas** le modèle (un HistGradientBoosting standard). C'est **tout le reste** : la qualité des données, l'honnêteté statistique, et l'industrialisation. Retiens ça : en entretien, on ne t'embauchera pas pour « avoir utilisé XGBoost », mais pour **avoir refusé les conclusions faciles et fausses**.

---

## 2. Rigueur analytique : les 4 biais qu'on a corrigés

C'est le cœur de ce qu'on a fait cette session. Un **biais**, c'est quand un chiffre te raconte une histoire qui a l'air vraie mais qui est fausse à cause de *comment les données sont faites*. On en a corrigé quatre.

### 2.1 Le biais de composition

**Le problème.** Tu traces le prix médian par âge, marque par marque, et une courbe *monte* avec l'âge (une voiture qui prend de la valeur en vieillissant ?!). C'est absurde.

**Pourquoi ça arrive.** À l'âge 2, ta marque est surtout représentée par des petites citadines ; à l'âge 6, par des SUV. Ce n'est pas la même voiture qui vieillit — c'est **le mélange (la composition) qui change**. La courbe mesure le changement de mix, pas la dépréciation.

**L'outil/méthode.** Une **régression linéaire à effets fixes** :
```
log(prix) ~ effets_fixes(modèle) + effets_fixes(âge)
```
En français : « à modèle donné, quel est l'effet propre de l'âge ? ». Les indicatrices de modèle (`C(modèle)`) *absorbent* le changement de composition. On l'a codée avec `numpy.linalg.lstsq` (moindres carrés). On travaille sur `log(prix)` pour que les effets soient **multiplicatifs** (une dépréciation, c'est un %, pas un montant fixe).

**L'interprétation.** Le coefficient de l'âge te donne la vraie décote annuelle, *nette* du changement de parc. On a aussi ajouté un mode **« modèle représentatif »** : au lieu de la régression, on trace la décote littérale d'un seul modèle dominant par marque (Golf 7, 208…). Les deux doivent raconter la même histoire — si oui, c'est robuste.

**En entretien :** « Je ne compare jamais des médianes brutes par groupe sans vérifier que les groupes sont comparables. Sur la dépréciation, j'ai contrôlé la composition par une régression à effets fixes modèle + âge, sinon un changement de mix se lit comme une hausse de valeur. »

### 2.2 Le petit échantillon (small-sample noise)

**Le problème.** La courbe de prix médian par âge du segment luxe *montait* de l'âge 0 à l'âge 1 (179k → 247k).

**Pourquoi.** Le point « luxe, âge 0 » ne reposait que sur **9 annonces**. Avec si peu de données, la médiane est instable — un ou deux véhicules atypiques la font bouger de dizaines de milliers de dinars.

**L'outil/méthode.** Un **seuil d'effectif minimum** : on n'affiche un point que s'il repose sur `n ≥ 10` annonces. Simple, mais c'est de la rigueur : *on ne trace pas ce qu'on ne peut pas mesurer de façon fiable*.

**L'interprétation.** En retirant les points flimsy, la fausse montée disparaît et la courbe redevient monotone décroissante (la réalité). Toujours afficher les tailles d'échantillon pour que le lecteur juge.

**En entretien :** « J'affiche systématiquement `n`, et je masque les agrégats sous un seuil d'effectif, parce qu'une médiane sur 9 points n'est pas un signal, c'est du bruit. »

### 2.3 La prime du neuf (vs dépréciation d'occasion)

**Le problème.** On mesurait une « falaise » de dépréciation de −17 %/an sur les premières années. Le chiffre était juste mais **mal interprété**.

**Pourquoi.** La grosse chute entre l'âge 0-1 et l'âge 2, ce n'est pas de la dépréciation d'occasion : c'est la **prime du neuf** qui s'évapore (une voiture quasi neuve se vend presque au prix du neuf, puis rejoint brutalement le marché de l'occasion). Ce n'est pas ce qu'un acheteur d'occasion subit.

**L'outil/méthode.** On a **déplacé le point de départ** de la mesure : la décote d'occasion se mesure **à partir de 2 ans**, pas 0. Un simple choix de fenêtre, mais qui change l'histoire.

**L'interprétation.** Une fois la prime du neuf exclue, on découvre que **le généraliste tunisien tient très bien sa valeur** de 2 à 4 ans (~3 %/an) puis décroche, tandis que **le luxe décroche tôt**. C'est un conseil d'achat *utile et vrai*, pas le cliché « la voiture perd 20 % en sortant du garage ».

**En entretien :** « J'ai séparé la prime du neuf de la vraie dépréciation d'occasion. Confondre les deux fait dire n'importe quoi sur le bon moment pour acheter. »

### 2.4 Le biais de survie (survivorship bias)

**Le problème.** Pour un « prix médian glissant » du marché dans le temps, l'approche évidente est : grouper par **date de dépôt** de l'annonce et lisser. Ça donnait une belle courbe… qui montrait une chute de 30 % du marché. **Fausse.**

**Pourquoi.** Une annonce déposée il y a 8 semaines et **encore en ligne aujourd'hui** est, par définition, une annonce **qui ne s'est pas vendue** — souvent parce qu'elle est **surcotée**. Donc les vieilles dates de dépôt sont sur-représentées par des invendus chers → la médiane paraît « haute avant, basse maintenant ». Ce n'est pas le marché qui baisse, c'est **qui survit dans ton échantillon** qui est biaisé.

**L'outil/méthode.** On ne mesure PAS par date de dépôt. On mesure **à travers les runs nocturnes** : chaque nuit, le pipeline enregistre le prix médian du marché *actuellement en ligne* (un snapshot non biaisé), et on lisse ces snapshots avec une **médiane glissante sur 7 runs**.

**L'interprétation.** Chaque point = état réel du marché à cette date. La glissante enlève le bruit run-à-run. Contrepartie : ça se remplit dans le temps (il faut des runs).

**En entretien :** « Je me méfie du biais de survie : les données qui *restent* ne sont pas représentatives des données qui *existaient*. Pour une tendance de prix, j'utilise des snapshots successifs du marché, pas la date de mise en ligne. » ← **c'est le genre de phrase qui impressionne un data scientist senior.**

---

## 3. Le raisonnement quant (page Samsar)

La page Samsar s'adresse au *revendeur* (achat-revente). On y a mis de la vraie logique financière.

### 3.1 Gain absolu vs ROI par tranche de prix

**Le problème.** « Dans quelle gamme de prix un revendeur doit-il chasser ? »

**L'outil/méthode.** On segmente les affaires par **tranche de prix d'achat** (`pandas.cut`) et on calcule, par tranche : le **gain médian en dinars** et le **ROI médian** (gain / prix).

**L'interprétation — deux courbes qui se croisent :**
- Le **gain absolu monte** avec le prix (une grosse voiture = grosse marge en DT).
- Le **ROI % descend** (le petit budget rapporte plus *en pourcentage*).

Décision : petit capital → chasse le bas (ROI 60 %, l'argent tourne vite) ; gros capital → le haut (grosse marge par flip). Et le **milieu de gamme (40-60k) est un piège** (ROI le plus bas). C'est de l'**analyse marginale** : chaque dinar de plus investi rapporte de moins en moins de %.

### 3.2 La « régularité » (un Sharpe-*like*) — et l'honnêteté du nom

**L'idée.** Toi tu as proposé un ratio type Sharpe : « pour 1 dinar de plus investi, combien je récupère au regard du risque ? ». Bonne intuition.

**L'outil/méthode.** Par tranche : `régularité = ROI médian ÷ dispersion du ROI (écart-type)`. Un ROI élevé avec peu de dispersion = des affaires **consistantes**.

**L'interprétation ET la nuance cruciale.** On a trouvé que le 90k+ a la meilleure régularité (peu de dispersion) et le 40-60k la pire. **MAIS** — et c'est le point d'honnêteté — **ce n'est PAS un Sharpe financier**, pour deux raisons :
1. La dispersion mélange la *vraie* variété des affaires **et l'erreur du modèle** (~10,7 %). Une partie du « risque » mesuré, c'est juste ton modèle qui se trompe.
2. Ça n'inclut **pas le risque de revente** (est-ce que ça se vend ? en combien de temps ?), qui est le vrai risque du revendeur.

Donc on l'a appelé **« régularité »**, pas « Sharpe ». 

**En entretien :** « J'ai construit un indicateur rendement/dispersion, mais je l'ai nommé honnêtement ‘régularité' et pas ‘Sharpe', parce qu'un Sharpe suppose un risque de marché que je ne mesure pas encore. Sur-vendre un indicateur, c'est le meilleur moyen de se faire coincer. » ← **encore une phrase qui vaut de l'or.**

### 3.3 L'arbitrage géographique à confiance graduée

**Le problème.** « Acheter dans une région pas chère, revendre dans une région chère » — mais est-ce un vrai écart, ou juste des voitures différentes / trop peu d'annonces ?

**L'outil/méthode.** Régression `log(prix) ~ âge + km + région` : le coefficient « région » donne l'écart de prix **à âge et kilométrage comparables**. Puis on classe chaque play par **niveau de confiance** (🟢 Forte / 🟡 Moyenne / 🔴 Faible) selon le **nombre d'annonces des deux côtés** et la taille de l'écart.

**Deux angles :**
- **Par modèle** : composition parfaitement contrôlée (même modèle exact), mais peu d'annonces → souvent 🔴.
- **Par segment** : la même régression sur tout un segment (plus d'annonces → 🟢), au prix d'un contrôle de composition plus grossier.

**L'interprétation.** Le résultat honnête : la plupart des arbitrages par modèle sortent 🔴 (basés sur 3-7 annonces). L'outil **empêche** de foncer sur un « +32 % » qui n'est que du bruit. Et le profit est affiché **brut, avant frais** (transport + mutation).

**En entretien :** « Je ne présente jamais un écart sans un niveau de confiance basé sur la taille d'échantillon. Un outil qui dit ‘fonce' sur 3 données est dangereux ; le mien dit ‘vérifie'. »

### 3.4 Les cartes KPI adaptatives

**Le problème.** En haut de la page : « 196 opportunités ». Avec les filtres actifs : « 1 opportunité ». Incohérent — le lecteur ne sait pas quel chiffre croire.

**La solution.** On a remonté les filtres *au-dessus* des cartes, et les 4 cartes se recalculent sur la sélection, avec le marché global gardé en légende pour le contexte. C'est de l'**UX de dashboard** : un chiffre doit toujours décrire ce que l'utilisateur regarde.

---

## 4. Le modèle et son interprétabilité

### 4.1 SHAP, et le repli par perturbation

**Le problème.** La décomposition « pourquoi ce prix ? » (SHAP) ne s'affichait pas sur ton déploiement, alors que `shap` était dans `requirements.txt`.

**Pourquoi.** `shap` est un package fragile (dépendances lourdes, versions capricieuses). Sur Streamlit Cloud, il échouait silencieusement (`try/except` qui renvoyait `None`), donc le bloc disparaissait sans erreur visible.

**L'outil/méthode.** On a rendu l'explication **robuste** : elle essaie SHAP, et **si ça échoue pour n'importe quelle raison**, elle bascule sur un **repli par perturbation** :
```
contribution d'une variable ≈ prédiction(réelle) − prédiction(cette variable ramenée à sa médiane)
```
En clair : « de combien le prix change si je remplace le kilométrage par le kilométrage médian ? » = l'effet marginal de cette variable. Ça ne dépend d'**aucun package**.

**L'interprétation.** Barres vertes = tirent le prix vers le haut, rouges = vers le bas. La légende dit honnêtement si c'est du SHAP (rigoureux) ou du repli (approximation). Résultat : le bloc s'affiche **toujours**.

**En entretien :** « J'ai découplé une fonctionnalité clé d'une dépendance fragile. Plutôt que de dépendre d'un package qui peut casser en prod, j'ai un repli maison qui approxime la même chose. » (C'est une leçon de **robustesse logicielle**, pas juste de ML.)

### 4.2 Rappels (déjà en place, mais à savoir défendre)

- **Scoring out-of-fold (`cross_val_predict`)** : chaque annonce est notée par un modèle qui **ne l'a pas vue** à l'entraînement. Sinon le modèle « mémorise » l'annonce et déclare qu'elle est parfaitement estimée → fausses affaires. C'est la garantie que tes opportunités sont réelles.
- **MdAPE** (Median Absolute Percentage Error, ~10,7 %) : l'erreur *relative médiane*. On l'utilise plutôt que le MAE (sensible aux grosses valeurs) ou le R² parce que sur un marché aux prix très dispersés, une erreur *en %* et *médiane* (robuste aux extrêmes) est la plus honnête.
- **Ablation de features** : on retire chaque variable une à une et on mesure la perte en validation croisée. C'est comme ça qu'on est passé de 15 à 10 features (certaines mortes ou redondantes). Plus rigoureux que de choisir par importance SHAP.

---

## 5. Industrialisation (les recommandations du review)

Un review externe a noté le projet 9/10 et listé des priorités. On les a traitées.

### 5.1 La CI (`ci.yml`) — pourquoi séparer code et données

**Le problème.** Tu avais 66 tests… que GitHub Actions **n'exécutait jamais**. Des tests non lancés automatiquement ne servent presque à rien.

**L'outil/méthode.** **GitHub Actions**, deux workflows séparés :
- **`ci.yml`** (nouveau) : à chaque *push / pull request* → installe, lance **pytest** (66 tests), `verifier_sync.py` (cohérence), **ruff** (lint). C'est la CI du **code**.
- **`scraping.yml`** (existant) : la nuit → scraping + pipeline + publication. C'est le pipeline de **données**.

**Pourquoi séparer ?** Le code et les données ont des cycles de vie différents. Tu veux valider ton code à *chaque modif* (rapide, sans scraper), et scraper *une fois par nuit*. Les mélanger, c'est soit tester trop rarement, soit scraper trop souvent.

**En entretien :** « J'ai séparé CI code (tests à chaque push) et pipeline données (cron nocturne). Deux responsabilités, deux workflows. »

### 5.2 Le quality gate dynamique (`core/quality_gate.py`)

**Le problème.** Le pipeline vérifiait « au moins 100 annonces ». Trop faible : si tu passes de 5 800 à 720 annonces (un scraper cassé), 720 > 100 → ✅, et tu publies des données pourries.

**L'outil/méthode.** Un contrôle **relatif et multi-dimensionnel** (pandas) :
- Volume total ≥ **60 % d'une référence glissante** (la dernière exécution réussie, stockée dans un JSON) → attrape l'effondrement.
- Planchers **par source**.
- **Complétude** : Prix/Marque/Modèle renseignés au-dessus d'un seuil.
- **Plausibilité** : part de prix hors bornes réalistes bornée.
- **Doublons** bornés.

Il sort en erreur (code 1) si un contrôle *dur* échoue → le workflow **bloque la publication** de `processed`/`models`. On l'a **branché** dans `scraping.yml`.

**Pourquoi relatif > absolu.** Un seuil absolu (« 100 ») ne connaît pas ton volume normal. Un seuil relatif (« 60 % d'hier ») s'adapte. C'est la différence entre une alarme incendie réglée sur « 1000°C » et une réglée sur « +50° d'un coup ».

**En entretien :** « Mes contrôles qualité sont relatifs à une référence glissante, pas des seuils absolus en dur — sinon on ne détecte pas une dégradation partielle. »

### 5.3 Le suivi de performance + drift (`core/suivi_performance.py`)

**Le problème.** Un modèle fiable *aujourd'hui* peut dériver *dans le temps* (le marché change). Un snapshot ne le prouve pas.

**L'outil/méthode.** À chaque run, on enregistre dans un historique JSON :
- la **MdAPE** (globale + par tranche),
- un **instantané du marché** (volume, prix médian, mix des sources, complétude),
- le **drift** : écart de l'instantané courant vs la **médiane des 7 derniers runs**. Si le prix médian, le volume ou le mix des sources bouge trop → alerte.

Le dashboard (page Admin) trace tout ça dans le temps.

**L'interprétation.** Une **MdAPE plate sur des semaines** = modèle fiable dans la durée (le vrai argument de fiabilité). Une hausse progressive ou un drift = le marché a bougé → réentraîner.

**Le concept clé : le drift.** C'est quand les données de production s'éloignent des données d'entraînement. C'est *la* raison n°1 pour laquelle un modèle ML se dégrade en prod. Le surveiller, c'est du **MLOps** (léger — voir glossaire).

**En entretien :** « Je surveille le drift des données et la stabilité de l'erreur dans le temps, parce qu'un modèle ne meurt pas d'un coup, il dérive. »

### 5.4 La modularisation d'`app.py`

**Le problème.** `app.py` faisait 2 500 lignes. Difficile à maintenir, à tester, à relire.

**L'outil/méthode.** On l'a découpé (via un script `ast` pour être fiable) en couches :
```
app.py          → point d'entrée (nav + dispatch), 100 lignes
ui/             → thème (couleurs, CSS) + helpers de rendu (graphes)
services/       → data_service, model_service, analytics_service
views/          → une page par fichier (7 pages)
```

**Pourquoi cette architecture (séparation des responsabilités).** Chaque couche a un rôle : `ui` ne connaît pas les données, `services` ne connaît pas l'affichage, `views` orchestre. Avantage entretien : **tu peux tester `services/` indépendamment de Streamlit**.

**Le piège qu'on a évité.** On a nommé le dossier des pages **`views/`** et pas `pages/`, parce que **`pages/` est réservé par Streamlit** (il génère une navigation multipage automatique qui aurait doublonné ta sidebar). Détail, mais c'est de la connaissance-outil précise.

**Statut.** La version modularisée est prête et vérifiée (imports OK, 66 tests), mais l'app active reste le **monolithe** (éprouvé) — tu adopteras la modularisée après l'avoir testée en local. Le monitoring est dans les deux.

**En entretien :** « J'ai refactoré une app monolithique en couches ui/services/views pour la testabilité, en évitant le piège du dossier `pages/` réservé par Streamlit. »

---

## 6. Git : l'incident du force-push

**Ce qui s'est passé.** Tu as fait `git push --force` et écrasé l'historique distant → perte apparente de tout. Heureusement, je t'avais cloné le repo *quelques minutes avant*, donc j'ai pu tout reconstruire.

**La leçon (importante).**
- `git push --force` **écrase** l'historique distant sans filet. Il ne fusionne pas, il remplace. Si ton local est incomplet, tu détruis ce qui était en ligne.
- **Ne l'utilise que si tu es certain** de ce que contient ton local.
- Pour publier des changements normaux : `git pull --rebase` puis `git push` (**sans** `--force`) — ça n'écrase rien.
- Réflexe de sécurité : `git status` et `git log` **avant** tout push force.

**En entretien (si on te demande un incident) :** « J'ai fait un force-push malheureux et compris de première main pourquoi il faut le manier avec précaution. Depuis, je vérifie systématiquement l'état local avant, et je privilégie le rebase. » (Assumer une erreur et sa leçon est *bien vu*.)

---

## 7. Le bug Mermaid

**Le problème.** Le diagramme d'architecture du README ne s'affichait pas sur GitHub : « Cannot read properties of undefined (reading 'render') ».

**La cause.** Les labels des nœuds contenaient des **apostrophes** (`d'état`) et des caractères spéciaux (`→ · ≈ ≥ %`) **sans guillemets**. Le parseur Mermaid de GitHub s'étrangle dessus.

**La solution.** Mettre **tous les labels entre guillemets** (`A1["automobile.tn"]`), ce qui autorise les caractères spéciaux, et neutraliser les symboles les plus risqués.

**La leçon.** Mermaid est du code : sa syntaxe a des règles. Quand un rendu échoue, c'est presque toujours un caractère non échappé. Le réflexe = quoter les labels.

---

## 8. Glossaire express

| Terme | En une phrase |
|---|---|
| **Biais de composition** | Un agrégat par groupe change parce que *le mélange* change, pas le phénomène. |
| **Biais de survie** | Les données qui *restent* ne représentent pas les données qui *existaient* (les invendus surcotés faussent la médiane). |
| **Prime du neuf** | La sur-valeur d'un véhicule quasi neuf, qui s'évapore la 1re année — ce n'est pas de la dépréciation d'occasion. |
| **Out-of-fold (OOF)** | Noter chaque donnée avec un modèle qui ne l'a pas vue à l'entraînement → pas de fuite, pas de sur-optimisme. |
| **MdAPE** | Erreur relative médiane (%). Robuste aux extrêmes ; meilleure que MAE/R² sur des prix dispersés. |
| **Ablation** | Retirer une variable et mesurer la perte de perf pour savoir si elle sert vraiment. |
| **Effets fixes** | Des indicatrices (dummies) dans une régression pour « neutraliser » un facteur (ici : le modèle, l'âge). |
| **log(prix)** | On modélise le log car la dépréciation est multiplicative (un %), pas additive (un montant). |
| **SHAP** | Méthode qui attribue à chaque variable sa contribution à une prédiction précise. |
| **Régularité (≠ Sharpe)** | Rendement ÷ dispersion. Pas un Sharpe financier car la dispersion mêle bruit modèle et vrai risque. |
| **Drift** | Les données de prod s'éloignent de celles d'entraînement → le modèle se dégrade. |
| **Quality gate** | Contrôles automatiques qui *bloquent* la publication de données dégradées. |
| **CI/CD** | Intégration continue : lancer tests/lint automatiquement à chaque changement de code. |
| **MLOps « léger »** | Retraining auto + CI + validation + monitoring. « Léger » car pas de model registry / champion-challenger / rollback. |
| **Cox (analyse de survie)** | Modèle qui estime si un événement (ici : la disparition d'une annonce = vente) arrive *plus vite* dans un groupe, en contrôlant d'autres facteurs. |

---

## 9. Ce qui reste à faire

**Rien de plus à coder** — le review est couvert (P0, P1, P2). Ce qui reste est du **temps** et du **manuel** :

1. **Tester** `streamlit run app.py` en local, puis pousser (sans `--force` de préférence, ou en vérifiant `git status` avant).
2. Remplir le **About GitHub** (description + lien Streamlit + topics).
3. **Laisser tourner** le pipeline : le monitoring et la validation de survie se remplissent run après run. C'est là que naît la preuve *dans la durée* :
   - « les opportunités détectées disparaissent-elles plus vite ? » (validation de survie, Cox),
   - « la MdAPE reste-t-elle plate sur des semaines ? » (suivi de performance).

**Le vrai prochain saut n'est plus technique — il est temporel.** Le code prouve que tu *sais construire* ; les courbes dans le temps prouveront que ça *marche vraiment*. C'est ce qui fait passer un projet de « joli » à « crédible ».

---

### Un dernier mot

Tu m'as demandé de faire, et c'est normal — c'est comme ça qu'on avance vite. Mais maintenant que tu as ce document, **relis le code en le tenant à côté** : ouvre `analyse_decote_segment`, `calculer_arbitrage_geo`, `expliquer_prix`, `quality_gate.py`, `suivi_performance.py`, et retrouve dans chaque fonction le *pourquoi* décrit ici. Le jour où tu peux expliquer chaque biais qu'on a corrigé **sans ce document sous les yeux**, le projet est vraiment le tien.
