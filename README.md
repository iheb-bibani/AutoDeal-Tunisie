# AutoDeal Tunisie 🚗

Plateforme d'intelligence du marché automobile d'occasion tunisien : collecte quotidienne des annonces, nettoyage et fiabilisation des données, modèle de « Juste Prix », détection d'opportunités d'achat-revente, alertes Telegram et application d'analyse.

Sources : **automobile.tn** (majoritairement des professionnels), **tayara.tn** (majoritairement des particuliers), **automax.tn**.

---

## Représentativité des données — à lire avant tout

Les données couvrent le marché **en ligne**, pas le marché tunisien dans son ensemble : les ventes de gré à gré hors internet, les souks auto et une partie du marché des concessionnaires n'y apparaissent pas. Les prix affichés sont en outre des prix **demandés**, pas des prix de transaction — la négociation réelle se fait en dessous. Les analyses restent pertinentes pour comparer, positionner et repérer des anomalies de prix, mais aucun chiffre ne doit être lu comme une cote officielle.

---

## Architecture

```
scrapers/            Collecte
├── scraper_tayara.py         requests + BeautifulSoup
├── scraper_automobile.py     Patchright (contournement Cloudflare Turnstile)
└── scraper_automax.py        pagination AJAX
        │   -> data/raw/{tayara,automobile,automax}.csv
        ▼
core/                Traitement (ordre imposé)
├── merging_files.py          fusion des 3 sources, déduplication par lien,
│                             normalisation marque/énergie/localisation,
│                             filtrage catégorie tayara par URL
├── nettoyer_base.py          bruit, valeurs aberrantes, inférence
│                             Marque/Modèle depuis les titres, fraîcheur ≤ 30 j
├── enrichir_base_avance.py   Segment, Zone, Âge véhicule, Presque-neuve
├── modele_prediction.py      modèle de prix (CV 5 folds), scoring out-of-fold
└── detect_deals.py           filtrage des vraies opportunités (15 % – 55 %)
        │   -> data/processed/*.csv + data/models/modele_prix.pkl
        ▼
app.py               Application Streamlit (3 espaces)
utils/send_telegram.py        alertes des nouvelles opportunités
```

`main.py` orchestre l'ensemble via `subprocess` : la sortie de chaque étape est visible, les codes retour sont vérifiés, et le pipeline s'arrête si une étape de traitement critique échoue — un scraper en échec, lui, n'arrête rien (les autres sources suffisent).

---

## Qualité des données

Le nettoyage traite les problèmes réellement observés sur les données collectées :

**Filtrage du bruit.** La catégorie « voitures » de tayara contient de vraies annonces immobilières (terrains, appartements), des pièces détachées (kits embrayage, amortisseurs, alternateurs), des locations et des annonces sans contenu. Tout cela est écarté par mots-clés et par vérification de la catégorie dans l'URL tayara. Les pages parasites capturées par erreur chez automobile.tn (« Présentation de la cote ») sont également éliminées.

**Inférence Marque / Modèle depuis les titres.** Sur tayara, le champ structuré Marque/Modèle est souvent vide ou rempli avec « Autres », alors que le titre contient l'information (« Golf 7 toutes options », « vente clio 4 », « Mercedes C180 »). Le pipeline devine alors la marque et le modèle en trois niveaux :

1. marques et modèles déjà connus dans la base (correspondance mot entier, insensible aux espaces et tirets : « Golf7 » = « Golf 7 ») ;
2. référentiel du marché tunisien (`config.py`) couvrant les marques absentes du champ structuré — Cupra, Haval, Dongfeng, Lada, Wallys, BYD… — et les modèles usuels de chaque marque ;
3. modèles qui impliquent la marque à eux seuls (« clio » → Renault Clio, « b9 » → Citroën Berlingo, « jolion » → Haval Jolion).

Les allemandes premium ont des règles dédiées, car le mot qui suit la marque y désigne une motorisation et non un modèle : codes moteur BMW (`320d` → Série 3), classes Mercedes (`C180` → Classe C), nomenclature Audi (A3, Q5, TT via le référentiel). Les erreurs de saisie récurrentes des vendeurs sont corrigées de façon ciblée (« Great Wall M4 » étiqueté Haval, « Gol » choisi à la place de « Golf »), et la casse des modèles est unifiée (« jolion » / « Jolion » / « JOLION » = une seule catégorie).

**Champs mal captés par les scrapers.** Deux corrections importantes, dont l'effet était invisible car silencieux :

- `Puissance_Fiscale` est stockée en texte par automobile.tn (`"5cv"`, `"12cv"`). Une conversion numérique directe renvoyait `NaN` sur la totalité de cette source : le champ passait de 86 % rempli avant nettoyage à 18 % après. Le nombre est désormais extrait avant conversion (96 % rempli).
- `Boite_Vitesse` contient, sur automobile.tn, le type de **transmission aux roues** (Traction/Intégrale/Propulsion) et jamais le type de boîte. Cette valeur est maintenant déplacée vers une colonne `Transmission` — c'est une vraie information, désormais utilisée comme variable du modèle — au lieu d'être jetée. La boîte est ensuite récupérée depuis le titre lorsqu'elle y est explicite (`"... 110 cv Boîte auto"`). automobile.tn ne mentionnant que les automatiques, seules celles-ci sont récupérées : supposer que l'absence de mention vaut « Manuelle » serait une invention, pas une déduction.

**Caractéristiques extraites des titres.** `Puissance_DIN` et `Cylindree` sont lues dans le titre, très normé chez automobile.tn (`"Golf 7 Smartline 1.2 TSI 16V S&S 110 cv"`), et disponibles sur environ deux tiers des annonces. À ne pas confondre avec la puissance fiscale, qui est une échelle administrative.

**Valeurs aberrantes.** Prix hors plage (3 000 – 500 000 DT) et valeurs de remplissage « chiffre répété » (11111, 999999) écartés ; kilométrages > 500 000 km, puissances fiscales impossibles (numéros de téléphone glissés dans le champ), années hors 1980 – année+1 mis à vide plutôt que la ligne supprimée. Les valeurs de `Boite_Vitesse` qui décrivent en réalité la transmission aux roues (Traction/Intégrale/Propulsion) sont vidées.

**Dates.** Les scrapers écrivent en ISO (AAAA-MM-JJ) ; le parsing n'utilise jamais `dayfirst=True`. L'âge des annonces est calculé par rapport à la date du jour, et la base d'analyse est limitée aux annonces de 60 jours ou moins (fenêtre `MAX_DAYS_OLD`, calibrée empiriquement — voir ci-dessous).

**Déduplication.** Une annonce scrapée plusieurs fois (relances, appends) n'apparaît qu'une fois, version la plus récente conservée (clé : le lien).

---

## Modèle de « Juste Prix »

Un seul modèle, sur toute la base, comparé par validation croisée (5 folds) entre Ridge, RandomForest et HistGradientBoosting. La sélection se fait sur l'**erreur relative médiane out-of-fold**, ni sur le R² ni sur le MAE en dinars :

- le R² mesure la variance expliquée en log-prix, très loin de ce que constate l'utilisateur ;
- le MAE en dinars est écrasé par le haut de gamme — sur les données réelles il vaut 6 700 DT sous 30 000 DT mais 75 000 DT au-dessus de 200 000 DT. Sélectionner dessus revient à choisir le modèle qui estime le mieux les Porsche, pas celui qui estime le mieux les voitures courantes ;
- `Score_Opportunite` est un écart **relatif** : la métrique de sélection doit l'être aussi, sinon on optimise autre chose que ce qu'on publie.

Deux points de méthode importants :

- **Scoring out-of-fold** : le `Prix_Theorique` et le `Score_Opportunite` de chaque annonce sont calculés par un modèle qui n'a jamais vu cette annonce pendant son entraînement (`cross_val_predict`). Les scores exportés sont donc aussi honnêtes que les métriques affichées.
- **Regroupement des rares réservé à l'entraînement** : les marques < 10 annonces et modèles < 5 annonces sont regroupés en « Autre » uniquement dans la matrice d'entraînement, pour éviter des colonnes quasi vides. Les colonnes Marque/Modèle exportées conservent les vrais noms — l'application n'affiche jamais « Autre_modele » à la place d'un modèle connu.

Le fichier scoré ajoute aussi un `Score_Liquidite` (volume d'annonces du couple Marque+Modèle, normalisé) : un proxy de facilité de revente.

### Fenêtre de fraîcheur (`MAX_DAYS_OLD`)

Elle a été calibrée par mesure, pas choisie à l'intuition. Protocole : l'ensemble d'évaluation est **figé** aux annonces de moins de 30 jours (celles qu'on score en production) ; seule varie la quantité d'annonces plus anciennes ajoutées à l'entraînement. Les anciennes annonces ne sont jamais en test, uniquement en apprentissage. Chaque fenêtre est mesurée sur 5 tirages aléatoires pour distinguer le signal du bruit.

| Fenêtre | Annonces d'entraînement | Erreur relative médiane |
|---|---|---|
| 30 j | 2 308 | 10,83 % ± 0,27 |
| 45 j | 2 842 | 10,39 % ± 0,15 |
| **60 j** | **2 872** | **10,28 % ± 0,12** |
| 90 j | 3 142 | 10,25 % ± 0,13 |
| 180 j | 3 288 | 10,39 % ± 0,14 |
| 365 j | 3 384 | 10,32 % ± 0,20 |
| toutes | 3 581 | 10,27 % ± 0,20 |

Seule la fenêtre à 30 jours se distingue vraiment : elle est la pire. Toutes les fenêtres de 45 jours et plus se valent, leurs intervalles se chevauchant largement. La valeur retenue est **60 jours** — le gain est acquis dès ce point, et allonger davantage n'apporte rien tout en augmentant le risque d'entraîner le modèle sur des prix périmés.

Une réserve, mesurée et non supposée : une annonce encore en ligne après plusieurs mois est le plus souvent une annonce **surévaluée qui ne se vend pas**. En isolant tayara (source homogène) et à véhicule comparable — même marque, même âge, même kilométrage — chaque mois d'ancienneté de l'annonce s'accompagne de **+3,8 % sur le prix affiché**. Ce n'est pas de l'inflation : c'est de la survie sélective, les annonces au bon prix disparaissant en se vendant. Élargir la fenêtre revient donc à sur-représenter les vendeurs trop gourmands, ce qui explique le plateau au-delà de 60 jours malgré 700 annonces supplémentaires.

Ce chiffre ne doit surtout pas être lu comme une dérive des prix du marché. La base est une **photographie à un instant donné** : on n'observe jamais les annonces déjà vendues, et la composition par source varie complètement avec l'ancienneté (78 % d'automobile.tn parmi les annonces de moins de 30 jours, 0 % au-delà de 60 jours). Sans contrôle de la source, le coefficient s'inverse même en −0,85 %/mois — un pur artefact de composition. Mesurer une vraie évolution des prix du marché demanderait des collectes répétées dans le temps, ce que l'historique de `Annonce-Detectee` permettra une fois plusieurs mois de scraping quotidien accumulés.

### Seuil de détection et fiabilité

L'erreur relative médiane du modèle est d'environ **12 %**. Un seuil de détection à 15 % serait donc *à l'intérieur* du bruit de l'estimation : il retenait 18 % de tout le marché comme « bonnes affaires », c'est-à-dire une annonce sur cinq — ce n'était pas un détecteur d'opportunités mais un détecteur d'erreurs d'estimation. `detect_deals.py` retient donc les annonces entre **25 % et 55 %** sous le prix théorique : le plancher est à environ deux fois l'erreur typique, le plafond écarte les erreurs de saisie et les annonces pour pièces.

Second garde-fou, la **fiabilité de l'estimation** : un prix théorique appuyé sur 60 annonces comparables et un prix appuyé sur 2 annonces d'un modèle rare n'ont pas la même valeur. Chaque annonce porte donc `Nb_Comparables` et `Fiabilite_Estimation` (Faible / Moyenne / Élevée). Les opportunités reposant sur moins de 8 comparables restent dans le fichier mais ne déclenchent pas d'alerte Telegram, et l'application les masque par défaut.

Le fichier d'alertes est réécrit à chaque exécution, même vide, pour ne jamais servir des opportunités périmées.

---

## Application (Streamlit)

Trois espaces, pour deux métiers :

**📊 Vue Marché — concessionnaire.** Volume et âge du parc, parts de marché par marque, prix médians (avec effectifs), courbe de dépréciation, structure du marché par gamme de prix, prix par énergie et par région avec **lissage bayésien** (les régions à petit échantillon sont ramenées vers la médiane nationale pour ne pas sur-interpréter le hasard).

Deux analyses agrégées y sont calculées à partir de toutes les annonces, et non de simples médianes :

- **Décote annuelle par modèle** — ajustement de `log(prix) ~ âge`, dont la pente donne un taux de perte de valeur en % par an. C'est un pourcentage de la valeur restante, ce qui correspond à la dépréciation réelle d'un véhicule, et la pente utilise toutes les annonces du modèle plutôt que deux points médians. Exportable en CSV. Sur les données actuelles : Fiat 500 −3,9 %/an, Golf 7 −5,4 %/an, Kia Sportage −6,6 %/an, Mercedes Classe C −9,6 %/an, Range Rover Evoque −11,8 %/an.
- **Prime professionnelle, à âge égal** — `log(prix) ~ âge + vendeur_pro` par modèle. Comparer directement les prix médians pros et particuliers donnait un écart médian de +15 % avec des valeurs absurdes (+169 % sur une Škoda Octavia) : c'était un **biais de composition**, les pros vendant des véhicules bien plus récents que les particuliers pour le même modèle. Une fois l'âge neutralisé — et en n'utilisant que la zone où les âges des deux populations se recouvrent d'au moins 2 ans — la prime médiane tombe à **+2 %**. L'arbitrage « acheter au particulier, revendre au pro » ne tient donc que sur une poignée de modèles (Seat Ibiza +23 %, Classe C +17 %, Clio +16 %).

**🎯 Vue Samsar — achat-revente.** Opportunités chiffrées en dinars (gain médian, meilleure affaire, gain cumulé dans le budget), filtre par budget/marque/zone, matrice décote × liquidité (en haut à droite : gros gain **et** revente facile), filtre de fiabilité des estimations, tableau détaillé avec liens directs vers les annonces, **rotation du marché** (modèles les plus présents, et modèles dont les annonces sont les plus fraîches — proxy d'écoulement rapide), et **arbitrage géographique** : pour un modèle donné, la région où l'acheter le moins cher et celle où le revendre le plus cher.

**💰 Calculateur de Juste Prix.** Estimation du modèle ML pour une configuration saisie, confrontée aux annonces comparables réelles du marché : médiane, fourchette interquartile et positionnement visuel de l'estimation parmi les points réels.

---

## Installation et utilisation

```bash
pip install -r requirements.txt
python -m playwright install chromium
python -m patchright install chromium

cp .env.example .env        # renseigner TELEGRAM_TOKEN / TELEGRAM_CHAT_ID (facultatif)

python main.py              # pipeline complet : scraping -> nettoyage -> modèle -> deals
streamlit run app.py        # application
python utils/send_telegram.py   # envoi manuel des alertes (dédupliquées)
```

Chaque étape peut aussi être lancée individuellement (`python core/merging_files.py`, etc.) dans l'ordre indiqué par l'architecture ci-dessus.

### Automatisation (GitHub Actions)

`.github/workflows/scraping.yml` lance le pipeline complet **chaque nuit à minuit heure tunisienne**. Le cron GitHub est toujours exprimé en UTC et la Tunisie est à UTC+1 toute l'année, sans changement d'heure : la valeur est donc `0 23 * * *`.

Secrets requis dans les paramètres du dépôt : `TELEGRAM_TOKEN`, `TELEGRAM_CHAT_ID`.

Quatre garde-fous, chacun couvrant une panne silencieuse constatée ou possible :

- **`concurrency`** — un scraping complet peut durer plusieurs heures. Sans ce verrou, un run lent et celui du lendemain tourneraient en parallèle et se pousseraient des commits contradictoires sur les mêmes fichiers.
- **Vérification des données produites** — le job échoue si `tunisia-cars-scored.csv` ou le modèle est absent, vide, ou contient moins de 100 annonces. Sans ce contrôle, un pipeline interrompu à mi-chemin laisse les anciens fichiers en place : le commit ne montre aucun changement et tout paraît normal alors que plus rien n'est à jour.
- **Telegram en `continue-on-error`** — une panne de notification ne doit jamais faire perdre les données scrapées.
- **`git pull --rebase -X theirs`** — pendant un *rebase*, les rôles sont inversés par rapport à un merge : `ours` désigne la branche sur laquelle on rejoue (le dépôt distant) et `theirs` les commits rejoués (les données tout juste scrapées). L'ancienne valeur `-X ours` jetait donc silencieusement le scraping de la nuit en cas de conflit, exactement l'inverse de l'intention.

### Déploiement Streamlit Cloud

**Versions épinglées.** `scikit-learn` est figé à l'exact dans `requirements.txt`. Le modèle `modele_prix.pkl` est entraîné par GitHub Actions et chargé par Streamlit Cloud : si les deux environnements résolvent des versions différentes, le dépickle échoue et le calculateur de Juste Prix tombe en erreur, sans qu'aucun autre signal n'apparaisse. Après toute mise à jour de cette version, réentraîner le modèle et le committer.

**Lecture des données depuis GitHub.** Le rafraîchissement automatique de Streamlit Cloud après un push n'est pas fiable — l'app conserve fréquemment l'ancien contenu du dépôt jusqu'à un redémarrage manuel. Comme les données sont réécrites chaque nuit, s'appuyer dessus reviendrait à afficher des annonces périmées sans signal visible. L'application lit donc les CSV et le modèle **directement depuis GitHub (raw)**, avec un cache d'une heure, et retombe automatiquement sur les fichiers locaux en cas d'échec réseau — ce qui est aussi le mode de développement.

Deux constantes en haut de `app.py` pilotent ce comportement : `DEPOT_GITHUB` (à changer en cas de fork) et `LIRE_DEPUIS_GITHUB` (mettre à `False` pour travailler hors ligne). La barre latérale affiche la date de dernière collecte, alerte si les données ont plus de deux jours, et propose un bouton de rafraîchissement manuel.

`.github/workflows/scraping.yml` lance le pipeline complet chaque jour à 03h00 UTC (ou manuellement depuis l'onglet Actions), envoie les alertes Telegram, puis commit et push les nouvelles données. L'envoi Telegram est en `continue-on-error` : une panne de notification ne doit jamais faire perdre les données scrapées. Le push utilise `git pull --rebase -X ours` pour survivre aux commits arrivés pendant le run. Secrets requis côté repo : `TELEGRAM_TOKEN`, `TELEGRAM_CHAT_ID`.

---

## Structure des fichiers de données

| Fichier | Contenu |
|---|---|
| `data/raw/{tayara,automobile,automax}.csv` | Annonces brutes par source (append, dédupliquées par lien au scraping) |
| `data/processed/tunisia-cars.csv` | Fusion normalisée des 3 sources |
| `data/processed/tunisia-cars-recent.csv` | Base nettoyée, Marque/Modèle complétés, ≤ 30 jours |
| `data/processed/tunisia-cars-final-features.csv` | + variables dérivées (segment, zone, âge…) |
| `data/processed/tunisia-cars-scored.csv` | + Prix_Theorique, Score_Opportunite, Score_Liquidite |
| `data/processed/alertes_bonnes_affaires.csv` | Opportunités retenues (15 % – 55 % sous l'argus) |
| `data/models/modele_prix.pkl` | Pipeline scikit-learn entraîné, utilisé par le calculateur |

---

## Dépannage

- **`tunisia-cars-scored.csv` introuvable dans l'app** → lancer `python main.py` (ou la chaîne `core/` à la main).
- **Erreur scikit-learn au calcul du Juste Prix** → le modèle sauvegardé a été entraîné avec une autre version : relancer `python core/modele_prediction.py`.
- **automobile.tn bloqué (Cloudflare)** → le scraper utilise Patchright avec `channel="chrome"` en mode non-headless (d'où `xvfb` dans le workflow CI) ; éviter de lancer Chrome en root (`--no-sandbox` est un signal de détection).
- **Aucune alerte Telegram** → vérifier `TELEGRAM_TOKEN`/`TELEGRAM_CHAT_ID` dans `.env` (en local) ou dans les secrets du repo (CI), et que `alertes_bonnes_affaires.csv` n'est pas vide.
