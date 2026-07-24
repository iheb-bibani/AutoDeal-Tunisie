# AutoDeal Tunisie 🚗

Plateforme d'intelligence du marché automobile d'occasion tunisien : collecte quotidienne des annonces, fiabilisation des données, modèle de « Juste Prix », détection d'opportunités, alertes Telegram et application d'analyse.

Sources : **automobile.tn** (majoritairement des professionnels), **tayara.tn** (majoritairement des particuliers), **automax.tn**.

---

## À lire avant tout : ce que ces données sont, et ne sont pas

Elles couvrent le marché **en ligne**, pas le marché tunisien dans son ensemble : les ventes de gré à gré, les souks auto et une partie des concessionnaires n'y figurent pas.

Ce sont des prix **demandés**, jamais des prix de transaction — la négociation réelle se fait en dessous, et davantage chez les particuliers que chez les professionnels.

Enfin, la base est une **photographie** : on n'observe jamais les annonces déjà vendues. Toute lecture d'évolution dans le temps à partir d'un seul scraping est un artefact (voir « Fenêtre de fraîcheur »).

Ces analyses servent à comparer, positionner et repérer des anomalies. Aucun chiffre ne doit être lu comme une cote officielle.

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
│                             normalisation marque/énergie/localisation
├── nettoyer_base.py          bruit, valeurs aberrantes, inférence
│                             Marque/Modèle, fraîcheur ≤ 60 j
├── enrichir_base_avance.py   segment, zone, âge, puissance DIN, cylindrée
├── modele_prediction.py      modèle de prix (CV 5 folds), scoring out-of-fold
├── detect_deals.py           filtrage des opportunités
└── suivi_annonces.py         apparition / disparition des annonces
        │   -> data/processed/*.csv + data/models/modele_prix.pkl
        ▼
app.py               Application Streamlit (4 espaces)
utils/send_telegram.py        alertes des nouvelles opportunités
```

`main.py` orchestre l'ensemble via `subprocess` : la sortie de chaque étape reste visible, les codes retour sont vérifiés, et le pipeline s'arrête si une étape critique échoue. Un scraper en échec n'arrête rien — les autres sources suffisent.

> **Attention au placement des fichiers.** Les scripts de `core/` et `utils/` ajoutent la racine du projet à `sys.path` avant d'importer `config` et `logger`. Lancé par `python utils/send_telegram.py`, Python place le dossier du *script* en tête du chemin de recherche, jamais la racine : sans cet ajout, l'import échoue avec `ModuleNotFoundError`. Déplacer un script d'un dossier à l'autre casse aussi la référence correspondante dans `main.py`.

---

## Qualité des données

Le nettoyage traite des problèmes réellement observés, pas des cas théoriques.

**Filtrage du bruit.** La catégorie « voitures » de tayara contient de vraies annonces immobilières, des pièces détachées, des locations et des annonces vides. Tout cela est écarté par mots-clés et par vérification de la catégorie dans l'URL. Les pages parasites d'automobile.tn (« Présentation de la cote ») sont également éliminées.

**Inférence Marque / Modèle depuis les titres.** Sur tayara, le champ structuré est souvent vide ou rempli avec « Autres » alors que le titre contient l'information. Trois niveaux :

1. marques et modèles déjà connus dans la base (mot entier, insensible aux espaces et tirets : « Golf7 » = « Golf 7 ») ;
2. référentiel du marché tunisien (`config.py`) couvrant les marques absentes du champ structuré — Cupra, Haval, Dongfeng, Lada, Wallys, BYD… ;
3. modèles qui impliquent la marque à eux seuls (« clio » → Renault Clio, « b9 » → Citroën Berlingo).

Les allemandes premium ont des règles dédiées, car le mot suivant la marque y désigne une motorisation : codes moteur BMW (`320d` → Série 3), classes Mercedes (`C180` → Classe C). Les erreurs de saisie récurrentes sont corrigées de façon ciblée, et la casse des modèles unifiée.

**Champs mal captés par les scrapers.** Deux corrections dont l'effet était invisible car silencieux :

- `Puissance_Fiscale` est stockée en texte par automobile.tn (`"5cv"`). Une conversion numérique directe renvoyait `NaN` sur toute cette source : le champ passait de 86 % rempli à 18 %. Le nombre est désormais extrait avant conversion (96 %).
- `Boite_Vitesse` contient sur automobile.tn le type de **transmission aux roues** (Traction/Intégrale/Propulsion), jamais la boîte. Cette valeur est déplacée vers une colonne `Transmission` — vraie information, utilisée comme variable du modèle — au lieu d'être jetée. La boîte est ensuite relue dans le titre lorsqu'elle y figure. automobile.tn ne mentionnant que les automatiques, seules celles-ci sont récupérées : supposer que l'absence de mention vaut « Manuelle » serait une invention.

**Équipements du véhicule.** `scraper_automobile.py` visitait déjà la page détail de chaque annonce : l'extraction des options n'ajoute **aucune requête**. Dans le bloc `equipments-wrapper`, chaque `<li>` est une option *possible* pour le modèle ; seules celles portant la classe `highlighted` équipent réellement la voiture. Lire tous les `<li>` reviendrait à enregistrer le catalogue constructeur, identique pour deux véhicules très différemment équipés. Deux colonnes : `Nb_Options` (variable du modèle) et `Options` (liste lisible). La couverture se construit au fil des scrapings ; `Nb_Options` est créée vide tant qu'aucune donnée n'existe, pour que le modèle garde toujours le même jeu de variables.

**Caractéristiques extraites des titres.** `Puissance_DIN` et `Cylindree` sont lues dans le titre, très normé chez automobile.tn (`"Golf 7 Smartline 1.2 TSI 16V S&S 110 cv"`), disponibles sur environ deux tiers des annonces. À ne pas confondre avec la puissance fiscale, qui est une échelle administrative.

**Valeurs aberrantes.** Prix hors plage (3 000 – 500 000 DT) et valeurs de remplissage (11111, 999999) écartés ; kilométrages > 500 000 km, puissances fiscales impossibles et années hors 1980 – année+1 mis à vide plutôt que la ligne supprimée.

**Dates.** Les scrapers écrivent en ISO (AAAA-MM-JJ) ; le parsing n'utilise jamais `dayfirst=True`.

**Déduplication.** Une annonce scrapée plusieurs fois n'apparaît qu'une fois, version la plus récente conservée (clé : le lien).

---

## Modèle de « Juste Prix »

Comparaison par validation croisée (5 folds) entre Ridge, RandomForest et HistGradientBoosting. La sélection se fait sur l'**erreur relative médiane out-of-fold**, ni sur le R² ni sur le MAE en dinars :

- le R² mesure la variance expliquée en log-prix, très loin de ce que constate l'utilisateur ;
- le MAE en dinars est écrasé par le haut de gamme — 6 700 DT sous 30 000 DT, mais 75 000 DT au-dessus de 200 000 DT. Sélectionner dessus revient à choisir le modèle qui estime le mieux les Porsche, pas les voitures courantes ;
- `Score_Opportunite` est un écart **relatif** : la métrique de sélection doit l'être aussi.

Deux points de méthode :

- **Scoring out-of-fold** : `Prix_Theorique` et `Score_Opportunite` sont calculés par un modèle qui n'a jamais vu l'annonce concernée pendant son entraînement (`cross_val_predict`). Les scores exportés sont aussi honnêtes que les métriques affichées.
- **Regroupement des rares réservé à l'entraînement** : marques < 10 annonces et modèles < 5 annonces sont regroupés en « Autre » uniquement dans la matrice d'entraînement. Les colonnes exportées conservent les vrais noms — l'application n'affiche jamais « Autre_modele » à la place d'un modèle connu.

### Fenêtre de fraîcheur (`MAX_DAYS_OLD`)

Calibrée par mesure. Protocole : l'ensemble d'évaluation est **figé** aux annonces de moins de 30 jours ; seule varie la quantité d'annonces plus anciennes ajoutées à l'entraînement. Les anciennes ne sont jamais en test. Chaque fenêtre est mesurée sur 5 tirages aléatoires pour séparer le signal du bruit.

| Fenêtre | Annonces d'entraînement | Erreur relative médiane |
|---|---|---|
| 30 j | 2 308 | 10,83 % ± 0,27 |
| 45 j | 2 842 | 10,39 % ± 0,15 |
| **60 j** | **2 872** | **10,28 % ± 0,12** |
| 90 j | 3 142 | 10,25 % ± 0,13 |
| 365 j | 3 384 | 10,32 % ± 0,20 |
| toutes | 3 581 | 10,27 % ± 0,20 |

Seule la fenêtre à 30 jours se distingue : elle est la pire. Au-delà de 45 jours tout se vaut — l'écart entre 60 j et 90 j (0,03 point) est quatre fois plus petit que le bruit entre graines (0,12), et 90 j ne bat 60 j que 3 fois sur 5. Valeur retenue : **60 jours**.

Réserve mesurée : une annonce encore en ligne après plusieurs mois est le plus souvent **surévaluée**. Sur tayara seul, à véhicule comparable, chaque mois d'ancienneté s'accompagne de **+3,8 %** sur le prix affiché. Ce n'est pas de l'inflation, c'est de la survie sélective — les annonces au bon prix disparaissent en se vendant. Élargir la fenêtre sur-représente donc les vendeurs trop gourmands, ce qui explique le plateau au-delà de 60 jours.

### Seuil de détection et fiabilité

L'erreur relative médiane du modèle est d'environ 10 %. Un seuil à 15 % serait donc *à l'intérieur* du bruit : il retenait 18 % de tout le marché comme « bonnes affaires », soit une annonce sur cinq. `detect_deals.py` retient les annonces entre **25 % et 55 %** sous le prix théorique — le plancher à environ deux fois l'erreur typique, le plafond écartant les erreurs de saisie et les annonces pour pièces.

Second garde-fou, la **fiabilité** : un prix appuyé sur 60 comparables et un prix appuyé sur 2 annonces d'un modèle rare n'ont pas la même valeur. Chaque annonce porte `Nb_Comparables` et `Fiabilite_Estimation`. Sous 8 comparables, l'opportunité reste dans le fichier mais ne déclenche pas d'alerte Telegram, et l'application la masque par défaut.

Le fichier d'alertes est réécrit à chaque exécution, même vide, pour ne jamais servir d'opportunités périmées.

---

## Suivi des annonces — la seule mesure factuelle

`core/suivi_annonces.py` tient `data/processed/suivi_annonces.csv`, qui enregistre pour chaque annonce sa première apparition, sa dernière, sa disparition éventuelle et sa durée en ligne. Tout le reste du projet porte sur des prix *demandés* ; ici on observe un fait.

**Aucune requête HTTP.** Les scrapers parcourent déjà l'intégralité du catalogue de chaque site (303 pages tayara, 160 automobile.tn). Une annonce présente hier et absente aujourd'hui a été retirée — l'information est gratuite. L'approche précédente envoyait des milliers de requêtes HEAD par nuit, se faisait bloquer par Cloudflare, et un incident réseau marquait une annonce « vendue » définitivement. Elle écrivait de plus dans `tunisia-cars.csv`, régénéré à chaque exécution : son travail était effacé au run suivant.

**« Disparue » n'est pas « vendue ».** L'annonce peut avoir été retirée, avoir expiré ou été supprimée. La durée avant disparition est un proxy d'écoulement : utile pour comparer des modèles, pas pour affirmer qu'un véhicule s'est vendu.

**Garde-fou.** Si le volume scrapé tombe sous 60 % des annonces actives connues, le script considère le scraping incomplet et n'enregistre aucune disparition — sinon un scraping interrompu marquerait des centaines d'annonces encore en ligne comme disparues, définitivement. Les annonces qui réapparaissent sont remises en actif.

Les annonces signalées comme opportunités sont marquées (`Etait_Opportunite`). Après quelques semaines, l'onglet Admin compare leur durée en ligne à celle des autres. Si les deux sont identiques, le détecteur ne détecte rien d'utile — et il vaut mieux le savoir.

---

## Application (Streamlit)

Un principe traverse toutes les analyses : **comparer des véhicules comparables**. Un prix médian brut mesure surtout la composition de l'échantillon. Plusieurs graphiques ont été refaits pour cette raison, chaque fois avec la mesure de l'erreur qu'ils produisaient.

### 📊 Vue Marché

**Courbe de dépréciation.** `log(prix) ~ effets fixes modèle + effets fixes âge`, en indice base 100. Le prix médian par âge donnait une courbe fausse : celle de Peugeot *montait* entre 3 et 5 ans, parce qu'à 3 ans l'échantillon était dominé par des 301 (berline économique) et à 5 ans par des 3008 (SUV). Les indicatrices de modèle absorbent la composition. L'âge reste catégoriel pour conserver la chute des premières années.

**Décote annuelle par modèle.** Ajustement `log(prix) ~ âge`, dont la pente donne un taux de perte en % par an — un pourcentage de la valeur restante, ce qui correspond à la dépréciation réelle. Exportable en CSV. Les deux panneaux (« perdent le plus vite » / « tiennent le mieux ») prennent au plus la moitié des modèles disponibles chacun : avec un `head(15)` fixe, sous 30 modèles les listes se recouvraient et à 15 elles devenaient identiques.

**Prime professionnelle, à âge égal.** `log(prix) ~ âge + kilométrage + vendeur_pro` par modèle. Comparer les prix médians donnait +15 % avec des valeurs absurdes (+169 % sur une Škoda Octavia) : les pros vendent des véhicules bien plus récents. Une fois l'âge neutralisé, la prime médiane tombe à **+4 %**. Le kilométrage est indispensable : sans lui, 5 modèles sur 35 affichaient un écart de signe opposé à la réalité. Garde-fou : recouvrement d'âge d'au moins 2 ans entre les deux populations — sans quoi corriger de l'âge revient à extrapoler.

**Niveau de prix par région.** `log(prix) ~ modèle + âge + kilométrage + région`. La corrélation entre le prix médian d'une région et sa part de marques premium était de **+0,70** : Tunis n'est pas chère, 34 % de ses annonces sont des Mercedes, BMW ou Audi contre 10 % à Médenine. Corrigés, les écarts régionaux tiennent dans une fourchette de 9 points.

Également : parts de marché, structure par gamme de prix, prix par énergie avec lissage bayésien (les catégories peu représentées sont ramenées vers la médiane nationale), tendance par jour de collecte.

### 🎯 Vue Samsar

Opportunités chiffrées en dinars, filtre budget/marque/zone, matrice décote × liquidité, tableau avec liens directs, et filtre de fiabilité actif par défaut.

**Rotation du marché**, restreinte à **tayara.tn**. L'âge des annonces en ligne n'est comparable qu'à source égale : automobile.tn ne conserve pas d'annonces anciennes, et la corrélation entre part de professionnels d'un modèle et âge médian de ses annonces est de **−0,43**. En mélangeant les sources, les modèles « les plus rapides » étaient l'Audi A5, la BMW Série 5 et le Range Rover Sport — tous à 100 % de pros. Cette mesure indirecte disparaîtra quand le suivi des annonces aura accumulé assez d'historique.

**Arbitrage géographique.** Même régression que le niveau régional, mais par modèle : où acheter et où revendre un véhicule donné, à âge et kilométrage comparables. Le graphique précédent comparait des prix médians bruts — sur l'Audi A3 Sportback, Sfax affichait 120 000 DT contre 99 250 DT à Nabeul parce que les voitures de Sfax avaient 2 ans et celles de Nabeul 4. 48 % des modèles présentaient ce biais.

### 💰 Calculateur de Juste Prix

Estimation du modèle pour une configuration saisie, confrontée aux annonces comparables réelles : médiane, fourchette interquartile et positionnement visuel parmi les points réels.

### 🛠️ Admin

Comparaison des trois modèles candidats (métriques out-of-fold, modèle retenu), erreur par gamme de prix et par modèle, courbe de calibration de la fenêtre, validation par les disparitions réelles, taux de remplissage des colonnes. Rien ici n'est destiné à un utilisateur final : c'est de quoi juger si les chiffres affichés ailleurs méritent confiance.

---

## Installation et utilisation

```bash
pip install -r requirements.txt
python -m playwright install chromium
python -m patchright install chromium

cp .env.example .env        # TELEGRAM_TOKEN / TELEGRAM_CHAT_ID (facultatif)

python main.py              # pipeline complet
streamlit run app.py        # application
python utils/send_telegram.py   # envoi manuel des alertes
```

Chaque étape peut être lancée individuellement, dans l'ordre de l'architecture.

### Automatisation (GitHub Actions)

`.github/workflows/scraping.yml` lance le pipeline **chaque nuit à minuit heure tunisienne**. Le cron GitHub est en UTC et la Tunisie est à UTC+1 sans changement d'heure : la valeur est `0 23 * * *`.

Secrets requis : `TELEGRAM_TOKEN`, `TELEGRAM_CHAT_ID`.

Cinq garde-fous, chacun couvrant une panne constatée :

- **`fetch-depth: 0`** — `actions/checkout` ne récupère qu'un commit par défaut. Un rebase sur un historique tronqué ne peut pas se raccorder au distant, et le push est rejeté avec « fetch first ».
- **`concurrency`** — un scraping dure plusieurs heures ; sans verrou, le run du lendemain démarrerait par-dessus.
- **Vérification des données produites** — le job échoue si la base scorée est absente, vide ou sous 100 annonces. Sans ce contrôle, un pipeline interrompu laisse les anciens fichiers en place et le commit ne montre aucun changement : tout paraît normal alors que plus rien n'est à jour.
- **Publication différenciée** — `data/raw` est append-only et jamais incohérent : il est publié même en cas d'échec, pour ne pas jeter des heures de collecte. `data/processed` et `data/models` ne sont publiés que si tout a réussi. Sur le chemin d'échec, les fichiers traités modifiés sont restaurés avant le rebase : laissés modifiés mais non stagés, ils rendraient l'arbre de travail sale et `git rebase` refuserait de tourner.
- **Boucle de publication (5 tentatives)** — `fetch` + `rebase -X theirs` + `push`. Pendant un *rebase* les rôles sont inversés par rapport à un merge : `ours` désigne la branche sur laquelle on rejoue (le distant) et `theirs` les commits rejoués (les données scrapées). `-X ours` jetterait donc silencieusement le scraping de la nuit.

### Déploiement Streamlit Cloud

**Versions épinglées.** `scikit-learn` est figé à l'exact dans `requirements.txt`. Le modèle est entraîné par GitHub Actions et chargé par Streamlit Cloud : si les deux environnements résolvent des versions différentes, le dépickle échoue et le calculateur tombe en erreur, sans autre signal. Après toute mise à jour de cette version, réentraîner le modèle et le committer.

**Lecture des données depuis GitHub.** Le rafraîchissement automatique de Streamlit Cloud après un push n'est pas fiable — l'app conserve fréquemment l'ancien contenu jusqu'à un redémarrage manuel. Les données étant réécrites chaque nuit, s'y fier reviendrait à afficher des annonces périmées sans signal visible. L'application lit donc les CSV et le modèle **directement depuis GitHub (raw)**, avec un cache d'une heure, et retombe sur les fichiers locaux en cas d'échec réseau.

Deux constantes en haut de `app.py` : `DEPOT_GITHUB` (à changer en cas de fork) et `LIRE_DEPUIS_GITHUB` (`False` pour travailler hors ligne). La barre latérale affiche la date de dernière collecte, alerte au-delà de deux jours, et propose un rafraîchissement manuel.

---

## Fichiers de données

| Fichier | Contenu |
|---|---|
| `data/raw/{tayara,automobile,automax}.csv` | Annonces brutes par source (append, dédupliquées par lien) |
| `data/processed/tunisia-cars.csv` | Fusion normalisée des 3 sources |
| `data/processed/tunisia-cars-recent.csv` | Base nettoyée, Marque/Modèle complétés, ≤ 60 jours |
| `data/processed/tunisia-cars-final-features.csv` | + variables dérivées |
| `data/processed/tunisia-cars-scored.csv` | + Prix_Theorique, Score_Opportunite, Score_Liquidite, fiabilité |
| `data/processed/alertes_bonnes_affaires.csv` | Opportunités retenues (25 % – 55 % sous l'argus) |
| `data/processed/suivi_annonces.csv` | Historique apparition / disparition de chaque annonce |
| `data/processed/diagnostics_modele.json` | Métriques des modèles candidats (onglet Admin) |
| `data/processed/calibration_fenetre.json` | Mesures de calibration de `MAX_DAYS_OLD` |
| `data/models/modele_prix.pkl` | Pipeline scikit-learn entraîné |

---

## Dépannage

- **`tunisia-cars-scored.csv` introuvable dans l'app** → lancer `python main.py`.
- **L'app affiche des données anciennes** → `LIRE_DEPUIS_GITHUB = True` fait lire le dépôt distant même en local. Vérifier que le dernier run CI a bien poussé, ou passer la constante à `False`.
- **`ModuleNotFoundError: No module named 'logger'`** → le script concerné n'ajoute pas la racine à `sys.path` (voir l'encadré de la section Architecture).
- **Push CI rejeté (« fetch first »)** → vérifier `fetch-depth: 0` et la présence de la boucle de publication ; le log doit afficher `Publication, tentative 1/5...`.
- **Erreur scikit-learn au calcul du Juste Prix** → modèle entraîné avec une autre version : relancer `python core/modele_prediction.py`.
- **automobile.tn bloqué (Cloudflare)** → Patchright avec `channel="chrome"` en mode non-headless, d'où `xvfb` en CI ; éviter de lancer Chrome en root.
- **Aucune alerte Telegram** → vérifier les secrets et que `alertes_bonnes_affaires.csv` n'est pas vide.

---

## Limites connues

**Mesures transversales.** Toutes les analyses de dépréciation comparent des véhicules d'âges différents à un instant donné ; elles ne suivent pas un véhicule dans le temps. Si les millésimes récents sont mieux équipés, une part de l'écart attribué à l'âge vient de l'équipement. C'est la méthode standard faute de données longitudinales, mais c'en est une limite.

**Boucle de rétroaction ouverte.** Le système prédit des prix demandés et signale ce qui est en dessous. Il n'observe jamais si une opportunité s'est vendue, ni à quel prix. Les seuils (25 %, 8 comparables) sont calés sur le bruit mesuré du modèle, pas sur des affaires réellement conclues. Le suivi des annonces est la première brique pour fermer cette boucle.

**Plancher de précision.** Sur des véhicules quasi identiques (même modèle, même année, kilométrage proche), la dispersion des prix demandés est déjà de 4 %. Sur ce segment le modèle atteint 6 % : la marge restante y est mince. L'erreur se concentre sur les véhicules anciens (13,1 % au-delà de 13 ans, contre 9,2 % sous 3 ans), où l'état domine le prix et où aucune donnée n'est disponible — `Etat_Vehicule` est vide à 100 %.
