# Fiabilité du modèle AutoDeal

Ce document formalise la lecture des diagnostics ML affichés dans l'espace Admin.

## Ce que signifie le MdAPE

Le **MdAPE** est la médiane de `|prix_estimé - prix_observé| / prix_observé`.
Un MdAPE de 8,1 % signifie que la moitié des observations évaluées ont une erreur
relative inférieure ou égale à 8,1 %. Cela **ne signifie pas** « 91,9 % de
précision » et ne signifie pas non plus que chaque véhicule a 8,1 % d'erreur.

Exemple : 8,1 % de 30 000 DT représente 2 430 DT. C'est un ordre de grandeur de
l'erreur relative médiane, pas une erreur garantie sur chaque annonce.

## Pourquoi la tendance temporelle compte

Une amélioration simultanée de la MdAPE et du volume est un signal plus solide
qu'un excellent run isolé. On surveille donc ensemble :

- MdAPE ;
- volume évalué ;
- stabilité entre runs ;
- drift de prix, volume et mix des sources.

Une courbe durablement basse et stable est plus rassurante qu'un minimum ponctuel.

## Pourquoi regarder les tranches de prix

La performance globale masque l'hétérogénéité. AutoDeal suit donc les erreurs par
tranche de prix. Une forme en U est plausible : les véhicules très bon marché
peuvent dépendre davantage de l'état réel, de réparations ou d'accidents mal
capturés ; le haut de gamme est souvent plus rare et hétérogène (options,
configurations, importation, motorisation).

Le cœur de marché peut ainsi être beaucoup mieux estimé que les extrêmes.

## Toujours afficher N

Une MdAPE de 6 % calculée sur 1 500 observations est beaucoup plus crédible que
6 % sur 12 observations. Toute lecture segmentée doit donc être accompagnée du
nombre d'observations `N`.

## Métriques complémentaires obligatoires

Le MdAPE seul ne décrit pas les mauvais cas. Le monitoring calcule désormais :

- **MdAE en DT** : erreur absolue médiane ;
- **MAE en DT** : erreur absolue moyenne ;
- **P90 de l'erreur relative** : 90 % des observations ont une erreur inférieure à ce seuil ;
- **% à ±5 %, ±10 %, ±15 %** : lecture métier compréhensible ;
- **biais médian signé** : détecte une tendance à sur-estimer ou sous-estimer.

Exemple de formulation produit recommandée : « 74 % des véhicules évalués sont à
moins de ±10 % de leur prix observé », si les données du run le confirment.

## Où chercher les prochaines améliorations

Avant de changer d'algorithme, localiser les erreurs par :

- marque ;
- modèle ;
- âge ;
- kilométrage ;
- énergie ;
- tranche de prix ;
- profondeur/nombre de comparables.

L'objectif est d'identifier les segments où le modèle manque de données ou de
variables explicatives. C'est généralement plus utile que d'optimiser aveuglément
la métrique globale.

## Validation réellement hors-échantillon

Les métriques utilisées pour communiquer la qualité du modèle doivent provenir de
prédictions hors-échantillon. Le projet dispose aussi de stress-tests : validation
croisée, GroupKFold marque-modèle, holdout temporel et holdout par source.

Le **holdout temporel** est particulièrement important : il répond à la question
métier « avec ce que le modèle savait hier, estime-t-il correctement les annonces
les plus récentes ? ».

## Confiance affichée à l'utilisateur

La confiance d'une estimation ne doit pas dépendre uniquement du prix prédit. Elle
doit combiner au minimum :

1. erreur historique de la tranche ;
2. nombre de comparables ;
3. homogénéité des comparables ;
4. accord entre estimation ML et marché comparable.

Une voiture dans une tranche à ~6 % d'erreur avec beaucoup de comparables peut
être affichée comme **fiabilité élevée**. Une voiture rare ou dans une tranche à
~11–15 % doit afficher une prudence explicite.
