# Intégration produit AutoDeal — août 2026

## Nouveautés

1. Accueil grand public avec 3 parcours : trouver, estimer, bonnes affaires.
2. Cartes annonces visuelles avec prix, estimation, fourchette, écart, comparables et score AutoDeal.
3. Fiche détaillée d'une annonce avec explication du score et comparables réels.
4. Recherche orientée budget / année / kilométrage / énergie / boîte.
5. Comparateur de 2 à 4 véhicules avec recommandation du meilleur compromis.
6. Score AutoDeal explicable (55 % prix vs marché, 25 % confiance, 20 % liquidité).
7. Fourchette de prix autour de l'estimation plutôt qu'une fausse précision ponctuelle.
8. Fraîcheur, nombre de comparables, fiabilité et erreur typique affichés.
9. Historique du marché basé sur suivi_annonces.csv : baisses observées et tendances de cohortes.
10. Page Alertes : filtre de bonnes affaires en session + intégration avec l'existant Telegram global.

## Architecture

- `product_views.py` contient les nouvelles vues produit afin d'éviter de gonfler encore `app.py`.
- `app.py` conserve les écrans existants et adopte une navigation centrée acheteur.
- Les anciennes vues `Concessionnaire`, `Samsar`, `Carte`, `Recherche avancée`, `Admin` restent disponibles.

## Limite volontaire sur les alertes

Il n'existe pas encore de comptes utilisateurs ni de base persistante. Une alerte personnalisée configurée dans l'interface ne peut donc pas survivre de manière fiable à une nouvelle session. Le pipeline Telegram global existant reste actif. Pour des alertes personnalisées persistantes, brancher une base (par exemple Supabase) et faire lire les critères par GitHub Actions avant envoi Telegram/email.

## Validation

- `python -m py_compile app.py product_views.py` : OK
- `pytest -q` : 67 tests passés
