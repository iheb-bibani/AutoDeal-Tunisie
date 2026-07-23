"""
detect_deals.py
Filtre data/tunisia-cars-scored.csv (produit par modele_prediction.py) pour
n'garder que les vraies opportunités, et écrit data/alertes_bonnes_affaires.csv
-- le fichier que lisent utils/send_telegram.py et app.py.

Ce script remplace l'ancienne version qui lisait "data/voitures_clean.csv"
(un fichier qui n'existe nulle part dans le pipeline réel) avec des noms de
colonnes d'une génération antérieure du schéma (Prix_DT, Date_Detection).

Point important ajouté ici, absent de toutes les versions précédentes :
un PLAFOND de plausibilité. Sur les vraies données, les "meilleures
opportunités" à >60% sous le prix théorique sont presque toujours des erreurs
(prix mal saisi, annonce pour pièces détachées ayant échappé au filtre de
bruit) plutôt que de vraies affaires -- les envoyer sur Telegram ferait plus
de mal que de bien à la confiance dans l'outil.
"""

import os
import pandas as pd

IN_FICHIER = "data/processed/tunisia-cars-scored.csv"
FICHIER_ALERTES = "data/processed/alertes_bonnes_affaires.csv"

# SEUIL_MIN était fixé à 0.15. Problème mesuré sur les données réelles :
# l'erreur relative médiane du modèle est de ~12%. Un seuil à 15% est donc
# À L'INTÉRIEUR du bruit de l'estimation -- il retenait 18% de TOUT le marché
# (426 annonces sur 2308) comme "bonnes affaires", ce qui n'a aucun sens :
# une annonce sur cinq n'est pas une opportunité, c'est simplement une annonce
# que le modèle estime mal. Le seuil est porté à 25%, soit environ deux fois
# l'erreur typique -- au-dessus du bruit, l'écart devient un vrai signal.
SEUIL_MIN = 0.25
SEUIL_MAX = 0.55  # au-delà de 55%, c'est presque toujours une erreur de prix, pas une affaire

# Une estimation appuyée sur trop peu d'annonces comparables n'est pas fiable :
# les "affaires" sur les modèles quasi absents du marché sont majoritairement
# des erreurs d'estimation. On les garde dans le fichier (colonne
# Fiabilite_Estimation) mais elles ne déclenchent pas d'alerte Telegram.
COMPARABLES_MIN_POUR_ALERTE = 8


def calculer_argus_et_liquidite():
    if not os.path.exists(IN_FICHIER):
        print(f"❌ Fichier introuvable : {IN_FICHIER} (lance merging_files.py -> nettoyer_base.py -> enrichir_base_avance.py -> modele_prediction.py avant)")
        return

    df = pd.read_csv(IN_FICHIER, sep=";", encoding="utf-8-sig")
    print(f"📊 {len(df)} annonces scorées chargées depuis {IN_FICHIER}")

    if df.empty:
        print("⚠️ Fichier vide -- rien à filtrer.")
        return

    deals = df[(df["Score_Opportunite"] >= SEUIL_MIN) & (df["Score_Opportunite"] <= SEUIL_MAX)]

    exclues_trop_belles = (df["Score_Opportunite"] > SEUIL_MAX).sum()
    if exclues_trop_belles:
        print(f"⚠️ {exclues_trop_belles} annonces avec un score > {SEUIL_MAX:.0%} écartées "
              f"(quasi certainement des erreurs de prix, pas de vraies affaires).")

    print(f"🔎 {len(deals)} opportunité(s) retenue(s) entre {SEUIL_MIN:.0%} et {SEUIL_MAX:.0%} sous le prix théorique.")

    if "Nb_Comparables" in deals.columns:
        deals = deals.copy()
        deals["Alerte_Telegram"] = deals["Nb_Comparables"] >= COMPARABLES_MIN_POUR_ALERTE
        nb_solides = int(deals["Alerte_Telegram"].sum())
        print(f"   dont {nb_solides} appuyée(s) sur au moins {COMPARABLES_MIN_POUR_ALERTE} annonces "
              f"comparables (les seules qui déclencheront une alerte Telegram) ; "
              f"{len(deals) - nb_solides} reposent sur trop peu de comparables pour être fiables.")
        # Les plus fiables en premier, puis le gain décroissant
        deals = deals.sort_values(["Alerte_Telegram", "Score_Opportunite"], ascending=[False, False])

    # Toujours écrire le fichier, même vide (avec les en-têtes) : sinon
    # l'ancien fichier reste en place et l'app + Telegram continuent
    # d'afficher/envoyer des opportunités périmées d'un scraping précédent.
    if "Alerte_Telegram" not in deals.columns:
        deals = deals.sort_values("Score_Opportunite", ascending=False)
    deals.to_csv(FICHIER_ALERTES, index=False, sep=";", encoding="utf-8-sig")
    if len(deals):
        print(f"✅ Fichier écrit : {FICHIER_ALERTES}")
    else:
        print(f"⚠️ Aucun deal dans la fourchette retenue -> {FICHIER_ALERTES} écrit vide (en-têtes seuls).")


if __name__ == "__main__":
    calculer_argus_et_liquidite()
