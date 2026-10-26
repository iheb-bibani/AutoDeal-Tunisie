"""
suivi_annonces.py
Suit chaque annonce dans le temps : première/dernière observation, variations
de prix, disparition et réapparition.

Une disparition est un proxy d'écoulement, jamais une vente certifiée. Une
annonce peut être vendue, retirée, expirée ou supprimée.

Le point critique est la qualité du scraping : une source qui tombe ne doit
jamais transformer toutes ses annonces en fausses « disparitions ». Les
garde-fous sont donc appliqués à deux niveaux : volume global ET volume par
source.
"""

import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

from config import PROCESSED_FILES
from core.io_atomique import ecrire_csv_atomique
from scrapers.liens_vus import SOURCE_LABELS, charger_liens_vus
from scrapers.parsing import parser_prix_tnd

MANIFESTE = "data/processed/pipeline_run.json"

FICHIER_SUIVI = "data/processed/suivi_annonces.csv"
FICHIER_ALERTES = "data/processed/alertes_bonnes_affaires.csv"

# Si le catalogue courant tombe sous 60 % du volume actif précédent, on
# considère le scraping comme incomplet. Ce seuil s'applique au total et à
# chaque source indépendamment.
SEUIL_VOLUME_SUSPECT = 0.60

COLONNES_SNAPSHOT = ["Source", "Marque", "Modèle", "Année", "Localisation"]


def charger_suivi():
    if os.path.exists(FICHIER_SUIVI):
        suivi = pd.read_csv(FICHIER_SUIVI, sep=";", encoding="utf-8-sig")
        for col in ["Date_Disparition", "Premiere_Vue", "Derniere_Vue", "Statut"]:
            if col in suivi.columns:
                suivi[col] = suivi[col].astype(object)
        return suivi
    return pd.DataFrame(columns=[
        "Lien", "Source", "Marque", "Modèle", "Année", "Localisation",
        "Prix_Initial", "Prix_Dernier", "Premiere_Vue", "Derniere_Vue",
        "Statut", "Date_Disparition", "Jours_En_Ligne", "Nb_Reapparitions",
    ])


def _sources_suspectes(suivi: pd.DataFrame, merged: pd.DataFrame) -> set[str]:
    """Détecte les sources dont le scraping courant est probablement partiel.

    Le contrôle global seul est insuffisant : si une source représentant 25 %
    du marché tombe totalement, le volume global peut rester au-dessus de 60 %
    et ses annonces seraient alors toutes marquées disparues.
    """
    if suivi.empty or "Source" not in suivi.columns or "Source" not in merged.columns:
        return set()

    actifs = suivi[suivi["Statut"] == "Active"].copy()
    if actifs.empty:
        return set()

    avant = actifs["Source"].fillna("<inconnue>").astype(str).value_counts()
    courant = merged["Source"].fillna("<inconnue>").astype(str).value_counts()
    suspectes: set[str] = set()
    for source, n_avant in avant.items():
        n_courant = int(courant.get(source, 0))
        if n_avant > 0 and n_courant < n_avant * SEUIL_VOLUME_SUSPECT:
            suspectes.add(str(source))
            print(
                f"⚠️  Source {source!r} : {n_courant} annonce(s) aujourd'hui vs "
                f"{n_avant} active(s) auparavant. Les disparitions de cette source "
                "sont gelées pour ce run."
            )
    return suspectes


def _sources_echouees() -> set[str]:
    """Scrapers en échec/timeout dans le manifeste du run courant."""
    try:
        import json

        with open(MANIFESTE, encoding="utf-8") as f:
            scrapers = (json.load(f) or {}).get("scrapers") or {}
    except (OSError, ValueError):
        return set()
    return {k for k, v in scrapers.items() if isinstance(v, dict) and v.get("status") != "success"}


def observations_du_jour() -> tuple[set[str], dict[str, float], set[str]] | None:
    """Liens réellement vus aujourd'hui par les scrapers ayant abouti.

    Renvoie (liens, prix_affiches, labels_sources_observees) ou None si aucun
    scraper n'a publié ses liens vus (mode historique).
    """
    vus = charger_liens_vus()
    echecs = _sources_echouees()
    liens: set[str] = set()
    prix: dict[str, float] = {}
    labels: set[str] = set()
    for source, liens_prix in vus.items():
        if source in echecs or source not in SOURCE_LABELS:
            continue
        labels.add(SOURCE_LABELS[source])
        for lien, p in liens_prix.items():
            liens.add(str(lien))
            v = parser_prix_tnd(f"{p} DT") if p not in (None, "", "N/A") else None
            if v:
                prix[str(lien)] = float(v)
    return (liens, prix, labels) if labels else None


def _source_observee(sources: pd.Series, labels: set[str]) -> pd.Series:
    s = sources.fillna("").astype(str)
    masque = pd.Series(False, index=sources.index)
    for label in labels:
        masque |= s.str.startswith(label)
    return masque


def _finaliser(suivi, merged, connus, aujourd_hui, nb_disparues):
    """Ajoute les nouvelles annonces, marque les opportunités et écrit le suivi."""
    # ---- Nouvelles annonces --------------------------------------------
    nouveaux = merged[~merged["Lien"].astype(str).isin(connus)].copy()
    if len(nouveaux):
        ajout = pd.DataFrame({
            "Lien": nouveaux["Lien"].astype(str),
            "Prix_Initial": nouveaux["Prix"],
            "Prix_Dernier": nouveaux["Prix"],
            "Premiere_Vue": aujourd_hui,
            "Derniere_Vue": aujourd_hui,
            "Statut": "Active",
            "Date_Disparition": pd.NA,
            "Jours_En_Ligne": pd.NA,
            "Nb_Reapparitions": 0,
        })
        for col in COLONNES_SNAPSHOT:
            ajout[col] = nouveaux[col].values if col in nouveaux.columns else pd.NA
        suivi = pd.concat([suivi, ajout], ignore_index=True)

    # ---- Marquage historique des opportunités ---------------------------
    if os.path.exists(FICHIER_ALERTES):
        try:
            alertes = pd.read_csv(FICHIER_ALERTES, sep=";", encoding="utf-8-sig")
            if len(alertes) and "Lien" in alertes.columns:
                if "Etait_Opportunite" not in suivi.columns:
                    suivi["Etait_Opportunite"] = False
                est_deal = suivi["Lien"].astype(str).isin(
                    alertes["Lien"].astype(str)
                )
                suivi.loc[est_deal, "Etait_Opportunite"] = True
        except (pd.errors.EmptyDataError, KeyError):
            pass
    if "Etait_Opportunite" not in suivi.columns:
        suivi["Etait_Opportunite"] = False
    suivi["Etait_Opportunite"] = (
        suivi["Etait_Opportunite"]
        .astype(str).str.strip().str.lower()
        .isin({"true", "1", "vrai"})
    )

    # Historique longitudinal non reconstructible : écriture atomique.
    ecrire_csv_atomique(suivi, FICHIER_SUIVI)

    actives = int((suivi["Statut"] == "Active").sum())
    disparues_total = int((suivi["Statut"] == "Disparue").sum())
    print("-" * 30)
    print(f"Nouvelles annonces suivies : {len(nouveaux)}")
    print(f"Disparues aujourd'hui      : {nb_disparues}")
    print(
        f"Total suivi                : {len(suivi)} "
        f"({actives} actives, {disparues_total} disparues)"
    )
    mesurables = pd.to_numeric(
        suivi["Jours_En_Ligne"], errors="coerce"
    ).dropna()
    if len(mesurables) >= 10:
        print(
            f"Durée médiane en ligne     : {mesurables.median():.0f} jours "
            f"(sur {len(mesurables)} annonces disparues)"
        )
    else:
        print(
            "Durée médiane en ligne     : pas encore assez d'historique "
            "(il faut plusieurs jours de collecte)."
        )
    print(f"Fichier : {FICHIER_SUIVI}")
    print("-" * 30)


def mettre_a_jour_depuis_observations(suivi, merged, connus, aujourd_hui, liens_vus, prix_vus, labels):
    """Suivi fondé sur les liens réellement observés par chaque scraper.

    Une annonce ne peut disparaître que si SA source a été observée
    aujourd'hui, et que le volume observé de cette source est crédible.
    """
    print(f"Annonces vues aujourd'hui (pages de liste) : {len(liens_vus)} — sources : {', '.join(sorted(labels))}")
    prix_merged = merged.set_index(merged["Lien"].astype(str))["Prix"] if "Prix" in merged else pd.Series(dtype=float)

    if len(suivi):
        source_suivi = suivi["Source"] if "Source" in suivi.columns else pd.Series(pd.NA, index=suivi.index)
        observee = _source_observee(source_suivi, labels)
        vue = suivi["Lien"].astype(str).isin(liens_vus)

        # Garde-fou par source : volume vu vs actifs connus de cette source.
        suspects = set()
        for label in labels:
            de_la_source = source_suivi.fillna("").astype(str).str.startswith(label)
            actifs = int((de_la_source & (suivi["Statut"] == "Active")).sum())
            vus_source = int((de_la_source & vue).sum())
            if actifs and vus_source < actifs * SEUIL_VOLUME_SUSPECT:
                suspects.add(label)
                print(f"⚠️  {label} : {vus_source} annonce(s) suivie(s) revue(s) sur {actifs} actives — "
                      "disparitions gelées pour cette source.")
        observee &= ~_source_observee(source_suivi, suspects)

        suivi.loc[vue, "Derniere_Vue"] = aujourd_hui
        liens = suivi.loc[vue, "Lien"].astype(str)
        nouveau_prix = liens.map(prix_vus).fillna(liens.map(prix_merged))
        suivi.loc[vue, "Prix_Dernier"] = nouveau_prix.where(nouveau_prix.notna(), suivi.loc[vue, "Prix_Dernier"]).values

        revenues = vue & (suivi["Statut"] == "Disparue")
        if revenues.sum():
            suivi.loc[revenues, "Statut"] = "Active"
            suivi.loc[revenues, "Date_Disparition"] = pd.NA
            suivi.loc[revenues, "Jours_En_Ligne"] = pd.NA
            suivi.loc[revenues, "Nb_Reapparitions"] = (
                pd.to_numeric(suivi.loc[revenues, "Nb_Reapparitions"], errors="coerce").fillna(0) + 1
            )
            print(f"↩️  {int(revenues.sum())} annonce(s) réapparue(s) — remises en actif.")

        disparues = observee & ~vue & (suivi["Statut"] == "Active")
        nb_disparues = int(disparues.sum())
        if nb_disparues:
            suivi.loc[disparues, "Statut"] = "Disparue"
            suivi.loc[disparues, "Date_Disparition"] = aujourd_hui
            duree = (pd.to_datetime(aujourd_hui) - pd.to_datetime(suivi.loc[disparues, "Premiere_Vue"], errors="coerce")).dt.days
            suivi.loc[disparues, "Jours_En_Ligne"] = duree.values
    else:
        nb_disparues = 0

    _finaliser(suivi, merged, connus, aujourd_hui, nb_disparues)


def mettre_a_jour():
    aujourd_hui = datetime.now().strftime("%Y-%m-%d")

    merged = pd.read_csv(PROCESSED_FILES["merged"], sep=";", encoding="utf-8-sig")
    merged = merged.dropna(subset=["Lien"]).drop_duplicates(subset=["Lien"], keep="last")
    suivi = charger_suivi()
    connus = set(suivi["Lien"].astype(str)) if len(suivi) else set()

    observations = observations_du_jour()
    if observations is not None:
        mettre_a_jour_depuis_observations(suivi, merged, connus, aujourd_hui, *observations)
        return

    # Mode historique (aucun scraper n'a publié ses liens vus) : le fichier
    # fusionné sert de catalogue du jour.
    liens_du_jour = set(merged["Lien"].astype(str))
    print(f"Annonces vues aujourd'hui : {len(liens_du_jour)}")

    # ---- Garde-fous anti faux-positifs -----------------------------------
    actifs_avant = int((suivi["Statut"] == "Active").sum()) if len(suivi) else 0
    scraping_suspect = (
        actifs_avant > 0
        and len(liens_du_jour) < actifs_avant * SEUIL_VOLUME_SUSPECT
    )
    if scraping_suspect:
        print(
            f"⚠️  Volume global du jour ({len(liens_du_jour)}) très inférieur aux "
            f"{actifs_avant} annonces actives connues : aucune disparition ne sera "
            "enregistrée sur ce run."
        )
    sources_suspectes = _sources_suspectes(suivi, merged)

    # ---- Annonces déjà suivies et revues aujourd'hui --------------------
    if len(suivi):
        vue = suivi["Lien"].astype(str).isin(liens_du_jour)
        prix_du_jour = merged.set_index(merged["Lien"].astype(str))["Prix"]
        suivi.loc[vue, "Derniere_Vue"] = aujourd_hui
        suivi.loc[vue, "Prix_Dernier"] = (
            suivi.loc[vue, "Lien"].astype(str).map(prix_du_jour).values
        )

        revenues = vue & (suivi["Statut"] == "Disparue")
        if revenues.sum():
            suivi.loc[revenues, "Statut"] = "Active"
            suivi.loc[revenues, "Date_Disparition"] = pd.NA
            suivi.loc[revenues, "Jours_En_Ligne"] = pd.NA
            suivi.loc[revenues, "Nb_Reapparitions"] = (
                pd.to_numeric(
                    suivi.loc[revenues, "Nb_Reapparitions"], errors="coerce"
                ).fillna(0) + 1
            )
            print(f"↩️  {int(revenues.sum())} annonce(s) réapparue(s) — remises en actif.")

    # ---- Disparitions ---------------------------------------------------
    nb_disparues = 0
    if len(suivi) and not scraping_suspect:
        disparues = (
            ~suivi["Lien"].astype(str).isin(liens_du_jour)
            & (suivi["Statut"] == "Active")
        )
        if sources_suspectes and "Source" in suivi.columns:
            source_norm = suivi["Source"].fillna("<inconnue>").astype(str)
            disparues &= ~source_norm.isin(sources_suspectes)

        if disparues.sum():
            suivi.loc[disparues, "Statut"] = "Disparue"
            suivi.loc[disparues, "Date_Disparition"] = aujourd_hui
            duree = (
                pd.to_datetime(aujourd_hui)
                - pd.to_datetime(
                    suivi.loc[disparues, "Premiere_Vue"], errors="coerce"
                )
            ).dt.days
            suivi.loc[disparues, "Jours_En_Ligne"] = duree.values
            nb_disparues = int(disparues.sum())

    _finaliser(suivi, merged, connus, aujourd_hui, nb_disparues)

if __name__ == "__main__":
    mettre_a_jour()
