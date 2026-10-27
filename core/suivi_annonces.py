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
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from core.dates import aujourd_hui as date_du_jour  # noqa: E402

import pandas as pd

from config import PROCESSED_FILES
from core.doublons import lien_canonique
from core.io_atomique import ecrire_csv_atomique
from scrapers.liens_vus import SOURCE_LABELS, charger_catalogues
from scrapers.parsing import parser_prix_tnd

MANIFESTE = "data/processed/pipeline_run.json"

FICHIER_SUIVI = "data/processed/suivi_annonces.csv"
FICHIER_ALERTES = "data/processed/alertes_bonnes_affaires.csv"

# Si le catalogue courant tombe sous 60 % du volume actif précédent, on
# considère le scraping comme incomplet. Ce seuil s'applique au total et à
# chaque source indépendamment.
SEUIL_VOLUME_SUSPECT = 0.60

# Catalogue partiel (plafond de pages) : les pages de liste sont triées par
# date. On ne juge les disparitions que pour les annonces déposées après le
# 1er quartile des dates de dépôt observées (prudent : les annonces anciennes
# « remontées » en tête de liste tirent le minimum vers le bas), plus une marge.
QUANTILE_COUVERTURE = 0.25
MARGE_COUVERTURE_JOURS = 2
MIN_LIENS_DATES_COUVERTURE = 50

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


def observations_du_jour():
    """Liens réellement vus aujourd'hui par les scrapers ayant abouti.

    Renvoie (liens, prix_affiches, sources_completes, sources_partielles) ou
    None si aucun scraper n'a publié de catalogue (mode historique).
    ``sources_partielles`` : {label: ensemble des liens vus pour cette source}.
    """
    catalogues = charger_catalogues()
    echecs = _sources_echouees()
    liens: set[str] = set()
    prix: dict[str, float] = {}
    completes: set[str] = set()
    partielles: dict[str, set[str]] = {}
    for source, cat in catalogues.items():
        if source in echecs or source not in SOURCE_LABELS:
            continue
        label = SOURCE_LABELS[source]
        vus_source = {str(lien) for lien in cat["liens"]}
        if cat.get("complet"):
            completes.add(label)
        else:
            partielles[label] = vus_source
        liens |= vus_source
        for lien, p in cat["liens"].items():
            v = parser_prix_tnd(f"{p} DT") if p not in (None, "", "N/A") else None
            if v:
                prix[str(lien)] = float(v)
    return (liens, prix, completes, partielles) if (completes or partielles) else None


def date_couverture(liens_vus_source: set[str], dates_depot: pd.Series) -> pd.Timestamp | None:
    """Date à partir de laquelle un catalogue partiel est jugé exhaustif."""
    dates = dates_depot.reindex(list(liens_vus_source)).dropna()
    if len(dates) < MIN_LIENS_DATES_COUVERTURE:
        return None
    return dates.quantile(QUANTILE_COUVERTURE) + pd.Timedelta(days=MARGE_COUVERTURE_JOURS)


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


def mettre_a_jour_depuis_observations(suivi, merged, connus, aujourd_hui, liens_vus, prix_vus, labels,
                                      partielles=None):
    """Suivi fondé sur les liens réellement observés par chaque scraper.

    Une annonce ne peut disparaître que si SA source a été observée
    aujourd'hui (catalogue complet, ou catalogue partiel couvrant sa date de
    dépôt), et que le volume observé de cette source est crédible.
    """
    partielles = partielles or {}
    tous = sorted(labels | set(partielles))
    print(f"Annonces vues aujourd'hui (pages de liste) : {len(liens_vus)} — sources : {', '.join(tous)}")
    liens_merged = merged["Lien"].astype(str)
    prix_merged = merged.set_index(liens_merged)["Prix"] if "Prix" in merged else pd.Series(dtype=float)
    depot_merged = (
        pd.to_datetime(merged.set_index(liens_merged)["Annonce-Deposee"], errors="coerce")
        if "Annonce-Deposee" in merged else pd.Series(dtype="datetime64[ns]")
    )
    depot_merged = depot_merged[~depot_merged.index.duplicated(keep="last")]

    if len(suivi):
        source_suivi = suivi["Source"] if "Source" in suivi.columns else pd.Series(pd.NA, index=suivi.index)
        observee = _source_observee(source_suivi, labels)
        # Comparaison sur le lien canonique (tayara : identifiant d'annonce,
        # indépendant de la ville présente dans l'URL).
        canon_suivi = lien_canonique(suivi["Lien"])
        canon_vus = set(lien_canonique(pd.Series(sorted(liens_vus), dtype=str))) if liens_vus else set()
        vue = canon_suivi.isin(canon_vus)
        prix_vus_canon = {}
        if prix_vus:
            cles = pd.Series(list(prix_vus.keys()), dtype=str)
            prix_vus_canon = dict(zip(lien_canonique(cles), prix_vus.values(), strict=True))

        # Catalogues partiels : seulement la fenêtre de dates réellement couverte.
        depot_suivi = suivi["Lien"].astype(str).map(depot_merged)
        for label, vus_label in partielles.items():
            debut = date_couverture(vus_label, depot_merged)
            if debut is None:
                print(f"ℹ️  {label} : catalogue partiel sans dates exploitables — pas de disparition.")
                continue
            couverte = _source_observee(source_suivi, {label}) & (depot_suivi >= debut)
            observee |= couverte
            print(f"ℹ️  {label} : catalogue partiel, disparitions jugées pour les dépôts ≥ {debut.date()}.")

        # Garde-fou par source : volume vu vs actifs connus de cette source
        # (dans la fenêtre observée).
        suspects = set()
        for label in tous:
            de_la_source = source_suivi.fillna("").astype(str).str.startswith(label) & observee
            actifs = int((de_la_source & (suivi["Statut"] == "Active")).sum())
            vus_source = int((de_la_source & vue).sum())
            if actifs and vus_source < actifs * SEUIL_VOLUME_SUSPECT:
                suspects.add(label)
                print(f"⚠️  {label} : {vus_source} annonce(s) suivie(s) revue(s) sur {actifs} actives — "
                      "disparitions gelées pour cette source.")
        observee &= ~_source_observee(source_suivi, suspects)

        suivi.loc[vue, "Derniere_Vue"] = aujourd_hui
        liens = suivi.loc[vue, "Lien"].astype(str)
        nouveau_prix = canon_suivi[vue].map(prix_vus_canon).fillna(liens.map(prix_merged))
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
    aujourd_hui = date_du_jour()

    merged = pd.read_csv(PROCESSED_FILES["merged"], sep=";", encoding="utf-8-sig")
    merged = merged.dropna(subset=["Lien"]).drop_duplicates(subset=["Lien"], keep="last")
    suivi = charger_suivi()
    connus = set(suivi["Lien"].astype(str)) if len(suivi) else set()

    observations = observations_du_jour()
    if observations is not None:
        liens, prix, completes, partielles = observations
        mettre_a_jour_depuis_observations(suivi, merged, connus, aujourd_hui, liens, prix, completes, partielles)
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
