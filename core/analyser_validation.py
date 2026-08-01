"""
analyser_validation.py
Répond à LA question du projet : les annonces signalées comme "bonnes affaires"
en sont-elles vraiment ?

Deux niveaux, selon ce que l'historique permet :

  1. TRANSVERSAL (disponible dès le premier jour) -- plausibilité :
     les deals se concentrent-ils là où le modèle est le moins fiable
     (peu de comparables, marques rares/luxe à forte erreur absolue) ?
     Ce n'est PAS une preuve, c'est un contrôle de cohérence.

  2. LONGITUDINAL (nécessite plusieurs semaines de suivi) -- la vraie preuve :
     les annonces flaggées disparaissent-elles plus vite que les autres ?
     Une disparition n'est pas forcément une vente, mais si les deals
     s'écoulent significativement plus vite, le Score_Opportunite capte un
     signal réel. Test de Mann-Whitney (pas de normalité supposée).

Usage : python core/analyser_validation.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from config import PROCESSED_FILES, SEUIL_DEAL_MIN, SEUIL_DEAL_MAX, COMPARABLES_MIN_POUR_ALERTE

FICHIER_SUIVI = "data/processed/suivi_annonces.csv"


def analyse_transversale(df: pd.DataFrame) -> None:
    print("=" * 64)
    print("1. VALIDATION TRANSVERSALE (plausibilité, disponible aujourd'hui)")
    print("=" * 64)

    deals = df[df["Score_Opportunite"].between(SEUIL_DEAL_MIN, SEUIL_DEAL_MAX)]
    taux = len(deals) / len(df) if len(df) else 0
    print(f"Marché scoré : {len(df)} | deals [{SEUIL_DEAL_MIN:.0%}-{SEUIL_DEAL_MAX:.0%}] : "
          f"{len(deals)} ({taux:.1%})")

    if "Fiabilite_Estimation" in deals.columns:
        print("\nRépartition des deals par fiabilité :")
        for k, v in deals["Fiabilite_Estimation"].value_counts(dropna=False).items():
            print(f"  {k!s:10s} {v}")

    # Test 1 : les deals s'appuient-ils sur MOINS de comparables que le marché ?
    if "Nb_Comparables" in df.columns:
        pct_deals = (deals["Nb_Comparables"] < COMPARABLES_MIN_POUR_ALERTE).mean()
        pct_marche = (df["Nb_Comparables"] < COMPARABLES_MIN_POUR_ALERTE).mean()
        print(f"\nComparables médians — marché : {df['Nb_Comparables'].median():.0f} | "
              f"deals : {deals['Nb_Comparables'].median():.0f}")
        print(f"Part appuyée sur <{COMPARABLES_MIN_POUR_ALERTE} comparables — "
              f"marché : {pct_marche:.0%} | deals : {pct_deals:.0%}")
        if pct_deals > pct_marche + 0.10:
            print("  ⚠️  Les deals sont sur-concentrés sur des modèles peu comparables :")
            print("      une partie est probablement une erreur d'estimation, pas une affaire.")

    # Test 2 : marques sur-représentées (signal de biais du modèle)
    part_marche = df["Marque"].value_counts(normalize=True)
    part_deals = deals["Marque"].value_counts(normalize=True)
    comp = pd.DataFrame({"marche": part_marche, "deals": part_deals}).fillna(0)
    comp["nb_deals"] = deals["Marque"].value_counts()
    comp["sur_repr"] = (comp["deals"] / comp["marche"]).replace([np.inf], np.nan)
    comp = comp[comp["nb_deals"] >= 3].sort_values("sur_repr", ascending=False)
    if len(comp):
        print("\nMarques sur-représentées dans les deals (>1.5 = à regarder de près) :")
        for marque, r in comp.head(6).iterrows():
            flag = "  ⚠️" if r["sur_repr"] > 1.5 else "    "
            print(f"{flag} {marque:14s} x{r['sur_repr']:.1f}  ({int(r['nb_deals'])} deals)")


def analyse_longitudinale(suivi: pd.DataFrame) -> None:
    print("\n" + "=" * 64)
    print("2. VALIDATION LONGITUDINALE (la vraie preuve, nécessite de l'historique)")
    print("=" * 64)
    print("Question : à PRIX COMPARABLE, les opportunités s'écoulent-elles plus vite ?")
    print("La stratification par tranche de prix est indispensable : les voitures")
    print("chères se vendent naturellement plus lentement, comparer les durées brutes")
    print("mélangerait 'c'est une affaire' et 'c'est juste une Clio pas chère'.")

    if "Etait_Opportunite" not in suivi.columns:
        print("\nColonne Etait_Opportunite absente -- relance le pipeline pour la créer.")
        return

    est_deal = suivi["Etait_Opportunite"].astype(str).str.strip().str.lower().isin({"true", "1", "vrai"})
    jours = pd.to_numeric(suivi.get("Jours_En_Ligne"), errors="coerce")
    if "Statut" in suivi.columns:
        disparue = suivi["Statut"].astype(str).str.strip().str.lower().eq("disparue")
    else:
        disparue = jours.notna()

    # Durée de survie AVEC censure : une annonce encore en ligne n'est pas
    # ignorée, elle est censurée à (aujourd'hui - première vue). Ne garder que
    # les disparues (comme l'ancienne version) créait un biais de survie.
    pv = pd.to_datetime(suivi.get("Premiere_Vue"), errors="coerce")
    duree_censuree = (pd.Timestamp.now().normalize() - pv).dt.days
    duree = jours.where(disparue, duree_censuree)
    event = disparue.astype(int)

    # Stratification par tranche de prix
    prix = pd.to_numeric(suivi.get("Prix_Dernier"), errors="coerce")
    if prix.isna().all():
        prix = pd.to_numeric(suivi.get("Prix_Initial"), errors="coerce")
    bornes = [0, 25000, 50000, 100000, float("inf")]
    libelles = ["< 25k", "25–50k", "50–100k", "100k +"]
    strate = pd.cut(prix, bins=bornes, labels=libelles)

    valide = duree.notna() & (duree >= 0) & strate.notna()
    d = pd.DataFrame({
        "duree": duree, "event": event.astype(int),
        "deal": est_deal.astype(int), "strate": strate.astype(str),
    })[valide]

    n_ev_deal = int(d.loc[d["deal"] == 1, "event"].sum())
    n_ev_autre = int(d.loc[d["deal"] == 0, "event"].sum())
    print(f"\nDisparitions observées — opportunités : {n_ev_deal} | autres : {n_ev_autre}")

    if n_ev_deal < 10 or n_ev_autre < 10:
        jours_hist = pv.dropna().dt.date.nunique() if pv.notna().any() else 0
        print(f"\n⏳ Pas encore assez de disparitions ({jours_hist} jour(s) de collecte).")
        print("   Il faut ~10 disparitions par groupe, et la stratification par prix")
        print("   en exige davantage encore -- compte plusieurs semaines.")
        return

    # --- Méthode 1 (préférée) : modèle de Cox stratifié par tranche de prix ---
    # Contrôle le prix par régression, gère la censure. HR>1 = l'opportunité a un
    # risque de disparition plus élevé À PRIX ÉGAL = elle s'écoule plus vite.
    try:
        from lifelines import CoxPHFitter
        cph = CoxPHFitter()
        cph.fit(d[["duree", "event", "deal", "strate"]], duration_col="duree",
                event_col="event", strata=["strate"], formula="deal")
        hr = float(np.exp(cph.params_["deal"]))
        p = float(cph.summary.loc["deal", "p"])
        print("\nModèle de Cox stratifié par tranche de prix :")
        print(f"  Hazard ratio (opportunité) = {hr:.2f}  (>1 = s'écoule plus vite, à prix comparable)")
        print(f"  p = {p:.4f}")
        if p < 0.05 and hr > 1:
            print("  ✅ À prix comparable, les opportunités partent significativement plus vite")
            print("     -> le Score_Opportunite capte un vrai signal.")
        elif p < 0.05 and hr < 1:
            print("  ⚠️ Les opportunités restent PLUS longtemps en ligne -> le score va à l'envers.")
        else:
            print("  ❌ Aucune différence significative une fois le prix contrôlé.")
        return
    except ImportError:
        print("\n(lifelines absent -> repli sur un Mann-Whitney STRATIFIÉ par tranche de")
        print(" prix, sur les seules disparues ; correct pour le prix, mais ignore la censure.")
        print(" `pip install lifelines` active le modèle de Cox, plus rigoureux.)")
    except Exception as e:
        print(f"\n(Cox indisponible : {type(e).__name__}. Repli Mann-Whitney stratifié.)")

    # --- Méthode 2 (repli) : Mann-Whitney stratifié + combinaison de Fisher ---
    try:
        from scipy.stats import mannwhitneyu, combine_pvalues
        ps = []
        for lab in libelles:
            sub = d[(d["strate"] == lab) & (d["event"] == 1)]
            a = sub.loc[sub["deal"] == 1, "duree"]
            b = sub.loc[sub["deal"] == 0, "duree"]
            if len(a) >= 5 and len(b) >= 5:
                _, p = mannwhitneyu(a, b, alternative="less")
                print(f"  {lab:8s} : opp {a.median():.0f} j vs autres {b.median():.0f} j "
                      f"(p={p:.3f}, n={len(a)}/{len(b)})")
                ps.append(p)
        if ps:
            _, p_comb = combine_pvalues(ps, method="fisher")
            print(f"  p combiné (Fisher, {len(ps)} strates) = {p_comb:.4f}")
            print("  ✅ Signal réel à prix comparable." if p_comb < 0.05
                  else "  ❌ Pas de différence nette une fois le prix contrôlé.")
        else:
            print("  Pas assez d'annonces par strate pour un test fiable (≥5 par groupe).")
    except ImportError:
        print("\n(scipy absent -> comparaison descriptive par strate seulement)")
        for lab in libelles:
            sub = d[(d["strate"] == lab) & (d["event"] == 1)]
            a = sub.loc[sub["deal"] == 1, "duree"]
            b = sub.loc[sub["deal"] == 0, "duree"]
            if len(a) and len(b):
                print(f"  {lab:8s} : opp {a.median():.0f} j vs autres {b.median():.0f} j")


def main():
    df = pd.read_csv(PROCESSED_FILES["scored"], sep=";", encoding="utf-8-sig")
    analyse_transversale(df)

    if Path(FICHIER_SUIVI).exists():
        suivi = pd.read_csv(FICHIER_SUIVI, sep=";", encoding="utf-8-sig")
        analyse_longitudinale(suivi)
    else:
        print("\n(suivi_annonces.csv absent -- validation longitudinale impossible)")


if __name__ == "__main__":
    main()