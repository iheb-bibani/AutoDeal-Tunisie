"""Analyses agrégées : décote, prime pro, indices de dépréciation, capital, géo."""
import unicodedata
import numpy as np
import pandas as pd

MARQUES_LUXE = ["Porsche", "Land Rover", "Mercedes-Benz", "Jaguar", "Audi", "Bmw", "Tesla"]
GRAND_TUNIS = ["Tunis", "Ariana", "Ben Arous", "Manouba"]
K_LISSAGE = 15
MIN_ANNONCES_DECOTE = 15      # points nécessaires pour ajuster une pente crédible
MIN_AGES_DISTINCTS = 4        # sans plusieurs âges différents, aucune pente n'a de sens
MIN_PAR_COTE = 3              # annonces minimum de chaque type de vendeur
MIN_RECOUVREMENT_ANS = 2      # recouvrement d'âge minimum entre pros et particuliers
NOMS_MODELE_COURT = {
    "HistGradientBoostingRegressor": "HGB",
    "RandomForestRegressor": "RandomForest",
    "Ridge": "Ridge",
}
GOUVERNORATS_COORD = {
    "Tunis": (36.80, 10.18), "Ariana": (36.86, 10.19), "Ben Arous": (36.75, 10.23),
    "Manouba": (36.81, 9.84), "Nabeul": (36.45, 10.74), "Zaghouan": (36.40, 10.14),
    "Bizerte": (37.27, 9.87), "Béja": (36.73, 9.18), "Jendouba": (36.50, 8.78),
    "Le Kef": (36.17, 8.70), "Siliana": (36.08, 9.37), "Kairouan": (35.68, 10.10),
    "Kasserine": (35.17, 8.83), "Sidi Bouzid": (35.04, 9.48), "Sousse": (35.83, 10.63),
    "Monastir": (35.77, 10.83), "Mahdia": (35.50, 11.06), "Sfax": (34.74, 10.76),
    "Gafsa": (34.42, 8.78), "Tozeur": (33.92, 8.13), "Kébili": (33.70, 8.97),
    "Gabès": (33.88, 10.10), "Médenine": (33.35, 10.50), "Tataouine": (32.93, 10.45),
}
_GOUV_NORM = {"".join(c for c in unicodedata.normalize("NFD", g.lower())
                      if unicodedata.category(c) != "Mn"): g for g in GOUVERNORATS_COORD}
def nom_modele_court(nom):
    return NOMS_MODELE_COURT.get(str(nom), str(nom))
def calculer_prix_ajuste(df, colonne, k=K_LISSAGE):
    """Lissage bayésien : ramène la médiane d'une catégorie peu représentée
    vers la médiane nationale, pour ne pas se faire piéger par le hasard d'un
    petit échantillon."""
    mediane_nat = df["Prix"].median()
    stats = df.groupby(colonne)["Prix"].agg(["median", "count"])
    stats["prix_ajuste"] = (stats["count"] * stats["median"] + k * mediane_nat) / (stats["count"] + k)
    return stats.sort_values("prix_ajuste", ascending=False)
def analyse_rendement_capital(deals):
    """Par tranche de PRIX D'ACHAT : gain médian (DT), ROI médian (%), et
    'régularité' = ROI médian ÷ dispersion du ROI (un Sharpe-*like*).
    ⚠️ Ce n'est PAS un Sharpe financier : la dispersion mêle la variété réelle des
    affaires ET l'erreur du modèle (~10,7 %), et n'inclut pas le risque de revente
    (qui viendra du suivi des disparitions). À lire : 'consistance de l'écart estimé'."""
    d = deals.dropna(subset=["Prix", "Prix_Theorique"]).copy()
    d["gain"] = d["Prix_Theorique"] - d["Prix"]
    d["roi"] = d["gain"] / d["Prix"] * 100
    bornes = [0, 25000, 40000, 60000, 90000, float("inf")]
    libelles = ["< 25k", "25-40k", "40-60k", "60-90k", "90k +"]
    d["_tr"] = pd.cut(d["Prix"], bornes, labels=libelles)
    rows = []
    for tr, g in d.groupby("_tr", observed=True):
        if len(g) < 8:
            continue
        roi_med, roi_sd = g["roi"].median(), g["roi"].std()
        rows.append({
            "tranche": str(tr), "n": len(g), "gain": g["gain"].median(),
            "roi": roi_med, "regularite": roi_med / roi_sd if roi_sd else np.nan,
        })
    return pd.DataFrame(rows)
def analyse_decote_segment(df, seg_label):
    """Décote annuelle par phase (falaise 2→4 ans, plateau 4→7, longue traîne
    7→10) et âges les plus liquides, pour un segment. La falaise démarre à
    l'âge 2 (et non 1) car l'âge 1 est rare et dominé par des quasi-neuves
    haut de gamme : partir de là surestimerait la falaise (la 'prime du neuf'
    qui s'évapore n'est pas de la dépréciation). Basé sur des phases pour
    rester robuste au bruit. Renvoie None si données insuffisantes."""
    if "Age_Vehicule" not in df.columns or "Segment_Vehicule" not in df.columns:
        return None
    seg_val = 1 if "luxe" in seg_label.lower() else 0
    d = df.dropna(subset=["Age_Vehicule", "Prix"])
    d = d[d["Segment_Vehicule"] == seg_val]
    g = (d[(d["Age_Vehicule"] >= 0) & (d["Age_Vehicule"] <= 18)]
         .groupby("Age_Vehicule").agg(prix=("Prix", "median"), n=("Prix", "size")))
    g = g[g["n"] >= 8]
    if len(g) < 4:
        return None
    ages, prix = g.index.values.astype(float), g["prix"].values.astype(float)

    def px(a):
        return float(prix[int(np.argmin(np.abs(ages - a)))])

    def cagr(a, b):
        p0, p1 = px(a), px(b)
        return (1 - (p1 / p0) ** (1 / (b - a))) * 100 if p0 > 0 and b > a else float("nan")

    liq = g[g.index >= 3].sort_values("n", ascending=False)
    ages_liquides = sorted(int(x) for x in liq.head(3).index) if len(liq) else []
    return {
        "falaise": cagr(2, 4), "plateau": cagr(4, 7), "traine": cagr(7, 10),
        "ages_liquides": ages_liquides,
        "n_liquide": int(liq["n"].head(3).sum()) if len(liq) else 0,
    }
def calculer_niveau_regional(df, n_min=30):
    """Niveau de prix par région, à véhicule comparable.

    Comparer les prix médians régionaux revient à comparer des paniers de
    véhicules différents : la corrélation entre le prix médian d'une région
    et sa part de marques premium est de +0,70 sur les données réelles.

    On estime donc log(prix) ~ modèle + âge + kilométrage + région. Les
    indicatrices de modèle absorbent la composition ; le coefficient de
    région mesure ce qui reste, c'est-à-dire le niveau de prix local pour un
    véhicule identique.
    """
    d = df.dropna(subset=["Prix", "Age_Vehicule", "Kilométrage", "Modèle", "Localisation"])
    d = d[(d["Prix"] > 0) & d["Age_Vehicule"].between(0, 25)].copy()
    vc = d["Localisation"].value_counts()
    d = d[d["Localisation"].isin(vc[vc >= n_min].index)]
    vm = d["Modèle"].value_counts()
    d = d[d["Modèle"].isin(vm[vm >= 5].index)]
    if d["Localisation"].nunique() < 3 or len(d) < 200:
        return pd.DataFrame()

    X = pd.get_dummies(d[["Modèle", "Localisation"]], drop_first=True).astype(float)
    X["age"] = d["Age_Vehicule"].values
    X["km"] = d["Kilométrage"].values / 10000
    X.insert(0, "const", 1.0)
    try:
        coef, *_ = np.linalg.lstsq(X.values, np.log(d["Prix"].values), rcond=None)
    except np.linalg.LinAlgError:
        return pd.DataFrame()

    cols = list(X.columns)
    lignes = []
    for region in sorted(d["Localisation"].unique()):
        nom = f"Localisation_{region}"
        effet = coef[cols.index(nom)] if nom in cols else 0.0
        lignes.append({
            "Localisation": region,
            "prime_pct": (np.exp(effet) - 1) * 100,
            "n": int((d["Localisation"] == region).sum()),
        })
    return pd.DataFrame(lignes)
def calculer_decote_annuelle(df):
    """Décote annuelle moyenne par modèle.

    On ajuste log(prix) ~ âge : la pente donne directement un taux de décote
    constant en pourcentage par an (exp(pente) - 1), ce qui correspond à la
    façon dont une voiture se déprécie réellement -- un pourcentage du prix
    restant, pas un montant fixe en dinars.

    Une simple différence de prix médian entre deux âges serait beaucoup plus
    fragile : elle ne repose que sur deux points, alors que la pente utilise
    toutes les annonces du modèle.
    """
    d = df.dropna(subset=["Prix", "Age_Vehicule", "Marque", "Modèle"])
    d = d[d["Age_Vehicule"].between(0, 20) & (d["Prix"] > 0)]

    lignes = []
    for (marque, modele), g in d.groupby(["Marque", "Modèle"]):
        if len(g) < MIN_ANNONCES_DECOTE or g["Age_Vehicule"].nunique() < MIN_AGES_DISTINCTS:
            continue
        pente = np.polyfit(g["Age_Vehicule"], np.log(g["Prix"]), 1)[0]
        decote = (np.exp(pente) - 1) * 100
        if not np.isfinite(decote) or decote > 0:   # une décote positive = données incohérentes
            continue
        lignes.append({
            "libelle": f"{marque} {modele}",
            "Marque": marque, "Modèle": modele,
            "decote_pct_an": decote,
            "n": len(g),
            "prix_median": g["Prix"].median(),
            "age_median": g["Age_Vehicule"].median(),
        })
    return pd.DataFrame(lignes)
def calculer_prime_pro(df):
    """Écart de prix pro vs particulier pour un même modèle, à âge égal.

    Pour chaque modèle : log(prix) ~ âge + kilométrage + vendeur_pro. Le
    coefficient du vendeur donne l'écart une fois l'âge ET le kilométrage
    neutralisés.

    Le kilométrage est indispensable ici : à âge égal, pros et particuliers ne
    vendent pas des véhicules au même compteur. Sans ce contrôle, 5 modèles
    sur 35 affichaient un écart de signe opposé à la réalité -- un pro vendant
    des exemplaires plus roulés paraissait « vendre moins cher » alors qu'à
    kilométrage comparable il vend plus cher.

    Garde-fou décisif : les deux populations doivent se recouvrir en âge sur
    au moins MIN_RECOUVREMENT_ANS années, et la régression n'est faite que sur
    cette zone de recouvrement. Sans cela, "corriger de l'âge" reviendrait à
    extrapoler hors des données -- c'est exactement ce qui produisait des
    écarts absurdes (+169 %) dans la version précédente, où les pros vendaient
    des véhicules récents et les particuliers des véhicules deux fois plus
    vieux du même modèle.
    """
    d = df.dropna(subset=["Prix", "Age_Vehicule", "Kilométrage", "Marque", "Modèle", "Source"])
    d = d[d["Age_Vehicule"].between(0, 25) & (d["Prix"] > 0)].copy()
    d["est_pro"] = d["Source"].str.lower().str.contains("automobile", na=False).astype(int)

    lignes = []
    for (marque, modele), g in d.groupby(["Marque", "Modèle"]):
        pro, part = g[g["est_pro"] == 1], g[g["est_pro"] == 0]
        if len(pro) < MIN_PAR_COTE or len(part) < MIN_PAR_COTE:
            continue
        age_min = max(pro["Age_Vehicule"].min(), part["Age_Vehicule"].min())
        age_max = min(pro["Age_Vehicule"].max(), part["Age_Vehicule"].max())
        if age_max - age_min < MIN_RECOUVREMENT_ANS:
            continue
        zone = g[g["Age_Vehicule"].between(age_min, age_max)]
        n_pro = int((zone["est_pro"] == 1).sum())
        n_part = int((zone["est_pro"] == 0).sum())
        if n_pro < MIN_PAR_COTE or n_part < MIN_PAR_COTE:
            continue
        X = np.column_stack([
            np.ones(len(zone)),
            zone["Age_Vehicule"],
            zone["Kilométrage"] / 10000,
            zone["est_pro"],
        ])
        try:
            coef, *_ = np.linalg.lstsq(X, np.log(zone["Prix"]), rcond=None)
        except np.linalg.LinAlgError:
            continue
        prime = (np.exp(coef[3]) - 1) * 100
        if not np.isfinite(prime):
            continue
        lignes.append({
            "libelle": f"{marque} {modele}",
            "Marque": marque, "Modèle": modele,
            "prime_pct": prime,
            "n_pro": n_pro, "n_particulier": n_part,
            "age_min": age_min, "age_max": age_max,
            "km_median_pro": float(zone[zone["est_pro"] == 1]["Kilométrage"].median()),
            "km_median_particulier": float(zone[zone["est_pro"] == 0]["Kilométrage"].median()),
        })
    return pd.DataFrame(lignes)
def calculer_indice_depreciation(df, marques, age_max=20, n_min_par_age=5):
    """Profil de dépréciation par marque, corrigé de la composition.

    Une courbe de prix médian par âge est trompeuse : le panier de modèles
    change avec l'âge. Sur les données réelles, la courbe Peugeot *monte*
    entre 3 et 5 ans -- non parce qu'une Peugeot prend de la valeur, mais
    parce qu'à 3 ans l'échantillon est dominé par des 301 (berline
    économique) et à 5 ans par des 3008 (SUV). C'est un changement de
    composition, pas de la dépréciation.

    Ici : log(prix) ~ effets fixes MODÈLE + effets fixes ÂGE. Les indicatrices
    de modèle absorbent la composition ; le profil d'âge restant est celui
    d'un même modèle qui vieillit. Il est exprimé en indice base 100 à l'âge
    le plus bas observé, et l'âge reste catégoriel pour conserver la forme
    réelle de la courbe (chute forte les premières années) plutôt que de
    l'aplatir en droite.
    """
    d = df.dropna(subset=["Prix", "Age_Vehicule", "Marque", "Modèle"])
    d = d[d["Age_Vehicule"].between(0, age_max) & (d["Prix"] > 0)]

    sorties = []
    for marque in marques:
        sub = d[d["Marque"] == marque].copy()
        # Modèles trop rares : ils ne peuvent pas servir de référence stable
        vc = sub["Modèle"].value_counts()
        sub = sub[sub["Modèle"].isin(vc[vc >= 5].index)]
        # Âges trop peu représentés : une médiane sur 2 annonces n'est pas un point
        va = sub["Age_Vehicule"].value_counts()
        sub = sub[sub["Age_Vehicule"].isin(va[va >= n_min_par_age].index)]
        if len(sub) < 40 or sub["Modèle"].nunique() < 2 or sub["Age_Vehicule"].nunique() < 4:
            continue

        # L'âge est traité en catégoriel, mais la référence doit être l'âge le
        # PLUS JEUNE. Encoder l'âge en texte et laisser drop_first choisir
        # prendrait la première valeur par ordre alphabétique -- soit "10"
        # avant "2" -- et l'indice serait rapporté à une base absurde.
        ages = sorted(sub["Age_Vehicule"].astype(int).unique())
        age_ref = ages[0]
        for age in ages[1:]:
            sub[f"_age_{age}"] = (sub["Age_Vehicule"].astype(int) == age).astype(float)
        colonnes_age = [f"_age_{a}" for a in ages[1:]]

        X = pd.get_dummies(sub[["Modèle"]], drop_first=True).astype(float)
        for col in colonnes_age:
            X[col] = sub[col].values
        X.insert(0, "const", 1.0)
        try:
            coef, *_ = np.linalg.lstsq(X.values, np.log(sub["Prix"].values), rcond=None)
        except np.linalg.LinAlgError:
            continue

        cols = list(X.columns)
        for age in ages:
            nom = f"_age_{age}"
            effet = coef[cols.index(nom)] if nom in cols else 0.0  # 0 pour l'âge de référence
            sorties.append({
                "Marque": marque, "age": age,
                "indice": float(np.exp(effet) * 100),
                "n": int((sub["Age_Vehicule"] == age).sum()),
                "age_ref": age_ref,
            })
    return pd.DataFrame(sorties)
def calculer_indice_modele_representatif(df, marques, age_max=20, n_min_par_age=5):
    """Décote d'UN modèle représentatif par marque : le plus fréquent qui a une
    couverture d'âge suffisante. Vue la plus littérale (le même modèle qui
    vieillit, sans régression) — au prix de moins de données. Indexé base 100
    au plus jeune âge observé de ce modèle."""
    d = df.dropna(subset=["Prix", "Age_Vehicule", "Marque", "Modèle"])
    d = d[d["Age_Vehicule"].between(0, age_max) & (d["Prix"] > 0)]
    sorties = []
    for marque in marques:
        sub = d[d["Marque"] == marque]
        meilleur = None
        for modele, g in sub.groupby("Modèle"):
            va = g["Age_Vehicule"].astype(int).value_counts()
            ages_ok = sorted(int(a) for a in va[va >= n_min_par_age].index)
            if len(ages_ok) >= 4 and (meilleur is None or len(g) > meilleur[2]):
                meilleur = (modele, ages_ok, len(g))
        if not meilleur:
            continue
        modele, ages, _ = meilleur
        gm = sub[(sub["Modèle"] == modele) & (sub["Age_Vehicule"].astype(int).isin(ages))]
        med = gm.groupby(gm["Age_Vehicule"].astype(int))["Prix"].median()
        base = med.loc[min(ages)]
        for a in sorted(med.index):
            sorties.append({
                "Marque": f"{marque} — {modele}", "age": a,
                "indice": float(med.loc[a] / base * 100) if base else np.nan,
                "n": int((gm["Age_Vehicule"].astype(int) == a).sum()),
            })
    return pd.DataFrame(sorties)
def _match_gouvernorat(localisation):
    """Rattache une localisation libre (ex 'Sfax Ville') à un gouvernorat."""
    s = "".join(c for c in unicodedata.normalize("NFD", str(localisation).lower())
                if unicodedata.category(c) != "Mn")
    for norm, disp in _GOUV_NORM.items():
        if norm in s:
            return disp
    return None
