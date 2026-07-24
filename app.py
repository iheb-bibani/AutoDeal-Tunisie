"""
app.py
AutoDeal Tunisie -- application d'intelligence du marché de l'occasion.

Trois espaces, pensés pour deux métiers différents :

  1. 📊 Vue Marché (concessionnaire) : structure du parc, parts de marché,
     dépréciation, écart pros/particuliers, prix par région et par énergie.
  2. 🎯 Vue Samsar (achat-revente)   : opportunités chiffrées en dinars,
     rotation par modèle, arbitrage géographique, filtre par budget.
  3. 💰 Calculateur de Juste Prix    : estimation par le modèle ML +
     positionnement par rapport aux annonces comparables réelles.

Pas de comptes ni d'authentification -- usage personnel. Les alertes
Telegram restent gérées par utils/send_telegram.py.
"""

import json
import numpy as np
import joblib
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="AutoDeal Tunisie",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

SCORED_PATH = "data/processed/tunisia-cars-scored.csv"
DEALS_PATH = "data/processed/alertes_bonnes_affaires.csv"
MODELE_PATH = "data/models/modele_prix.pkl"

# ---------------------------------------------------------------------------
# Source des données
# ---------------------------------------------------------------------------
# Le rafraîchissement automatique de Streamlit Cloud après un push GitHub n'est
# pas fiable : l'app conserve fréquemment l'ancien contenu du dépôt jusqu'à un
# redémarrage manuel. Comme les données sont réécrites chaque nuit par GitHub
# Actions, s'appuyer dessus reviendrait à afficher des annonces périmées sans
# aucun signal visible.
#
# On lit donc les fichiers directement depuis GitHub (raw), avec un cache d'une
# heure : l'app se met à jour d'elle-même, redémarrage ou pas. En cas d'échec
# réseau, on retombe silencieusement sur les fichiers locaux -- ce qui est aussi
# le mode utilisé en développement.
DEPOT_GITHUB = "iheb-bibani/AutoDeal-Tunisie"
BRANCHE_GITHUB = "main"
BASE_RAW = f"https://raw.githubusercontent.com/{DEPOT_GITHUB}/{BRANCHE_GITHUB}/"

# Mettre à False pour forcer la lecture locale (développement hors ligne).
LIRE_DEPUIS_GITHUB = True
DUREE_CACHE = 3600  # secondes


def _url(chemin_relatif):
    return BASE_RAW + chemin_relatif


@st.cache_data(ttl=DUREE_CACHE, show_spinner=False)
def lire_csv(chemin):
    """Lit un CSV depuis GitHub si possible, sinon depuis le disque local."""
    if LIRE_DEPUIS_GITHUB:
        try:
            return pd.read_csv(_url(chemin), sep=";", encoding="utf-8-sig")
        except Exception:
            pass
    return pd.read_csv(chemin, sep=";", encoding="utf-8-sig")


@st.cache_data(ttl=DUREE_CACHE, show_spinner=False)
def lire_json_distant(chemin):
    if LIRE_DEPUIS_GITHUB:
        try:
            import urllib.request
            with urllib.request.urlopen(_url(chemin), timeout=15) as r:
                return json.loads(r.read().decode("utf-8"))
        except Exception:
            pass
    try:
        with open(chemin, encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None

MARQUES_LUXE = ["Porsche", "Land Rover", "Mercedes-Benz", "Jaguar", "Audi", "Bmw", "Tesla"]
GRAND_TUNIS = ["Tunis", "Ariana", "Ben Arous", "Manouba"]
K_LISSAGE = 15

# ---------------------------------------------------------------------------
# Identité visuelle : palette "asphalte & dinar"
# ---------------------------------------------------------------------------

C_ENCRE = "#15232E"      # texte, barres neutres
C_ASPHALTE = "#2C3E50"   # barres principales
C_GAIN = "#0E9F6E"       # vert "gain" -- opportunités, hausses
C_ALERTE = "#D9480F"     # orange brûlé -- points d'attention
C_SABLE = "#C9A227"      # accent secondaire (or/sable)
C_GRIS = "#8A97A3"

SEQ_CATEGORIELLE = [C_ASPHALTE, C_GAIN, C_SABLE, C_ALERTE, "#5C7A99", "#A3B2BF"]

st.markdown(
    """
    <style>
    /* Cartes KPI */
    div[data-testid="stMetric"] {
        background: linear-gradient(180deg, #FFFFFF 0%, #F7F9FA 100%);
        border: 1px solid #E4E9ED;
        border-left: 4px solid #0E9F6E;
        border-radius: 10px;
        padding: 14px 18px 10px 18px;
        box-shadow: 0 1px 3px rgba(21, 35, 46, .06);
    }
    div[data-testid="stMetric"] label { color: #5C6B78 !important; }

    /* Barre latérale sombre */
    section[data-testid="stSidebar"] {
        background: #15232E;
    }
    section[data-testid="stSidebar"] * { color: #E8EDF1 !important; }
    section[data-testid="stSidebar"] hr { border-color: #2C3E50; }

    /* Titres */
    h1, h2, h3 { color: #15232E; }
    h1 { letter-spacing: -0.5px; }

    /* Onglets */
    button[data-baseweb="tab"] { font-weight: 600; }

    /* Liens dans les tableaux */
    a { color: #0E9F6E; }
    </style>
    """,
    unsafe_allow_html=True,
)


def style_figure(fig, hauteur=380):
    """Applique le gabarit visuel commun à toutes les figures Plotly."""
    fig.update_layout(
        height=hauteur,
        margin=dict(l=10, r=10, t=48, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Source Sans Pro, sans-serif", color=C_ENCRE, size=13),
        title_font=dict(size=15, color=C_ENCRE),
        colorway=SEQ_CATEGORIELLE,
        hoverlabel=dict(bgcolor="#FFFFFF", font_color=C_ENCRE, bordercolor="#E4E9ED"),
    )
    fig.update_xaxes(gridcolor="#EDF1F4", zerolinecolor="#E4E9ED")
    fig.update_yaxes(gridcolor="#EDF1F4", zerolinecolor="#E4E9ED")
    return fig


# ---------------------------------------------------------------------------
# Chargement des données
# ---------------------------------------------------------------------------

@st.cache_data(ttl=DUREE_CACHE)
def charger_scored():
    try:
        df = lire_csv(SCORED_PATH)
    except Exception:
        return None
    # Colonnes lisibles si absentes (compatibilité avec d'anciens fichiers)
    if "Segment_Libelle" not in df.columns:
        df["Segment_Libelle"] = np.where(df["Marque"].isin(MARQUES_LUXE), "Luxe", "Standard")
    if "Zone_Libelle" not in df.columns:
        df["Zone_Libelle"] = np.where(df["Localisation"].isin(GRAND_TUNIS), "Grand Tunis", "Province")
    if "Age_Vehicule" not in df.columns and "Année" in df.columns:
        df["Age_Vehicule"] = (pd.Timestamp.now().year - df["Année"]).clip(lower=0)
    return df


@st.cache_data(ttl=DUREE_CACHE)
def charger_deals():
    try:
        df = lire_csv(DEALS_PATH)
        return df if not df.empty else None
    except Exception:
        return None


@st.cache_resource(ttl=DUREE_CACHE)
def charger_modele():
    """Le modèle est réentraîné chaque nuit : on le recharge depuis GitHub avec
    la même durée de cache que les données, pour ne pas scorer des annonces
    fraîches avec un modèle de la semaine dernière."""
    bundle = None
    if LIRE_DEPUIS_GITHUB:
        try:
            import io, urllib.request
            with urllib.request.urlopen(_url(MODELE_PATH), timeout=30) as r:
                bundle = joblib.load(io.BytesIO(r.read()))
        except Exception:
            bundle = None
    if bundle is None:
        try:
            bundle = joblib.load(MODELE_PATH)
        except Exception:
            return None
    if not isinstance(bundle, dict) or "pipeline" not in bundle:
        return None
    return bundle


def calculer_prix_ajuste(df, colonne, k=K_LISSAGE):
    """Lissage bayésien : ramène la médiane d'une catégorie peu représentée
    vers la médiane nationale, pour ne pas se faire piéger par le hasard d'un
    petit échantillon."""
    mediane_nat = df["Prix"].median()
    stats = df.groupby(colonne)["Prix"].agg(["median", "count"])
    stats["prix_ajuste"] = (stats["count"] * stats["median"] + k * mediane_nat) / (stats["count"] + k)
    return stats.sort_values("prix_ajuste", ascending=False)


def fmt_dt(v):
    return f"{v:,.0f}".replace(",", " ") + " DT"


# ---------------------------------------------------------------------------
# Analyses agrégées : décote annuelle et prime professionnelle
# ---------------------------------------------------------------------------

MIN_ANNONCES_DECOTE = 15      # points nécessaires pour ajuster une pente crédible
MIN_AGES_DISTINCTS = 4        # sans plusieurs âges différents, aucune pente n'a de sens
MIN_PAR_COTE = 3              # annonces minimum de chaque type de vendeur
MIN_RECOUVREMENT_ANS = 2      # recouvrement d'âge minimum entre pros et particuliers


@st.cache_data
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


@st.cache_data
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


# ===========================================================================
# 1. VUE MARCHÉ (concessionnaire)
# ===========================================================================

@st.cache_data
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


def page_marche(df):
    st.title("📊 Vue Marché")
    st.caption("Structure et niveaux de prix du marché de l'occasion — pensé concessionnaire : "
               "que vaut le parc, où se vend quoi, à quel prix.")

    if df is None:
        st.error(f"`{SCORED_PATH}` introuvable — lance `python main.py` d'abord.")
        return

    # ---- KPIs ------------------------------------------------------------
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Annonces actives (≤ 30 j)", f"{len(df):,}".replace(",", " "))
    c2.metric("Prix médian du marché", fmt_dt(df["Prix"].median()))
    age_median = df["Age_Vehicule"].median()
    c3.metric("Âge médian du parc", f"{age_median:.0f} ans" if pd.notna(age_median) else "—")
    km_median = df["Kilométrage"].median()
    c4.metric("Kilométrage médian", f"{km_median:,.0f} km".replace(",", " ") if pd.notna(km_median) else "—")

    st.divider()

    # ---- Parts de marché et niveaux de prix ------------------------------
    col_a, col_b = st.columns(2)
    with col_a:
        parts = df["Marque"].value_counts().head(15).sort_values()
        fig = px.bar(
            x=parts.values, y=parts.index, orientation="h",
            title="Parts de marché — volume d'annonces par marque (top 15)",
            labels={"x": "Annonces", "y": ""},
        )
        fig.update_traces(marker_color=C_ASPHALTE)
        st.plotly_chart(style_figure(fig, 430), width="stretch")

    with col_b:
        top_m = df.groupby("Marque").agg(prix=("Prix", "median"), n=("Prix", "count"))
        top_m = top_m[top_m["n"] >= 10].sort_values("prix").tail(15)
        fig = px.bar(
            x=top_m["prix"], y=top_m.index, orientation="h",
            title="Prix médian par marque (≥ 10 annonces, top 15)",
            labels={"x": "Prix médian (DT)", "y": ""},
        )
        fig.update_traces(
            marker_color=[C_SABLE if m in MARQUES_LUXE else C_ASPHALTE for m in top_m.index],
            customdata=top_m["n"], hovertemplate="%{y} : %{x:,.0f} DT (n=%{customdata})<extra></extra>",
        )
        st.plotly_chart(style_figure(fig, 430), width="stretch")

    # ---- Dépréciation ----------------------------------------------------
    st.subheader("Courbe de dépréciation")
    st.caption("Perte de valeur selon l'âge, **à modèle constant** — la composition du parc "
               "change avec l'âge, et la corriger est indispensable pour lire une vraie "
               "dépréciation.")

    marques_dispo = df["Marque"].value_counts()
    marques_courbe = st.multiselect(
        "Marques à comparer (≥ 15 annonces)",
        options=list(marques_dispo[marques_dispo >= 15].index),
        default=[m for m in ["Volkswagen", "Peugeot", "Kia", "Mercedes-Benz"] if marques_dispo.get(m, 0) >= 15][:4],
    )

    indice = calculer_indice_depreciation(df, marques_courbe) if marques_courbe else pd.DataFrame()
    if len(indice):
        fig = px.line(
            indice, x="age", y="indice", color="Marque", markers=True,
            title="Indice de valeur selon l'âge (base 100 au plus jeune âge observé)",
            labels={"age": "Âge (années)", "indice": "Indice de valeur"},
            custom_data=["n"],
        )
        fig.update_traces(hovertemplate="%{y:.0f} (base 100)<br>%{x} ans — "
                                        "%{customdata[0]} annonces<extra></extra>")
        st.plotly_chart(style_figure(fig), width="stretch")

        with st.expander("ℹ️ Pourquoi un indice, et pas le prix médian par âge"):
            st.write(
                """
Tracer le prix médian par âge donne une courbe fausse, parce que le panier de
modèles change avec l'âge. Sur les données réelles, la courbe Peugeot
**montait** entre 3 et 5 ans :

| Âge | Prix médian | Modèles dominants |
|---|---|---|
| 3 ans | 34 700 DT | **301** ×14, 208 ×6 |
| 5 ans | 62 500 DT | **3008** ×5, 208 ×5 |

Une Peugeot ne prend pas de la valeur en vieillissant : à 3 ans l'échantillon
est dominé par des 301 (berline économique), à 5 ans par des 3008 (SUV). C'est
un changement de composition, pas de la dépréciation.

L'indice est calculé par `log(prix) ~ effets fixes modèle + effets fixes âge`.
Les indicatrices de modèle absorbent la composition ; le profil d'âge restant
est celui d'un même modèle qui vieillit. L'âge reste catégoriel pour conserver
la forme réelle de la courbe — la chute des premières années — au lieu de
l'aplatir en droite. Les âges représentés par moins de 5 annonces sont exclus :
une médiane sur 2 annonces n'est pas un point de courbe.

**Limite.** La mesure reste transversale : on compare des véhicules d'âges
différents à un instant donné, on ne suit pas un véhicule dans le temps. Si les
millésimes récents sont mieux équipés que les anciens, une part de l'écart vient
de l'équipement et non de l'âge.
"""
            )
    elif marques_courbe:
        st.info("Pas assez d'annonces pour ces marques (il faut au moins 40 annonces, "
                "2 modèles distincts et 4 âges différents avec 5 annonces chacun).")

    # ---- Décote annuelle par modèle --------------------------------------
    st.subheader("Décote annuelle par modèle")
    st.caption("Combien un modèle perd, en pourcentage de sa valeur, chaque année. "
               "C'est le chiffre qui sert à fixer une reprise ou à choisir un véhicule "
               "qui tiendra sa valeur.")

    decote = calculer_decote_annuelle(df)
    if len(decote):
        col_g, col_h = st.columns(2)
        with col_g:
            top_d = decote.sort_values("decote_pct_an").head(15).sort_values("decote_pct_an", ascending=False)
            fig = px.bar(
                top_d, x="decote_pct_an", y="libelle", orientation="h",
                title="Perdent le plus vite leur valeur",
                labels={"decote_pct_an": "Décote (% par an)", "libelle": ""},
                custom_data=["n", "prix_median"],
            )
            fig.update_traces(
                marker_color=C_ALERTE,
                hovertemplate="%{y} : %{x:.1f} %/an<br>%{customdata[0]} annonces — "
                              "prix médian %{customdata[1]:,.0f} DT<extra></extra>",
            )
            st.plotly_chart(style_figure(fig, 460), width="stretch")
        with col_h:
            garde = decote.sort_values("decote_pct_an", ascending=True).head(15)
            fig = px.bar(
                garde, x="decote_pct_an", y="libelle", orientation="h",
                title="Tiennent le mieux leur valeur",
                labels={"decote_pct_an": "Décote (% par an)", "libelle": ""},
                custom_data=["n", "prix_median"],
            )
            fig.update_traces(
                marker_color=C_GAIN,
                hovertemplate="%{y} : %{x:.1f} %/an<br>%{customdata[0]} annonces — "
                              "prix médian %{customdata[1]:,.0f} DT<extra></extra>",
            )
            st.plotly_chart(style_figure(fig, 460), width="stretch")

        st.download_button(
            "⬇️ Télécharger le tableau des décotes (CSV)",
            decote.sort_values("decote_pct_an")[
                ["Marque", "Modèle", "decote_pct_an", "n", "prix_median", "age_median"]
            ].round(2).to_csv(index=False, sep=";").encode("utf-8-sig"),
            file_name="decote_annuelle_par_modele.csv",
            mime="text/csv",
        )
        st.caption(f"{len(decote)} modèles ont assez d'annonces (≥ {MIN_ANNONCES_DECOTE}, "
                   f"réparties sur ≥ {MIN_AGES_DISTINCTS} âges différents) pour une décote fiable.")
    else:
        st.info("Pas encore assez d'annonces par modèle pour estimer une décote annuelle.")
    st.subheader("Prime professionnelle, à âge égal")
    st.caption("automobile.tn est surtout alimenté par les professionnels, tayara.tn par les "
               "particuliers. La question utile : pour un même modèle **et un même âge**, "
               "de combien un pro affiche-t-il au-dessus d'un particulier ?")

    prime = calculer_prime_pro(df)
    if len(prime):
        top = prime.reindex(prime["prime_pct"].abs().sort_values(ascending=False).index).head(15)
        top = top.sort_values("prime_pct")
        fig = px.bar(
            top, x="prime_pct", y="libelle", orientation="h",
            title="Écart de prix pro vs particulier, à âge comparable",
            labels={"prime_pct": "Prime professionnelle (%)", "libelle": ""},
            custom_data=["n_pro", "n_particulier", "age_min", "age_max",
                         "km_median_pro", "km_median_particulier"],
        )
        fig.update_traces(
            marker_color=[C_GAIN if v > 0 else C_ALERTE for v in top["prime_pct"]],
            hovertemplate="%{y} : %{x:+.0f} %<br>%{customdata[0]} annonces pro / "
                          "%{customdata[1]} particulier<br>Âges comparés : "
                          "%{customdata[2]:.0f}–%{customdata[3]:.0f} ans"
                          "<br>Km médian : %{customdata[4]:,.0f} (pro) / "
                          "%{customdata[5]:,.0f} (particulier)<extra></extra>",
        )
        fig.add_vline(x=0, line_color=C_GRIS, line_width=1)
        st.plotly_chart(style_figure(fig, 430), width="stretch")

        mediane = prime["prime_pct"].median()
        st.markdown(f"**Prime pro médiane à âge égal : {mediane:+.0f} %** "
                    f"sur {len(prime)} modèles comparables.")

        with st.expander("ℹ️ Pourquoi « à âge égal » change tout"):
            st.write(
                """
Comparer directement le prix médian des pros à celui des particuliers donne un
écart médian de **+15 %**, avec des valeurs absurdes (+169 % sur une Škoda
Octavia, +129 % sur une Mercedes Classe C). Ce n'est pas une marge : c'est un
**biais de composition**. Les pros vendent des Classe C de 3 ans, les
particuliers des Classe C de 15 ans. On compare des voitures différentes et on
appelle ça un écart de prix.

Ici, pour chaque modèle, le prix est expliqué par l'âge **et** par le type de
vendeur (`log(prix) ~ âge + vendeur_pro`). L'écart affiché est ce qui reste une
fois l'âge neutralisé. Deux garde-fous : au moins 3 annonces de chaque côté, et
surtout un **recouvrement d'au moins 2 ans** entre les âges des deux
populations — sans recouvrement, corriger de l'âge reviendrait à extrapoler
hors des données observées.

Résultat : la prime médiane tombe à **+2 %**. L'arbitrage « acheter au
particulier, revendre au pro » est donc bien plus étroit que ne le suggérait le
graphique brut, et ne tient réellement que sur les quelques modèles en tête de
liste.
"""
            )
    else:
        st.info("Pas encore assez de modèles où pros et particuliers vendent des véhicules "
                "d'âges comparables (minimum 3 annonces de chaque côté et 2 ans de recouvrement).")

    # ---- Structure par gamme de prix ------------------------------------
    col_c, col_d = st.columns(2)
    with col_c:
        tranches = pd.cut(
            df["Prix"],
            bins=[0, 20000, 35000, 50000, 80000, 120000, 200000, np.inf],
            labels=["< 20k", "20–35k", "35–50k", "50–80k", "80–120k", "120–200k", "> 200k"],
        )
        repartition = tranches.value_counts().sort_index()
        fig = px.bar(
            x=repartition.index.astype(str), y=repartition.values,
            title="Structure du marché par gamme de prix (DT)",
            labels={"x": "Gamme de prix", "y": "Annonces"},
        )
        fig.update_traces(marker_color=C_ASPHALTE)
        st.plotly_chart(style_figure(fig), width="stretch")

    with col_d:
        if "Energie" in df.columns:
            stats_e = calculer_prix_ajuste(df, "Energie")
            fig = px.bar(
                x=stats_e.index, y=stats_e["prix_ajuste"],
                title="Prix médian par énergie (ajusté petits échantillons)",
                labels={"x": "", "y": "Prix ajusté (DT)"},
            )
            fig.update_traces(
                marker_color=C_ASPHALTE, customdata=stats_e["count"],
                hovertemplate="%{x} : %{y:,.0f} DT (n=%{customdata})<extra></extra>",
            )
            st.plotly_chart(style_figure(fig), width="stretch")

    # ---- Prix par région -------------------------------------------------
    if "Localisation" in df.columns:
        stats_r = calculer_prix_ajuste(df, "Localisation").head(15)
        fig = px.bar(
            x=stats_r["prix_ajuste"], y=stats_r.index, orientation="h",
            title="Prix médian par région, ajusté (top 15)",
            labels={"x": "Prix ajusté (DT)", "y": ""},
        )
        fig.update_traces(
            marker_color=C_ASPHALTE, customdata=stats_r["count"],
            hovertemplate="%{y} : %{x:,.0f} DT (n=%{customdata})<extra></extra>",
        )
        st.plotly_chart(style_figure(fig, 430), width="stretch")

        with st.expander("ℹ️ Pourquoi « ajusté » ?"):
            st.write(
                f"""
Une région avec 4-6 annonces peut avoir une médiane faussée par une seule
voiture atypique. Le **lissage bayésien** corrige ça : plus l'échantillon
d'une région est petit, plus son prix est ramené vers la médiane nationale
({df['Prix'].median():,.0f} DT).

`prix_ajusté = (n × médiane_région + {K_LISSAGE} × médiane_nationale) / (n + {K_LISSAGE})`

Une région à n=6 est tirée à ~71 % vers la médiane nationale ; une région à
n=200 garde quasiment sa propre médiane.
"""
            )

    # ---- Tendance temporelle --------------------------------------------
    if "Annonce-Detectee" in df.columns and df["Annonce-Detectee"].nunique() >= 2:
        tendance = df.groupby("Annonce-Detectee").agg(prix=("Prix", "median"), n=("Prix", "count")).reset_index()
        fig = px.line(
            tendance, x="Annonce-Detectee", y="prix", markers=True,
            title="Tendance du prix médian par jour de collecte",
            labels={"Annonce-Detectee": "", "prix": "Prix médian (DT)"},
        )
        fig.update_traces(line_color=C_GAIN)
        st.plotly_chart(style_figure(fig, 320), width="stretch")


# ===========================================================================
# 2. VUE SAMSAR (achat-revente)
# ===========================================================================

def page_samsar(df_scored, df_deals):
    st.title("🎯 Vue Samsar")
    st.caption("Achat-revente : les affaires chiffrées en dinars, ce qui tourne vite, "
               "et où acheter moins cher.")

    if df_scored is None:
        st.error(f"`{SCORED_PATH}` introuvable — lance `python main.py` d'abord.")
        return

    # ---- Opportunités ----------------------------------------------------
    if df_deals is None:
        st.info("Aucune opportunité détectée sur le dernier scraping — les KPIs et la matrice "
                "ci-dessous apparaîtront dès qu'il y en aura.")
    else:
        deals = df_deals.copy()
        deals["Gain_DT"] = (deals["Prix_Theorique"] - deals["Prix"]).clip(lower=0)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Opportunités détectées", len(deals))
        if "Nb_Comparables" in deals.columns:
            solides = int((deals["Nb_Comparables"] >= 8).sum())
            c2.metric("Dont estimations solides", solides,
                      help="Appuyées sur au moins 8 annonces comparables. Les autres reposent "
                           "sur trop peu de véhicules similaires pour être fiables.")
        else:
            c2.metric("Gain potentiel médian", fmt_dt(deals["Gain_DT"].median()))
        c3.metric("Gain médian", fmt_dt(deals["Gain_DT"].median()))
        c4.metric("Décote médiane vs argus", f"{deals['Score_Opportunite'].median():.0%}")

        st.divider()

        # ---- Filtres -----------------------------------------------------
        colf1, colf2, colf3 = st.columns([2, 1, 1])
        with colf1:
            budget = st.slider(
                "Budget d'achat maximum (DT)",
                min_value=5000, max_value=int(deals["Prix"].max()) + 5000,
                value=min(80000, int(deals["Prix"].max()) + 5000), step=5000,
            )
        with colf2:
            marques_f = st.multiselect("Marques", sorted(deals["Marque"].dropna().unique()))
        with colf3:
            zones_f = st.multiselect(
                "Zone", sorted(deals["Zone_Libelle"].dropna().unique())
                if "Zone_Libelle" in deals.columns else [],
            )

        fiables_seulement = st.checkbox(
            "Estimations solides uniquement (≥ 8 annonces comparables)", value=True,
            help="Le prix théorique d'un modèle presque absent du marché est peu fiable : "
                 "une décote apparente y est le plus souvent une erreur d'estimation, pas une affaire.",
        )

        sel = deals[deals["Prix"] <= budget]
        if fiables_seulement and "Nb_Comparables" in sel.columns:
            sel = sel[sel["Nb_Comparables"] >= 8]
        if marques_f:
            sel = sel[sel["Marque"].isin(marques_f)]
        if zones_f and "Zone_Libelle" in sel.columns:
            sel = sel[sel["Zone_Libelle"].isin(zones_f)]

        st.markdown(f"**{len(sel)} opportunités** dans ton budget — gain potentiel cumulé "
                    f"si tout était acheté/revendu à l'argus : **{fmt_dt(sel['Gain_DT'].sum())}**")

        # ---- Matrice gain × liquidité ------------------------------------
        if len(sel):
            fig = px.scatter(
                sel, x="Score_Opportunite", y="Score_Liquidite", size="Gain_DT",
                color="Score_Liquidite", color_continuous_scale=["#C9CFD6", C_GAIN],
                hover_data={"Titre": True, "Marque": True, "Modèle": True,
                            "Prix": ":,.0f", "Gain_DT": ":,.0f",
                            "Score_Opportunite": ":.0%", "Score_Liquidite": ":.2f"},
                title="Matrice des affaires — en haut à droite : gros gain ET revente facile",
                labels={"Score_Opportunite": "Décote vs prix théorique",
                        "Score_Liquidite": "Liquidité (facilité de revente)"},
            )
            fig.update_layout(coloraxis_showscale=False)
            fig.update_xaxes(tickformat=".0%")
            st.plotly_chart(style_figure(fig, 430), width="stretch")

            # ---- Tableau -------------------------------------------------
            colonnes = ["Titre", "Marque", "Modèle", "Année", "Kilométrage", "Prix",
                        "Prix_Theorique", "Gain_DT", "Score_Opportunite",
                        "Fiabilite_Estimation", "Nb_Comparables", "Localisation", "Lien"]
            colonnes = [c for c in colonnes if c in sel.columns]
            st.dataframe(
                sel[colonnes].sort_values("Gain_DT", ascending=False),
                width="stretch", hide_index=True,
                column_config={
                    "Prix": st.column_config.NumberColumn("Prix affiché", format="%d DT"),
                    "Prix_Theorique": st.column_config.NumberColumn("Prix théorique", format="%d DT"),
                    "Gain_DT": st.column_config.NumberColumn("Gain potentiel", format="%d DT"),
                    "Score_Opportunite": st.column_config.NumberColumn("Décote", format="percent"),
                    "Kilométrage": st.column_config.NumberColumn("Km", format="%d"),
                    "Année": st.column_config.NumberColumn("Année", format="%d"),
                    "Fiabilite_Estimation": st.column_config.TextColumn("Fiabilité"),
                    "Nb_Comparables": st.column_config.NumberColumn("Comparables", format="%d"),
                    "Lien": st.column_config.LinkColumn("Annonce", display_text="Ouvrir ↗"),
                },
            )

    st.divider()

    # ---- Rotation : qu'est-ce qui tourne vite ? --------------------------
    st.subheader("Rotation du marché — qu'est-ce qui part vite ?")
    st.caption("Volume d'annonces = demande et facilité de revente. Âge médian des annonces "
               "encore en ligne = proxy de vitesse d'écoulement (plus c'est bas, plus ça tourne).")

    rotation = (
        df_scored.dropna(subset=["Marque", "Modèle"])
        .groupby(["Marque", "Modèle"])
        .agg(volume=("Prix", "count"),
             age_annonce_median=("Age_Annonce_Jours", "median"),
             prix_median=("Prix", "median"))
        .reset_index()
    )
    rotation = rotation[rotation["volume"] >= 8]
    rotation["libelle"] = rotation["Marque"] + " " + rotation["Modèle"].astype(str)

    if len(rotation):
        col_r1, col_r2 = st.columns(2)
        with col_r1:
            top_vol = rotation.sort_values("volume").tail(12)
            fig = px.bar(
                top_vol, x="volume", y="libelle", orientation="h",
                title="Modèles les plus présents sur le marché",
                labels={"volume": "Annonces actives", "libelle": ""},
                custom_data=["prix_median"],
            )
            fig.update_traces(
                marker_color=C_ASPHALTE,
                hovertemplate="%{y} : %{x} annonces<br>Prix médian : %{customdata[0]:,.0f} DT<extra></extra>",
            )
            st.plotly_chart(style_figure(fig, 420), width="stretch")
        with col_r2:
            rapides = rotation.sort_values("age_annonce_median").head(12).sort_values(
                "age_annonce_median", ascending=False)
            fig = px.bar(
                rapides, x="age_annonce_median", y="libelle", orientation="h",
                title="Modèles dont les annonces sont les plus fraîches (écoulement rapide)",
                labels={"age_annonce_median": "Âge médian des annonces (jours)", "libelle": ""},
                custom_data=["volume", "prix_median"],
            )
            fig.update_traces(
                marker_color=C_GAIN,
                hovertemplate="%{y} : %{x:.0f} j (n=%{customdata[0]})"
                              "<br>Prix médian : %{customdata[1]:,.0f} DT<extra></extra>",
            )
            st.plotly_chart(style_figure(fig, 420), width="stretch")

    # ---- Arbitrage géographique -----------------------------------------
    st.subheader("Arbitrage géographique — où acheter, où revendre")
    st.caption("Pour un modèle donné : dans quelle région son prix est-il le plus bas "
               "et le plus haut, **à âge et kilométrage comparables** ?")

    @st.cache_data
    def calculer_arbitrage_geo(df_):
        """Régression log(prix) ~ âge + km + région par modèle.
        Le coefficient région donne l'écart une fois l'âge et le km neutralisés.
        Garde-fous : ≥ 3 annonces par région, ≥ 12 annonces au total,
        recouvrement d'âge ≥ 2 ans entre toutes les paires de régions."""
        rows = []
        base = df_.dropna(subset=["Prix", "Age_Vehicule", "Kilométrage", "Localisation"])
        for (m, mo), g in base.groupby(["Marque", "Modèle"]):
            regs_ok = g.groupby("Localisation").filter(lambda x: len(x) >= 3)["Localisation"].value_counts()
            regs_ok = regs_ok[regs_ok >= 3].index.tolist()
            g2 = g[g["Localisation"].isin(regs_ok)]
            if g2["Localisation"].nunique() < 2 or len(g2) < 12:
                continue
            ar = g2.groupby("Localisation")["Age_Vehicule"].agg(["min", "max"])
            ok = all(
                min(r1["max"], r2["max"]) - max(r1["min"], r2["min"]) >= 2
                for i, r1 in ar.iterrows() for j, r2 in ar.iterrows() if i < j
            )
            if not ok:
                continue
            X = pd.get_dummies(g2[["Localisation"]], drop_first=True).astype(float)
            X["age"] = g2["Age_Vehicule"]
            X["km"] = g2["Kilométrage"] / 10000
            X.insert(0, "const", 1.0)
            try:
                c, *_ = np.linalg.lstsq(X.values, np.log(g2["Prix"].values), rcond=None)
            except np.linalg.LinAlgError:
                continue
            coefs = {col.replace("Localisation_", ""): (np.exp(c[list(X.columns).index(col)]) - 1) * 100
                     for col in X.columns if col.startswith("Localisation_")}
            ref = [r for r in regs_ok if r not in coefs]
            if ref:
                coefs[ref[0]] = 0.0
            if len(coefs) < 2:
                continue
            best = max(coefs, key=coefs.get)
            worst = min(coefs, key=coefs.get)
            ecart = coefs[best] - coefs[worst]
            prix_base = float(np.exp(
                c[0]
                + c[list(X.columns).index("age")] * g2["Age_Vehicule"].median()
                + c[list(X.columns).index("km")] * g2["Kilométrage"].median() / 10000
            ))
            rows.append({
                "libelle": f"{m} {mo}", "Marque": m, "Modèle": str(mo),
                "n": len(g2), "n_regs": g2["Localisation"].nunique(),
                "acheter": worst, "revendre": best,
                "ecart_pct": round(ecart, 1),
                "ecart_dt": int(prix_base * ecart / 100),
                "coefs": coefs,
            })
        return pd.DataFrame(rows).sort_values("ecart_pct", ascending=False) if rows else pd.DataFrame()

    arb = calculer_arbitrage_geo(df_scored)
    if len(arb):
        choix = arb["libelle"].tolist()
        modele_choisi = st.selectbox("Modèle à analyser", choix)
        ligne = arb[arb["libelle"] == modele_choisi].iloc[0]
        coefs = ligne["coefs"]
        coefs_df = pd.DataFrame({"region": list(coefs.keys()), "prime_pct": list(coefs.values())})
        coefs_df = coefs_df.sort_values("prime_pct")

        fig = px.bar(
            coefs_df, x="prime_pct", y="region", orientation="h",
            title=f"{modele_choisi} — prime de prix par région, à âge et km comparables",
            labels={"prime_pct": "Prime vs référence (%)", "region": ""},
        )
        couleurs = [C_GAIN if i == 0 else (C_ALERTE if i == len(coefs_df) - 1 else C_ASPHALTE)
                    for i in range(len(coefs_df))]
        fig.update_traces(
            marker_color=couleurs,
            hovertemplate="%{y} : %{x:+.1f} %<extra></extra>",
        )
        fig.add_vline(x=0, line_color=C_GRIS, line_width=1)
        st.plotly_chart(style_figure(fig, 340), width="stretch")
        st.markdown(
            f"🟢 **Acheter :** {ligne['acheter']} — "
            f"🟠 **Revendre :** {ligne['revendre']} — "
            f"écart à véhicule comparable : **{ligne['ecart_pct']:.0f} %** "
            f"(environ **{ligne['ecart_dt']:,} DT**) — ".replace(",", " ")
            + f"basé sur **{ligne['n']} annonces** dans **{ligne['n_regs']} régions**"
        )

        with st.expander("ℹ️ Pourquoi « à âge et km comparables » change tout"):
            st.write(
                """
Le graphique précédent comparait les prix médians bruts par région. Conséquence :
sur l'Audi A3 Sportback, Sfax affichait 120 000 DT et Nabeul 99 250 DT —
**parce que les voitures de Sfax avaient 2 ans en moyenne et celles de Nabeul 4 ans**.
L'écart venait de l'âge, pas de la région. 48 % des modèles avaient ce biais.

Ici, la même régression que pour la prime pro/particulier :
`log(prix) ~ âge + kilométrage + région`. Le coefficient région indique combien
le même véhicule — à âge et km identiques — serait affiché différemment selon
l'endroit. La barre à zéro est la région de référence ; toutes les autres sont
lues comme des écarts par rapport à elle.

Garde-fous appliqués : ≥ 3 annonces par région, ≥ 12 au total, et les plages
d'âge de chaque région doivent se recouvrir d'au moins 2 ans entre toutes les
paires — sans recouvrement, « corriger de l'âge » revient à extrapoler hors des
données. **{} modèles** retenus sur les données actuelles.
""".format(len(arb))
            )
    else:
        st.info("Pas encore assez de modèles avec au moins 3 annonces dans plusieurs régions "
                "et un recouvrement d'âge suffisant.")


# ===========================================================================
# 3. CALCULATEUR DE JUSTE PRIX
# ===========================================================================

def page_calculateur(df, bundle):
    st.title("💰 Calculateur de Juste Prix")
    st.caption("Estimation par le modèle entraîné sur le marché récent (≤ 30 jours), "
               "confrontée aux annonces comparables réelles.")

    if df is None or bundle is None:
        st.warning("Données ou modèle introuvables — relance `python main.py`.")
        return

    col1, col2, col3 = st.columns(3)
    with col1:
        marques = sorted(df["Marque"].dropna().unique())
        marque = st.selectbox("Marque", marques, index=marques.index("Volkswagen") if "Volkswagen" in marques else 0)
        modeles_m = sorted(df[df["Marque"] == marque]["Modèle"].dropna().astype(str).unique())
        modele = st.selectbox("Modèle", modeles_m) if modeles_m else st.text_input("Modèle")
        annee = st.number_input("Année", min_value=1990, max_value=pd.Timestamp.now().year + 1, value=2020)
    with col2:
        km = st.number_input("Kilométrage", min_value=0, max_value=500000, value=80000, step=5000)
        cv = st.number_input("Puissance fiscale (CV)", min_value=2, max_value=30, value=7)
        zone = st.selectbox("Zone", ["Grand Tunis", "Province"])
    with col3:
        boites = sorted(df["Boite_Vitesse"].dropna().unique())
        boite = st.selectbox("Boîte", boites)
        energies = sorted(df["Energie"].dropna().unique())
        energie = st.selectbox("Énergie", energies)

    segment = "Luxe" if marque in MARQUES_LUXE else "Standard"
    annee_courante = pd.Timestamp.now().year
    presque_neuve = annee >= annee_courante - 1
    st.caption(f"Segment déduit : **{segment}** — Presque neuve : **{'Oui' if presque_neuve else 'Non'}**")

    if st.button("Calculer le prix théorique", type="primary"):
        colonnes = bundle.get("features_numeriques", []) + bundle.get("features_categorielles", [])
        saisie = {
            "Kilométrage": km, "Année": annee, "Age_Vehicule": max(annee_courante - annee, 0),
            "Puissance_Fiscale": cv,
            "Segment_Vehicule": int(segment == "Luxe"),
            "Est_Presque_Neuve": int(presque_neuve),
            "Zone_Economique": int(zone == "Grand Tunis"),
            "Marque": marque, "Modèle": modele,
            "Boite_Vitesse": boite, "Energie": energie,
        }
        X = pd.DataFrame([{c: saisie.get(c) for c in colonnes}])
        try:
            prix_log = bundle["pipeline"].predict(X)[0]
            prix_theorique = float(np.expm1(prix_log))
        except Exception as e:
            st.error(f"Erreur lors de la prédiction : {str(e)[:120]}")
            st.info("Relance `python core/modele_prediction.py` pour réentraîner le modèle "
                    "avec ta version de scikit-learn.")
            return

        if np.isnan(prix_theorique) or prix_theorique <= 0:
            st.error("Le modèle n'a pas pu produire une estimation cohérente pour cette combinaison.")
            return

        # ---- Résultat + comparables réels --------------------------------
        comparables = df[(df["Marque"] == marque) & (df["Modèle"].astype(str) == str(modele))]

        r1, r2, r3 = st.columns(3)
        r1.metric("Prix théorique estimé", fmt_dt(prix_theorique))
        if len(comparables) >= 3:
            r2.metric(f"Médiane des {len(comparables)} annonces comparables", fmt_dt(comparables["Prix"].median()))
            r3.metric("Fourchette observée",
                      f"{comparables['Prix'].quantile(.25):,.0f} – {comparables['Prix'].quantile(.75):,.0f} DT".replace(",", " "))

            fig = go.Figure()
            fig.add_trace(go.Box(
                x=comparables["Prix"], name="", marker_color=C_ASPHALTE,
                boxpoints="all", jitter=0.4, pointpos=0,
                hovertemplate="%{x:,.0f} DT<extra></extra>",
            ))
            fig.add_vline(x=prix_theorique, line_color=C_GAIN, line_width=3,
                          annotation_text="Estimation", annotation_font_color=C_GAIN)
            fig.update_layout(title=f"Ton estimation face aux {len(comparables)} annonces "
                                    f"{marque} {modele} du marché")
            fig.update_xaxes(title="Prix (DT)")
            fig.update_yaxes(visible=False)
            st.plotly_chart(style_figure(fig, 280), width="stretch")
        else:
            st.info("Moins de 3 annonces comparables sur le marché récent — l'estimation repose "
                    "surtout sur des véhicules proches, à prendre avec plus de prudence.")

        st.caption("💡 Une annonce réelle nettement sous ce prix ? Regarde la Vue Samsar : "
                   "elle y est probablement déjà signalée.")


# ===========================================================================
# 4. ADMIN — diagnostics du modèle et de la donnée
# ===========================================================================

DIAG_PATH = "data/processed/diagnostics_modele.json"
CALIB_PATH = "data/processed/calibration_fenetre.json"


def charger_json(chemin):
    return lire_json_distant(chemin)


def page_admin(df, df_deals):
    st.title("🛠️ Admin")
    st.caption("Diagnostics du modèle et de la donnée. Rien ici n'est destiné à un utilisateur "
               "final : c'est de quoi juger si les chiffres affichés ailleurs méritent confiance.")

    if df is None:
        st.error(f"`{SCORED_PATH}` introuvable — lance `python main.py` d'abord.")
        return

    diag = charger_json(DIAG_PATH)

    # ---- Modèles comparés ------------------------------------------------
    st.subheader("Comparaison des modèles")
    if diag:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Modèle retenu", diag["modele_retenu"].split(" (")[0])
        c2.metric("Annonces d'entraînement", f"{diag['n_annonces']:,}".replace(",", " "))
        c3.metric("Fenêtre de fraîcheur", f"{diag['fenetre_jours']} j")
        c4.metric("Validation croisée", f"{diag['n_folds']} folds")

        cand = pd.DataFrame(diag["candidats"]).sort_values("mdape_pct")
        cand["retenu"] = cand["nom"] == diag["modele_retenu"]

        fig = px.bar(
            cand.sort_values("mdape_pct", ascending=False),
            x="mdape_pct", y="nom", orientation="h",
            title="Erreur relative médiane par modèle candidat (out-of-fold)",
            labels={"mdape_pct": "Erreur relative médiane (%)", "nom": ""},
            custom_data=["mdae_dt", "mae_dt"],
        )
        fig.update_traces(
            marker_color=[C_GAIN if r else C_GRIS
                          for r in cand.sort_values("mdape_pct", ascending=False)["retenu"]],
            hovertemplate="%{y}<br>Erreur relative médiane : %{x:.2f} %"
                          "<br>Erreur absolue médiane : %{customdata[0]:,.0f} DT"
                          "<br>MAE : %{customdata[1]:,.0f} DT<extra></extra>",
        )
        st.plotly_chart(style_figure(fig, 280), width="stretch")

        st.dataframe(
            cand[["nom", "mdape_pct", "mdae_dt", "mae_dt"]].rename(columns={
                "nom": "Modèle", "mdape_pct": "Erreur relative médiane (%)",
                "mdae_dt": "Erreur absolue médiane (DT)", "mae_dt": "MAE (DT)"}),
            width="stretch", hide_index=True,
        )
        st.caption(f"Entraînement du {diag['date_entrainement'].replace('T', ' à ')}. "
                   "La sélection se fait sur l'erreur relative médiane : `Score_Opportunite` "
                   "étant un écart relatif, la métrique de sélection doit l'être aussi. "
                   "Le MAE est affiché pour information — il est dominé par le haut de gamme.")
    else:
        st.warning(f"`{DIAG_PATH}` introuvable — relance `python core/modele_prediction.py` "
                   "pour générer les diagnostics.")

    st.divider()

    # ---- Erreur par gamme de prix ---------------------------------------
    st.subheader("Où le modèle se trompe")
    df_e = df.dropna(subset=["Prix", "Prix_Theorique"]).copy()
    df_e["err"] = (df_e["Prix"] - df_e["Prix_Theorique"]).abs()
    df_e["err_rel"] = df_e["err"] / df_e["Prix"]
    df_e["gamme"] = pd.cut(
        df_e["Prix"], bins=[0, 30000, 50000, 80000, 120000, 200000, np.inf],
        labels=["< 30k", "30–50k", "50–80k", "80–120k", "120–200k", "> 200k"],
    )
    par_gamme = df_e.groupby("gamme", observed=True).agg(
        n=("err", "size"), err_rel=("err_rel", "median"), err_dt=("err", "median"),
        mae=("err", "mean")).reset_index()

    col_a, col_b = st.columns(2)
    with col_a:
        fig = px.bar(par_gamme, x="gamme", y="err_rel",
                     title="Erreur relative médiane par gamme de prix",
                     labels={"gamme": "", "err_rel": "Erreur relative médiane"},
                     custom_data=["n", "err_dt"])
        fig.update_traces(marker_color=C_ASPHALTE,
                          hovertemplate="%{x} : %{y:.1%}<br>n=%{customdata[0]} — "
                                        "médiane %{customdata[1]:,.0f} DT<extra></extra>")
        fig.update_yaxes(tickformat=".0%")
        st.plotly_chart(style_figure(fig, 340), width="stretch")
    with col_b:
        fig = px.bar(par_gamme, x="gamme", y="mae",
                     title="Erreur absolue moyenne (MAE) par gamme",
                     labels={"gamme": "", "mae": "MAE (DT)"}, custom_data=["n"])
        fig.update_traces(marker_color=C_ALERTE,
                          hovertemplate="%{x} : %{y:,.0f} DT (n=%{customdata[0]})<extra></extra>")
        st.plotly_chart(style_figure(fig, 340), width="stretch")

    st.caption("Le contraste entre ces deux graphiques explique pourquoi la sélection ne se fait "
               "pas sur le MAE : en relatif l'erreur est à peu près stable, en dinars elle explose "
               "sur le haut de gamme. Optimiser le MAE reviendrait à optimiser pour les voitures "
               "de luxe, qui sont une petite minorité des annonces.")

    # ---- Pires modèles ---------------------------------------------------
    pires = (df_e.groupby(["Marque", "Modèle"])
             .agg(n=("err_rel", "size"), err_rel=("err_rel", "median"))
             .reset_index())
    pires = pires[pires["n"] >= 10].sort_values("err_rel", ascending=False).head(12)
    if len(pires):
        pires["libelle"] = pires["Marque"] + " " + pires["Modèle"].astype(str)
        fig = px.bar(pires.sort_values("err_rel"), x="err_rel", y="libelle", orientation="h",
                     title="Modèles les moins bien estimés (≥ 10 annonces)",
                     labels={"err_rel": "Erreur relative médiane", "libelle": ""},
                     custom_data=["n"])
        fig.update_traces(marker_color=C_ALERTE,
                          hovertemplate="%{y} : %{x:.1%} (n=%{customdata[0]})<extra></extra>")
        fig.update_xaxes(tickformat=".0%")
        st.plotly_chart(style_figure(fig, 400), width="stretch")

    st.divider()

    # ---- Calibration de la fenêtre --------------------------------------
    st.subheader("Calibration de la fenêtre de fraîcheur")
    calib = charger_json(CALIB_PATH)
    if calib:
        c = pd.DataFrame(calib["resultats"])
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=c["fenetre"], y=c["err_rel_moy"] + c["ecart_type"],
            line=dict(width=0), showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(
            x=c["fenetre"], y=c["err_rel_moy"] - c["ecart_type"],
            fill="tonexty", fillcolor="rgba(138,151,163,0.20)", line=dict(width=0),
            showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(
            x=c["fenetre"], y=c["err_rel_moy"], mode="lines+markers",
            line=dict(color=C_GAIN, width=2), marker=dict(size=9, color=C_GAIN),
            customdata=c[["n_train", "ecart_type"]],
            hovertemplate="%{x} : %{y:.2f} % ± %{customdata[1]:.2f}"
                          "<br>%{customdata[0]:,} annonces d'entraînement<extra></extra>",
            showlegend=False))
        fig.update_layout(title="Erreur selon la quantité de données d'entraînement")
        fig.update_yaxes(title="Erreur relative médiane (%)")
        st.plotly_chart(style_figure(fig, 360), width="stretch")

        st.dataframe(
            c[["fenetre", "n_train", "err_rel_moy", "ecart_type"]].rename(columns={
                "fenetre": "Fenêtre", "n_train": "Annonces d'entraînement",
                "err_rel_moy": "Erreur relative médiane (%)", "ecart_type": "Écart-type (5 graines)"}),
            width="stretch", hide_index=True,
        )
        with st.expander("ℹ️ Protocole et lecture"):
            st.write(f"**Protocole.** {calib['protocole']}\n\n**Conclusion.** {calib['conclusion']}")
    else:
        st.info(f"`{CALIB_PATH}` introuvable — la calibration est une mesure ponctuelle, "
                "pas une étape du pipeline.")

    st.divider()

    # ---- Validation par les disparitions réelles -------------------------
    st.subheader("Validation — les opportunités partent-elles plus vite ?")
    st.caption("Le seul endroit du projet où l'on confronte les prédictions à un fait observé. "
               "Une annonce qui disparaît n'est pas forcément vendue — elle peut avoir été "
               "retirée ou avoir expiré — mais c'est le meilleur proxy disponible.")

    suivi = None
    try:
        suivi = lire_csv("data/processed/suivi_annonces.csv")
    except Exception:
        pass

    if suivi is None or "Jours_En_Ligne" not in suivi.columns:
        st.info("Le suivi des annonces n'a pas encore tourné. Il démarre au prochain "
                "`python main.py`.")
    else:
        jours = pd.to_numeric(suivi["Jours_En_Ligne"], errors="coerce")
        mesurees = suivi[jours.notna()].copy()
        mesurees["jours"] = jours[jours.notna()]

        c1, c2, c3 = st.columns(3)
        c1.metric("Annonces suivies", f"{len(suivi):,}".replace(",", " "))
        c2.metric("Disparues (mesurables)", len(mesurees))
        c3.metric("Signalées comme opportunité",
                  int(suivi["Etait_Opportunite"].sum()) if "Etait_Opportunite" in suivi.columns else 0)

        if len(mesurees) < 30:
            st.info(
                f"**{len(mesurees)} annonces disparues** — il en faut plusieurs dizaines pour "
                "comparer quoi que ce soit. Le suivi s'enrichit à chaque exécution nocturne : "
                "compte quelques semaines avant que cette section devienne lisible.\n\n"
                "Ce qui apparaîtra ici : la durée en ligne des annonces signalées comme "
                "opportunités face à celle des autres. Si les deux sont identiques, le "
                "détecteur ne détecte rien d'utile — et il vaudra mieux le savoir."
            )
        else:
            deals = mesurees[mesurees["Etait_Opportunite"] == True]      # noqa: E712
            autres = mesurees[mesurees["Etait_Opportunite"] != True]     # noqa: E712
            if len(deals) >= 10 and len(autres) >= 10:
                ca, cb = st.columns(2)
                ca.metric("Durée médiane — opportunités", f"{deals['jours'].median():.0f} j")
                cb.metric("Durée médiane — autres annonces", f"{autres['jours'].median():.0f} j")

                fig = go.Figure()
                fig.add_trace(go.Box(x=autres["jours"], name="Autres annonces",
                                     marker_color=C_GRIS, boxmean=True))
                fig.add_trace(go.Box(x=deals["jours"], name="Signalées opportunité",
                                     marker_color=C_GAIN, boxmean=True))
                fig.update_layout(title="Durée en ligne avant disparition")
                fig.update_xaxes(title="Jours en ligne")
                st.plotly_chart(style_figure(fig, 320), width="stretch")

                ecart = autres["jours"].median() - deals["jours"].median()
                if ecart > 1:
                    st.success(f"Les annonces signalées disparaissent environ **{ecart:.0f} jours "
                               "plus vite** que les autres — le détecteur capte bien quelque chose.")
                elif ecart < -1:
                    st.error("Les annonces signalées restent **plus longtemps** en ligne que les "
                             "autres. Le détecteur sélectionne probablement des véhicules peu "
                             "demandés plutôt que des bonnes affaires — seuils à revoir.")
                else:
                    st.warning("Aucune différence nette entre les deux groupes. En l'état, le "
                               "détecteur n'apporte pas de signal mesurable sur la vitesse "
                               "d'écoulement.")
            else:
                st.info("Pas encore assez d'annonces disparues dans chacun des deux groupes "
                        "(minimum 10 de chaque côté).")

            # Liquidité réelle par modèle
            par_modele = (mesurees.groupby(["Marque", "Modèle"])["jours"]
                          .agg(["size", "median"]).reset_index())
            par_modele = par_modele[par_modele["size"] >= 5].sort_values("median")
            if len(par_modele):
                par_modele["libelle"] = par_modele["Marque"] + " " + par_modele["Modèle"].astype(str)
                fig = px.bar(par_modele.head(15).sort_values("median", ascending=False),
                             x="median", y="libelle", orientation="h",
                             title="Modèles qui partent le plus vite (durée réelle en ligne)",
                             labels={"median": "Jours en ligne (médiane)", "libelle": ""},
                             custom_data=["size"])
                fig.update_traces(marker_color=C_GAIN,
                                  hovertemplate="%{y} : %{x:.0f} j (n=%{customdata[0]})<extra></extra>")
                st.plotly_chart(style_figure(fig, 420), width="stretch")
                st.caption("Cette mesure remplacera à terme le `Score_Liquidite`, qui n'est "
                           "aujourd'hui qu'un proxy fondé sur le volume d'annonces.")

    st.divider()

    # ---- Qualité de la donnée -------------------------------------------
    st.subheader("Qualité de la donnée")
    taux = (df.notna().mean() * 100).round(0).sort_values()
    taux = taux[taux < 100]
    if len(taux):
        fig = px.bar(x=taux.values, y=taux.index, orientation="h",
                     title="Taux de remplissage des colonnes incomplètes (%)",
                     labels={"x": "Rempli (%)", "y": ""})
        fig.update_traces(marker_color=[C_ALERTE if v < 50 else C_ASPHALTE for v in taux.values],
                          hovertemplate="%{y} : %{x:.0f} %<extra></extra>")
        st.plotly_chart(style_figure(fig, max(280, 26 * len(taux))), width="stretch")

    col_c, col_d = st.columns(2)
    with col_c:
        if "Fiabilite_Estimation" in df.columns:
            rep = df["Fiabilite_Estimation"].value_counts()
            fig = px.bar(x=rep.index.astype(str), y=rep.values,
                         title="Fiabilité des estimations (nb de comparables)",
                         labels={"x": "", "y": "Annonces"})
            fig.update_traces(marker_color=C_ASPHALTE)
            st.plotly_chart(style_figure(fig, 300), width="stretch")
    with col_d:
        if df_deals is not None and "Nb_Comparables" in df_deals.columns:
            solides = int((df_deals["Nb_Comparables"] >= 8).sum())
            st.metric("Opportunités détectées", len(df_deals))
            st.metric("Dont estimations solides", solides)
            st.metric("Écartées faute de comparables", len(df_deals) - solides)
            st.caption("Seules les opportunités solides déclenchent une alerte Telegram.")

    if diag:
        with st.expander("Variables utilisées par le modèle"):
            st.write("**Numériques** : " + ", ".join(f"`{f}`" for f in diag["features_numeriques"]))
            st.write("**Catégorielles** : " + ", ".join(f"`{f}`" for f in diag["features_categorielles"]))


# ===========================================================================
# Navigation
# ===========================================================================

def main():
    with st.sidebar:
        st.markdown("## 🚗 AutoDeal Tunisie")
        st.caption("Intelligence du marché de l'occasion — automobile.tn · tayara.tn · automax.tn")
        st.divider()
        page = st.radio(
            "Espace de travail",
            ["📊 Vue Marché", "🎯 Vue Samsar", "💰 Calculateur", "🛠️ Admin"],
            label_visibility="collapsed",
        )
        st.divider()
        df_temp = charger_scored()
        if df_temp is not None and "Annonce-Detectee" in df_temp.columns:
            derniere = df_temp["Annonce-Detectee"].max()
            st.caption(f"Dernière collecte : **{derniere}**")
            st.caption(f"Base : **{len(df_temp):,} annonces**".replace(",", " "))
            try:
                retard = (pd.Timestamp.now().normalize() - pd.to_datetime(derniere)).days
                if retard > 2:
                    st.warning(f"Données vieilles de {retard} jours — le scraping nocturne "
                               "a peut-être échoué. Vérifie l'onglet Actions du dépôt.")
            except Exception:
                pass
        if st.button("🔄 Rafraîchir les données"):
            st.cache_data.clear()
            st.cache_resource.clear()
            st.rerun()
        st.caption(f"Source : {'GitHub' if LIRE_DEPUIS_GITHUB else 'fichiers locaux'} · "
                   f"cache {DUREE_CACHE // 60} min")

    df_scored = charger_scored()
    df_deals = charger_deals()
    bundle = charger_modele()

    if page == "📊 Vue Marché":
        page_marche(df_scored)
    elif page == "🎯 Vue Samsar":
        page_samsar(df_scored, df_deals)
    elif page == "💰 Calculateur":
        page_calculateur(df_scored, bundle)
    else:
        page_admin(df_scored, df_deals)


if __name__ == "__main__":
    main()
