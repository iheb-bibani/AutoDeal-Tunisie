"""Page concessionnaire."""
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from config import MAX_DAYS_OLD

from ui.theme import (C_ASPHALTE, C_GAIN, C_ALERTE, C_SABLE, C_GRIS)
from ui.charts import style_figure, fmt_dt
from services.data_service import SCORED_PATH
from services.analytics_service import (calculer_prix_ajuste, analyse_decote_segment,
    calculer_niveau_regional, calculer_decote_annuelle, calculer_prime_pro,
    calculer_indice_depreciation, calculer_indice_modele_representatif, MARQUES_LUXE,
    MIN_ANNONCES_DECOTE, MIN_AGES_DISTINCTS)


def page_marche(df):
    st.title("🏢 Concessionnaire")
    st.caption("Comprendre et valoriser le parc : structure et niveaux de prix, "
               "que vaut le marché, où se vend quoi, à quel prix.")

    if df is None:
        st.error(f"`{SCORED_PATH}` introuvable — lance `python main.py` d'abord.")
        return

    # ---- KPIs ------------------------------------------------------------
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(f"Annonces récentes (≤ {MAX_DAYS_OLD} j)", f"{len(df):,}".replace(",", " "))
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
            title="Part des annonces observées par marque (top 15)",
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

    vue_depr = st.radio(
        "Méthode",
        ["Profil marque (composition corrigée par régression)", "Modèle représentatif par marque"],
        horizontal=True,
        help="La régression garde toutes les données et corrige le mix de modèles. "
             "Le modèle représentatif est plus littéral (un seul modèle qui vieillit) "
             "mais repose sur moins d'annonces.",
    )
    if not marques_courbe:
        indice = pd.DataFrame()
    elif vue_depr.startswith("Modèle"):
        indice = calculer_indice_modele_representatif(df, marques_courbe)
    else:
        indice = calculer_indice_depreciation(df, marques_courbe)
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
        # Les deux panneaux ne doivent JAMAIS montrer les mêmes modèles.
        # Avec un head(15) fixe, dès que la fonction retourne moins de 30
        # modèles les deux listes se recouvrent, et à 15 elles deviennent
        # identiques : on affiche alors deux fois la même chose, l'une
        # étiquetée "perdent le plus vite" et l'autre "tiennent le mieux".
        # En prenant au plus la moitié de chaque côté, le recouvrement est
        # structurellement impossible.
        k = min(15, len(decote) // 2)

        if k < 3:
            # Trop peu de modèles pour opposer deux groupes : un seul
            # graphique, honnête, avec tout ce qui est disponible.
            tout = decote.sort_values("decote_pct_an")
            fig = px.bar(
                tout, x="decote_pct_an", y="libelle", orientation="h",
                title=f"Décote annuelle — les {len(tout)} modèles disponibles",
                labels={"decote_pct_an": "Décote (% par an)", "libelle": ""},
                custom_data=["n", "prix_median"],
            )
            fig.update_traces(
                marker_color=C_ASPHALTE,
                hovertemplate="%{y} : %{x:.1f} %/an<br>%{customdata[0]} annonces — "
                              "prix médian %{customdata[1]:,.0f} DT<extra></extra>",
            )
            st.plotly_chart(style_figure(fig, max(280, 30 * len(tout))), width="stretch")
            st.warning(f"Seuls {len(decote)} modèles ont assez d'annonces pour estimer une "
                       "décote. Trop peu pour opposer « perdent le plus vite » et « tiennent "
                       "le mieux » sans afficher deux fois les mêmes véhicules.")
        else:
            col_g, col_h = st.columns(2)
            with col_g:
                top_d = decote.sort_values("decote_pct_an").head(k).sort_values("decote_pct_an", ascending=False)
                fig = px.bar(
                    top_d, x="decote_pct_an", y="libelle", orientation="h",
                    title=f"Perdent le plus vite leur valeur (top {k})",
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
                garde = decote.sort_values("decote_pct_an", ascending=False).head(k)
                fig = px.bar(
                    garde, x="decote_pct_an", y="libelle", orientation="h",
                    title=f"Tiennent le mieux leur valeur (top {k})",
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

    # ---- Prix par région, à véhicule comparable --------------------------
    if "Localisation" in df.columns:
        niveau = calculer_niveau_regional(df)
        if len(niveau):
            fig = px.bar(
                niveau.sort_values("prime_pct"), x="prime_pct", y="Localisation",
                orientation="h",
                title="Niveau de prix par région, à véhicule comparable",
                labels={"prime_pct": "Écart vs référence (%)", "Localisation": ""},
                custom_data=["n"],
            )
            fig.update_traces(
                marker_color=[C_GAIN if v < 0 else C_ALERTE
                              for v in niveau.sort_values("prime_pct")["prime_pct"]],
                hovertemplate="%{y} : %{x:+.1f} %<br>%{customdata[0]} annonces<extra></extra>",
            )
            fig.add_vline(x=0, line_color=C_GRIS, line_width=1)
            st.plotly_chart(style_figure(fig, 430), width="stretch")
            with st.expander("ℹ️ Pourquoi pas simplement le prix médian par région"):
                st.write(
                    """
Le prix médian brut par région ne mesure pas le niveau de prix local, mais la
composition du parc qui y est vendu. Sur les données réelles, la corrélation
entre le prix médian d'une région et sa **part de marques premium** est de
**+0,70** : Tunis n'est pas « chère », 34 % de ses annonces sont des Mercedes,
BMW ou Audi, contre 10 % à Médenine.

Le graphique ci-dessus vient de `log(prix) ~ modèle + âge + kilométrage +
région`. Le coefficient de région indique combien **le même véhicule** se
négocie plus ou moins cher selon l'endroit. Le vert signale les régions où
acheter, l'orange celles où revendre.
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

    st.divider()

    # ---- Courbe de décote : comment le prix baisse avec l'âge ------------
    st.subheader("Courbe de décote — comment le prix baisse avec l'âge")
    st.caption("Prix médian selon l'âge du véhicule, séparé par segment. La pente donne "
               "la dépréciation ; les toutes premières années reflètent surtout la prime du neuf.")
    if "Age_Vehicule" in df.columns and "Segment_Vehicule" in df.columns:
        d = df.dropna(subset=["Age_Vehicule", "Prix"]).copy()
        d = d[(d["Age_Vehicule"] >= 0) & (d["Age_Vehicule"] <= 25)]
        d["_seg"] = d["Segment_Vehicule"].map({1: "Segment luxe", 0: "Généraliste"}).fillna("Généraliste")
        courbe = (d.groupby(["_seg", "Age_Vehicule"], observed=True)
                  .agg(prix=("Prix", "median"), n=("Prix", "size")).reset_index())
        courbe = courbe[courbe["n"] >= 10]  # points trop peu peuplés = bruités (ex luxe âge 0, n=9)
        if len(courbe):
            fig = px.line(courbe, x="Age_Vehicule", y="prix", color="_seg", markers=True,
                          title="Prix médian selon l'âge, par segment",
                          labels={"Age_Vehicule": "Âge (années)", "prix": "Prix médian (DT)", "_seg": "Segment"})
            st.plotly_chart(style_figure(fig, 380), width="stretch")
            gen = (courbe[(courbe["_seg"] == "Généraliste") & courbe["Age_Vehicule"].between(2, 8)]
                   .sort_values("Age_Vehicule"))
            if len(gen) >= 3:
                p0, p1 = gen["prix"].iloc[0], gen["prix"].iloc[-1]
                ans = gen["Age_Vehicule"].iloc[-1] - gen["Age_Vehicule"].iloc[0]
                if p0 > 0 and ans > 0:
                    taux = (1 - (p1 / p0) ** (1 / ans)) * 100
                    st.caption(f"Décote annuelle moyenne (généraliste, **2–8 ans**) : **≈ {taux:.0f} % par an**. "
                               "La chute plus marquée avant 2 ans est surtout la « prime du neuf » "
                               "qui s'évapore, pas de la dépréciation d'occasion.")
        else:
            st.info("Pas encore assez de données par âge pour tracer la courbe.")
    else:
        st.info("Colonnes d'âge/segment absentes des données scorées.")

    # ---- Décote par segment + stratégie propriétaire ---------------------
    st.divider()
    st.subheader("Décote par segment & stratégie d'achat propriétaire")
    infos = {s: analyse_decote_segment(df, s) for s in ["Généraliste", "Segment luxe"]}
    infos = {s: v for s, v in infos.items() if v}
    if infos:
        st.caption("Décote annuelle réelle par tranche d'âge (mesurée à partir de 2 ans : "
                   "l'âge 0-1, quasi-neuves rares et chères, relève de la « prime du neuf » "
                   "et non de la dépréciation d'occasion).")
        tab = pd.DataFrame([{
            "Segment": seg,
            "2→4 ans": f"−{v['falaise']:.0f} %/an",
            "4→7 ans": f"−{v['plateau']:.0f} %/an",
            "7→10 ans": f"−{v['traine']:.0f} %/an",
        } for seg, v in infos.items()])
        st.dataframe(tab, hide_index=True, width="stretch")

        gen = infos.get("Généraliste")
        lux = infos.get("Segment luxe") or infos.get("Luxe")
        st.markdown("**🧍 Pour un propriétaire — ce que disent vraiment les chiffres**")
        if gen:
            st.markdown(
                f"- **Le généraliste tient sa valeur au début, puis décroche.** De 2 à 4 ans, "
                f"il ne perd que **~{gen['falaise']:.0f} %/an** ; la décote **accélère ensuite** "
                f"(~{gen['plateau']:.0f} %/an de 4 à 7 ans, ~{gen['traine']:.0f} %/an après). "
                f"Il n'y a donc pas de « falaise » d'occasion — la seule vraie chute, c'est la "
                f"**prime du neuf** qui s'évapore la 1re année, que tu évites simplement en "
                f"achetant d'occasion.\n"
                f"- **Point d'entrée optimal : 2 à 4 ans** — perte de détention la plus faible. "
                f"Si tu comptes revendre, fais-le **avant que la décote accélère** (~6-7 ans).")
        if lux:
            st.markdown(
                f"- **Le luxe, lui, décroche plus tôt** (~{lux['falaise']:.0f} %/an dès 2-4 ans) : "
                f"la vieille règle « laisser le premier propriétaire payer la chute » s'y applique "
                f"vraiment. Achète-le plutôt vers 4-5 ans, une fois la grosse perte encaissée.")
    else:
        st.info("Pas assez de données par segment pour l'analyse de décote.")
