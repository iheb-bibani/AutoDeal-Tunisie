"""Page samsar."""
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from ui.theme import (C_ASPHALTE, C_GAIN, C_ALERTE, C_GRIS)
from ui.charts import style_figure, explication, fmt_dt
from services.data_service import SCORED_PATH
from services.model_service import (charger_modele, expliquer_prix)
from services.analytics_service import (analyse_rendement_capital, analyse_decote_segment,
    _match_gouvernorat)


def page_samsar(df_scored, df_deals):
    st.title("🤝 Samsar")
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

        # ---- Filtres (au-dessus des cartes : elles s'adaptent à la sélection) ----
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

        # ---- Cartes : reflètent LA SÉLECTION (marché global gardé en contexte) ----
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Opportunités (sélection)", len(sel))
        c2.metric("Gain médian", fmt_dt(sel["Gain_DT"].median()) if len(sel) else "—")
        c3.metric("Gain cumulé", fmt_dt(sel["Gain_DT"].sum()) if len(sel) else "—")
        c4.metric("Décote médiane vs argus",
                  f"{sel['Score_Opportunite'].median():.0%}" if len(sel) else "—")
        if "Nb_Comparables" in deals.columns:
            solides_g = int((deals["Nb_Comparables"] >= 8).sum())
            ctx = f"**{len(deals)}** opportunités détectées, dont **{solides_g}** sur ≥ 8 comparables"
        else:
            ctx = f"**{len(deals)}** opportunités détectées"
        st.caption(f"Les cartes s'adaptent à tes filtres ci-dessus. Marché global : {ctx}.")

        st.divider()

        # ---- Où chasser selon ton capital --------------------------------
        st.subheader("Où chasser selon ton capital")
        st.caption("Le gain absolu monte avec le prix d'achat, mais le rendement (%) et la "
                   "régularité, non. À toi de choisir la tranche selon ton capital et ton appétit.")
        rc = analyse_rendement_capital(deals)
        if len(rc) >= 2:
            fig_rc = go.Figure()
            fig_rc.add_bar(x=rc["tranche"], y=rc["gain"], name="Gain médian (DT)",
                           marker_color=C_ASPHALTE)
            fig_rc.add_scatter(x=rc["tranche"], y=rc["roi"], name="ROI médian (%)", yaxis="y2",
                               mode="lines+markers", line=dict(color=C_GAIN, width=3))
            fig_rc.update_layout(
                title_text="",
                yaxis=dict(title="Gain médian (DT)"),
                yaxis2=dict(title="ROI médian (%)", overlaying="y", side="right", showgrid=False),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
            )
            st.plotly_chart(style_figure(fig_rc, 380), width="stretch")

            tab_rc = pd.DataFrame({
                "Tranche d'achat": rc["tranche"],
                "Gain médian": rc["gain"].map(fmt_dt),
                "ROI médian": rc["roi"].map(lambda x: f"{x:.0f} %"),
                "Régularité": rc["regularite"].map(lambda x: f"{x:.2f}"),
                "Nb affaires": rc["n"],
            })
            st.dataframe(tab_rc, hide_index=True, width="stretch")

            best_roi = rc.loc[rc["roi"].idxmax()]
            best_reg = rc.loc[rc["regularite"].idxmax()]
            worst = rc.loc[rc["roi"].idxmin()]
            st.markdown(
                f"- **Petit capital → maximise le rendement.** La tranche **{best_roi['tranche']}** "
                f"donne le meilleur ROI (**{best_roi['roi']:.0f} %**) : tu fais tourner ton argent vite.\n"
                f"- **Gros capital → maximise l'absolu.** Les tranches hautes rapportent le plus gros "
                f"gain par flip (jusqu'à **{fmt_dt(rc['gain'].max())}**), et la meilleure régularité est "
                f"en **{best_reg['tranche']}** (mais peu d'affaires : {int(best_reg['n'])}).\n"
                f"- **Évite {worst['tranche']}** : le ROI le plus bas (**{worst['roi']:.0f} %**) sans "
                f"régularité qui compense — le piège du milieu de gamme."
            )
            explication(
                "Comment lire « gain », « ROI » et « régularité »",
                "**Ce que ça montre.** Pour chaque tranche de prix d'achat : le gain médian en "
                "dinars, le rendement médian (%), et la *régularité* du rendement.\n\n"
                "**Comment c'est calculé.** Gain = prix théorique − prix affiché ; ROI = gain / prix. "
                "**Régularité = ROI médian ÷ dispersion du ROI** (un Sharpe-*like*).\n\n"
                "**Comment décider.** Gros ROI = capital qui tourne vite (petit budget) ; gros gain "
                "absolu = grosse marge par flip (gros budget) ; forte régularité = affaires "
                "consistantes.\n\n"
                "⚠️ **Ce n'est pas un Sharpe financier.** La dispersion mêle la variété réelle des "
                "affaires et l'erreur du modèle (~10,7 %), et n'intègre pas le risque de revente — "
                "celui-ci viendra quand le suivi des disparitions aura accumulé de l'historique."
            )

        st.divider()

        # ---- Matrice gain × profondeur de marché ------------------------------------
        if len(sel):
            fig = px.scatter(
                sel, x="Score_Opportunite", y="Score_Liquidite", size="Gain_DT",
                color="Score_Liquidite", color_continuous_scale=["#C9CFD6", C_GAIN],
                hover_data={"Titre": True, "Marque": True, "Modèle": True,
                            "Prix": ":,.0f", "Gain_DT": ":,.0f",
                            "Score_Opportunite": ":.0%", "Score_Liquidite": ":.2f"},
                title="Matrice des affaires — en haut à droite : gros gain ET revente facile",
                labels={"Score_Opportunite": "Décote vs prix théorique",
                        "Score_Liquidite": "Profondeur du marché (proxy)"},
            )
            fig.update_layout(coloraxis_showscale=False)
            fig.update_xaxes(tickformat=".0%")
            st.plotly_chart(style_figure(fig, 430), width="stretch")
            explication(
                "Comment lire la matrice des affaires",
                "**Ce que ça montre.** Chaque point est une opportunité. En abscisse, sa décote "
                "(à quel point elle est sous le prix théorique) ; en ordonnée, sa profondeur de marché "
                "(facilité de revente) ; la taille du point = le gain en dinars.\n\n"
                "**Comment c'est calculé.** Décote = 1 − prix/prix_théorique ; profondeur de marché = proxy "
                "basé sur le volume d'annonces du modèle (à affiner avec la vraie vitesse de revente).\n\n"
                "**Comment décider.** Vise **en haut à droite** : grosse décote *et* revente facile. "
                "Les gros points en haut à droite sont les affaires idéales ; un gros point en bas "
                "(gros gain mais peu liquide) peut rester longtemps sur les bras."
            )

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

            # ---- Pourquoi cette affaire ? (SHAP local) -------------------
            sel_shap = sel.sort_values("Gain_DT", ascending=False).reset_index(drop=True)

            def _label_deal(r):
                an = "" if pd.isna(r.get("Année")) else f" {int(r['Année'])}"
                return f"{r['Marque']} {r['Modèle']}{an} — {r['Prix']:,.0f} DT".replace(",", " ")

            sel_shap["_label"] = sel_shap.apply(_label_deal, axis=1)
            choix = st.selectbox("🔍 Analyser une affaire — pourquoi le modèle l'estime plus chère",
                                 sel_shap["_label"].tolist())
            ligne = sel_shap[sel_shap["_label"] == choix].iloc[0]

            bundle_s = charger_modele()
            contrib = expliquer_prix(bundle_s, ligne.to_dict(), df_scored) if bundle_s is not None else None
            if contrib is not None and len(contrib):
                decote = ligne.get("Score_Opportunite")
                st.caption(
                    f"Estimée à **{ligne['Prix_Theorique']:,.0f} DT**, affichée à "
                    f"**{ligne['Prix']:,.0f} DT** → décote de **{decote:.0%}**. "
                    "Voici ce qui, selon le modèle, justifie sa valeur estimée :".replace(",", " ")
                )
                top = contrib.head(6).iloc[::-1]
                fig = go.Figure(go.Bar(
                    x=top["shap"], y=top["label"], orientation="h",
                    marker_color=[C_GAIN if v > 0 else C_ALERTE for v in top["shap"]],
                    hovertemplate="%{y}<extra></extra>", showlegend=False))
                fig.update_layout(title="Ce qui valorise (vert) ou dévalorise (rouge) cette voiture")
                fig.update_xaxes(title="contribution (échelle log-prix)")
                st.plotly_chart(style_figure(fig, 300), width="stretch")
            elif bundle_s is None:
                st.caption("Modèle indisponible — l'explication par annonce nécessite le modèle chargé.")
            else:
                st.caption("Explication indisponible pour cette annonce.")

    st.divider()

    # ---- Rotation : qu'est-ce qui tourne vite ? --------------------------
    st.subheader("Rotation du marché — qu'est-ce qui part vite ?")
    st.caption("Volume d'annonces = demande et facilité de revente. L'âge des annonces encore "
               "en ligne sert de proxy d'écoulement — mais uniquement sur tayara.tn, pour la "
               "raison expliquée sous le second graphique.")

    # L'âge des annonces n'est comparable QU'À SOURCE ÉGALE. automobile.tn ne
    # conserve pas d'annonces anciennes : la corrélation entre "part de pros"
    # d'un modèle et l'âge médian de ses annonces est de -0,43. Mélanger les
    # sources faisait apparaître Audi A5, BMW Série 5 ou Range Rover Sport
    # comme les modèles "les plus rapides" -- tous à 100 % de pros. On ne
    # mesurait pas la vitesse d'écoulement mais le site d'origine.
    base_rotation = df_scored[
        df_scored["Source"].astype(str).str.lower().str.contains("tayara", na=False)
    ]

    rotation = (
        base_rotation.dropna(subset=["Marque", "Modèle"])
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
                title="Écoulement le plus rapide (tayara.tn uniquement)",
                labels={"age_annonce_median": "Âge médian des annonces (jours)", "libelle": ""},
                custom_data=["volume", "prix_median"],
            )
            fig.update_traces(
                marker_color=C_GAIN,
                hovertemplate="%{y} : %{x:.0f} j (n=%{customdata[0]})"
                              "<br>Prix médian : %{customdata[1]:,.0f} DT<extra></extra>",
            )
            st.plotly_chart(style_figure(fig, 420), width="stretch")

        with st.expander("ℹ️ Pourquoi seulement tayara.tn ici"):
            st.write(
                """
L'âge des annonces encore en ligne n'est comparable qu'à **source égale**.
automobile.tn ne conserve pas d'annonces anciennes : sur les données réelles,
la corrélation entre la part de professionnels d'un modèle et l'âge médian de
ses annonces est de **−0,43**. En mélangeant les sources, les modèles
« les plus rapides » étaient l'Audi A5 Sportback, la BMW Série 5, le X3 et le
Range Rover Sport — tous à **100 % de pros**. Ce n'était pas de la vitesse
d'écoulement, c'était le site d'origine.

Restreindre à tayara.tn rend la comparaison honnête, au prix d'un périmètre
plus étroit. Cette mesure indirecte disparaîtra dès que le suivi des annonces
(`core/suivi_annonces.py`) aura accumulé quelques semaines : il donnera la
durée réelle entre publication et disparition, sans proxy.
"""
            )

    # ---- Fenêtre d'achat samsar : cheap mais revendable ------------------
    st.divider()
    st.subheader("Fenêtre d'achat samsar — cheap, mais qui se revend")
    st.caption("Le compromis achat-revente : assez vieux pour acheter bas (falaise passée), "
               "assez demandé pour ne pas rester en stock.")
    infos_s = {s: analyse_decote_segment(df_scored, s) for s in ["Généraliste", "Segment luxe"]}
    infos_s = {s: v for s, v in infos_s.items() if v}
    if infos_s:
        ref = infos_s.get("Généraliste") or next(iter(infos_s.values()))
        liq = ref["ages_liquides"]
        if liq:
            a_min, a_max = min(liq), max(liq)
            st.markdown(
                f"- **Vise la zone {a_min}-{a_max} ans (généraliste).** C'est là que le marché a le "
                f"plus d'annonces (**{ref['n_liquide']} sur ces âges**) : le plus d'acheteurs, donc "
                f"**revente rapide et risque d'invendu minimal**.\n"
                f"- **La prime du neuf est déjà encaissée** : la grosse perte de la 1re année a "
                f"été payée par le vendeur précédent — tu achètes bas, sans la subir.\n"
                f"- **Faible perte pendant la détention** (~{ref['plateau']:.0f} %/an) : ta marge n'est "
                f"pas grignotée si la voiture reste quelques semaines en vitrine.\n"
                f"- **⚠️ Le piège du très vieux (> 12 ans)** : encore moins cher, mais beaucoup moins "
                f"d'acheteurs. Une voiture cheap qui ne se revend pas immobilise ton cash — ça coûte "
                f"plus qu'une marge plus fine qui tourne vite."
            )
        else:
            st.info("Pas assez d'annonces par âge pour cerner la fenêtre liquide.")
    else:
        st.info("Pas assez de données par segment pour la stratégie samsar.")

    # ---- Arbitrage géographique -----------------------------------------
    st.subheader("Arbitrage géographique — où acheter, où revendre")
    st.caption("Dans quelle région le prix est-il le plus bas / le plus haut, **à âge et "
               "kilométrage comparables** ? Deux angles : **par modèle** (précis mais peu "
               "d'annonces) ou **par segment** (robuste statistiquement, composition plus grossière).")

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
            n_ach = int((g2["Localisation"] == worst).sum())
            n_rev = int((g2["Localisation"] == best).sum())
            rows.append({
                "libelle": f"{m} {mo}", "Marque": m, "Modèle": str(mo),
                "n": len(g2), "n_regs": g2["Localisation"].nunique(),
                "acheter": worst, "revendre": best,
                "n_acheter": n_ach, "n_revendre": n_rev,
                "ecart_pct": round(ecart, 1),
                "ecart_dt": int(prix_base * ecart / 100),
                "prix_achat_est": int(prix_base),
                "coefs": coefs,
            })
        return pd.DataFrame(rows).sort_values("ecart_pct", ascending=False) if rows else pd.DataFrame()

    @st.cache_data
    def calculer_arbitrage_geo_segment(df_, seg_val):
        """Même régression log(prix) ~ âge + km + gouvernorat, mais poolée sur TOUT
        un segment (au lieu d'un seul modèle). Beaucoup plus d'annonces par région
        -> plus robuste ; en contrepartie la composition modèle n'est pas contrôlée
        (âge et km le sont). Localisation agrégée au gouvernorat."""
        base = df_.dropna(subset=["Prix", "Age_Vehicule", "Kilométrage", "Localisation"]).copy()
        base = base[base["Segment_Vehicule"] == seg_val]
        base["gouv"] = base["Localisation"].map(_match_gouvernorat)
        base = base.dropna(subset=["gouv"])
        vc = base["gouv"].value_counts()
        regs = vc[vc >= 15].index.tolist()
        base = base[base["gouv"].isin(regs)]
        if base["gouv"].nunique() < 2 or len(base) < 40:
            return None
        X = pd.get_dummies(base[["gouv"]], drop_first=True).astype(float)
        X["age"] = base["Age_Vehicule"]
        X["km"] = base["Kilométrage"] / 10000
        X.insert(0, "const", 1.0)
        try:
            c, *_ = np.linalg.lstsq(X.values, np.log(base["Prix"].values), rcond=None)
        except np.linalg.LinAlgError:
            return None
        coefs = {col.replace("gouv_", ""): (np.exp(c[list(X.columns).index(col)]) - 1) * 100
                 for col in X.columns if col.startswith("gouv_")}
        ref = [r for r in regs if r not in coefs]
        if ref:
            coefs[ref[0]] = 0.0
        best, worst = max(coefs, key=coefs.get), min(coefs, key=coefs.get)
        prix_base = float(np.exp(
            c[0] + c[list(X.columns).index("age")] * base["Age_Vehicule"].median()
            + c[list(X.columns).index("km")] * base["Kilométrage"].median() / 10000))
        ecart = coefs[best] - coefs[worst]
        return {
            "coefs": coefs, "acheter": worst, "revendre": best, "n": len(base),
            "n_acheter": int((base["gouv"] == worst).sum()),
            "n_revendre": int((base["gouv"] == best).sum()), "n_regs": len(regs),
            "ecart_pct": round(ecart, 1), "ecart_dt": int(prix_base * ecart / 100),
        }

    def _rendre_arbitrage(titre, coefs, acheter, revendre, n_ach, n_rev, n_tot, ecart_pct, ecart_dt):
        """Carte d'action à confiance graduée + graphe, commune aux deux vues."""
        min_reg = min(n_ach, n_rev)
        if min_reg >= 8 and n_tot >= 30 and ecart_pct >= 15:
            emoji, niveau, action = "🟢", "Forte", (
                f"**Play net.** Achète à **{acheter}**, revends à **{revendre}**. L'écart est "
                f"solide et repose sur assez d'annonces des deux côtés — il dépasse largement le "
                f"bruit. Il te reste à couvrir transport + mutation avec la marge.")
        elif min_reg >= 5 and n_tot >= 20 and ecart_pct >= 8:
            emoji, niveau, action = "🟡", "Moyenne", (
                f"**Écart réel, à confirmer.** Achète à **{acheter}**, revends à **{revendre}** — "
                f"mais la marge est plus fine : vérifie l'état et chiffre les frais (mutation, "
                f"transport) avant de t'engager.")
        else:
            emoji, niveau, action = "🔴", "Faible", (
                f"**Signal indicatif seulement** ({n_ach} annonce(s) à l'achat, {n_rev} à la "
                f"revente). Traite-le comme une piste à confirmer à la main, pas une certitude.")
        m1, m2, m3 = st.columns(3)
        m1.metric("Profit potentiel (brut)", f"{ecart_dt:,} DT".replace(",", " "))
        m2.metric("Profit %", f"+{ecart_pct:.0f} %")
        m3.metric("Confiance", f"{emoji} {niveau}")
        st.info(action)
        st.caption(f"À âge et km comparables, sur {n_tot} annonces ({n_ach} à {acheter}, "
                   f"{n_rev} à {revendre}). **Profit brut, avant frais** de transport et de "
                   "mutation — un plafond théorique, pas un net dans ta poche.")
        coefs_df = pd.DataFrame({"region": list(coefs.keys()),
                                 "prime_pct": list(coefs.values())}).sort_values("prime_pct")
        fig = px.bar(coefs_df, x="prime_pct", y="region", orientation="h",
                     title=f"{titre} — prime de prix par région, à âge et km comparables",
                     labels={"prime_pct": "Prime vs référence (%)", "region": ""})
        couleurs = [C_GAIN if i == 0 else (C_ALERTE if i == len(coefs_df) - 1 else C_ASPHALTE)
                    for i in range(len(coefs_df))]
        fig.update_traces(marker_color=couleurs, hovertemplate="%{y} : %{x:+.1f} %<extra></extra>")
        fig.add_vline(x=0, line_color=C_GRIS, line_width=1)
        st.plotly_chart(style_figure(fig, 340), width="stretch")

    vue_arb = st.radio("Vue", ["Par modèle (précis)", "Par segment (robuste)"], horizontal=True)
    if vue_arb.startswith("Par modèle"):
        arb = calculer_arbitrage_geo(df_scored)
        if len(arb):
            modele_choisi = st.selectbox("Modèle à analyser", arb["libelle"].tolist())
            l = arb[arb["libelle"] == modele_choisi].iloc[0]
            _rendre_arbitrage(modele_choisi, l["coefs"], l["acheter"], l["revendre"],
                              l["n_acheter"], l["n_revendre"], l["n"], l["ecart_pct"], l["ecart_dt"])
        else:
            st.info("Pas assez de données par modèle pour un arbitrage fiable.")
    else:
        seg_label = st.selectbox("Segment", ["Généraliste", "Segment luxe"])
        res = calculer_arbitrage_geo_segment(df_scored, 1 if "luxe" in seg_label.lower() else 0)
        if res:
            _rendre_arbitrage(seg_label, res["coefs"], res["acheter"], res["revendre"],
                              res["n_acheter"], res["n_revendre"], res["n"], res["ecart_pct"], res["ecart_dt"])
        else:
            st.info("Pas assez de données pour ce segment.")

    with st.expander("ℹ️ Pourquoi « à âge et km comparables » change tout"):
        st.write(
            """
Comparer les prix médians *bruts* par région est trompeur : une région peut
sembler moins chère simplement parce que ses voitures y sont plus vieilles ou
plus kilométrées. Exemple observé : sur l'Audi A3 Sportback, un écart Sfax/Nabeul
venait surtout de l'âge (2 ans en moyenne vs 4 ans), pas de la région.

Ici, une régression `log(prix) ~ âge + kilométrage + région` isole l'effet région
une fois l'âge et le km neutralisés. La barre à zéro est la région de référence ;
les autres se lisent comme des écarts par rapport à elle.

- **Vue par modèle** — garde-fous : ≥ 3 annonces/région, ≥ 12 au total, recouvrement
  d'âge ≥ 2 ans entre régions. Précis (même modèle), mais souvent peu d'annonces →
  confiance basse.
- **Vue par segment** — même régression poolée sur tout un segment (≥ 15 annonces
  par gouvernorat). Beaucoup plus robuste statistiquement, au prix d'un contrôle de
  composition plus grossier (les modèles d'un même segment sont mélangés).
"""
        )
