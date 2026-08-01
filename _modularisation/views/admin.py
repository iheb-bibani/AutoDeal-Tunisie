"""Page admin."""
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from ui.theme import (C_ASPHALTE, C_GAIN, C_ALERTE, C_GRIS)
from ui.charts import style_figure, explication
from services.data_service import (charger_json, lire_csv, SCORED_PATH, DIAG_PATH,
    CALIB_PATH, SHAP_PATH, HISTO_PATH)
from services.model_service import LABELS_FEATURES
from services.analytics_service import nom_modele_court


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

    # ---- Fiabilité de l'estimation par tranche de prix -------------------
    st.subheader("Fiabilité de l'estimation par tranche de prix")
    st.caption("Erreur du modèle (prédiction hors-échantillon vs prix réel) selon la "
               "gamme de prix. Plus la courbe est basse, plus l'estimation est fiable "
               "dans cette tranche — à lire avant de se fier à un prix estimé.")

    if "Prix_Theorique" in df.columns:
        d_fia = df.dropna(subset=["Prix", "Prix_Theorique"]).copy()
        d_fia = d_fia[(d_fia["Prix"] > 0) & (d_fia["Prix_Theorique"] > 0)]
    else:
        d_fia = pd.DataFrame()

    if len(d_fia) >= 50:
        d_fia["_err"] = (d_fia["Prix_Theorique"] - d_fia["Prix"]).abs() / d_fia["Prix"]
        bornes = [0, 15000, 25000, 35000, 50000, 75000, 100000, float("inf")]
        libelles = ["< 15k", "15–25k", "25–35k", "35–50k", "50–75k", "75–100k", "100k +"]
        d_fia["_tranche"] = pd.cut(d_fia["Prix"], bins=bornes, labels=libelles)
        gr = (d_fia.groupby("_tranche", observed=True)
              .agg(n=("_err", "size"), mdape=("_err", lambda x: 100 * x.median()))
              .reset_index())
        gr = gr[gr["n"] >= 10]  # une tranche trop peu peuplée donne une mesure instable

        mdape_global = 100 * d_fia["_err"].median()

        fig = go.Figure()
        fig.add_hline(y=mdape_global, line=dict(color="rgba(138,151,163,0.6)", dash="dot"),
                      annotation_text=f"erreur globale {mdape_global:.0f}%",
                      annotation_position="top left")
        fig.add_trace(go.Scatter(
            x=gr["_tranche"], y=gr["mdape"], mode="lines+markers",
            line=dict(color=C_GAIN, width=2), marker=dict(size=10, color=C_GAIN),
            customdata=gr[["n"]],
            hovertemplate="%{x} DT : erreur médiane %{y:.1f} %"
                          "<br>%{customdata[0]:,} annonces<extra></extra>",
            showlegend=False))
        fig.update_layout(title="Erreur relative médiane par tranche de prix (DT)")
        fig.update_yaxes(title="Erreur relative médiane (%)")
        fig.update_xaxes(title="Tranche de prix")
        st.plotly_chart(style_figure(fig, 360), width="stretch")

        best = gr.loc[gr["mdape"].idxmin()]
        pire = gr.loc[gr["mdape"].idxmax()]
        st.caption(
            f"Zone la plus fiable : **{best['_tranche']} DT** ({best['mdape']:.0f} % d'erreur). "
            f"La moins fiable : **{pire['_tranche']} DT** ({pire['mdape']:.0f} %) — une estimation "
            "dans cette gamme est à prendre avec prudence."
        )
    else:
        st.info("Pas assez de données scorées (avec `Prix_Theorique`) pour mesurer la "
                "fiabilité par tranche.")

    st.divider()

    # ---- Performance & drift dans le temps -------------------------------
    st.subheader("Performance du modèle dans le temps")
    st.caption("MdAPE et composition du marché à chaque run nocturne. "
               "La vraie preuve de fiabilité, ce n'est pas un snapshot : c'est la stabilité "
               "dans la durée. Alimenté par `core/suivi_performance.py`.")
    histo = charger_json(HISTO_PATH)
    if not histo or len(histo) < 2:
        n = 0 if not histo else len(histo)
        st.info(f"Pas encore assez d'historique ({n} run(s)) — la courbe apparaîtra après "
                "quelques exécutions nocturnes. Ajoute `python core/suivi_performance.py` "
                "au pipeline pour commencer à accumuler.")
    else:
        hp = pd.DataFrame([{
            "date": pd.to_datetime(h.get("date")),
            "MdAPE (%)": h.get("mdape_global"),
            "Volume": h.get("instantane", {}).get("n"),
            "Prix médian": h.get("instantane", {}).get("prix_median"),
        } for h in histo]).sort_values("date")

        fig = go.Figure()
        fig.add_scatter(x=hp["date"], y=hp["MdAPE (%)"], mode="lines+markers",
                        name="MdAPE (%)", line=dict(color=C_GAIN, width=3))
        fig.add_scatter(x=hp["date"], y=hp["Volume"], mode="lines", name="Volume",
                        yaxis="y2", line=dict(color=C_GRIS, width=1, dash="dot"))
        fig.update_layout(
            yaxis=dict(title="MdAPE (%)"),
            yaxis2=dict(title="Volume", overlaying="y", side="right", showgrid=False),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0), title_text="",
        )
        st.plotly_chart(style_figure(fig, 340), width="stretch")

        c1, c2, c3 = st.columns(3)
        derniere = hp.iloc[-1]
        c1.metric("MdAPE (dernier run)", f"{derniere['MdAPE (%)']:.1f} %"
                  if pd.notna(derniere["MdAPE (%)"]) else "—")
        if len(hp) >= 2 and pd.notna(hp.iloc[-2]["MdAPE (%)"]) and pd.notna(derniere["MdAPE (%)"]):
            delta = derniere["MdAPE (%)"] - hp.iloc[-2]["MdAPE (%)"]
            c2.metric("Évolution vs run précédent", f"{delta:+.1f} pts",
                      delta=f"{delta:+.1f}", delta_color="inverse")
        c3.metric("Runs enregistrés", len(hp))

        drift = histo[-1].get("drift", [])
        if drift:
            st.warning("⚠️ Drift détecté au dernier run : " + " · ".join(drift))
        else:
            st.caption("✓ Aucun drift détecté au dernier run.")

        # ---- Prix médian glissant (à travers les runs, non biaisé) --------
        hp["glissant"] = hp["Prix médian"].rolling(7, min_periods=2, center=True).median()
        figp = go.Figure()
        figp.add_scatter(x=hp["date"], y=hp["Prix médian"], mode="markers",
                         name="Prix médian (par run)", marker=dict(color=C_GRIS, size=5))
        figp.add_scatter(x=hp["date"], y=hp["glissant"], mode="lines",
                         name="Médiane glissante (7 runs)", line=dict(color=C_ASPHALTE, width=3))
        figp.update_layout(yaxis=dict(title="Prix médian marché (DT)"), title_text="",
                           legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0))
        st.plotly_chart(style_figure(figp, 300), width="stretch")
        st.caption("Prix médian du marché à chaque run (points) et sa **médiane glissante** (ligne). "
                   "Chaque run mesure le marché *actuellement en ligne* → pas de biais de survie, "
                   "contrairement à une médiane par date de dépôt (où les vieilles annonces encore "
                   "présentes sont les invendues, souvent surcotées).")

        explication(
            "Comment lire ce suivi",
            "**Ce que ça montre.** L'erreur du modèle (MdAPE) et le volume de données à chaque "
            "run nocturne.\n\n"
            "**Comment c'est calculé.** Chaque nuit, `suivi_performance.py` enregistre la MdAPE "
            "hors-échantillon et un instantané du marché (prix médian, mix des sources, "
            "complétude), puis compare à la médiane des 7 derniers runs pour détecter un drift.\n\n"
            "**Comment décider.** Une **MdAPE plate** = modèle fiable dans la durée. Une **hausse "
            "progressive** ou un **drift** signalé = le marché a bougé → il est temps de réentraîner "
            "ou d'investiguer (nouvelle source, changement de composition)."
        )

    st.divider()

    # ---- Interprétation du modèle (SHAP) ---------------------------------
    st.subheader("Interprétation — qu'est-ce qui fait le prix ?")
    st.caption("Importance globale de chaque variable dans les prédictions "
               "(valeurs SHAP : moyenne des contributions absolues au prix estimé). "
               "Plus la barre est longue, plus la variable pèse.")
    shap_data = charger_json(SHAP_PATH)
    if shap_data and shap_data.get("importances"):
        imp = pd.DataFrame(shap_data["importances"])
        imp["label"] = imp["feature"].map(LABELS_FEATURES).fillna(imp["feature"])
        imp = imp.sort_values("pct")  # ascendant -> plus important en haut du barh
        fig = go.Figure(go.Bar(
            x=imp["pct"], y=imp["label"], orientation="h",
            marker=dict(color=C_GAIN),
            hovertemplate="%{y} : %{x:.1f} %<extra></extra>", showlegend=False))
        fig.update_layout(title="Poids de chaque variable dans le prix estimé")
        fig.update_xaxes(title="Part de l'importance totale (%)")
        st.plotly_chart(style_figure(fig, 420), width="stretch")

        top3 = imp.sort_values("pct", ascending=False).head(3)["label"].tolist()
        st.caption(f"Le prix est surtout piloté par : **{', '.join(top3)}**. "
                   "Marque et modèle pèsent étonnamment peu — le modèle s'appuie "
                   "davantage sur le millésime, l'équipement et la motorisation que sur le badge. "
                   "(Le champ d'état réel `Etat_Vehicule` n'est pas rempli par les scrapers, donc non utilisé.)")
        with st.expander("ℹ️ Comment lire ces valeurs"):
            st.write(
                f"Modèle : `{nom_modele_court(shap_data.get('modele', '?'))}`, échantillon de "
                f"{shap_data.get('n_lignes', '?')} annonces. Échelle : "
                f"{shap_data.get('echelle', '')}. Les valeurs SHAP mesurent la "
                "contribution moyenne (en valeur absolue) de chaque variable à "
                "l'écart de prix prédit, normalisées ici en % du total."
            )
    else:
        st.info(f"`{SHAP_PATH}` introuvable — relance `python core/modele_prediction.py` "
                "(avec `shap` installé) pour générer l'interprétation.")

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

    st.divider()

    # ---- Santé du scraping : volume par source (et par jour) -------------
    st.subheader("Santé du scraping — volume par source")
    st.caption("Annonces détectées par source. Un effondrement soudain sur une source = "
               "scraper probablement cassé (changement de structure du site, anti-bot…).")
    try:
        suivi_sc = lire_csv("data/processed/suivi_annonces.csv")
    except Exception:
        suivi_sc = None
    if suivi_sc is not None and {"Source", "Premiere_Vue"}.issubset(suivi_sc.columns):
        s = suivi_sc.copy()
        s["_jour"] = pd.to_datetime(s["Premiere_Vue"], errors="coerce").dt.date
        s = s.dropna(subset=["_jour"])
        if s["_jour"].nunique() >= 2:
            par_jour = s.groupby(["_jour", "Source"], observed=True).size().reset_index(name="n")
            fig = px.line(par_jour, x="_jour", y="n", color="Source", markers=True,
                          title="Annonces détectées par source et par jour",
                          labels={"_jour": "", "n": "Annonces détectées", "Source": "Source"})
            st.plotly_chart(style_figure(fig, 320), width="stretch")
        else:
            par_src = (s.groupby("Source", observed=True).size()
                       .reset_index(name="n").sort_values("n", ascending=False))
            fig = px.bar(par_src, x="Source", y="n", title="Annonces par source (collecte du jour)",
                         labels={"Source": "", "n": "Annonces"})
            fig.update_traces(marker_color=C_GAIN)
            st.plotly_chart(style_figure(fig, 300), width="stretch")
            st.caption(f"Une seule journée d'historique — ce graphe deviendra une courbe temporelle "
                       f"par source dès plusieurs jours de collecte. Total : {len(s)} annonces sur "
                       f"{s['Source'].nunique()} sources.")
    else:
        st.info("`suivi_annonces.csv` indisponible ou sans colonnes Source / Premiere_Vue.")

    if diag:
        with st.expander("Variables utilisées par le modèle"):
            st.write("**Numériques** : " + ", ".join(f"`{f}`" for f in diag["features_numeriques"]))
            st.write("**Catégorielles** : " + ", ".join(f"`{f}`" for f in diag["features_categorielles"]))
