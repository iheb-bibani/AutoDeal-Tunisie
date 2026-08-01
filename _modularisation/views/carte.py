"""Page carte."""
import streamlit as st
import plotly.express as px

from ui.charts import explication
from services.analytics_service import (_match_gouvernorat, GOUVERNORATS_COORD)


def page_carte(df):
    st.title("🗺️ Carte du marché")
    st.caption("Le marché par gouvernorat : où les prix sont hauts, où se concentrent "
               "les opportunités. Croise avec l'arbitrage géographique de la page Samsar.")
    if df is None or df.empty or "Localisation" not in df.columns:
        st.info("Données de localisation indisponibles.")
        return

    metrique = st.radio("Colorer par", ["Prix médian", "Densité d'opportunités"],
                        horizontal=True)

    d = df.dropna(subset=["Localisation", "Prix"]).copy()
    d["_gouv"] = d["Localisation"].map(_match_gouvernorat)
    d = d.dropna(subset=["_gouv"])
    if d.empty:
        st.info("Aucune localisation n'a pu être rattachée à un gouvernorat.")
        return

    agg = d.groupby("_gouv").agg(
        n=("Prix", "size"),
        prix_median=("Prix", "median"),
        n_deals=("Score_Opportunite", lambda s: int(s.between(0.25, 0.55).sum())),
    ).reset_index()
    agg = agg[agg["n"] >= 5]
    agg["taux_deals"] = (100 * agg["n_deals"] / agg["n"]).round(1)
    agg["lat"] = agg["_gouv"].map(lambda g: GOUVERNORATS_COORD[g][0])
    agg["lon"] = agg["_gouv"].map(lambda g: GOUVERNORATS_COORD[g][1])

    couleur = "prix_median" if metrique.startswith("Prix") else "taux_deals"
    labels = {"prix_median": "Prix médian (DT)", "taux_deals": "% d'opportunités",
              "n": "Annonces", "_gouv": "Gouvernorat"}
    fig = px.scatter_mapbox(
        agg, lat="lat", lon="lon", size="n", color=couleur,
        hover_name="_gouv",
        hover_data={"n": True, "prix_median": ":,.0f", "taux_deals": True,
                    "lat": False, "lon": False},
        color_continuous_scale="Turbo", size_max=45, zoom=5.1,
        center={"lat": 34.5, "lon": 9.6}, labels=labels, height=560,
    )
    fig.update_layout(mapbox_style="open-street-map",
                      margin={"l": 0, "r": 0, "t": 10, "b": 0})
    st.plotly_chart(fig, width="stretch")

    if metrique.startswith("Prix"):
        cher = agg.loc[agg["prix_median"].idxmax()]
        pas_cher = agg.loc[agg["prix_median"].idxmin()]
        st.caption(f"Le plus cher : **{cher['_gouv']}** ({cher['prix_median']:,.0f} DT médian). "
                   f"Le moins cher : **{pas_cher['_gouv']}** ({pas_cher['prix_median']:,.0f} DT). "
                   "L'écart, c'est le potentiel d'arbitrage géographique.".replace(",", " "))
    else:
        top = agg.loc[agg["taux_deals"].idxmax()]
        st.caption(f"Plus forte densité d'opportunités : **{top['_gouv']}** "
                   f"({top['taux_deals']:.0f} % des annonces sous le prix estimé). "
                   "C'est là qu'il y a le plus à chasser.")

    explication(
        "Comment lire la carte",
        "**Ce que ça montre.** Le marché par gouvernorat. Chaque bulle = un gouvernorat ; sa "
        "**taille** = le nombre d'annonces, sa **couleur** = la métrique choisie (prix médian ou "
        "densité d'opportunités).\n\n"
        "**Comment c'est calculé.** Les localisations libres sont rattachées à leur gouvernorat, "
        "puis agrégées (prix médian, % d'annonces sous le prix estimé). Seuls les gouvernorats à "
        "≥ 5 annonces sont affichés.\n\n"
        "**Comment décider.** En mode *prix médian*, l'écart entre zones chères (Grand Tunis) et "
        "bon marché (intérieur) est le potentiel d'arbitrage géographique. En mode *densité "
        "d'opportunités*, c'est là où chasser. C'est une vue à l'échelle du gouvernorat (pas de "
        "coordonnées plus fines disponibles)."
    )
