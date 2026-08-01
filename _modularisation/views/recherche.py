"""Page recherche."""
import pandas as pd
import streamlit as st



def page_recherche(df):
    st.title("🔎 Recherche")
    st.caption("Filtre tout le marché scoré selon tes critères. Chaque résultat montre "
               "son prix affiché, l'estimation du modèle et l'écart (= l'opportunité).")
    if df is None or df.empty:
        st.info("Données indisponibles.")
        return

    d = df.copy()
    c1, c2, c3 = st.columns(3)
    with c1:
        sel_marques = st.multiselect("Marque", sorted(d["Marque"].dropna().unique()))
        sel_energie = st.multiselect(
            "Énergie", sorted(d["Energie"].dropna().unique()) if "Energie" in d.columns else [])
    with c2:
        sel_lieux = st.multiselect(
            "Région", sorted(d["Localisation"].dropna().unique()) if "Localisation" in d.columns else [])
        sel_boite = st.multiselect(
            "Boîte", sorted(d["Boite_Vitesse"].dropna().unique()) if "Boite_Vitesse" in d.columns else [])
    with c3:
        seulement_deals = st.checkbox("Opportunités uniquement (25–55 % sous le prix)")
        tri = st.selectbox("Trier par",
                           ["Meilleure affaire", "Prix croissant", "Prix décroissant", "Plus récent"])

    s1, s2, s3 = st.columns(3)
    prix_ok = d["Prix"].dropna()
    with s1:
        pmin = int(prix_ok.min()) if len(prix_ok) else 0
        pmax = int(prix_ok.quantile(0.99)) if len(prix_ok) else 500000
        budget = st.slider("Budget (DT)", pmin, max(pmax, pmin + 1000), (pmin, pmax), step=1000)
    with s2:
        an_ok = d["Année"].dropna()
        amin = int(an_ok.min()) if len(an_ok) else 1990
        amax = int(an_ok.max()) if len(an_ok) else pd.Timestamp.now().year
        annee = st.slider("Année", amin, max(amax, amin + 1), (amin, amax))
    with s3:
        km_ok = d["Kilométrage"].dropna()
        kmax = int(km_ok.quantile(0.99)) if len(km_ok) else 500000
        km_max = st.slider("Kilométrage max", 0, max(kmax, 10000), max(kmax, 10000), step=10000)

    if sel_marques:
        d = d[d["Marque"].isin(sel_marques)]
    if sel_energie:
        d = d[d["Energie"].isin(sel_energie)]
    if sel_lieux:
        d = d[d["Localisation"].isin(sel_lieux)]
    if sel_boite:
        d = d[d["Boite_Vitesse"].isin(sel_boite)]
    d = d[d["Prix"].between(budget[0], budget[1])]
    d = d[d["Année"].between(annee[0], annee[1])]
    d = d[d["Kilométrage"].fillna(0) <= km_max]
    if seulement_deals:
        d = d[d["Score_Opportunite"].between(0.25, 0.55)]

    if tri == "Meilleure affaire":
        d = d.sort_values("Score_Opportunite", ascending=False)
    elif tri == "Prix croissant":
        d = d.sort_values("Prix")
    elif tri == "Prix décroissant":
        d = d.sort_values("Prix", ascending=False)
    elif tri == "Plus récent" and "Annonce-Detectee" in d.columns:
        d = d.sort_values("Annonce-Detectee", ascending=False)

    st.markdown(f"**{len(d)} annonces** correspondent" +
                (f" · {min(len(d), 200)} affichées" if len(d) > 200 else "") + ".")

    aff = d.head(200).copy()
    aff["Écart"] = (aff["Score_Opportunite"] * 100).round(0)
    cols = [c for c in ["Marque", "Modèle", "Année", "Kilométrage", "Prix",
                        "Prix_Theorique", "Écart", "Localisation", "Lien"] if c in aff.columns]
    st.dataframe(
        aff[cols], hide_index=True, width="stretch",
        column_config={
            "Prix": st.column_config.NumberColumn("Prix", format="%d DT"),
            "Prix_Theorique": st.column_config.NumberColumn("Estimé", format="%d DT"),
            "Kilométrage": st.column_config.NumberColumn("Km", format="%d"),
            "Écart": st.column_config.NumberColumn("Écart", format="%d %%",
                                                   help="% sous le prix estimé par le modèle"),
            "Lien": st.column_config.LinkColumn("Annonce", display_text="ouvrir"),
        },
    )
