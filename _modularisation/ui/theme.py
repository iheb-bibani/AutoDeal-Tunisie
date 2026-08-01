"""Thème visuel : palette de couleurs + configuration de page (set_page_config + CSS)."""
import streamlit as st

C_ENCRE = "#15232E"      # texte, barres neutres
C_ASPHALTE = "#2C3E50"   # barres principales
C_GAIN = "#0E9F6E"       # vert "gain" -- opportunités, hausses
C_ALERTE = "#D9480F"     # orange brûlé -- points d'attention
C_SABLE = "#C9A227"      # accent secondaire (or/sable)
C_GRIS = "#8A97A3"
SEQ_CATEGORIELLE = [C_ASPHALTE, C_GAIN, C_SABLE, C_ALERTE, "#5C7A99", "#A3B2BF"]


def configurer_page():
    """À appeler en TOUT PREMIER dans app.py : set_page_config + injection du CSS global."""
    st.set_page_config(
        page_title="AutoDeal Tunisie",
        page_icon="🚗",
        layout="wide",
        initial_sidebar_state="expanded",
    )
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

        /* Masquer la barre d'outils + logo Plotly (le petit bloc au coin des graphes) */
        .modebar { display: none !important; }
        .js-plotly-plot .plotly .modebar-container { display: none !important; }

        /* Boutons de navigation (sidebar) : texte foncé sur fond blanc (inactifs) */
        section[data-testid="stSidebar"] .stButton button,
        section[data-testid="stSidebar"] .stButton button p {
            color: #15232E !important;
        }
        /* Bouton actif (primary, vert) : texte blanc */
        section[data-testid="stSidebar"] .stButton button[kind="primary"],
        section[data-testid="stSidebar"] .stButton button[kind="primary"] p {
            color: #FFFFFF !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
