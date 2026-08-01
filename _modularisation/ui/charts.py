"""Helpers de rendu partagés : gabarit Plotly, conteneur d'explication, format DT."""
import streamlit as st
from ui.theme import C_ENCRE, SEQ_CATEGORIELLE

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
def explication(titre, corps):
    """Conteneur d'explication standard sous un graphe : montre → calcule → décide."""
    with st.expander(f"ℹ️ {titre}"):
        st.markdown(corps)
def fmt_dt(v):
    return f"{v:,.0f}".replace(",", " ") + " DT"
