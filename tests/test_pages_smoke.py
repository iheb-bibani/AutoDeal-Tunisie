"""Smoke test : chaque page Streamlit s'affiche sans exception sur les données locales.

Aurait détecté la page Carte cassée par plotly 7 (scatter_mapbox retiré).
"""
import pytest

pytest.importorskip("streamlit.testing.v1")
from streamlit.testing.v1 import AppTest  # noqa: E402

PAGES = {
    "page_accueil": "(df,)", "page_acheter": "(df,)", "page_comparateur": "(df,)",
    "page_sante_marche": "(df,)", "page_historique": "(df,)", "page_alertes": "(df, deals)",
    "page_tarifs": "(df,)", "page_calculateur": "(df, modele)", "page_assistant": "(df, modele)",
    "page_marche": "(df,)", "page_recherche": "(df,)", "page_carte": "(df,)",
    "page_samsar": "(df, deals)", "page_admin": "(df, deals)",
}


@pytest.mark.parametrize("page,args", PAGES.items())
def test_page_s_affiche_sans_exception(page, args):
    code = f"""
import views.legacy_dashboard as L
L.LIRE_DEPUIS_GITHUB = False
df, deals, modele = L.charger_scored(), L.charger_deals(), L.charger_modele()
L.{page}(*{args})
"""
    at = AppTest.from_string(code, default_timeout=180).run()
    assert not at.exception, str(at.exception[0].value)[:500]
