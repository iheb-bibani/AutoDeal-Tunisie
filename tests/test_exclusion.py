"""Tests des règles métier de détection des deals."""
import pandas as pd

from core.detect_deals import annonces_exclues, sources_trop_risquees


def _exclu(titre):
    return bool(annonces_exclues(pd.Series([titre])).iloc[0])


def test_exclut_accidentee():
    for t in ["Voiture accidentée", "BMW accidenté", "VOITURE ACCIDENTEE"]:
        assert _exclu(t), t


def test_exclut_pieces():
    for t in ["Golf pour pièces", "voiture a piece", "pieces détachées"]:
        assert _exclu(t), t


def test_exclut_moteur_hs():
    for t in ["Clio moteur HS", "boite hs", "moteur à refaire"]:
        assert _exclu(t), t


def test_exclut_sans_papiers_export_et_non_dedouanee():
    for t in [
        "voiture sans papiers",
        "non dédouanée",
        "pas dedouanee",
        "sans dédouanement",
        "voiture pour export",
    ]:
        assert _exclu(t), t


def test_ne_exclut_pas_voiture_dedouanee():
    for t in ["Voiture dédouanée", "BMW importée et dedouanee", "Dédouanée Tunisie"]:
        assert not _exclu(t), t


def test_ne_exclut_pas_annonce_saine():
    for t in [
        "Toyota Agya",
        "Volkswagen Golf 7",
        "Peugeot 208 première main",
        "Mercedes Classe C toutes options",
    ]:
        assert not _exclu(t), t


def test_insensible_accents_casse():
    assert _exclu("ACCIDENTÉE") and _exclu("accidentee") and _exclu("Accidentée")


def test_serie_et_valeurs_manquantes():
    s = pd.Series(["Voiture accidentée", None, "Toyota Yaris"], index=[5, 6, 7])
    out = annonces_exclues(s)
    assert list(out.index) == [5, 6, 7]
    assert out.loc[5] and not out.loc[6] and not out.loc[7]


def test_source_holdout_trop_faible_bloque_notification():
    diagnostics = {
        "validation_robuste": {
            "source_holdout": [
                {"source": "source-ok", "n_test": 100, "mdape": 0.12},
                {"source": "source-risquee", "n_test": 100, "mdape": 0.27},
                # Échantillon trop petit : pas de quarantaine automatique.
                {"source": "source-petite", "n_test": 10, "mdape": 0.40},
            ]
        }
    }
    assert sources_trop_risquees(diagnostics) == {"source-risquee"}
