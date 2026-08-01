"""
Tests de core/enrichir_base_avance.deriver_drapeaux_etat : extraction de
drapeaux d'état (1ère main, non accidentée, full options, état d'origine)
depuis le texte libre des annonces, insensible aux accents et à la casse.
"""
import pandas as pd

from core.enrichir_base_avance import deriver_drapeaux_etat


def _flags(texte):
    return deriver_drapeaux_etat(pd.Series([texte])).iloc[0].to_dict()


def test_premiere_main_variantes():
    for t in ["1ère main", "1ere main", "première main", "1 ere main", "1er main"]:
        assert _flags(t)["Premiere_Main"] == 1, t


def test_non_accidentee_variantes():
    for t in ["jamais accidentée", "non accidenté", "pas accidentée", "aucun accident"]:
        assert _flags(t)["Non_Accidentee"] == 1, t


def test_full_options_variantes():
    for t in ["full option", "toutes options", "ttes options", "full options"]:
        assert _flags(t)["Full_Options"] == 1, t


def test_etat_origine_variantes():
    for t in ["état d'origine", "peinture d'origine", "tout d'origine", "etat d origine"]:
        assert _flags(t)["Etat_Origine"] == 1, t


def test_titre_nu_tout_a_zero():
    f = _flags("Toyota Agya")
    assert sum(f.values()) == 0


def test_insensible_accents_et_casse():
    a = _flags("PREMIÈRE MAIN JAMAIS ACCIDENTÉE")
    assert a["Premiere_Main"] == 1 and a["Non_Accidentee"] == 1


def test_cumul_plusieurs_drapeaux():
    f = _flags("Peugeot 208 première main, jamais accidentée, full option, état d'origine")
    assert f == {"Premiere_Main": 1, "Non_Accidentee": 1, "Full_Options": 1, "Etat_Origine": 1}


def test_valeurs_manquantes_et_index_preserve():
    s = pd.Series(["1ère main", None, "full option"], index=[10, 20, 30])
    out = deriver_drapeaux_etat(s)
    assert list(out.index) == [10, 20, 30]
    assert out.loc[20].sum() == 0          # NaN -> tous à 0, pas d'erreur
    assert out.loc[10, "Premiere_Main"] == 1
    assert out.loc[30, "Full_Options"] == 1
