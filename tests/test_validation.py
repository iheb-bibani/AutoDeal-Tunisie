"""
Tests de core/analyser_validation.analyse_longitudinale (version stratifiée) :
- avec peu de disparitions -> message d'attente, pas de crash ;
- avec un signal fort (opportunités qui partent 4x plus vite), à prix contrôlé
  -> la méthode le détecte (Cox si lifelines présent, sinon Mann-Whitney
  stratifié via scipy).
"""
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from core.analyser_validation import analyse_longitudinale


def _suivi_synthetique(n_par_groupe, base_deal, base_autre, seed=0):
    """Construit un suivi simulé sur 4 tranches de prix. Les opportunités
    s'écoulent plus vite (base_deal < base_autre)."""
    rng = np.random.default_rng(seed)
    today = datetime.now()
    lignes = []
    for pm in (18000, 38000, 75000, 140000):
        for deal in (1, 0):
            base = base_deal if deal else base_autre
            for _ in range(n_par_groupe):
                dur = max(1, int(rng.exponential(base)))
                depose = today - timedelta(days=int(rng.uniform(5, 60)))
                disparue = dur <= (today - depose).days
                lignes.append({
                    "Premiere_Vue": depose.strftime("%Y-%m-%d"),
                    "Statut": "Disparue" if disparue else "Active",
                    "Jours_En_Ligne": dur if disparue else np.nan,
                    "Etait_Opportunite": bool(deal),
                    "Prix_Dernier": pm * rng.uniform(0.85, 1.15),
                })
    return pd.DataFrame(lignes)


def test_peu_de_disparitions_message_attente(capsys):
    # base très grande + peu de lignes -> quasi aucune disparition
    suivi = _suivi_synthetique(n_par_groupe=3, base_deal=500, base_autre=500)
    analyse_longitudinale(suivi)
    out = capsys.readouterr().out
    assert "Pas encore assez de disparitions" in out


def test_signal_fort_detecte(capsys):
    # opportunités 4x plus rapides, gros échantillon -> doit conclure au signal
    suivi = _suivi_synthetique(n_par_groupe=60, base_deal=10, base_autre=40)
    analyse_longitudinale(suivi)
    out = capsys.readouterr().out
    assert "Disparitions observées" in out
    assert "✅" in out  # Cox ou Mann-Whitney stratifié : signal détecté


def test_ne_crashe_pas_sans_colonne_opportunite(capsys):
    analyse_longitudinale(pd.DataFrame({"Premiere_Vue": ["2026-01-01"]}))
    out = capsys.readouterr().out
    assert "Etait_Opportunite absente" in out