"""
verifier_sync.py — à lancer À LA RACINE de ton repo : `python verifier_sync.py`
Compare ton dépôt local à la version complète attendue et liste ce qui manque.
Ne modifie rien, ne fait que lire.
"""
import os
import re


def lire(chemin):
    try:
        with open(chemin, encoding="utf-8") as f:
            return f.read()
    except Exception:
        return ""


def nb_features_num():
    t = lire("config.py")
    m = re.search(r"FEATURES_NUMERIQUES\s*=\s*\[(.*?)\]", t, re.S)
    return len(re.findall(r'"[^"]+"', m.group(1))) if m else -1


CHECKS = [
    ("config.py : 10 features réduites (6 numériques)", lambda: nb_features_num() == 6,
     "config.py contient encore l'ancienne liste (15 features)."),
    ("core/detect_deals.py : règle d'exclusion", lambda: "MOTIFS_EXCLUSION" in lire("core/detect_deals.py"),
     "detect_deals.py n'a pas la règle d'exclusion (accidentée/pièces/HS)."),
    ("app.py : page Recherche", lambda: "def page_recherche" in lire("app.py"),
     "app.py n'a pas la page Recherche."),
    ("app.py : nom court du modèle (HGB)", lambda: "nom_modele_court" in lire("app.py"),
     "app.py affiche encore le nom long du modèle."),
    ("core/enrichir_base_avance.py : drapeaux d'état", lambda: "deriver_drapeaux_etat" in lire("core/enrichir_base_avance.py"),
     "enrichir n'a pas la dérivation des drapeaux d'état."),
    ("core/analyser_validation.py : validation stratifiée (Cox)",
     lambda: "CoxPHFitter" in lire("core/analyser_validation.py") or "stratifié par tranche" in lire("core/analyser_validation.py"),
     "analyser_validation utilise encore le Mann-Whitney brut (non stratifié)."),
    ("core/modele_prediction.py : calcul SHAP intégré", lambda: "calculer_importance_shap" in lire("core/modele_prediction.py"),
     "modele_prediction ne génère pas l'importance SHAP."),
    ("requirements.txt : shap", lambda: bool(re.search(r"^shap", lire("requirements.txt"), re.M)),
     "requirements.txt n'a pas la ligne 'shap>=0.44' -> SHAP jamais installé."),
    ("requirements.txt : scikit-learn 1.7.2", lambda: "scikit-learn==1.7.2" in lire("requirements.txt"),
     "requirements.txt n'est pas figé sur scikit-learn 1.7.2."),
    ("data/processed/shap_importance.json présent", lambda: os.path.exists("data/processed/shap_importance.json"),
     "shap_importance.json manquant -> relance modele_prediction.py avec shap installé."),
    ("tests/test_exclusion.py", lambda: os.path.exists("tests/test_exclusion.py"), "fichier de test manquant."),
    ("tests/test_drapeaux_etat.py", lambda: os.path.exists("tests/test_drapeaux_etat.py"), "fichier de test manquant."),
    ("tests/test_validation.py", lambda: os.path.exists("tests/test_validation.py"), "fichier de test manquant."),
    ("README.md exhaustif (table des matières)", lambda: "Table des matières" in lire("README.md"),
     "README.md n'est pas la version exhaustive."),
    ("LICENSE présent", lambda: os.path.exists("LICENSE"), "fichier LICENSE manquant (référencé par le README)."),
]


def main():
    print("=" * 60)
    print("Vérification du sync AutoDeal-Tunisie")
    print("=" * 60)
    manquants = []
    for libelle, test, aide in CHECKS:
        try:
            ok = bool(test())
        except Exception:
            ok = False
        print(f"  {'✅' if ok else '❌'}  {libelle}")
        if not ok:
            manquants.append((libelle, aide))
    print("-" * 60)
    if not manquants:
        print("🎉 Tout est là. Ton repo est à jour avec la version complète.")
    else:
        print(f"⚠️  {len(manquants)} élément(s) manquant(s) ou en retard :\n")
        for libelle, aide in manquants:
            print(f"   • {libelle}\n     -> {aide}")
        print("\nLe plus simple : recopie tout le contenu du zip 'AutoDeal-Tunisie-complet'")
        print("par-dessus ce dossier, puis relance :")
        print("   pip install -r requirements.txt")
        print("   python core/modele_prediction.py")
        print("   python core/detect_deals.py")
        print("   python -m pytest tests/ -q      (doit afficher 66 passed)")


if __name__ == "__main__":
    main()