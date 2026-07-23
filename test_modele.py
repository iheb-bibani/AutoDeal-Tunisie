"""
test_modele.py
Script de diagnostic pour tester et dépanner le modèle sauvegardé
"""

import os
import sys
import joblib
import pandas as pd
import numpy as np
from logger import get_logger
from config import PROCESSED_FILES, MODEL_PATH, FEATURES_NUMERIQUES, FEATURES_CATEGORIELLES

logger = get_logger(__name__)


def tester_modele():
    """Teste le modèle sauvegardé"""
    
    print("\n" + "="*60)
    print("🧪 TEST DU MODÈLE")
    print("="*60)
    
    # 1. Vérifier que le modèle existe
    if not os.path.exists(MODEL_PATH):
        logger.error(f"❌ Modèle introuvable: {MODEL_PATH}")
        return False
    
    logger.info(f"✅ Modèle trouvé: {MODEL_PATH}")
    
    # 2. Charger le modèle
    try:
        modele_bundle = joblib.load(MODEL_PATH)
        logger.info("✅ Modèle chargé avec succès")
    except Exception as e:
        logger.error(f"❌ Erreur lors du chargement: {e}")
        return False
    
    # 3. Vérifier la structure
    if not isinstance(modele_bundle, dict):
        logger.error("❌ Le modèle n'est pas un dictionnaire")
        return False
    
    logger.info("✅ Structure correcte (dictionnaire)")
    
    # 4. Vérifier les clés requises
    cles_requises = ["pipeline", "features_numeriques", "features_categorielles"]
    for cle in cles_requises:
        if cle not in modele_bundle:
            logger.error(f"❌ Clé manquante: {cle}")
            return False
    
    logger.info(f"✅ Toutes les clés requises présentes: {cles_requises}")
    
    # 5. Vérifier les features
    features_num = modele_bundle["features_numeriques"]
    features_cat = modele_bundle["features_categorielles"]
    
    logger.info(f"✅ Features numériques: {features_num}")
    logger.info(f"✅ Features catégoriques: {features_cat}")
    
    # 6. Essayer une prédiction test
    logger.info("\n🔬 Test de prédiction...")
    
    try:
        # Créer des données test
        test_data = {
            "Kilométrage": 80000,
            "Année": 2020,
            "Puissance_Fiscale": 7,
            "Segment_Vehicule": 0,
            "Est_Presque_Neuve": 0,
            "Zone_Economique": 0,
            "Marque": "Peugeot",
            "Modèle": "3008",
            "Boite_Vitesse": "Automatique",
            "Energie": "Diesel",
        }
        
        colonnes = features_num + features_cat
        X = pd.DataFrame([{c: test_data.get(c) for c in colonnes}])
        
        logger.info(f"Données test: {X.to_dict('records')[0]}")
        
        # Prédiction
        prix_log = modele_bundle["pipeline"].predict(X)[0]
        prix_theorique = np.expm1(prix_log)
        
        logger.info(f"✅ Prédiction réussie: {prix_theorique:,.0f} DT")
        
    except AttributeError as e:
        logger.error(f"❌ Erreur AttributeError (incompatibilité sklearn): {e}")
        logger.warning("💡 Le modèle a été entraîné avec une version différente de scikit-learn")
        return False
        
    except Exception as e:
        logger.error(f"❌ Erreur lors de la prédiction: {e}")
        return False
    
    print("\n" + "="*60)
    print("✅ TOUS LES TESTS RÉUSSIS!")
    print("="*60)
    return True


def verifier_donnees_entrainement():
    """Vérifie que les données d'entraînement existent"""
    
    print("\n" + "="*60)
    print("📊 VÉRIFICATION DES DONNÉES")
    print("="*60)
    
    fichier_enrichi = PROCESSED_FILES["enriched"]
    
    if not os.path.exists(fichier_enrichi):
        logger.error(f"❌ Fichier d'entraînement introuvable: {fichier_enrichi}")
        logger.info("💡 Lance d'abord: python core/merging_files.py && python core/nettoyer_base.py && python core/enrichir_base_avance.py")
        return False
    
    try:
        df = pd.read_csv(fichier_enrichi, sep=";", encoding="utf-8-sig")
        logger.info(f"✅ Données chargées: {len(df)} annonces")
        
        # Vérifier les colonnes
        colonnes_attendues = FEATURES_NUMERIQUES + FEATURES_CATEGORIELLES + ["Prix"]
        manquantes = set(colonnes_attendues) - set(df.columns)
        
        if manquantes:
            logger.error(f"❌ Colonnes manquantes: {manquantes}")
            return False
        
        logger.info(f"✅ Toutes les colonnes présentes")
        
        # Vérifier les valeurs nulles
        nulls = df[colonnes_attendues].isna().sum()
        if (nulls > 0).any():
            logger.warning(f"⚠️ Valeurs nulles détectées:\n{nulls[nulls > 0]}")
        
        logger.info(f"✅ Données valides pour réentraînement")
        return True
        
    except Exception as e:
        logger.error(f"❌ Erreur lors de la vérification: {e}")
        return False


def recommander_action():
    """Recommande une action basée sur les tests"""
    
    print("\n" + "="*60)
    print("💡 RECOMMANDATIONS")
    print("="*60)
    
    # Test du modèle
    modele_ok = tester_modele()
    
    # Test des données
    donnees_ok = verifier_donnees_entrainement()
    
    if modele_ok and donnees_ok:
        print("\n✅ Tout semble OK!")
        print("   Si vous avez toujours une erreur dans l'app Streamlit,")
        print("   c'est peut-être une incompatibilité d'environnement.")
        print("\n💡 Actions recommandées:")
        print("   1. Vérifiez votre version de scikit-learn: pip list | grep scikit")
        print("   2. Réentraînez le modèle: python core/modele_prediction.py")
        print("   3. Relancez Streamlit: streamlit run app.py")
        
    elif not donnees_ok:
        print("\n⚠️ Les données d'entraînement sont manquantes!")
        print("\n💡 Actions requises:")
        print("   1. Lance le pipeline complet:")
        print("      python core/merging_files.py")
        print("      python core/nettoyer_base.py")
        print("      python core/enrichir_base_avance.py")
        print("   2. Puis réentraîne le modèle:")
        print("      python core/modele_prediction.py")
        
    elif not modele_ok:
        print("\n❌ Le modèle a un problème!")
        print("\n💡 Actions requises:")
        print("   1. Assurez-vous que les données existent:")
        print("      python core/enrichir_base_avance.py")
        print("   2. Réentraîne le modèle:")
        print("      python core/modele_prediction.py")
        print("   3. Si ça ne fonctionne pas, recréez l'environnement:")
        print("      pip install --upgrade scikit-learn")
        print("      python core/modele_prediction.py")


if __name__ == "__main__":
    recommander_action()
