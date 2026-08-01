"""Rend la racine du projet importable pour les tests (`import config`,
`from core... import ...`) quel que soit le répertoire d'où pytest est lancé."""
import sys
from pathlib import Path

RACINE = Path(__file__).parent.resolve()
if str(RACINE) not in sys.path:
    sys.path.insert(0, str(RACINE))
