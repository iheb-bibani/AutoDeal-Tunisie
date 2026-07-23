"""
logger.py
Système de logging centralisé et cohérent pour tout le projet
"""

import logging
import logging.config
from pathlib import Path
from config import LOGGING_CONFIG


def configure_logging():
    """Configure le logging selon LOGGING_CONFIG"""
    logging.config.dictConfig(LOGGING_CONFIG)


def get_logger(name: str) -> logging.Logger:
    """Retourne un logger configuré avec le nom donné"""
    configure_logging()
    return logging.getLogger(name)


class LoggerMixin:
    """Mixin pour ajouter du logging automatique à une classe"""

    @property
    def logger(self) -> logging.Logger:
        """Retourne le logger de la classe"""
        if not hasattr(self, "_logger"):
            self._logger = get_logger(self.__class__.__name__)
        return self._logger


# Configuration automatique au import
configure_logging()
