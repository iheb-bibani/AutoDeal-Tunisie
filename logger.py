"""
logger.py
Système de logging centralisé et cohérent pour tout le projet
"""

import logging
import logging.config
from config import LOGGING_CONFIG


_CONFIGURED = False


def configure_logging(force: bool = False) -> None:
    """Configure le logging selon LOGGING_CONFIG (une seule fois par process).

    Rappeler dictConfig à chaque get_logger() fermait/rouvrait le FileHandler
    et réinitialisait les handlers à chaque import de module.
    """
    global _CONFIGURED
    if _CONFIGURED and not force:
        return
    logging.config.dictConfig(LOGGING_CONFIG)
    _CONFIGURED = True


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
