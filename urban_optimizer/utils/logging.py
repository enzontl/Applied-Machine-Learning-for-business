"""Logger centralisé pour tout le projet."""

import logging
import sys


def get_logger(name: str = "urban_optimizer", level: int = logging.INFO) -> logging.Logger:
    """Renvoie un logger configuré de manière cohérente.

    Args:
        name: nom du logger (typiquement __name__)
        level: niveau de log (logging.DEBUG, INFO, WARNING, ERROR)

    Returns:
        Logger prêt à être utilisé.
    """
    logger = logging.getLogger(name)

    if not logger.handlers:
        logger.setLevel(level)
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "%(asctime)s | %(name)s | %(levelname)s | %(message)s",
            datefmt="%H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.propagate = False

    return logger
