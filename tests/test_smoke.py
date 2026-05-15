"""Tests de fumée — vérifient que le package s'importe correctement."""


def test_import_package():
    """Le package se charge sans erreur."""
    import urban_optimizer

    assert hasattr(urban_optimizer, "__version__")
    assert urban_optimizer.__version__ == "0.1.0"


def test_import_subpackages():
    """Tous les sous-packages se chargent."""
    from urban_optimizer import (
        assignment,
        demand,
        diagnosis,
        network,
        optimization,
        utils,
        viz,
    )

    assert all([assignment, demand, diagnosis, network, optimization, utils, viz])


def test_config_paths():
    """Les chemins de la config existent."""
    from urban_optimizer import config

    assert config.PROJECT_ROOT.exists()
    assert config.DATA_DIR.exists()


def test_logger():
    """Le logger se crée et n'a pas de handler dupliqué."""
    from urban_optimizer.utils.logging import get_logger

    logger1 = get_logger("test_logger")
    logger2 = get_logger("test_logger")

    assert logger1 is logger2
    assert len(logger1.handlers) == 1
