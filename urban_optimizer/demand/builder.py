"""Façade publique : ``generate_od_matrix(network, …)`` orchestre la brique 2."""

from __future__ import annotations

from urban_optimizer.network.urban_network import UrbanNetwork
from urban_optimizer.utils.logging import get_logger

from .gravity import gravity_od
from .od_matrix import ODMatrix
from .zoning import Zoning, build_grid_zoning, build_h3_zoning, build_iris_zoning

logger = get_logger(__name__)


def generate_od_matrix(
    network: UrbanNetwork,
    hour: int = 8,
    scenario: str = "weekday",
    method: str = "grid",
    *,
    n_cells: int = 20,
    h3_resolution: int = 8,
    insee_codes_dept: list[str] | None = None,
    beta: float | None = None,
    scale_factor: float = 1.0,
) -> ODMatrix:
    """Pipeline complet de la brique 2 : zonage → demande gravitaire.

    Args:
        network: réseau urbain construit par la brique 1.
        hour: heure ciblée (0-23).
        scenario: étiquette de scénario (libre, propagée dans ODMatrix).
        method: "grid", "iris" ou "h3".
        n_cells: nombre de cellules par côté pour la grille.
        h3_resolution: résolution H3 (8 ≈ hexagone de 0.7 km²).
        insee_codes_dept: codes département à charger (méthode "iris" uniquement).
        beta: paramètre de friction du modèle gravitaire (1/m).
        scale_factor: homothétie globale de la demande.

    Returns:
        ODMatrix calibrée prête pour la brique 3 (affectation).
    """
    logger.info(f"=== Brique 2 : génération OD ({method}, h={hour}) ===")

    zoning = _build_zoning(
        network,
        method=method,
        n_cells=n_cells,
        h3_resolution=h3_resolution,
        insee_codes_dept=insee_codes_dept,
    )
    zoning.summary()

    od = gravity_od(
        zoning,
        hour=hour,
        scenario=scenario,
        beta=beta,
        scale_factor=scale_factor,
    )
    od.summary()

    logger.info("=== Brique 2 terminée ===")
    return od


def _build_zoning(
    network: UrbanNetwork,
    method: str,
    n_cells: int,
    h3_resolution: int,
    insee_codes_dept: list[str] | None,
) -> Zoning:
    if method == "grid":
        return build_grid_zoning(network, n_cells=n_cells)
    if method == "iris":
        return build_iris_zoning(network, insee_codes_dept=insee_codes_dept)
    if method == "h3":
        return build_h3_zoning(network, resolution=h3_resolution)
    raise ValueError(f"Méthode de zonage inconnue : {method!r} (attendu : grid|iris|h3)")
