"""Modèle gravitaire doublement contraint pour générer une matrice OD."""

from __future__ import annotations

import numpy as np

from urban_optimizer.config import GRAVITY_BETA, HOURLY_DEMAND_SHARE
from urban_optimizer.utils.logging import get_logger

from .od_matrix import ODMatrix
from .zoning import Zoning

logger = get_logger(__name__)


def _euclidean_cost_matrix(centroids: np.ndarray) -> np.ndarray:
    """Coût d'impédance entre centroïdes = distance euclidienne (m).

    Pour passer à un coût en temps libre, remplacer par la matrice des plus
    courts chemins sur le réseau — plus coûteux mais plus juste.
    """
    diff = centroids[:, None, :] - centroids[None, :, :]
    return np.sqrt((diff ** 2).sum(axis=-1))


def _furness_balancing(
    seed: np.ndarray,
    o_targets: np.ndarray,
    d_targets: np.ndarray,
    max_iter: int = 50,
    tol: float = 1e-4,
) -> np.ndarray:
    """Furness (IPF) — équilibre simultanément lignes et colonnes.

    À chaque itération, on rééchelonne les lignes pour viser ``o_targets`` puis
    les colonnes pour viser ``d_targets``. Converge en 5-20 itérations.
    """
    T = seed.copy().astype(float)
    np.fill_diagonal(T, 0.0)

    for it in range(max_iter):
        row_sum = T.sum(axis=1)
        safe_row = np.where(row_sum > 0, row_sum, 1.0)
        row_factors = np.where(row_sum > 0, o_targets / safe_row, 1.0)
        T = T * row_factors[:, None]

        col_sum = T.sum(axis=0)
        safe_col = np.where(col_sum > 0, col_sum, 1.0)
        col_factors = np.where(col_sum > 0, d_targets / safe_col, 1.0)
        T = T * col_factors[None, :]

        err = max(
            np.abs(T.sum(axis=1) - o_targets).max(),
            np.abs(T.sum(axis=0) - d_targets).max(),
        )
        if err < tol:
            logger.debug(f"Furness convergé en {it + 1} itérations (err={err:.2e})")
            break

    return T


def gravity_od(
    zoning: Zoning,
    hour: int = 8,
    scenario: str = "weekday",
    beta: float | None = None,
    trips_per_capita_day: float = 3.5,
    scale_factor: float = 1.0,
    balance: bool = True,
) -> ODMatrix:
    """Construit une matrice OD horaire par modèle gravitaire.

    Formule de base :
        T_ij ∝ O_i · D_j · exp(-β · c_ij)

    avec :
      - O_i ∝ population de la zone i (origines, contraint par la trip-rate)
      - D_j ∝ emplois de la zone j (destinations)
      - c_ij : distance euclidienne entre centroïdes (m)
      - β : paramètre de décroissance (1/m), calibré ville-par-ville

    Args:
        zoning: zonage avec ``population`` et ``jobs`` par zone.
        hour: heure à laquelle on calibre la matrice (0-23).
        scenario: scénario nommé (libre, propagé dans ODMatrix).
        beta: paramètre du modèle, exprimé en 1/m ; défaut = config.GRAVITY_BETA / 1000.
        trips_per_capita_day: nombre moyen de déplacements par habitant et par jour.
        scale_factor: multiplicateur global (homothétie de la demande).
        balance: si True, équilibre lignes/colonnes par Furness (recommandé).

    Returns:
        ODMatrix calibrée pour l'heure indiquée.
    """
    n = zoning.n_zones
    if n < 2:
        raise ValueError("Zonage trop petit pour générer une OD (< 2 zones).")

    centroids = np.array([zoning.centroids_xy[z] for z in zoning.zone_ids])
    pop = zoning.gdf["population"].to_numpy(dtype=float)
    jobs = zoning.gdf["jobs"].to_numpy(dtype=float)

    if pop.sum() <= 0:
        raise ValueError("Population totale nulle — zonage non exploitable.")
    if jobs.sum() <= 0:
        logger.warning("Emplois totaux nuls — utilisation de la population comme proxy.")
        jobs = pop.copy()

    # Conversion population → déplacements horaires
    hour_share = HOURLY_DEMAND_SHARE.get(hour, 0.05)
    o_targets = pop * trips_per_capita_day * hour_share * scale_factor

    # Destinations re-normalisées pour que sum(D) = sum(O)
    d_targets = jobs * (o_targets.sum() / jobs.sum())

    # Friction
    if beta is None:
        # config.GRAVITY_BETA est exprimé en 1/km ; on convertit en 1/m
        beta_m = GRAVITY_BETA / 1000.0
    else:
        beta_m = float(beta)

    cost = _euclidean_cost_matrix(centroids)
    friction = np.exp(-beta_m * cost)
    np.fill_diagonal(friction, 0.0)

    seed = np.outer(o_targets, d_targets) * friction

    if balance:
        T = _furness_balancing(seed, o_targets, d_targets)
    else:
        # Normalise simplement pour égaler le total des origines
        total = seed.sum()
        T = seed * (o_targets.sum() / total) if total > 0 else seed

    np.fill_diagonal(T, 0.0)
    T = np.maximum(T, 0.0)

    logger.info(
        f"OD gravitaire — zones: {n}, h={hour}, scénario={scenario}, "
        f"total: {T.sum():,.0f} véh/h, β={beta_m:.2e} 1/m"
    )

    return ODMatrix(
        matrix=T,
        zone_ids=list(zoning.zone_ids),
        zone_to_node=dict(zoning.zone_to_node),
        hour=hour,
        scenario=scenario,
    )
