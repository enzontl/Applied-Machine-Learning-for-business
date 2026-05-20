"""Solveur Frank-Wolfe pour User Equilibrium et System Optimum.

UE minimise la somme des intégrales BPR (objectif de Beckmann) — chaque
utilisateur choisit son meilleur chemin.
SO minimise le coût total du système — c'est le même algorithme mais on
remplace la fonction de coût par son **marginal**.
"""

from __future__ import annotations

from typing import Callable

import igraph as ig
import numpy as np

from urban_optimizer.config import FRANK_WOLFE_MAX_ITER, FRANK_WOLFE_TOLERANCE
from urban_optimizer.demand.od_matrix import ODMatrix
from urban_optimizer.network.urban_network import UrbanNetwork
from urban_optimizer.utils.logging import get_logger

from .aon import assign_aon
from .bpr import beckmann_objective, bpr_marginal_time, bpr_time
from .result import AssignmentResult

logger = get_logger(__name__)


# ────────────────────────────────────────────────────────────────────────────
# Extraction des paramètres BPR depuis le graphe
# ────────────────────────────────────────────────────────────────────────────

def _bpr_params(graph: ig.Graph) -> tuple[np.ndarray, ...]:
    t0 = np.asarray(graph.es["t0_s"], dtype=float)
    capacity = np.asarray(graph.es["capacity"], dtype=float)
    alpha = np.asarray(graph.es["bpr_alpha"], dtype=float)
    beta = np.asarray(graph.es["bpr_beta"], dtype=float)
    return t0, capacity, alpha, beta


# ────────────────────────────────────────────────────────────────────────────
# Line search (bissection sur la dérivée du Beckmann)
# ────────────────────────────────────────────────────────────────────────────

def _line_search(
    x: np.ndarray,
    y: np.ndarray,
    cost_fn: Callable[[np.ndarray], np.ndarray],
    n_iter: int = 25,
) -> float:
    """Trouve λ ∈ [0,1] qui minimise Z(x + λ·(y-x)).

    Z' à minimiser : (y - x)·t(x + λ·(y-x)). On bissectionne sur ce signe.
    Si Z'(0) ≥ 0, prendre λ=0 (déjà optimal). Si Z'(1) ≤ 0, prendre λ=1.
    """
    d = y - x

    def zprime(lam: float) -> float:
        return float(np.sum(d * cost_fn(x + lam * d)))

    if zprime(0.0) >= 0.0:
        return 0.0
    if zprime(1.0) <= 0.0:
        return 1.0

    lo, hi = 0.0, 1.0
    for _ in range(n_iter):
        mid = 0.5 * (lo + hi)
        if zprime(mid) > 0.0:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


# ────────────────────────────────────────────────────────────────────────────
# Gap relatif (critère d'arrêt)
# ────────────────────────────────────────────────────────────────────────────

def _relative_gap(
    flows: np.ndarray,
    target_flows: np.ndarray,
    times: np.ndarray,
) -> float:
    """Gap relatif AoN-UE : (TT(x) - TT_aon(x)) / TT(x).

    TT(x)     = somme des x_a · t_a(x)  — temps total des flux actuels.
    TT_aon(x) = somme des y_a · t_a(x)  — temps total si tout le monde repassait
                 par AoN sur les temps actuels (borne inférieure).
    Plus le gap est petit, plus on est proche de Wardrop.
    """
    tt = float((flows * times).sum())
    tt_aon = float((target_flows * times).sum())
    if tt <= 0.0:
        return 0.0
    return abs(tt - tt_aon) / tt


# ────────────────────────────────────────────────────────────────────────────
# Boucle Frank-Wolfe générique
# ────────────────────────────────────────────────────────────────────────────

def _frank_wolfe(
    network: UrbanNetwork,
    od: ODMatrix,
    cost_fn: Callable[[np.ndarray], np.ndarray],
    *,
    max_iter: int,
    tol: float,
    method: str,
    initial_flows: np.ndarray | None = None,
) -> AssignmentResult:
    g = network.graph
    t0, capacity, alpha, beta = _bpr_params(g)

    # Initialisation : warm-start si initial_flows fourni et compatible, sinon AoN free-flow
    if initial_flows is not None and len(initial_flows) == g.ecount():
        flows = np.asarray(initial_flows, dtype=float).copy()
        logger.info(f"  warm-start depuis flows fournis ({len(flows)} arêtes)")
    else:
        flows = assign_aon(g, od, weights=t0.tolist())
    perceived = cost_fn(flows)

    converged = False
    gap = np.inf
    k = 0

    for k in range(1, max_iter + 1):
        # On convertit perceived en list UNE seule fois par itération
        # (au lieu d'une fois par appel AoN ; igraph est plus rapide avec list que np.array)
        perceived_list = perceived.tolist()
        target = assign_aon(g, od, weights=perceived_list)
        gap = _relative_gap(flows, target, perceived)
        if gap < tol:
            converged = True
            logger.info(f"  iter {k:>3}: gap={gap:.2e} ← converged")
            break

        lam = _line_search(flows, target, cost_fn)
        flows = (1.0 - lam) * flows + lam * target
        perceived = cost_fn(flows)

        if k % 10 == 0 or k == 1:
            logger.info(f"  iter {k:>3}: gap={gap:.2e}, λ={lam:.3f}")

    # Pour le reporting on utilise toujours le coût BPR moyen (pas marginal),
    # pour que VHT(SO) soit comparable à VHT(UE) sur la même métrique physique.
    actual_times = bpr_time(flows, t0, capacity, alpha, beta)
    vht = float((flows * actual_times).sum() / 3600.0)
    return AssignmentResult(
        flows=flows,
        travel_times=actual_times,
        free_flow_times=t0,
        vht=vht,
        iterations=k,
        converged=converged,
        final_gap=float(gap),
        method=method,
    )


# ────────────────────────────────────────────────────────────────────────────
# Façades publiques
# ────────────────────────────────────────────────────────────────────────────

def solve_user_equilibrium(
    network: UrbanNetwork,
    od: ODMatrix,
    max_iter: int = FRANK_WOLFE_MAX_ITER,
    tol: float = FRANK_WOLFE_TOLERANCE,
    *,
    initial_flows: np.ndarray | None = None,
) -> AssignmentResult:
    """User Equilibrium (Wardrop) par Frank-Wolfe sur le coût BPR moyen.

    Args:
        initial_flows: si fourni et de taille g.ecount(), warm-start depuis ces
            flux au lieu d'AoN free-flow. Utile pour ré-évaluer un plan perturbé
            (gain typique : 2-3× moins d'itérations).
    """
    t0, capacity, alpha, beta = _bpr_params(network.graph)

    def cost_fn(f: np.ndarray) -> np.ndarray:
        return bpr_time(f, t0, capacity, alpha, beta)

    logger.info(f"=== User Equilibrium (Frank-Wolfe, max_iter={max_iter}) ===")
    return _frank_wolfe(
        network, od, cost_fn, max_iter=max_iter, tol=tol, method="ue",
        initial_flows=initial_flows,
    )


def solve_system_optimum(
    network: UrbanNetwork,
    od: ODMatrix,
    max_iter: int = FRANK_WOLFE_MAX_ITER,
    tol: float = FRANK_WOLFE_TOLERANCE,
    *,
    initial_flows: np.ndarray | None = None,
) -> AssignmentResult:
    """System Optimum par Frank-Wolfe sur le coût marginal."""
    t0, capacity, alpha, beta = _bpr_params(network.graph)

    def cost_fn(f: np.ndarray) -> np.ndarray:
        return bpr_marginal_time(f, t0, capacity, alpha, beta)

    logger.info(f"=== System Optimum (Frank-Wolfe, max_iter={max_iter}) ===")
    return _frank_wolfe(
        network, od, cost_fn, max_iter=max_iter, tol=tol, method="so",
        initial_flows=initial_flows,
    )


def solve_all_or_nothing(
    network: UrbanNetwork,
    od: ODMatrix,
) -> AssignmentResult:
    """Affectation triviale (pas de congestion endogène) — baseline."""
    g = network.graph
    t0, capacity, alpha, beta = _bpr_params(g)

    flows = assign_aon(g, od, weights=t0)
    times = bpr_time(flows, t0, capacity, alpha, beta)
    vht = float((flows * times).sum() / 3600.0)
    logger.info(f"AoN — flux max: {flows.max():,.0f}, VHT: {vht:,.1f}")

    return AssignmentResult(
        flows=flows,
        travel_times=times,
        free_flow_times=t0,
        vht=vht,
        iterations=1,
        converged=True,
        final_gap=0.0,
        method="aon",
    )


# ────────────────────────────────────────────────────────────────────────────
# Métrique inter-modèles
# ────────────────────────────────────────────────────────────────────────────

def price_of_anarchy(ue: AssignmentResult, so: AssignmentResult) -> float:
    """Ratio VHT(UE) / VHT(SO) — coût social de l'égoïsme individuel.

    Vaut ~1.0 si les choix individuels coïncident avec l'optimum collectif,
    > 1 dès qu'il y a divergence (typiquement 1.05-1.30 sur des villes).
    """
    if so.vht <= 0:
        return 1.0
    return float(ue.vht / so.vht)


def beckmann(network: UrbanNetwork, flows: np.ndarray) -> float:
    """Valeur du Beckmann pour des flux donnés."""
    t0, capacity, alpha, beta = _bpr_params(network.graph)
    return beckmann_objective(flows, t0, capacity, alpha, beta)
