"""Fonctions de coût Bureau of Public Roads (BPR).

t(v)  = t0 · (1 + α · (v / c)^β)            — coût moyen, utilisé pour UE.
t̃(v) = t0 · (1 + α · (β + 1) · (v / c)^β)   — coût marginal, utilisé pour SO.

Toutes les fonctions sont vectorisées (numpy) — un seul appel par itération
de Frank-Wolfe traite tous les arcs en une fois.
"""

from __future__ import annotations

import numpy as np


def bpr_time(
    flow: np.ndarray,
    t0: np.ndarray,
    capacity: np.ndarray,
    alpha: np.ndarray,
    beta: np.ndarray,
) -> np.ndarray:
    """Temps de parcours BPR moyen.

    Args:
        flow: véh/h par arc.
        t0: temps à vide (s) par arc.
        capacity: capacité (véh/h) par arc.
        alpha, beta: paramètres BPR par arc (typiquement 0.15 et 4).

    Returns:
        Temps de parcours en secondes par arc.
    """
    ratio = flow / np.maximum(capacity, 1.0)
    return t0 * (1.0 + alpha * ratio ** beta)


def bpr_marginal_time(
    flow: np.ndarray,
    t0: np.ndarray,
    capacity: np.ndarray,
    alpha: np.ndarray,
    beta: np.ndarray,
) -> np.ndarray:
    """Coût marginal BPR : t(v) + v · t'(v) = t0 · (1 + α (β+1) (v/c)^β).

    Utilisé pour System Optimum : chaque arc est tarifé à son externalité.
    """
    ratio = flow / np.maximum(capacity, 1.0)
    return t0 * (1.0 + alpha * (beta + 1.0) * ratio ** beta)


def beckmann_objective(
    flow: np.ndarray,
    t0: np.ndarray,
    capacity: np.ndarray,
    alpha: np.ndarray,
    beta: np.ndarray,
) -> float:
    """Fonction objectif de Beckmann (intégrale du temps marginal).

    Σ_a t0_a · (x_a + α·c_a · (x_a/c_a)^(β+1) / (β+1))

    Sa minimisation sous contrainte OD donne l'équilibre utilisateur.
    """
    ratio = flow / np.maximum(capacity, 1.0)
    integ = flow + capacity * alpha * ratio ** (beta + 1.0) / (beta + 1.0)
    return float((t0 * integ).sum())
