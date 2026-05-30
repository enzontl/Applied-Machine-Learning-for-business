"""Ranking marginal des candidats d'amélioration.

Pour chaque candidat, on applique l'action sur une copie du réseau, on
re-résout UE et on mesure ΔVHT. Le score multi-critère est calculé à partir
des **gains** (en €/an équivalents) selon les poids fournis :

    score = w_time · gain_time(€) + w_fuel · gain_fuel(€) − cost_eur

Ce qui revient à mettre tout en valeur monétaire. Le top-N est ensuite
filtré sous une contrainte de budget par sélection gloutonne (sac à dos
relaxé — suffisant en première approche, on pourra remplacer par PuLP).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from urban_optimizer.assignment.frank_wolfe import (
    bpr_time,
    solve_user_equilibrium,
)
from urban_optimizer.assignment.result import AssignmentResult
from urban_optimizer.demand.od_matrix import ODMatrix
from urban_optimizer.network.urban_network import UrbanNetwork
from urban_optimizer.utils.logging import get_logger

from .candidates import Candidate

logger = get_logger(__name__)


# Constantes économiques (à exposer en config plus tard)
VALUE_OF_TIME_EUR_PER_H = 12.0           # rendement social € / h-passager
FUEL_LITERS_PER_VEH_PER_H = 4.5          # ~4-6 L/h en milieu urbain congestionné
FUEL_PRICE_EUR_PER_L = 1.85
WORKING_DAYS_PER_YEAR = 220              # pour amortir le coût sur 1 an de pointes


@dataclass
class CandidateEvaluation:
    candidate: Candidate
    delta_vht_h: float                   # gain en véh·h (positif = amélioration)
    delta_vht_share: float               # gain / VHT baseline
    new_vht: float
    annual_time_value_eur: float         # gain · 220 jours · 12 €/h
    annual_fuel_value_eur: float
    annual_benefit_eur: float
    cost_eur: float
    score: float                         # bénéfice net annuel - coût (mesure 1 an)
    bcr: float                           # benefit-cost ratio (annual_benefit / cost)
    is_braess: bool                      # vrai si action="remove" et ΔVHT > 0


# ────────────────────────────────────────────────────────────────────────────
# Patching du graphe : applique une action sans muter le réseau original
# ────────────────────────────────────────────────────────────────────────────

def _apply_action_in_place(
    capacity: np.ndarray,
    t0: np.ndarray,
    edge_id: int,
    action: str,
    magnitude: float,
    length_m: float,
) -> None:
    if action == "capacity_boost":
        capacity[edge_id] *= magnitude
    elif action == "speed_boost":
        # vitesse libre × magnitude → t0 / magnitude
        t0[edge_id] /= magnitude
    elif action == "remove":
        # Plutôt que de retirer l'arc (changerait la structure du graphe et
        # casserait les indices), on simule un coût quasi-infini :
        capacity[edge_id] = 1.0
        t0[edge_id] = length_m * 100.0   # pseudo-fermeture
    else:
        raise ValueError(f"action inconnue : {action!r}")


def _solve_with_patch(
    network: UrbanNetwork,
    od: ODMatrix,
    candidate: Candidate,
    *,
    warm_start_flows: np.ndarray | None,
    max_iter: int,
    tol: float,
) -> AssignmentResult:
    """Re-résoud UE en patchant temporairement capacity / t0 sur l'arc cible.

    Le graphe igraph n'est pas modifié structurellement — on échange
    seulement les vecteurs d'attributs.
    """
    g = network.graph
    original_cap = np.asarray(g.es["capacity"], dtype=float)
    original_t0 = np.asarray(g.es["t0_s"], dtype=float)

    cap = original_cap.copy()
    t0 = original_t0.copy()
    _apply_action_in_place(
        cap, t0, candidate.edge_id,
        candidate.action, candidate.magnitude, candidate.length_m,
    )

    try:
        g.es["capacity"] = cap.tolist()
        g.es["t0_s"] = t0.tolist()
        res = solve_user_equilibrium(
            network, od, max_iter=max_iter, tol=tol,
        )
    finally:
        g.es["capacity"] = original_cap.tolist()
        g.es["t0_s"] = original_t0.tolist()

    return res


# ────────────────────────────────────────────────────────────────────────────
# Évaluation d'un candidat
# ────────────────────────────────────────────────────────────────────────────

def _evaluate(
    network: UrbanNetwork,
    od: ODMatrix,
    candidate: Candidate,
    baseline_vht_h: float,
    *,
    warm_start_flows: np.ndarray | None,
    max_iter: int,
    tol: float,
) -> CandidateEvaluation:
    res = _solve_with_patch(
        network, od, candidate,
        warm_start_flows=warm_start_flows, max_iter=max_iter, tol=tol,
    )
    delta_h = baseline_vht_h - res.vht
    delta_share = delta_h / baseline_vht_h if baseline_vht_h > 0 else 0.0

    # Valorisations annuelles (heure de pointe répétée WORKING_DAYS_PER_YEAR fois)
    annual_time = delta_h * VALUE_OF_TIME_EUR_PER_H * WORKING_DAYS_PER_YEAR
    annual_fuel = (
        delta_h * FUEL_LITERS_PER_VEH_PER_H * FUEL_PRICE_EUR_PER_L
        * WORKING_DAYS_PER_YEAR
    )
    annual_benefit = annual_time + annual_fuel
    cost = candidate.cost_eur

    score = annual_benefit - cost  # marge sur 1 an
    bcr = annual_benefit / cost if cost > 0 else (np.inf if annual_benefit > 0 else 0.0)

    return CandidateEvaluation(
        candidate=candidate,
        delta_vht_h=delta_h,
        delta_vht_share=delta_share,
        new_vht=res.vht,
        annual_time_value_eur=annual_time,
        annual_fuel_value_eur=annual_fuel,
        annual_benefit_eur=annual_benefit,
        cost_eur=cost,
        score=score,
        bcr=bcr,
        is_braess=(candidate.action == "remove" and delta_h > 0),
    )


# ────────────────────────────────────────────────────────────────────────────
# Façade publique
# ────────────────────────────────────────────────────────────────────────────

def rank_candidates(
    network: UrbanNetwork,
    od: ODMatrix,
    candidates: list[Candidate],
    baseline: AssignmentResult,
    *,
    max_iter: int = 60,
    tol: float = 1e-3,
) -> list[CandidateEvaluation]:
    """Évalue tous les candidats et les classe par bénéfice-coût."""
    n = len(candidates)
    logger.info(f"=== Ranking de {n} candidats (FW max_iter={max_iter}) ===")
    evals: list[CandidateEvaluation] = []
    for i, c in enumerate(candidates, start=1):
        ev = _evaluate(
            network, od, c, baseline.vht,
            warm_start_flows=baseline.flows, max_iter=max_iter, tol=tol,
        )
        evals.append(ev)
        logger.info(
            f"  [{i:>2}/{n}] e{c.edge_id} {c.action:<14s} mag={c.magnitude:.2f} "
            f"ΔVHT={ev.delta_vht_h:>+9.1f}h  BCR={ev.bcr:.2f}  cost={c.cost_eur:>11,.0f}€"
        )
    evals.sort(key=lambda e: e.bcr, reverse=True)
    return evals
