"""Courbe Pareto budget → bénéfice annuel.

Une question classique de décideur : "pour 10/25/50/100/200 M€, quel bénéfice ?"

On évalue les candidats **une seule fois** (étape coûteuse), puis pour chaque
budget on relance la sélection gloutonne + une re-évaluation jointe (FW unique).

Le point d'inflexion de la courbe (rendements marginaux décroissants) donne
le "sweet spot" budgétaire — au-delà, chaque euro supplémentaire rapporte
moins.

Coût : ~max_fw_evals FW (évaluation individuelle) + N_budgets FW (re-éval jointe).
Par exemple : 15 + 6 = 21 FW pour 6 points de Pareto.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from urban_optimizer.assignment.result import AssignmentResult
from urban_optimizer.demand.od_matrix import ODMatrix
from urban_optimizer.diagnosis.accessibility import compute_accessibility
from urban_optimizer.network.buildings import ObstacleIndex
from urban_optimizer.network.urban_network import UrbanNetwork
from urban_optimizer.utils.logging import get_logger

from .mayor_profile import MayorProfile
from .network_design import (
    _evaluate_proposals,
    _generate_proposals,
    _greedy_select,
    _joint_re_evaluate,
    _quick_fw_screen,
)
from .score import score_network

logger = get_logger(__name__)


@dataclass
class ParetoPoint:
    """Évaluation d'un plan urbain pour un budget donné."""

    budget_eur: float                    # budget alloué
    capex_used_eur: float                # CAPEX effectivement dépensé (≤ budget)
    n_interventions: int                 # nb d'interventions retenues
    joint_annual_benefit_eur: float      # bénéfice annuel joint (FW unique)
    joint_vht_h: float                   # VHT après plan
    delta_vht_h: float                   # ΔVHT vs baseline
    bcr: float                           # bénéfice joint / CAPEX

    @property
    def marginal_efficiency_eur_per_meur(self) -> float:
        """Bénéfice annuel par M€ dépensé (1/payback × 1e6)."""
        if self.capex_used_eur <= 0:
            return 0.0
        return self.joint_annual_benefit_eur / (self.capex_used_eur / 1e6)


@dataclass
class ParetoFrontier:
    """Liste de points Pareto budget → bénéfice."""

    points: list[ParetoPoint] = field(default_factory=list)

    @property
    def best_marginal_point(self) -> ParetoPoint | None:
        """Point au meilleur ratio bénéfice / M€ — sweet spot budgétaire."""
        if not self.points:
            return None
        return max(self.points, key=lambda p: p.marginal_efficiency_eur_per_meur)


def compute_pareto_frontier(
    network: UrbanNetwork,
    od: ODMatrix,
    profile: MayorProfile,
    baseline_ue: AssignmentResult,
    *,
    budgets_eur: tuple[float, ...] = (
        5_000_000, 15_000_000, 30_000_000, 60_000_000, 120_000_000, 250_000_000,
    ),
    max_proposals: int = 60,
    max_fw_evals: int = 15,
    fw_max_iter: int = 25,                  # FW JOINT (par budget, strict)
    fw_tol: float = 5e-3,                   # FW JOINT
    fw_max_iter_cand: int = 10,             # FW par candidat (relâché)
    fw_tol_cand: float = 2e-2,              # FW par candidat — JAMAIS > 3e-2 (signal ΔVHT ~1-3%)
    n_jobs: int = -1,                       # workers joblib (-1 = tous cœurs)
    screen_factor: float = 2.0,             # pool initial = screen_factor × max_fw_evals
    obstacle_index: ObstacleIndex | None = None,
    soft_index: ObstacleIndex | None = None,
    periphery_margin_m: float = 600.0,
    accessibility_threshold_s: float = 15 * 60,
) -> ParetoFrontier:
    """Trace la courbe budget → bénéfice annuel pour plusieurs niveaux de budget.

    Génère les candidats + les évalue **une seule fois** (étape coûteuse),
    puis pour chaque budget : sélection gloutonne + re-évaluation jointe.

    Returns:
        ParetoFrontier avec un ParetoPoint par budget.
    """
    # 1. Baseline (accessibilité + score)
    baseline_access = compute_accessibility(
        network, od, baseline_ue, threshold_seconds=accessibility_threshold_s,
    )
    baseline_score = score_network(baseline_ue, profile, access=baseline_access)

    # 2. Candidats — pool plus large pour screening, puis un seul appel d'éval pour toute la courbe
    pool_size = max_fw_evals if screen_factor <= 1.0 else int(max_fw_evals * screen_factor)
    pre = _generate_proposals(
        network, od, baseline_ue,
        max_proposals=max_proposals, max_fw_evals=pool_size,
        obstacle_index=obstacle_index, soft_index=soft_index,
        periphery_margin_m=periphery_margin_m,
    )
    if not pre:
        logger.warning("Pareto : aucun candidat — courbe vide.")
        return ParetoFrontier()

    # 2b. Pré-filtre 1-iter FW (réduit aux max_fw_evals meilleurs)
    if len(pre) > max_fw_evals:
        pre = _quick_fw_screen(
            network, od, baseline_ue, pre, max_fw_evals, n_jobs=n_jobs,
        )

    # 3. Évaluation individuelle — UNE SEULE FOIS (réutilisée pour tous les budgets)
    evals = _evaluate_proposals(
        network, od, profile, baseline_ue, baseline_score, pre,
        fw_max_iter=fw_max_iter_cand, fw_tol=fw_tol_cand,
        n_jobs=n_jobs,
        baseline_access=baseline_access,
    )

    # 4. Pour chaque budget : greedy + joint
    # Mémoïsation : si 2 budgets consécutifs sélectionnent les MÊMES interventions
    # (cas fréquent quand le budget marginal ne permet pas de nouveau candidat),
    # on réutilise le résultat. Économise 1 FW joint + 1 compute_accessibility.
    points: list[ParetoPoint] = []
    joint_cache: dict[tuple[int, ...], object] = {}
    n_reused = 0

    for budget in budgets_eur:
        chosen, spent = _greedy_select(evals, budget)
        # Clé canonique : ids triés des propositions retenues
        chosen_key = tuple(sorted(id(e.proposal) for e in chosen)) if chosen else ()

        was_cached = chosen_key in joint_cache
        if was_cached:
            joint = joint_cache[chosen_key]
            n_reused += 1
        else:
            joint = _joint_re_evaluate(
                network, od, profile, baseline_ue, baseline_score,
                baseline_access.mean_reachable, baseline_access.gini,
                chosen, spent,
                fw_max_iter=fw_max_iter, fw_tol=fw_tol,
                accessibility_threshold_s=accessibility_threshold_s,
            )
            if joint is not None:
                joint_cache[chosen_key] = joint

        if joint is None:
            points.append(ParetoPoint(
                budget_eur=budget,
                capex_used_eur=0.0, n_interventions=0,
                joint_annual_benefit_eur=0.0,
                joint_vht_h=baseline_ue.vht,
                delta_vht_h=0.0, bcr=0.0,
            ))
        else:
            points.append(ParetoPoint(
                budget_eur=budget,
                capex_used_eur=spent,
                n_interventions=joint.n_interventions,
                joint_annual_benefit_eur=joint.joint_annual_benefit_eur,
                joint_vht_h=joint.joint_vht_h,
                delta_vht_h=joint.joint_delta_vht_h,
                bcr=joint.joint_bcr,
            ))
        tag = " (cache)" if was_cached else ""
        logger.info(
            f"Pareto budget {budget / 1e6:>6.0f} M€{tag} → "
            f"CAPEX {spent / 1e6:>6.1f} M€, "
            f"{len(chosen):>2} interventions, "
            f"bénéfice {points[-1].joint_annual_benefit_eur / 1e6:>+6.2f} M€/an, "
            f"BCR {points[-1].bcr:.2f}"
        )

    if n_reused > 0:
        logger.info(
            f"Pareto — {n_reused}/{len(budgets_eur)} budgets ont réutilisé un FW joint (cache)"
        )
    return ParetoFrontier(points=points)
