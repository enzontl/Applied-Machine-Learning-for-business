"""Test de robustesse d'un plan urbain face aux variations de demande.

Idée : le BCR du plan est calculé pour un niveau de demande **donné** (l'heure
de pointe courante). Mais si la population augmente de 20%, ou diminue parce
qu'une partie passe au télétravail, le plan tient-il toujours la route ?

On re-lance le Frank-Wolfe **avec et sans plan** sous différents facteurs
d'échelle de l'OD, et on rapporte :
- VHT baseline vs VHT plan, pour chaque scénario
- Bénéfice annuel correspondant
- Plan "robuste" si le bénéfice reste positif pour tous les scénarios

Coût : 2 FW supplémentaires par scénario — on peut limiter à 30 itérations
(les tendances restent fiables, l'équilibre exact n'est pas critique pour
juger de la direction).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from urban_optimizer.assignment.frank_wolfe import solve_user_equilibrium
from urban_optimizer.demand.od_matrix import ODMatrix
from urban_optimizer.network.urban_network import UrbanNetwork
from urban_optimizer.utils.logging import get_logger

from .mayor_profile import MayorProfile
from .network_design import NewArcEvaluation, _JointContext, _warm_start_flows
from .score import score_network

logger = get_logger(__name__)


@dataclass
class RobustnessPoint:
    """Évaluation pour un facteur d'échelle de la demande."""

    demand_scale: float
    baseline_vht_h: float
    plan_vht_h: float
    delta_vht_h: float                  # > 0 = plan profitable
    baseline_score_eur: float
    plan_score_eur: float
    annual_benefit_eur: float           # > 0 = plan tient le coup

    @property
    def is_beneficial(self) -> bool:
        return self.annual_benefit_eur > 0


@dataclass
class RobustnessReport:
    """Test de robustesse complet sous N scénarios de demande."""

    points: list[RobustnessPoint] = field(default_factory=list)

    @property
    def is_robust(self) -> bool:
        """True ssi le plan reste bénéfique pour TOUS les scénarios testés."""
        return bool(self.points) and all(p.is_beneficial for p in self.points)

    @property
    def worst_point(self) -> RobustnessPoint | None:
        """Scénario où le plan rapporte le moins (le plus fragile)."""
        if not self.points:
            return None
        return min(self.points, key=lambda p: p.annual_benefit_eur)

    @property
    def best_point(self) -> RobustnessPoint | None:
        """Scénario où le plan rapporte le plus."""
        if not self.points:
            return None
        return max(self.points, key=lambda p: p.annual_benefit_eur)


def evaluate_plan_robustness(
    network: UrbanNetwork,
    od: ODMatrix,
    plan: list[NewArcEvaluation],
    profile: MayorProfile,
    *,
    demand_scales: tuple[float, ...] = (0.8, 1.0, 1.2, 1.5),
    fw_max_iter: int = 30,
    fw_tol: float = 5e-3,
) -> RobustnessReport:
    """Évalue un plan urbain sous différents niveaux de demande.

    Pour chaque facteur d'échelle de l'OD, calcule deux FW :
    1. Baseline (réseau actuel)
    2. Plan appliqué (corridors élargis + nouvelles routes ajoutées)

    Returns:
        RobustnessReport contenant la liste des points évalués.

    Note: la valeur du bénéfice ici n'inclut PAS l'accessibilité/équité,
    par souci de rapidité. Le but est de juger la direction et la stabilité
    sous stress, pas l'amplitude exacte.
    """
    if not plan:
        return RobustnessReport()

    chosen_props = [ev.proposal for ev in plan]

    # On exécute les scales triés par proximité à 1.0 pour permettre un warm-start
    # inter-scale : la solution UE à demande ×s₁ est proche de celle à ×s₂ quand s₁≈s₂.
    # On garde l'ordre d'origine pour le rapport final.
    sorted_scales = sorted(demand_scales, key=lambda s: abs(s - 1.0))
    results_by_scale: dict[float, RobustnessPoint] = {}

    # Cache des flows précédents pour warm-start
    prev_base_flows = None
    prev_plan_flows = None

    for scale in sorted_scales:
        od_scaled = ODMatrix(
            matrix=od.matrix * scale,
            zone_ids=od.zone_ids,
            zone_to_node=od.zone_to_node,
            hour=od.hour,
            scenario=f"{od.scenario}_x{scale:.2f}",
        )
        # 1. Baseline — warm-start depuis la scale la plus proche déjà calculée
        base_ue = solve_user_equilibrium(
            network, od_scaled, max_iter=fw_max_iter, tol=fw_tol,
            initial_flows=prev_base_flows,
        )
        base_score = score_network(base_ue, profile)

        # 2. Plan appliqué — warm-start ordre de préférence :
        #    (a) flows plan d'une scale voisine (même ecount avec interventions)
        #    (b) sinon, base_ue de cette scale étendu pour les nouveaux arcs
        with _JointContext(network.graph, chosen_props):
            n_now = network.graph.ecount()
            if prev_plan_flows is not None and len(prev_plan_flows) == n_now:
                warm = prev_plan_flows
            else:
                warm = _warm_start_flows(base_ue.flows, n_now)
            plan_ue = solve_user_equilibrium(
                network, od_scaled, max_iter=fw_max_iter, tol=fw_tol,
                initial_flows=warm,
            )
            plan_score = score_network(plan_ue, profile)
            # Capturer les flows plan ICI (avant restauration du JointContext)
            prev_plan_flows = plan_ue.flows.copy()

        prev_base_flows = base_ue.flows.copy()

        benefit = base_score.total_annual_cost_eur - plan_score.total_annual_cost_eur
        delta_vht = base_ue.vht - plan_ue.vht
        results_by_scale[scale] = RobustnessPoint(
            demand_scale=scale,
            baseline_vht_h=base_ue.vht,
            plan_vht_h=plan_ue.vht,
            delta_vht_h=delta_vht,
            baseline_score_eur=base_score.total_annual_cost_eur,
            plan_score_eur=plan_score.total_annual_cost_eur,
            annual_benefit_eur=benefit,
        )
        logger.info(
            f"Robustesse ×{scale:.2f} — VHT base={base_ue.vht:,.0f}h plan={plan_ue.vht:,.0f}h "
            f"ΔVHT={delta_vht:+.1f}h bénéfice={benefit:+,.0f}€/an "
            f"(base iter={base_ue.iterations}, plan iter={plan_ue.iterations})"
        )

    # Restituer dans l'ordre demandé par l'utilisateur
    points = [results_by_scale[s] for s in demand_scales]
    return RobustnessReport(points=points)
