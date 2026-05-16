"""Network design problem (NDP) : propose de **nouvelles routes**.

Approche en 3 temps pour rester traitable malgré la NP-difficulté du NDP :

1. **Génération de candidats** — pour chaque paire de nœuds (u, v) :
   - distance euclidienne dans [min_length, max_length]
   - détour graphique actuel (length(shortest_path) / euclidean) > seuil
   - rang par proxy "détour gagné × demande approchée"

2. **Pré-filtrage AoN** — on évalue chaque candidat avec une affectation
   tout-ou-rien (très rapide). On garde les top-K par gain AoN.

3. **Évaluation fine** — on insère l'arc dans une copie du graphe, on
   relance Frank-Wolfe (warm-start), on calcule le score du nouveau réseau.
   Les arcs ajoutés sont restaurés à la fin (try/finally).

La sélection finale est gloutonne par BCR (Benefit / Cost Ratio).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import hypot

import igraph as ig
import numpy as np
from shapely.geometry import LineString

from urban_optimizer.assignment.frank_wolfe import (
    solve_all_or_nothing,
    solve_user_equilibrium,
)
from urban_optimizer.assignment.result import AssignmentResult
from urban_optimizer.config import BPR_ALPHA, BPR_BETA
from urban_optimizer.demand.od_matrix import ODMatrix
from urban_optimizer.network.buildings import BuildingIndex
from urban_optimizer.network.urban_network import UrbanNetwork
from urban_optimizer.utils.logging import get_logger

from .mayor_profile import MayorProfile
from .score import CityScore, score_network

logger = get_logger(__name__)


# ── Paramètres de design ────────────────────────────────────────────────────

# Catégorie de nouvel arc selon sa longueur
def _new_arc_specs(length_m: float) -> tuple[str, float, float, float]:
    """Renvoie (highway_type, capacité véh/h, vitesse km/h, coût €/m).

    Tarifs construction urbain (France, ordre de grandeur 2020-2025) :
      tertiary  : ~3 M€/km — voie urbaine 1 voie/sens
      secondary : ~6 M€/km — voie urbaine 2 voies/sens
      primary   : ~10 M€/km — boulevard ou rocade
    """
    if length_m < 500:
        return "tertiary", 1500.0, 50.0, 3_000.0
    if length_m < 2000:
        return "secondary", 2400.0, 70.0, 6_000.0
    return "primary", 3000.0, 90.0, 10_000.0


@dataclass
class NewArcProposal:
    """Une proposition de construction de route.

    Attributes:
        u_node, v_node: index des nœuds existants à connecter.
        length_m: distance euclidienne entre les nœuds (m).
        highway: catégorie déduite de la longueur.
        capacity: véh/h du nouvel arc.
        free_speed_kmh: vitesse libre.
        construction_cost_eur: CAPEX estimé.
        detour_before: ratio detour avant construction (length_path / euclidean).
        u_xy, v_xy: coordonnées Lambert-93, pour la viz.
    """

    u_node: int
    v_node: int
    length_m: float
    highway: str
    capacity: float
    free_speed_kmh: float
    construction_cost_eur: float
    detour_before: float
    u_xy: tuple[float, float]
    v_xy: tuple[float, float]

    def geometry(self) -> LineString:
        return LineString([self.u_xy, self.v_xy])


@dataclass
class NewArcEvaluation:
    """Résultat de l'évaluation Frank-Wolfe d'un candidat."""

    proposal: NewArcProposal
    new_vht_h: float
    baseline_vht_h: float
    delta_vht_h: float                       # > 0 = amélioration
    new_score_eur_year: float                # coût annuel pondéré profil
    baseline_score_eur_year: float
    annual_benefit_eur: float                # baseline − new
    payback_years: float                     # CAPEX / bénéfice annuel
    cost_eur: float                          # CAPEX pondéré profil
    bcr: float                               # benefit / cost annuel
    score: float = field(default=0.0)        # critère final = bénéfice − annuité

    @property
    def is_worth_it(self) -> bool:
        return self.annual_benefit_eur > 0 and self.cost_eur > 0


# ── Génération de candidats ─────────────────────────────────────────────────

def _shortest_length(g: ig.Graph, sources: list[int], targets: list[int]) -> np.ndarray:
    """Matrice (n_src, n_tgt) des longueurs de plus courts chemins (m)."""
    mat = g.distances(source=sources, target=targets, weights="length_m")
    return np.asarray(mat, dtype=float)


def _zone_demand_lookup(od: ODMatrix) -> dict[tuple[int, int], float]:
    lookup: dict[tuple[int, int], float] = {}
    for zo, zd, trips in od.iter_pairs():
        u = od.zone_to_node.get(zo)
        v = od.zone_to_node.get(zd)
        if u is None or v is None or u == v:
            continue
        lookup[(u, v)] = lookup.get((u, v), 0.0) + trips
    return lookup


def generate_new_arc_candidates(
    network: UrbanNetwork,
    od: ODMatrix,
    *,
    min_length_m: float = 150.0,
    max_length_m: float = 4_000.0,
    min_detour_ratio: float = 1.30,
    max_candidates: int = 80,
    building_index: BuildingIndex | None = None,
) -> list[NewArcProposal]:
    """Cherche les paires de nœuds (u, v) qui justifieraient un nouvel arc.

    On restreint aux nœuds **rattachés à une zone OD** (sinon il n'y a pas de
    demande directe et le gain est marginal). Pour chaque paire :

    - distance euclidienne dans la fenêtre
    - détour graphique actuel ≥ ``min_detour_ratio``
    - tracé rectiligne ne traversant aucun bâtiment (si *building_index* fourni)

    Score de proxy = (détour gagné) × max(1, demande directe OD).
    Top ``max_candidates`` retournés.
    """
    g = network.graph
    nodes_xy = network.nodes_xy

    demand_nodes = sorted({od.zone_to_node[z] for z in od.zone_ids})
    if len(demand_nodes) < 2:
        return []
    logger.info(f"NDP — {len(demand_nodes)} nœuds candidats")

    coords = np.array([nodes_xy[n] for n in demand_nodes])
    paths = _shortest_length(g, demand_nodes, demand_nodes)

    od_lookup = _zone_demand_lookup(od)

    proposals: list[tuple[float, NewArcProposal]] = []
    n_building_rejected = 0
    n = len(demand_nodes)
    for i in range(n):
        for j in range(i + 1, n):
            ux, uy = coords[i]
            vx, vy = coords[j]
            euclid = float(hypot(ux - vx, uy - vy))
            if euclid < min_length_m or euclid > max_length_m:
                continue

            path_len = min(paths[i, j], paths[j, i])  # graphe orienté → min
            if not np.isfinite(path_len) or path_len <= 0:
                continue
            detour = path_len / euclid
            if detour < min_detour_ratio:
                continue

            # Filtre bâtiments : on rejette si le tracé rectiligne traverse un bâtiment
            if building_index is not None:
                segment = LineString([(ux, uy), (vx, vy)])
                if building_index.crosses(segment):
                    n_building_rejected += 1
                    continue

            u_node = demand_nodes[i]
            v_node = demand_nodes[j]
            demand = od_lookup.get((u_node, v_node), 0.0) + od_lookup.get((v_node, u_node), 0.0)
            proxy = (path_len - euclid) * max(1.0, demand)

            hw, cap, speed, cost_per_m = _new_arc_specs(euclid)
            proposals.append((
                proxy,
                NewArcProposal(
                    u_node=u_node, v_node=v_node,
                    length_m=euclid, highway=hw,
                    capacity=cap, free_speed_kmh=speed,
                    construction_cost_eur=euclid * cost_per_m,
                    detour_before=detour,
                    u_xy=(ux, uy), v_xy=(vx, vy),
                ),
            ))

    if building_index is not None:
        logger.info(f"NDP — {n_building_rejected} candidats rejetés (traversent des bâtiments)")
    proposals.sort(key=lambda x: -x[0])
    top = [p for _, p in proposals[:max_candidates]]
    logger.info(f"NDP — {len(top)} candidats retenus (sur {len(proposals)} générés)")
    return top


# ── Insertion / retrait temporaire d'un arc ─────────────────────────────────

class _NewArcContext:
    """Ajoute un arc bidirectionnel temporaire au graphe. Restauration garantie."""

    def __init__(self, g: ig.Graph, prop: NewArcProposal):
        self.g = g
        self.prop = prop
        self.added_eids: list[int] = []

    def __enter__(self):
        before = self.g.ecount()
        p = self.prop
        t0 = p.length_m / (p.free_speed_kmh * 1000.0 / 3600.0)
        attrs = {
            "length_m": [p.length_m, p.length_m],
            "free_speed_kmh": [p.free_speed_kmh, p.free_speed_kmh],
            "capacity": [p.capacity, p.capacity],
            "t0_s": [t0, t0],
            "bpr_alpha": [BPR_ALPHA, BPR_ALPHA],
            "bpr_beta": [BPR_BETA, BPR_BETA],
            "highway": [p.highway, p.highway],
            "source": ["new", "new"],
        }
        self.g.add_edges([(p.u_node, p.v_node), (p.v_node, p.u_node)], attributes=attrs)
        self.added_eids = list(range(before, self.g.ecount()))
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.added_eids:
            self.g.delete_edges(self.added_eids)
            self.added_eids = []


# ── Évaluation d'un candidat ────────────────────────────────────────────────

def _evaluate(
    network: UrbanNetwork,
    od: ODMatrix,
    prop: NewArcProposal,
    baseline_ue: AssignmentResult,
    baseline_score: CityScore,
    profile: MayorProfile,
    *,
    max_iter: int,
    tol: float,
) -> NewArcEvaluation:
    g = network.graph
    with _NewArcContext(g, prop):
        new_ue = solve_user_equilibrium(network, od, max_iter=max_iter, tol=tol)

    new_score = score_network(new_ue, profile)
    delta_vht = baseline_ue.vht - new_ue.vht
    annual_benefit = baseline_score.total_annual_cost_eur - new_score.total_annual_cost_eur

    # Le profil maire amplifie ou atténue le coût ressenti de la construction
    cost = prop.construction_cost_eur * profile.w_construction

    bcr = annual_benefit / cost if cost > 0 else 0.0
    payback = cost / annual_benefit if annual_benefit > 0 else float("inf")
    # Score = bénéfice annuel net, après amortissement linéaire sur 20 ans
    amortization = cost / 20.0
    score = annual_benefit - amortization

    return NewArcEvaluation(
        proposal=prop,
        new_vht_h=new_ue.vht,
        baseline_vht_h=baseline_ue.vht,
        delta_vht_h=delta_vht,
        new_score_eur_year=new_score.total_annual_cost_eur,
        baseline_score_eur_year=baseline_score.total_annual_cost_eur,
        annual_benefit_eur=annual_benefit,
        payback_years=payback,
        cost_eur=cost,
        bcr=bcr,
        score=score,
    )


# ── Pré-filtrage AoN (rapide) pour ne garder que les K meilleurs ────────────

def _quick_aon_filter(
    network: UrbanNetwork,
    od: ODMatrix,
    candidates: list[NewArcProposal],
    keep_top: int,
) -> list[NewArcProposal]:
    """Estimation rapide : on évalue chaque candidat par All-or-Nothing.

    On utilise AoN sur les temps libres avec l'arc ajouté : la baisse de VHT
    AoN est un excellent proxy du gain UE (les chemins les plus directs
    deviennent réellement plus directs grâce au nouvel arc).
    """
    if len(candidates) <= keep_top:
        return candidates

    g = network.graph
    aon_baseline = solve_all_or_nothing(network, od)
    base_vht = aon_baseline.vht

    scored: list[tuple[float, NewArcProposal]] = []
    for p in candidates:
        with _NewArcContext(g, p):
            new_aon = solve_all_or_nothing(network, od)
        scored.append((base_vht - new_aon.vht, p))

    scored.sort(key=lambda x: -x[0])
    return [p for _, p in scored[:keep_top]]


# ── Façade ──────────────────────────────────────────────────────────────────

def propose_urban_plan(
    network: UrbanNetwork,
    od: ODMatrix,
    profile: MayorProfile,
    baseline_ue: AssignmentResult,
    *,
    budget_eur: float = 50_000_000.0,
    max_proposals: int = 60,
    max_fw_evals: int = 15,
    fw_max_iter: int = 25,
    fw_tol: float = 5e-3,
    building_index: BuildingIndex | None = None,
) -> tuple[list[NewArcEvaluation], CityScore]:
    """Pipeline complet de proposition de nouvelles routes.

    Args:
        building_index: index spatial des bâtiments OSM. Si fourni, les candidats
            dont le tracé rectiligne traverse un bâtiment sont rejetés avant
            toute évaluation Frank-Wolfe.

    Returns:
        ``(plan, baseline_score)`` où ``plan`` est la liste ordonnée des
        constructions retenues sous le budget.
    """
    baseline_score = score_network(baseline_ue, profile)
    logger.info(f"Score baseline ({profile.name}) : {baseline_score.composite_score:,.0f} €/an")

    raw = generate_new_arc_candidates(
        network, od, max_candidates=max_proposals, building_index=building_index,
    )
    if not raw:
        logger.warning("Aucun candidat de construction trouvé.")
        return [], baseline_score

    pre = _quick_aon_filter(network, od, raw, keep_top=max_fw_evals)
    logger.info(f"NDP — {len(pre)} candidats sélectionnés pour évaluation FW")

    evals: list[NewArcEvaluation] = []
    for i, prop in enumerate(pre, start=1):
        ev = _evaluate(
            network, od, prop, baseline_ue, baseline_score, profile,
            max_iter=fw_max_iter, tol=fw_tol,
        )
        evals.append(ev)
        logger.info(
            f"  [{i:>2}/{len(pre)}] {prop.highway:>9s} {prop.length_m:>5.0f}m "
            f"u={prop.u_node} v={prop.v_node} "
            f"ΔVHT={ev.delta_vht_h:>+7.1f}h benef={ev.annual_benefit_eur:>+12,.0f}€/an "
            f"BCR={ev.bcr:>5.2f} payback={ev.payback_years:>5.1f}y"
        )

    # Sélection gloutonne sous budget (par BCR décroissant, en €€ pondérés)
    evals_pos = [e for e in evals if e.is_worth_it]
    evals_pos.sort(key=lambda e: -e.bcr)
    chosen: list[NewArcEvaluation] = []
    spent = 0.0
    for ev in evals_pos:
        if spent + ev.cost_eur > budget_eur:
            continue
        chosen.append(ev)
        spent += ev.cost_eur

    logger.info(
        f"NDP — {len(chosen)} constructions retenues, coût total = {spent:,.0f}€ "
        f"(budget {budget_eur:,.0f}€)"
    )
    return chosen, baseline_score
