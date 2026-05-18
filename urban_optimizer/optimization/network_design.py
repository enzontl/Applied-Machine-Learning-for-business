"""Network design problem (NDP) : élargissement de corridors existants (Niveau 2).

Approche corridor — au lieu de proposer une ligne droite (qui peut traverser
des bâtiments), on cherche un chemin **sur le réseau existant** en pénalisant
les arcs saturés, puis on propose de doubler sa capacité.

Pipeline :
1. **Génération de corridors** — pour chaque paire de nœuds OD :
   - détour graphique ≥ seuil
   - routing sur le graphe réel, arcs saturés pénalisés ×30
   - rang par proxy (détour gagné × demande × saturation moyenne)

2. **Évaluation Frank-Wolfe** — pour chaque corridor candidat :
   - capacité des arcs doublée temporairement
   - FW relancé → nouveau VHT → bénéfice annuel
   - restauration garantie (try/finally)

3. **Sélection gloutonne** sous budget (BCR décroissant).

Avantage clé : la géométrie suit des rues réelles, aucun bâtiment n'est jamais
traversé.
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
from urban_optimizer.network.buildings import ObstacleIndex

# Alias de rétrocompatibilité pour les tests qui importent BuildingIndex directement
BuildingIndex = ObstacleIndex
from urban_optimizer.network.urban_network import UrbanNetwork
from urban_optimizer.utils.logging import get_logger

from .mayor_profile import MayorProfile
from .score import CityScore, score_network

logger = get_logger(__name__)


# ── Coûts de construction / élargissement ────────────────────────────────────

def _new_arc_specs(length_m: float) -> tuple[str, float, float, float]:
    """(highway, capacité, vitesse, coût €/m) pour une nouvelle route en ligne droite."""
    if length_m < 500:
        return "tertiary", 1500.0, 50.0, 3_000.0
    if length_m < 2000:
        return "secondary", 2400.0, 70.0, 6_000.0
    return "primary", 3000.0, 90.0, 10_000.0


def _widening_cost_per_m(highway: str) -> float:
    """Coût d'ajout d'une voie sur un arc existant (€/m) — 3-5× moins cher que neuf."""
    if highway in ("motorway", "trunk", "primary"):
        return 3_500.0
    if highway in ("secondary", "tertiary"):
        return 1_800.0
    return 1_200.0


# ── Dataclasses ──────────────────────────────────────────────────────────────

@dataclass
class NewArcProposal:
    """Une proposition d'intervention sur le réseau.

    En mode **corridor** (edge_ids non vide) : élargissement d'un corridor
    existant dont la géométrie suit les rues réelles.
    En mode **arc** (edge_ids vide) : construction d'une nouvelle route en ligne
    droite (mode hérité, conservé pour la compatibilité des tests).
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
    # Corridor mode — liste des edge IDs et coordonnées du tracé routé
    edge_ids: list[int] = field(default_factory=list)
    corridor_xy: list[tuple[float, float]] = field(default_factory=list)

    def geometry(self) -> LineString:
        if len(self.corridor_xy) >= 2:
            return LineString(self.corridor_xy)
        return LineString([self.u_xy, self.v_xy])

    @property
    def is_corridor(self) -> bool:
        return len(self.edge_ids) > 0


@dataclass
class NewArcEvaluation:
    """Résultat de l'évaluation Frank-Wolfe d'un candidat."""

    proposal: NewArcProposal
    new_vht_h: float
    baseline_vht_h: float
    delta_vht_h: float                       # > 0 = amélioration
    new_score_eur_year: float
    baseline_score_eur_year: float
    annual_benefit_eur: float
    payback_years: float
    cost_eur: float
    bcr: float
    score: float = field(default=0.0)

    @property
    def is_worth_it(self) -> bool:
        return self.annual_benefit_eur > 0 and self.cost_eur > 0


# ── Utilitaires ──────────────────────────────────────────────────────────────

def _shortest_length(g: ig.Graph, sources: list[int], targets: list[int]) -> np.ndarray:
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


_HW_RANK = {"residential": 0, "tertiary": 1, "secondary": 2, "primary": 3, "trunk": 4, "motorway": 5}


# ── Génération de corridors ───────────────────────────────────────────────────

def generate_corridor_candidates(
    network: UrbanNetwork,
    od: ODMatrix,
    *,
    ue: AssignmentResult | None = None,
    min_length_m: float = 300.0,
    max_length_m: float = 5_000.0,
    min_detour_ratio: float = 1.25,
    saturation_threshold: float = 0.65,
    max_candidates: int = 80,
) -> list[NewArcProposal]:
    """Identifie les corridors existants à élargir pour réduire la congestion.

    Pour chaque paire (u, v) de nœuds OD avec un fort détour :
    - Cherche le chemin sur le réseau réel en pénalisant les arcs saturés.
    - Le corridor trouvé suit des rues existantes (pas de traversée de bâtiment).
    - Scoré par (détour_gagné × demande × 1 + saturation_moy).

    Args:
        ue: résultat UE courant (pour calculer la saturation). Si None, on
            suppose aucune saturation (routing sur le plus court chemin libre).
    """
    g = network.graph
    nodes_xy = network.nodes_xy
    n_edges = g.ecount()

    capacity = np.asarray(g.es["capacity"], dtype=float)
    lengths = np.asarray(g.es["length_m"], dtype=float)
    hw_types = g.es["highway"]
    free_speeds = g.es["free_speed_kmh"]

    if ue is not None:
        sat_arr = ue.flows / np.maximum(capacity, 1.0)
    else:
        sat_arr = np.zeros(n_edges)

    # Poids pour le routing : on pénalise fortement les arcs saturés
    route_weights = lengths.copy()
    route_weights[sat_arr > saturation_threshold] *= 30.0

    demand_nodes = sorted({od.zone_to_node[z] for z in od.zone_ids})
    if len(demand_nodes) < 2:
        return []
    logger.info(f"NDP corridors — {len(demand_nodes)} nœuds OD candidats")

    od_lookup = _zone_demand_lookup(od)
    coords = np.array([nodes_xy[n] for n in demand_nodes])
    path_lengths = _shortest_length(g, demand_nodes, demand_nodes)

    proposals: list[tuple[float, NewArcProposal]] = []
    n = len(demand_nodes)

    for i in range(n):
        for j in range(i + 1, n):
            ux, uy = coords[i]
            vx, vy = coords[j]
            euclid = float(hypot(ux - vx, uy - vy))

            path_len = min(path_lengths[i, j], path_lengths[j, i])
            if not np.isfinite(path_len) or path_len <= 0:
                continue
            detour = path_len / euclid
            if detour < min_detour_ratio:
                continue

            u_node = demand_nodes[i]
            v_node = demand_nodes[j]

            # Trouver le corridor (routing avec pénalité saturation)
            epaths = g.get_shortest_paths(
                u_node, v_node,
                weights=route_weights.tolist(),
                output="epath",
            )
            if not epaths or not epaths[0]:
                continue
            edge_ids = epaths[0]

            corridor_length = float(sum(lengths[e] for e in edge_ids))
            if corridor_length < min_length_m or corridor_length > max_length_m:
                continue

            # Coordonnées réelles du tracé (pour la viz)
            vpaths = g.get_shortest_paths(
                u_node, v_node,
                weights=route_weights.tolist(),
                output="vpath",
            )
            vpath = vpaths[0] if vpaths else []
            corridor_xy = [nodes_xy[v] for v in vpath if v in nodes_xy]

            # Propriétés du corridor
            hw_list = [hw_types[e] for e in edge_ids]
            corridor_hw = max(hw_list, key=lambda h: _HW_RANK.get(h, 0))
            corridor_cap = float(np.min([capacity[e] for e in edge_ids]))
            corridor_speed = float(np.min([free_speeds[e] for e in edge_ids]))
            avg_sat = float(np.mean([sat_arr[e] for e in edge_ids]))

            cost_per_m = _widening_cost_per_m(corridor_hw)
            total_cost = corridor_length * cost_per_m

            demand = od_lookup.get((u_node, v_node), 0.0) + od_lookup.get((v_node, u_node), 0.0)
            proxy = (path_len - euclid) * max(1.0, demand) * (1.0 + avg_sat)

            proposals.append((
                proxy,
                NewArcProposal(
                    u_node=u_node, v_node=v_node,
                    length_m=corridor_length,
                    highway=corridor_hw,
                    capacity=corridor_cap,
                    free_speed_kmh=corridor_speed,
                    construction_cost_eur=total_cost,
                    detour_before=detour,
                    u_xy=(ux, uy), v_xy=(vx, vy),
                    edge_ids=edge_ids,
                    corridor_xy=corridor_xy,
                ),
            ))

    proposals.sort(key=lambda x: -x[0])
    top = [p for _, p in proposals[:max_candidates]]
    logger.info(f"NDP corridors — {len(top)} corridors retenus (sur {len(proposals)} générés)")
    return top


# ── Génération d'arcs en ligne droite (mode hérité, conservé pour les tests) ─

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
    """Mode hérité : propose des arcs en ligne droite (peut traverser des bâtiments).

    Conservé pour la rétrocompatibilité des tests. Le pipeline principal utilise
    désormais ``generate_corridor_candidates``.
    """
    g = network.graph
    nodes_xy = network.nodes_xy

    demand_nodes = sorted({od.zone_to_node[z] for z in od.zone_ids})
    if len(demand_nodes) < 2:
        return []

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

            path_len = min(paths[i, j], paths[j, i])
            if not np.isfinite(path_len) or path_len <= 0:
                continue
            detour = path_len / euclid
            if detour < min_detour_ratio:
                continue

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


# ── Contextes temporaires (modification réversible du graphe) ─────────────────

class _CorridorContext:
    """Double temporairement la capacité des arcs du corridor. Restauration garantie."""

    def __init__(self, g: ig.Graph, prop: NewArcProposal, uplift: float = 2.0):
        self.g = g
        self.edge_ids = prop.edge_ids
        self.uplift = uplift
        self.original_caps: list[float] = []

    def __enter__(self):
        caps = list(self.g.es["capacity"])
        for eid in self.edge_ids:
            self.original_caps.append(caps[eid])
            caps[eid] = caps[eid] * self.uplift
        self.g.es["capacity"] = caps
        return self

    def __exit__(self, exc_type, exc, tb):
        caps = list(self.g.es["capacity"])
        for eid, orig in zip(self.edge_ids, self.original_caps):
            caps[eid] = orig
        self.g.es["capacity"] = caps


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


# ── Évaluation Frank-Wolfe ────────────────────────────────────────────────────

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
    ctx = _CorridorContext(g, prop) if prop.is_corridor else _NewArcContext(g, prop)
    with ctx:
        new_ue = solve_user_equilibrium(network, od, max_iter=max_iter, tol=tol)

    new_score = score_network(new_ue, profile)
    delta_vht = baseline_ue.vht - new_ue.vht
    annual_benefit = baseline_score.total_annual_cost_eur - new_score.total_annual_cost_eur

    cost = prop.construction_cost_eur * profile.w_construction
    bcr = annual_benefit / cost if cost > 0 else 0.0
    payback = cost / annual_benefit if annual_benefit > 0 else float("inf")
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


# ── Pré-filtrage AoN (mode arc hérité uniquement) ────────────────────────────

def _quick_aon_filter(
    network: UrbanNetwork,
    od: ODMatrix,
    candidates: list[NewArcProposal],
    keep_top: int,
) -> list[NewArcProposal]:
    """Pré-filtre par All-or-Nothing pour le mode arc (hérité)."""
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


# ── Façade ────────────────────────────────────────────────────────────────────

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
    building_index: ObstacleIndex | None = None,
) -> tuple[list[NewArcEvaluation], CityScore]:
    """Pipeline complet : génère et évalue deux types d'interventions.

    - **Corridors** : élargissement (×2 capacité) d'un chemin sur le réseau existant.
    - **Nouvelles routes** : arc en ligne droite filtré par l'index d'obstacles
      (bâtiments, eau, parcs, voies ferrées) — uniquement si *building_index* fourni.

    Les deux types sont mélangés dans les évaluations FW (split 60/40).

    Returns:
        ``(plan, baseline_score)`` — liste des interventions retenues (BCR décroissant).
    """
    baseline_score = score_network(baseline_ue, profile)
    logger.info(f"Score baseline ({profile.name}) : {baseline_score.composite_score:,.0f} €/an")

    # ── Corridors à élargir ───────────────────────────────────────────────
    n_corridor_slots = max(1, int(max_fw_evals * 0.6))
    corridors = generate_corridor_candidates(
        network, od,
        ue=baseline_ue,
        max_candidates=max_proposals,
    )

    # ── Nouvelles routes (ligne droite filtrée obstacles) ─────────────────
    n_new_slots = max_fw_evals - n_corridor_slots
    new_arcs: list[NewArcProposal] = []
    if building_index is not None and n_new_slots > 0:
        new_arcs = generate_new_arc_candidates(
            network, od,
            max_candidates=max_proposals,
            building_index=building_index,
        )
        logger.info(f"NDP nouvelles routes — {len(new_arcs)} candidats (filtre obstacles actif)")
    elif n_new_slots > 0:
        logger.info("NDP nouvelles routes — désactivé (pas d'obstacle_index fourni)")

    # ── Sélection des candidats à évaluer (60 % corridors, 40 % nouvelles routes) ─
    pre: list[NewArcProposal] = []
    pre += corridors[:n_corridor_slots]
    pre += new_arcs[:n_new_slots]
    # Si un type en manque, on complète avec l'autre
    shortage = max_fw_evals - len(pre)
    if shortage > 0:
        extra_c = corridors[n_corridor_slots: n_corridor_slots + shortage]
        extra_n = new_arcs[n_new_slots: n_new_slots + shortage]
        pre += extra_c or extra_n

    if not pre:
        logger.warning("Aucun candidat trouvé.")
        return [], baseline_score

    logger.info(
        f"NDP — {len(pre)} candidats ({sum(p.is_corridor for p in pre)} corridors + "
        f"{sum(not p.is_corridor for p in pre)} nouvelles routes) pour évaluation FW"
    )

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

    evals_pos = [e for e in evals if e.is_worth_it]
    evals_pos.sort(key=lambda e: -e.bcr)
    chosen: list[NewArcEvaluation] = []
    spent = 0.0
    for ev in evals_pos:
        if spent + ev.cost_eur > budget_eur:
            continue
        chosen.append(ev)
        spent += ev.cost_eur

    n_c = sum(e.proposal.is_corridor for e in chosen)
    n_n = sum(not e.proposal.is_corridor for e in chosen)
    logger.info(
        f"NDP — {len(chosen)} interventions retenues ({n_c} corridors + {n_n} nouvelles routes), "
        f"coût total = {spent:,.0f}€ (budget {budget_eur:,.0f}€)"
    )
    return chosen, baseline_score
