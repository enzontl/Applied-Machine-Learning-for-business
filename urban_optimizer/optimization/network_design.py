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
from shapely.geometry import LineString, MultiPoint, box as _box

from urban_optimizer.assignment.frank_wolfe import (
    solve_all_or_nothing,
    solve_user_equilibrium,
)
from urban_optimizer.assignment.result import AssignmentResult
from urban_optimizer.config import BPR_ALPHA, BPR_BETA
from urban_optimizer.demand.od_matrix import ODMatrix
from shapely.strtree import STRtree as _STRtree
from shapely.geometry import Point as _Point

from urban_optimizer.network.buildings import BuildingIndex, ObstacleIndex
from urban_optimizer.network.urban_network import UrbanNetwork
from urban_optimizer.diagnosis.accessibility import compute_accessibility
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
    # Mode routé sur rues existantes — liste des edge IDs et coordonnées
    edge_ids: list[int] = field(default_factory=list)
    corridor_xy: list[tuple[float, float]] = field(default_factory=list)
    # "corridor" = élargissement axe saturé | "upgrade" = mise à niveau rues locales
    proposal_type: str = field(default="corridor")

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


@dataclass
class JointPlanResult:
    """Évaluation jointe d'un plan complet (toutes interventions simultanées).

    La somme des ΔVHT individuels surestime le gain réel (interactions négatives
    entre corridors parallèles, par ex.). Ce dataclass mesure le gain effectif.
    """

    n_interventions: int
    joint_vht_h: float                       # VHT total après plan
    baseline_vht_h: float                    # VHT avant plan
    naive_sum_delta_vht_h: float             # somme des ΔVHT individuels (overestimation)
    joint_delta_vht_h: float                 # ΔVHT effectif (joint)
    joint_annual_benefit_eur: float          # bénéfice annuel pondéré effectif
    naive_sum_annual_benefit_eur: float      # somme naïve des bénéfices individuels
    total_cost_eur: float                    # CAPEX cumulé du plan
    joint_bcr: float                         # bénéfice joint / CAPEX
    # Saturation v/c des arcs existants APRÈS plan (indexée comme network.graph.es)
    existing_sat_after: np.ndarray = field(
        default_factory=lambda: np.zeros(0, dtype=float)
    )
    # Accessibilité avant / après (# moyen de zones joignables sous le seuil)
    accessibility_before: float = 0.0
    accessibility_after: float = 0.0
    # Coefficient de Gini avant / après (0 = parfaite équité)
    gini_before: float = 0.0
    gini_after: float = 0.0
    # Induced demand (rebound effect)
    induced_iter: int = 0                    # nb d'itérations FW-induced (0 = désactivé)
    induced_trip_share: float = 0.0          # (T_final - T_baseline) / T_baseline
    induced_elasticity: float = 0.0          # ε utilisé (0 si désactivé)

    @property
    def redundancy_factor(self) -> float:
        """1.0 = pas de redondance. 0.7 = 30% du bénéfice naïf est illusoire."""
        if self.naive_sum_delta_vht_h <= 0:
            return 1.0
        return self.joint_delta_vht_h / self.naive_sum_delta_vht_h


# ── Utilitaires ──────────────────────────────────────────────────────────────

def _shortest_length(g: ig.Graph, sources: list[int], targets: list[int]) -> np.ndarray:
    # Pré-extraire les poids en liste (igraph est plus rapide avec list qu'un attribut string)
    w = g.es["length_m"]
    mat = g.distances(source=sources, target=targets, weights=w)
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


def _vpath_from_epath(g: ig.Graph, source: int, epath: list[int]) -> list[int]:
    """Reconstruit le vpath à partir du epath en marchant les arcs (O(len(epath))).

    Évite un second appel Dijkstra : on parcourt les arcs et on suit l'extrémité
    opposée à celle d'où on vient. Marche pour graphe orienté ou non.
    """
    vp = [source]
    cur = source
    for eid in epath:
        edge = g.es[eid]
        nxt = edge.target if edge.source == cur else edge.source
        vp.append(nxt)
        cur = nxt
    return vp


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
    max_candidates: int = 40,
    _shared: dict | None = None,
) -> list[NewArcProposal]:
    """Identifie les corridors existants à élargir pour réduire la congestion.

    Pour chaque paire (u, v) de nœuds OD avec un fort détour :
    - Cherche le chemin sur le réseau réel en pénalisant les arcs saturés.
    - Le corridor trouvé suit des rues existantes (pas de traversée de bâtiment).
    - Scoré par (détour_gagné × demande × 1 + saturation_moy).

    Args:
        ue: résultat UE courant (pour calculer la saturation). Si None, on
            suppose aucune saturation (routing sur le plus court chemin libre).
        _shared: dict pré-calculé par _generate_proposals (demand_nodes,
            path_lengths, od_lookup, coords). Évite les recalculs O(n²).
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

    if _shared is not None:
        demand_nodes = _shared["demand_nodes"]
        path_lengths = _shared["path_lengths"]
        od_lookup = _shared["od_lookup"]
        coords = _shared["coords"]
    else:
        demand_nodes = sorted({od.zone_to_node[z] for z in od.zone_ids})
        od_lookup = _zone_demand_lookup(od)
        coords = np.array([nodes_xy[n] for n in demand_nodes])
        path_lengths = _shortest_length(g, demand_nodes, demand_nodes)

    if len(demand_nodes) < 2:
        return []
    logger.info(f"NDP corridors — {len(demand_nodes)} nœuds OD candidats")

    # Conversion .tolist() UNE seule fois (au lieu d'une fois par paire)
    route_weights_list = route_weights.tolist()

    proposals: list[tuple[float, NewArcProposal]] = []
    n = len(demand_nodes)

    # Batché par source : 1 Dijkstra par source (n appels) au lieu de n²/2 paires
    for i in range(n):
        # Pré-filtrage des paires (j) par détour avant tout Dijkstra
        valid_js: list[tuple[int, float, float, float]] = []  # (j, euclid, path_len, detour)
        ux, uy = coords[i]
        for j in range(i + 1, n):
            vx, vy = coords[j]
            euclid = float(hypot(ux - vx, uy - vy))
            path_len = min(path_lengths[i, j], path_lengths[j, i])
            if not np.isfinite(path_len) or path_len <= 0:
                continue
            detour = path_len / euclid
            if detour < min_detour_ratio:
                continue
            valid_js.append((j, euclid, path_len, detour))

        if not valid_js:
            continue

        u_node = demand_nodes[i]
        targets = [demand_nodes[j] for j, _, _, _ in valid_js]
        # UN seul appel Dijkstra pour tous les targets, en epath
        epaths_list = g.get_shortest_paths(
            u_node, to=targets, weights=route_weights_list, output="epath",
        )

        for (j, euclid, path_len, detour), edge_ids in zip(valid_js, epaths_list):
            if not edge_ids:
                continue

            # Vectorisé : indexation numpy directe sur le corridor
            eids_arr = np.asarray(edge_ids, dtype=np.intp)
            corridor_length = float(lengths[eids_arr].sum())
            if corridor_length < min_length_m or corridor_length > max_length_m:
                continue

            v_node = demand_nodes[j]
            # vpath dérivé du epath (pas de second Dijkstra)
            vpath = _vpath_from_epath(g, u_node, edge_ids)
            corridor_xy = [nodes_xy[v] for v in vpath if v in nodes_xy]

            # Propriétés du corridor — indexation numpy vectorisée
            hw_list = [hw_types[e] for e in edge_ids]
            corridor_hw = max(hw_list, key=lambda h: _HW_RANK.get(h, 0))
            corridor_cap = float(capacity[eids_arr].min())
            corridor_speed = float(np.min([free_speeds[e] for e in edge_ids]))
            avg_sat = float(sat_arr[eids_arr].mean())

            cost_per_m = _widening_cost_per_m(corridor_hw)
            total_cost = corridor_length * cost_per_m

            demand = od_lookup.get((u_node, v_node), 0.0) + od_lookup.get((v_node, u_node), 0.0)
            proxy = (path_len - euclid) * max(1.0, demand) * (1.0 + avg_sat)

            vx, vy = coords[j]
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


def generate_upgrade_candidates(
    network: UrbanNetwork,
    od: ODMatrix,
    *,
    ue: AssignmentResult | None = None,
    min_length_m: float = 400.0,
    max_length_m: float = 6_000.0,
    min_detour_ratio: float = 1.25,
    capacity_threshold: float = 1_800.0,
    saturation_avoid: float = 0.35,
    max_candidates: int = 40,
    _shared: dict | None = None,
) -> list[NewArcProposal]:
    """Identifie des petites rues à transformer en axe principal (mise à niveau).

    Contrairement aux corridors (qui élargissent des axes déjà utilisés), cette
    fonction cherche des chemins passant par des rues de **faible capacité et peu
    chargées** (résidentiel, tertiary) qui pourraient devenir un itinéraire
    alternatif si elles étaient mises à niveau.

    Le tracé suit les rues existantes — aucun bâtiment ne peut être traversé.
    La simulation est identique aux corridors (capacité ×2 via _CorridorContext).

    Args:
        _shared: dict pré-calculé (demand_nodes, path_lengths, od_lookup, coords).
    """
    g = network.graph
    nodes_xy = network.nodes_xy
    n_edges = g.ecount()

    capacity = np.asarray(g.es["capacity"], dtype=float)
    lengths = np.asarray(g.es["length_m"], dtype=float)
    hw_types = g.es["highway"]
    free_speeds = g.es["free_speed_kmh"]

    sat_arr = ue.flows / np.maximum(capacity, 1.0) if ue is not None else np.zeros(n_edges)

    # Poids : privilégier rues peu chargées et faible capacité
    route_weights = lengths.copy()
    route_weights[sat_arr > saturation_avoid] *= 8.0
    route_weights[capacity > capacity_threshold] *= 4.0

    if _shared is not None:
        demand_nodes = _shared["demand_nodes"]
        path_lengths = _shared["path_lengths"]
        od_lookup = _shared["od_lookup"]
        coords = _shared["coords"]
    else:
        demand_nodes = sorted({od.zone_to_node[z] for z in od.zone_ids})
        od_lookup = _zone_demand_lookup(od)
        coords = np.array([nodes_xy[n] for n in demand_nodes])
        path_lengths = _shortest_length(g, demand_nodes, demand_nodes)

    if len(demand_nodes) < 2:
        return []
    logger.info(f"NDP mise à niveau — {len(demand_nodes)} nœuds OD candidats")

    # Conversion .tolist() UNE seule fois
    route_weights_list = route_weights.tolist()

    proposals: list[tuple[float, NewArcProposal]] = []
    n = len(demand_nodes)

    # Batché par source : 1 Dijkstra par source au lieu de n²/2 paires
    for i in range(n):
        valid_js: list[tuple[int, float, float, float]] = []
        ux, uy = coords[i]
        for j in range(i + 1, n):
            vx, vy = coords[j]
            euclid = float(hypot(ux - vx, uy - vy))
            path_len = min(path_lengths[i, j], path_lengths[j, i])
            if not np.isfinite(path_len) or path_len <= 0:
                continue
            detour = path_len / euclid
            if detour < min_detour_ratio:
                continue
            valid_js.append((j, euclid, path_len, detour))

        if not valid_js:
            continue

        u_node = demand_nodes[i]
        targets = [demand_nodes[j] for j, _, _, _ in valid_js]
        epaths_list = g.get_shortest_paths(
            u_node, to=targets, weights=route_weights_list, output="epath",
        )

        for (j, euclid, path_len, detour), edge_ids in zip(valid_js, epaths_list):
            if not edge_ids:
                continue

            # Vectorisé : indexation numpy directe
            eids_arr = np.asarray(edge_ids, dtype=np.intp)
            edge_caps = capacity[eids_arr]
            if float(np.median(edge_caps)) > capacity_threshold:
                continue  # déjà un axe principal — pas une mise à niveau

            corridor_length = float(lengths[eids_arr].sum())
            if corridor_length < min_length_m or corridor_length > max_length_m:
                continue

            v_node = demand_nodes[j]
            vpath = _vpath_from_epath(g, u_node, edge_ids)
            corridor_xy = [nodes_xy[v] for v in vpath if v in nodes_xy]

            hw_list = [hw_types[e] for e in edge_ids]
            corridor_hw = max(hw_list, key=lambda h: _HW_RANK.get(h, 0))
            corridor_cap = float(edge_caps.min())
            corridor_speed = float(np.min([free_speeds[e] for e in edge_ids]))
            avg_sat = float(sat_arr[eids_arr].mean())

            cost_per_m = _widening_cost_per_m(corridor_hw)
            total_cost = corridor_length * cost_per_m

            demand = od_lookup.get((u_node, v_node), 0.0) + od_lookup.get((v_node, u_node), 0.0)
            proxy = (path_len - euclid) * max(1.0, demand) * (1.0 + (1.0 - avg_sat))

            vx, vy = coords[j]
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
                    proposal_type="upgrade",
                ),
            ))

    proposals.sort(key=lambda x: -x[0])
    top = [p for _, p in proposals[:max_candidates]]
    logger.info(f"NDP mise à niveau — {len(top)} candidats retenus (sur {len(proposals)} générés)")
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


# ── Routage A* pour nouvelles routes (évitement d'obstacles) ─────────────────

# Coût de construction d'un pont (€/m) — voies ferrées ou cours d'eau
_BRIDGE_COST_PER_M = 18_000.0


def _extract_corners(geom) -> list[tuple[float, float]]:
    """Renvoie la liste des sommets (extérieur + trous) d'un polygone ou multipolygone."""
    pts: list[tuple[float, float]] = []
    if geom.geom_type == "Polygon":
        polys = [geom]
    elif geom.geom_type == "MultiPolygon":
        polys = list(geom.geoms)
    else:
        return pts
    for poly in polys:
        # Coordonnées de l'extérieur (on saute le point de fermeture)
        for x, y in list(poly.exterior.coords)[:-1]:
            pts.append((float(x), float(y)))
        for ring in poly.interiors:
            for x, y in list(ring.coords)[:-1]:
                pts.append((float(x), float(y)))
    return pts


def _route_avoiding_obstacles(
    u_xy: tuple[float, float],
    v_xy: tuple[float, float],
    hard_geoms: list,
    soft_geoms: list,
    *,
    margin_m: float = 400.0,
    max_corners: int = 80,
    bridge_factor: float = 5.0,
    buf_m: float = 7.0,
    shrink_m: float = 1.0,
    _hard_tree=None,
    _soft_tree=None,
) -> tuple[list[tuple[float, float]] | None, float]:
    """A* sur **graphe de visibilité** entre u et v.

    Contrairement à une grille uniforme (rigide, génère des zigzags), on utilise
    les **coins des obstacles** comme waypoints. Une arête est créée entre deux
    waypoints si une ligne droite peut les relier sans traverser d'obstacle dur.
    Cela produit des tracés naturels qui longent les bâtiments.

    Obstacles durs (bâtiments, parcs) : contournés.
    Obstacles doux (voies ferrées, eau) : traversés par pont, coût × bridge_factor.

    Returns:
        (chemin, longueur_pont_m) ou (None, 0) si aucun chemin trouvé.
    """
    from heapq import heappop, heappush

    # ── Bbox de recherche autour du segment u-v ──
    x_min = min(u_xy[0], v_xy[0]) - margin_m
    x_max = max(u_xy[0], v_xy[0]) + margin_m
    y_min = min(u_xy[1], v_xy[1]) - margin_m
    y_max = max(u_xy[1], v_xy[1]) + margin_m
    bbox = _box(x_min, y_min, x_max, y_max)

    hard_tree = _hard_tree if _hard_tree is not None else (_STRtree(hard_geoms) if hard_geoms else None)
    soft_tree = _soft_tree if _soft_tree is not None else (_STRtree(soft_geoms) if soft_geoms else None)

    # ── Extraction des coins d'obstacles dans le bbox ──
    relevant_indices: list[int] = []
    if hard_tree is not None:
        relevant_indices = [int(i) for i in hard_tree.query(bbox)]

    corners_with_owner: list[tuple[float, float, int]] = []
    for owner_idx in relevant_indices:
        for x, y in _extract_corners(hard_geoms[owner_idx]):
            if x_min <= x <= x_max and y_min <= y <= y_max:
                corners_with_owner.append((x, y, owner_idx))

    # Dédupliquer (tolérance 1 m — un coin partagé entre 2 polygones adjacents)
    seen: set[tuple[float, float]] = set()
    dedup: list[tuple[float, float, int]] = []
    for x, y, oid in corners_with_owner:
        key = (round(x, 0), round(y, 0))
        if key not in seen:
            seen.add(key)
            dedup.append((x, y, oid))

    # Limiter aux plus pertinents (proches du segment u-v direct)
    if len(dedup) > max_corners:
        line_uv = LineString([u_xy, v_xy])
        dedup.sort(key=lambda c: line_uv.distance(_Point(c[0], c[1])))
        dedup = dedup[:max_corners]

    # ── Waypoints : u, v, puis coins ──
    waypoints: list[tuple[float, float]] = [u_xy, v_xy] + [(x, y) for x, y, _ in dedup]
    owners: list[int] = [-1, -1] + [oid for _, _, oid in dedup]
    n = len(waypoints)
    start_i, end_i = 0, 1

    # ── Test de visibilité entre 2 waypoints ──
    def _edge_cost(i: int, j: int) -> float | None:
        """Coût d'une arête de visibilité (None si bloquée par un obstacle dur)."""
        xa, ya = waypoints[i]
        xb, yb = waypoints[j]
        # Longueur euclidienne directe (évite seg.length qui crée déjà LineString)
        dx, dy = xb - xa, yb - ya
        seg_len = hypot(dx, dy)
        if seg_len < 0.5:
            return None

        own_i, own_j = owners[i], owners[j]

        # Pré-check rapide par bbox : si aucun obstacle dur (hors owners) ne touche
        # la bbox élargie du segment, pas besoin de buffer + intersects (coûteux).
        has_potential_hard_hit = False
        if hard_tree is not None:
            seg_bbox = _box(
                min(xa, xb) - buf_m, min(ya, yb) - buf_m,
                max(xa, xb) + buf_m, max(ya, yb) + buf_m,
            )
            for h_idx in hard_tree.query(seg_bbox):
                h = int(h_idx)
                if h != own_i and h != own_j:
                    has_potential_hard_hit = True
                    break
            # Le cas same-owner force aussi un check précis
            if own_i >= 0 and own_i == own_j:
                has_potential_hard_hit = True

        # On ne construit le segment + buffer que si nécessaire
        seg = LineString([(xa, ya), (xb, yb)])
        buf_inner = None  # construit à la demande

        if has_potential_hard_hit:
            # On rétracte le segment de shrink_m pour permettre aux endpoints
            # d'être SUR un obstacle (coin de bâtiment) sans rejet immédiat
            if seg_len > 2 * shrink_m + 0.5:
                inner = LineString([
                    seg.interpolate(shrink_m),
                    seg.interpolate(seg_len - shrink_m),
                ])
            else:
                inner = seg
            buf_inner = inner.buffer(buf_m)

            for h_idx in hard_tree.query(buf_inner):
                h = int(h_idx)
                # On ignore les polygones qui possèdent un endpoint
                if h == own_i or h == own_j:
                    continue
                if hard_geoms[h].intersects(buf_inner):
                    return None
            # Si les 2 endpoints partagent le même owner, on vérifie quand même
            # que le segment ne traverse pas l'intérieur de ce polygone
            if own_i >= 0 and own_i == own_j:
                if hard_geoms[own_i].intersects(buf_inner):
                    return None

        # Surcoût pont si traversée d'un obstacle doux
        bridge_len = 0.0
        if soft_tree is not None:
            hits = soft_tree.query(seg)
            for h_idx in hits:
                if soft_geoms[int(h_idx)].intersects(seg):
                    bridge_len = seg_len
                    break

        return seg_len + bridge_len * (bridge_factor - 1.0)

    # ── Construction du graphe de visibilité (complet sur n waypoints) ──
    adj: list[list[tuple[int, float]]] = [[] for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            cost = _edge_cost(i, j)
            if cost is None:
                continue
            adj[i].append((j, cost))
            adj[j].append((i, cost))

    # ── A* heuristique = distance euclidienne ──
    ex, ey = v_xy

    def _h(i: int) -> float:
        return hypot(waypoints[i][0] - ex, waypoints[i][1] - ey)

    heap: list = [(float(_h(start_i)), 0.0, start_i)]
    dist: dict[int, float] = {start_i: 0.0}
    parent: dict[int, int | None] = {start_i: None}
    visited: set[int] = set()

    while heap:
        _, g, cur = heappop(heap)
        if cur in visited:
            continue
        visited.add(cur)
        if cur == end_i:
            path: list[tuple[float, float]] = []
            node: int | None = end_i
            while node is not None:
                path.append(waypoints[node])
                node = parent[node]
            path.reverse()
            # Longueur effective en pont sur le chemin final
            bridge_m = 0.0
            if soft_tree is not None and len(path) > 1:
                for k in range(len(path) - 1):
                    seg = LineString([path[k], path[k + 1]])
                    hits = soft_tree.query(seg)
                    if any(soft_geoms[int(h)].intersects(seg) for h in hits):
                        bridge_m += seg.length
            return path, bridge_m
        for nb, cost in adj[cur]:
            if nb in visited:
                continue
            new_g = g + cost
            if new_g < dist.get(nb, float("inf")):
                dist[nb] = new_g
                parent[nb] = cur
                heappush(heap, (new_g + _h(nb), new_g, nb))

    return None, 0.0


def _build_periphery_core(nodes_xy: dict, periphery_margin_m: float):
    """Retourne le polygone 'noyau intérieur' (convex hull érodé).

    Un nœud dans ce polygone est considéré en centre-ville.
    Un nœud hors de ce polygone est en périphérie.
    Retourne None si le réseau est trop petit pour être érodé.
    """
    all_pts = list(nodes_xy.values())
    if len(all_pts) < 3:
        return None
    hull = MultiPoint(all_pts).convex_hull
    core = hull.buffer(-periphery_margin_m)
    if core.is_empty:
        return None
    return core


def generate_new_route_candidates(
    network: UrbanNetwork,
    od: ODMatrix,
    *,
    ue: AssignmentResult | None = None,
    obstacle_index: ObstacleIndex | None = None,
    soft_index: ObstacleIndex | None = None,
    min_length_m: float = 500.0,
    max_length_m: float = 6_000.0,
    min_detour_ratio: float = 1.40,
    max_candidates: int = 40,
    periphery_margin_m: float = 600.0,
    _shared: dict | None = None,
) -> list[NewArcProposal]:
    """Propose de nouvelles routes en zone périphérique uniquement.

    Mode rapide (v2) : on utilise la ligne droite entre paires OD à fort
    détour. Si un obstacle dur est sur le trajet, la paire est simplement
    ignorée (au lieu de lancer un A* coûteux sur graphe de visibilité).
    L'A* est réservé aux top-K paires les plus prometteuses pour ne pas
    exploser le temps de génération.

    Obstacles doux (voies ferrées, cours d'eau) : traversés par pont, coût majoré.

    Args:
        _shared: dict pré-calculé (demand_nodes, path_lengths, od_lookup, coords).
    """
    if obstacle_index is None:
        return []

    g = network.graph
    nodes_xy = network.nodes_xy

    hard_geoms = list(obstacle_index._tree.geometries)
    soft_geoms = list(soft_index._tree.geometries) if soft_index else []

    if _shared is not None:
        demand_nodes = _shared["demand_nodes"]
        path_lengths = _shared["path_lengths"]
        od_lookup = _shared["od_lookup"]
        coords = _shared["coords"]
    else:
        demand_nodes = sorted({od.zone_to_node[z] for z in od.zone_ids})
        od_lookup = _zone_demand_lookup(od)
        coords = np.array([nodes_xy[n] for n in demand_nodes])
        path_lengths = _shortest_length(g, demand_nodes, demand_nodes)

    if len(demand_nodes) < 2:
        return []

    # Noyau intérieur : paires dont les DEUX nœuds sont dedans → ignorées
    core = _build_periphery_core(nodes_xy, periphery_margin_m)
    if core is not None:
        logger.info(
            f"NDP nouvelles routes — filtre périphérie activé "
            f"(marge {periphery_margin_m:.0f}m, {core.area / 1e6:.1f} km² de noyau exclu)"
        )

    # Pré-construction des STRtrees — une seule fois pour toute la boucle
    hard_tree_pre = _STRtree(hard_geoms) if hard_geoms else None
    soft_tree_pre = _STRtree(soft_geoms) if soft_geoms else None

    # Pré-calcul du filtre périphérie pour chaque nœud OD
    if core is not None:
        in_core = [core.contains(_Point(*nodes_xy[n])) for n in demand_nodes]
    else:
        in_core = [False] * len(demand_nodes)

    # ── Phase 1 : scoring rapide (ligne droite uniquement) ────────────
    # On évalue TOUTES les paires par proxy, mais sans A* coûteux.
    # Les paires bloquées par un obstacle dur sont simplement ignorées.
    proposals: list[tuple[float, NewArcProposal]] = []
    n = len(demand_nodes)
    n_skipped_core = 0
    n_blocked = 0

    for i in range(n):
        for j in range(i + 1, n):
            if in_core[i] and in_core[j]:
                n_skipped_core += 1
                continue

            ux, uy = coords[i]
            vx, vy = coords[j]
            euclid = float(hypot(ux - vx, uy - vy))

            path_len = min(path_lengths[i, j], path_lengths[j, i])
            if not np.isfinite(path_len) or path_len <= 0:
                continue
            detour = path_len / euclid
            if detour < min_detour_ratio:
                continue

            # Test obstacle dur : ligne droite seulement (pas d'A*)
            straight = LineString([(ux, uy), (vx, vy)])
            if hard_tree_pre is not None:
                buf_seg = straight.buffer(7.0)
                hits = hard_tree_pre.query(buf_seg)
                if any(hard_geoms[h].intersects(buf_seg) for h in hits):
                    n_blocked += 1
                    continue

            # Ligne droite libre d'obstacles durs
            route_length = euclid
            if route_length < min_length_m or route_length > max_length_m:
                continue

            # Check obstacle doux (pont)
            bridge_len = 0.0
            if soft_tree_pre is not None:
                hits = soft_tree_pre.query(straight)
                if any(soft_geoms[h].intersects(straight) for h in hits):
                    bridge_len = euclid

            u_node = demand_nodes[i]
            v_node = demand_nodes[j]

            hw, cap, speed, normal_cost_per_m = _new_arc_specs(route_length)
            normal_len = route_length - bridge_len
            total_cost = normal_len * normal_cost_per_m + bridge_len * _BRIDGE_COST_PER_M

            demand = od_lookup.get((u_node, v_node), 0.0) + od_lookup.get((v_node, u_node), 0.0)
            proxy = (path_len - euclid) * max(1.0, demand)

            proposals.append((
                proxy,
                NewArcProposal(
                    u_node=u_node, v_node=v_node,
                    length_m=route_length,
                    highway=hw, capacity=cap, free_speed_kmh=speed,
                    construction_cost_eur=total_cost,
                    detour_before=detour,
                    u_xy=(ux, uy), v_xy=(vx, vy),
                    edge_ids=[],
                    corridor_xy=[(ux, uy), (vx, vy)],
                    proposal_type="new_route",
                ),
            ))

    if n_skipped_core > 0:
        logger.info(f"NDP nouvelles routes — {n_skipped_core} paires ignorées (centre-ville)")
    if n_blocked > 0:
        logger.info(f"NDP nouvelles routes — {n_blocked} paires bloquées (obstacle dur)")
    proposals.sort(key=lambda x: -x[0])
    top = [p for _, p in proposals[:max_candidates]]
    logger.info(f"NDP nouvelles routes — {len(top)} candidats retenus")
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


class _JointContext:
    """Applique simultanément TOUTES les interventions d'un plan.

    Corridors / upgrades : capacité des arcs ×uplift.
    Nouvelles routes : arcs bidirectionnels ajoutés au graphe.
    Restauration garantie (try/finally).

    Sert à mesurer l'effet RÉEL d'un plan complet — la somme des effets
    individuels surestime systématiquement le gain (deux corridors parallèles
    se cannibalisent).
    """

    def __init__(self, g: ig.Graph, proposals: list[NewArcProposal], uplift: float = 2.0):
        self.g = g
        self.proposals = proposals
        self.uplift = uplift
        self.corridor_edges: dict[int, float] = {}
        self.added_eids: list[int] = []

    def __enter__(self):
        # 1. Élargissements : bulk read-modify-write (igraph bulk est plus rapide que per-edge)
        caps = list(self.g.es["capacity"])
        for p in self.proposals:
            if p.is_corridor:
                for eid in p.edge_ids:
                    if eid not in self.corridor_edges:
                        self.corridor_edges[eid] = caps[eid]
                    caps[eid] = self.corridor_edges[eid] * self.uplift
        self.g.es["capacity"] = caps

        # 2. Nouvelles routes : on ajoute tous les arcs neufs d'un coup
        new_props = [p for p in self.proposals if not p.is_corridor]
        if new_props:
            before = self.g.ecount()
            edges_to_add: list[tuple[int, int]] = []
            attrs_lists: dict[str, list] = {
                "length_m": [], "free_speed_kmh": [], "capacity": [], "t0_s": [],
                "bpr_alpha": [], "bpr_beta": [], "highway": [], "source": [],
            }
            for p in new_props:
                t0 = p.length_m / (p.free_speed_kmh * 1000.0 / 3600.0)
                for u, v in [(p.u_node, p.v_node), (p.v_node, p.u_node)]:
                    edges_to_add.append((u, v))
                    attrs_lists["length_m"].append(p.length_m)
                    attrs_lists["free_speed_kmh"].append(p.free_speed_kmh)
                    attrs_lists["capacity"].append(p.capacity)
                    attrs_lists["t0_s"].append(t0)
                    attrs_lists["bpr_alpha"].append(BPR_ALPHA)
                    attrs_lists["bpr_beta"].append(BPR_BETA)
                    attrs_lists["highway"].append(p.highway)
                    attrs_lists["source"].append("new")
            self.g.add_edges(edges_to_add, attributes=attrs_lists)
            self.added_eids = list(range(before, self.g.ecount()))
        return self

    def __exit__(self, exc_type, exc, tb):
        # Restaurer les capacités (bulk)
        if self.corridor_edges:
            caps = list(self.g.es["capacity"])
            for eid, orig in self.corridor_edges.items():
                if eid < len(caps):
                    caps[eid] = orig
            self.g.es["capacity"] = caps
        # Supprimer les arcs ajoutés
        if self.added_eids:
            self.g.delete_edges(self.added_eids)
            self.added_eids = []


# ── Évaluation MARGINALE (sans FW) ──────────────────────────────────────────
#
# Au lieu de relancer un Frank-Wolfe complet pour chaque candidat (~25s chacun),
# on estime l'impact de chaque intervention directement depuis les flows BPR
# baseline. C'est l'approche standard en ingénierie de transport : on calcule
# la variation de VHT par analyse marginale du coût BPR.
#
# Pour un corridor (capacité ×2), le temps BPR diminue sur les arcs touchés :
#   Δt(e) = t0 * α * [(v/c_old)^β - (v/c_new)^β]
# Le gain en VHT ≈ Σ_arcs_corridor flow(e) * Δt(e) / 3600
#
# C'est un proxy rapide (O(1) par candidat) qui capture 80-90% de l'effet
# sans aucun appel Dijkstra. Le FW joint final reste pour la validation.

from urban_optimizer.optimization.score import (
    ANNUAL_PEAK_HOURS,
    VALUE_OF_TIME_EUR_H,
    FUEL_LITERS_PER_VEH_HOUR,
    FUEL_PRICE_EUR_L,
    KG_CO2_PER_LITER,
    SOCIAL_COST_CO2_EUR_KG,
)


def _warm_start_flows(baseline_flows: np.ndarray, n_now: int) -> np.ndarray | None:
    n_base = len(baseline_flows)
    if n_base == n_now:
        return baseline_flows
    if n_base < n_now:
        return np.concatenate([baseline_flows, np.zeros(n_now - n_base, dtype=float)])
    return None


def _marginal_evaluate(
    network: UrbanNetwork,
    prop: NewArcProposal,
    baseline_ue: AssignmentResult,
    baseline_score: CityScore,
    profile: MayorProfile,
) -> NewArcEvaluation:
    """Estime le bénéfice d'une intervention par analyse marginale BPR (SANS FW).

    Pour un corridor/upgrade (capacité ×2 sur des arcs existants) :
    On calcule la réduction de temps BPR sur chaque arc du corridor et on multiplie
    par le flow existant pour obtenir le ΔVHT.

    Pour une nouvelle route : on estime le trafic capté par la demande OD concernée.
    """
    g = network.graph
    flows = baseline_ue.flows
    t0_arr = np.asarray(g.es["t0_s"], dtype=float)
    cap_arr = np.asarray(g.es["capacity"], dtype=float)

    if prop.is_corridor and prop.edge_ids:
        # Corridor / upgrade : on double la capacité → la congestion BPR diminue
        eids = np.asarray(prop.edge_ids, dtype=np.intp)
        f = flows[eids]
        t0 = t0_arr[eids]
        c_old = cap_arr[eids]
        c_new = c_old * 2.0  # uplift ×2

        # Temps BPR avant et après
        alpha, beta = BPR_ALPHA, BPR_BETA
        vc_old = f / np.maximum(c_old, 1.0)
        vc_new = f / np.maximum(c_new, 1.0)
        time_old = t0 * (1.0 + alpha * vc_old ** beta)
        time_new = t0 * (1.0 + alpha * vc_new ** beta)

        # Gain en véhicule-secondes sur ces arcs
        delta_veh_seconds = float(np.sum(f * (time_old - time_new)))
        delta_vht_h = delta_veh_seconds / 3600.0
    else:
        # Nouvelle route : estimation simplifiée basée sur le détour capté
        # Le flow capté ≈ proportionnel à la demande OD × ratio de temps gagné
        t0_new = prop.length_m / (prop.free_speed_kmh * 1000.0 / 3600.0)
        # Proxy : le détour moyen donne le temps "avant" sur le réseau
        # Le gain est proportionnel au détour et à la demande
        time_before = t0_new * prop.detour_before
        delta_time_s = time_before - t0_new
        # Estimation grossière du flow capté (~10% de la capacité pour une route neuve)
        estimated_flow = prop.capacity * 0.10
        delta_vht_h = estimated_flow * delta_time_s / 3600.0

    # Conversion en bénéfice annuel (même formule que score.py)
    annual_delta_vht = delta_vht_h * ANNUAL_PEAK_HOURS
    benefit_time = annual_delta_vht * VALUE_OF_TIME_EUR_H * profile.w_time
    benefit_fuel = annual_delta_vht * FUEL_LITERS_PER_VEH_HOUR * FUEL_PRICE_EUR_L * profile.w_fuel
    benefit_co2 = (
        annual_delta_vht * FUEL_LITERS_PER_VEH_HOUR * KG_CO2_PER_LITER
        * SOCIAL_COST_CO2_EUR_KG * profile.w_co2
    )
    annual_benefit = benefit_time + benefit_fuel + benefit_co2

    cost = prop.construction_cost_eur * profile.w_construction
    bcr = annual_benefit / cost if cost > 0 else 0.0
    payback = cost / annual_benefit if annual_benefit > 0 else float("inf")

    new_vht = baseline_ue.vht - delta_vht_h
    new_score = baseline_score.total_annual_cost_eur - annual_benefit

    return NewArcEvaluation(
        proposal=prop,
        new_vht_h=new_vht,
        baseline_vht_h=baseline_ue.vht,
        delta_vht_h=delta_vht_h,
        new_score_eur_year=new_score,
        baseline_score_eur_year=baseline_score.total_annual_cost_eur,
        annual_benefit_eur=annual_benefit,
        payback_years=payback,
        cost_eur=cost,
        bcr=bcr,
        score=annual_benefit - cost / 20.0,
    )


# ── Helpers internes ──────────────────────────────────────────────────────────

def _generate_proposals(
    network: UrbanNetwork,
    od: ODMatrix,
    baseline_ue: AssignmentResult,
    *,
    max_proposals: int,
    max_fw_evals: int,
    obstacle_index: ObstacleIndex | None,
    soft_index: ObstacleIndex | None,
    periphery_margin_m: float,
    _max_demand_nodes: int = 30,
) -> list[NewArcProposal]:
    """Génère + remplit (corridors, upgrades, new routes) → liste `pre`.

    Optimisation v2 : la matrice de distances, od_lookup et coords sont
    calculés UNE seule fois et partagés entre les 3 générateurs.
    Les demand_nodes sont capés à ``_max_demand_nodes`` pour garder la
    complexité O(n²) raisonnable.
    """
    import time as _time
    _t0 = _time.perf_counter()

    n_corr = max(1, int(max_fw_evals * 0.50))
    n_upgr = max(1, int(max_fw_evals * 0.25))
    n_new = max_fw_evals - n_corr - n_upgr

    # ── 1. Demand nodes (capés pour performance) ───────────────────────
    all_demand = sorted({od.zone_to_node[z] for z in od.zone_ids})
    if len(all_demand) > _max_demand_nodes:
        # Garder les nœuds avec le plus de flux OD (les plus importants)
        od_lookup_full = _zone_demand_lookup(od)
        node_flow: dict[int, float] = {}
        for (u, v), trips in od_lookup_full.items():
            node_flow[u] = node_flow.get(u, 0.0) + trips
            node_flow[v] = node_flow.get(v, 0.0) + trips
        all_demand.sort(key=lambda n: node_flow.get(n, 0.0), reverse=True)
        demand_nodes = sorted(all_demand[:_max_demand_nodes])
        logger.info(
            f"Demand nodes capés : {len(all_demand)} → {len(demand_nodes)} "
            f"(top flux OD)"
        )
    else:
        demand_nodes = all_demand

    if len(demand_nodes) < 2:
        return []

    # ── 2. Pré-calcul partagé (1 seule fois pour les 3 générateurs) ───
    g = network.graph
    nodes_xy = network.nodes_xy

    # Vérification défensive : graphe déconnecté = Dijkstra 10-100× plus lent
    n_comp = len(g.connected_components(mode="weak"))
    if n_comp > 1:
        logger.warning(
            f"⚠ GRAPHE DÉCONNECTÉ ({n_comp} composantes) — "
            f"Supprimez data/raw/network_cache/*.pkl et relancez. "
            f"Dijkstra sera extrêmement lent tant que le cache n'est pas purgé."
        )

    _t1 = _time.perf_counter()
    path_lengths = _shortest_length(g, demand_nodes, demand_nodes)
    logger.info(
        f"  Matrice distances {len(demand_nodes)}×{len(demand_nodes)} : "
        f"{_time.perf_counter() - _t1:.2f}s"
    )

    od_lookup = _zone_demand_lookup(od)
    coords = np.array([nodes_xy[n] for n in demand_nodes])

    shared = {
        "demand_nodes": demand_nodes,
        "path_lengths": path_lengths,
        "od_lookup": od_lookup,
        "coords": coords,
    }

    # ── 3. Corridors ──────────────────────────────────────────────────
    _t1 = _time.perf_counter()
    corridors = generate_corridor_candidates(
        network, od, ue=baseline_ue, max_candidates=max_proposals,
        _shared=shared,
    )
    logger.info(f"  Corridors ({len(corridors)}) : {_time.perf_counter() - _t1:.2f}s")

    # ── 4. Upgrades ───────────────────────────────────────────────────
    _t1 = _time.perf_counter()
    upgrades = generate_upgrade_candidates(
        network, od, ue=baseline_ue, max_candidates=max_proposals,
        _shared=shared,
    )
    logger.info(f"  Upgrades ({len(upgrades)}) : {_time.perf_counter() - _t1:.2f}s")

    # ── 5. Nouvelles routes ───────────────────────────────────────────
    new_routes: list[NewArcProposal] = []
    if obstacle_index is not None and n_new > 0:
        _t1 = _time.perf_counter()
        new_routes = generate_new_route_candidates(
            network, od, ue=baseline_ue,
            obstacle_index=obstacle_index, soft_index=soft_index,
            max_candidates=max_proposals,
            periphery_margin_m=periphery_margin_m,
            _shared=shared,
        )
        logger.info(f"  Nouvelles routes ({len(new_routes)}) : {_time.perf_counter() - _t1:.2f}s")

    logger.info(
        f"  _generate_proposals total : {_time.perf_counter() - _t0:.2f}s"
    )

    def _fill(primary, secondary, n):
        picked = list(primary[:n])
        if len(picked) < n:
            picked += list(secondary[: n - len(picked)])
        return picked

    return (
        _fill(corridors,   upgrades,    n_corr)
        + _fill(upgrades,  corridors,   n_upgr)
        + _fill(new_routes, corridors,  n_new)
    )


def _greedy_select(
    evals: list[NewArcEvaluation],
    budget_eur: float,
) -> tuple[list[NewArcEvaluation], float]:
    """Sélection gloutonne par BCR décroissant sous contrainte de budget."""
    evals_pos = [e for e in evals if e.is_worth_it]
    evals_pos.sort(key=lambda e: -e.bcr)
    chosen: list[NewArcEvaluation] = []
    spent = 0.0
    for ev in evals_pos:
        if spent + ev.cost_eur > budget_eur:
            continue
        chosen.append(ev)
        spent += ev.cost_eur
    return chosen, spent


def _joint_re_evaluate(
    network: UrbanNetwork,
    od: ODMatrix,
    profile: MayorProfile,
    baseline_ue: AssignmentResult,
    baseline_score: CityScore,
    baseline_access_mean: float,
    baseline_gini: float,
    chosen: list[NewArcEvaluation],
    spent: float,
    *,
    fw_max_iter: int,
    fw_tol: float,
    accessibility_threshold_s: float,
    enable_induced_demand: bool = False,
    induced_elasticity: float = -0.6,
    induced_max_iter: int = 3,
    induced_tol: float = 0.01,
) -> JointPlanResult | None:
    """Évaluation jointe : FW (+ boucle induced demand optionnelle).

    Si ``enable_induced_demand`` :
        1. FW joint sur ``od`` (demande baseline).
        2. Calcule un OD ajusté par élasticité (réf. demand/induced.py).
        3. Re-FW joint sur OD ajusté.
        4. Itère jusqu'à convergence du VHT (|ΔVHT|/VHT < induced_tol) ou
           ``induced_max_iter``.
    Le score final est calculé sur le **dernier** UE atteint (et l'OD induit
    correspondant), donc reflète l'équilibre demande↔congestion.
    """
    if not chosen:
        return None
    from urban_optimizer.demand.induced import apply_induced_demand

    chosen_props = [e.proposal for e in chosen]
    naive_sum_dvht = sum(e.delta_vht_h for e in chosen)
    naive_sum_benef = sum(e.annual_benefit_eur for e in chosen)
    n_existing_pre = network.graph.ecount()
    base_total_trips = od.total_trips

    with _JointContext(network.graph, chosen_props):
        warm = _warm_start_flows(baseline_ue.flows, network.graph.ecount())

        # --- Boucle FW-induced ---
        cur_od = od
        joint_ue = solve_user_equilibrium(
            network, cur_od, max_iter=fw_max_iter, tol=fw_tol, initial_flows=warm,
        )
        induced_iter = 0
        if enable_induced_demand:
            prev_vht = joint_ue.vht
            for k in range(induced_max_iter):
                cur_od, ind_stats = apply_induced_demand(
                    cur_od, network, baseline_ue, joint_ue,
                    elasticity=induced_elasticity,
                )
                logger.info(f"  [induced iter {k+1}] {ind_stats.summary()}")
                joint_ue = solve_user_equilibrium(
                    network, cur_od, max_iter=fw_max_iter, tol=fw_tol,
                    initial_flows=joint_ue.flows,
                )
                induced_iter = k + 1
                rel = abs(joint_ue.vht - prev_vht) / max(prev_vht, 1.0)
                if rel < induced_tol:
                    logger.info(
                        f"  [induced] convergé en {induced_iter} itération(s) "
                        f"(|ΔVHT|/VHT = {rel*100:.2f}% < {induced_tol*100:.1f}%)"
                    )
                    break
                prev_vht = joint_ue.vht
        # --- Fin boucle ---

        caps_after = np.asarray(network.graph.es["capacity"], dtype=float)
        joint_access = compute_accessibility(
            network, cur_od, joint_ue, threshold_seconds=accessibility_threshold_s,
        )
    sat_existing_after = (
        joint_ue.flows[:n_existing_pre]
        / np.maximum(caps_after[:n_existing_pre], 1.0)
    )
    joint_score = score_network(joint_ue, profile, access=joint_access)
    joint_dvht = baseline_ue.vht - joint_ue.vht
    joint_benef = baseline_score.total_annual_cost_eur - joint_score.total_annual_cost_eur
    joint_bcr = joint_benef / spent if spent > 0 else 0.0
    induced_share = (
        (cur_od.total_trips - base_total_trips) / base_total_trips
        if base_total_trips > 0 else 0.0
    )
    return JointPlanResult(
        n_interventions=len(chosen),
        joint_vht_h=joint_ue.vht,
        baseline_vht_h=baseline_ue.vht,
        naive_sum_delta_vht_h=naive_sum_dvht,
        joint_delta_vht_h=joint_dvht,
        joint_annual_benefit_eur=joint_benef,
        naive_sum_annual_benefit_eur=naive_sum_benef,
        total_cost_eur=spent,
        joint_bcr=joint_bcr,
        existing_sat_after=sat_existing_after,
        accessibility_before=baseline_access_mean,
        accessibility_after=joint_access.mean_reachable,
        gini_before=baseline_gini,
        gini_after=joint_access.gini,
        induced_iter=induced_iter,
        induced_trip_share=induced_share,
        induced_elasticity=induced_elasticity if enable_induced_demand else 0.0,
    )


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
    fw_max_iter: int = 15,
    fw_tol: float = 5e-3,
    obstacle_index: ObstacleIndex | None = None,
    soft_index: ObstacleIndex | None = None,
    periphery_margin_m: float = 600.0,
    accessibility_threshold_s: float = 15 * 60,
    selection_method: str = "knapsack",  # "greedy" | "knapsack"
    enable_2swap: bool = True,
    swap_max_iter: int = 3,
    swap_top_k_out: int = 3,
    swap_top_k_in: int = 5,
    swap_max_fw_calls: int = 12,
    enable_induced_demand: bool = False,
    induced_elasticity: float = -0.6,
    induced_max_iter: int = 3,
    induced_tol: float = 0.01,
    forecast_model=None,                  # FlowForecastModel | None
    forecast_features=None,               # pd.DataFrame (features par commune)
    forecast_omphale=None,                # pd.DataFrame OMPHALE | None
    forecast_iris_to_commune=None,        # dict[str, str] | None
    forecast_horizon_years: int = 0,      # 0 = pas de projection
    _baseline_access=None,
) -> tuple[list[NewArcEvaluation], CityScore, JointPlanResult | None]:
    """Pipeline rapide : évaluation marginale + sélection + FW joint final.

    1. Génère des candidats (corridors, upgrades, nouvelles routes).
    2. Évalue chaque candidat par **analyse marginale BPR** (O(1), pas de FW).
    3. Sélection sous budget : ``"knapsack"`` (DP 0/1 exact, défaut) ou
       ``"greedy"`` (BCR décroissant, historique).
    4. Optionnel — ``enable_2swap`` : post-traitement local-search 2-swap qui
       rejoue le FW joint pour casser les redondances entre interventions
       parallèles (ex. deux corridors quasi-parallèles qui se cannibalisent).
    5. FW joint final avec toutes les interventions appliquées simultanément.

    L'évaluation marginale capture 80-90% de l'effet réel en < 0.01s par candidat
    (vs ~20-30s pour un FW complet). Le FW joint final donne le vrai ΔVHT.

    Si ``forecast_horizon_years > 0`` et ``forecast_model`` est fourni, l'OD
    est d'abord projetée à l'horizon H (via le module forecast) et tout le
    pipeline (baseline UE, candidats, FW joints) s'appuie sur cette OD
    future. Les travaux proposés sont donc dimensionnés pour la demande
    attendue dans 5-10 ans, pas pour celle d'aujourd'hui.
    """
    from .selection import knapsack_dp_select, local_search_2swap
    import time as _time
    _t0 = _time.perf_counter()
    _obs = obstacle_index

    # ── Projection horizon (optionnelle) ──
    if forecast_horizon_years > 0 and forecast_model is not None:
        from urban_optimizer.forecast import project_od_future
        if forecast_features is None or forecast_iris_to_commune is None:
            raise ValueError(
                "forecast_horizon_years > 0 requiert forecast_features et "
                "forecast_iris_to_commune."
            )
        import pandas as _pd
        omphale = forecast_omphale if forecast_omphale is not None else _pd.DataFrame()
        od_proj, proj_stats = project_od_future(
            od, forecast_model, forecast_features, omphale,
            forecast_iris_to_commune,
            horizon_years=forecast_horizon_years,
        )
        logger.info(f"NDP — {proj_stats.summary()}")
        # On bascule sur l'OD future + on re-solve l'UE baseline dessus
        od = od_proj
        baseline_ue = solve_user_equilibrium(
            network, od, max_iter=fw_max_iter, tol=fw_tol,
        )
        _baseline_access = None  # invalidé car OD a changé

    baseline_access = _baseline_access or compute_accessibility(
        network, od, baseline_ue, threshold_seconds=accessibility_threshold_s,
    )
    baseline_score = score_network(baseline_ue, profile, access=baseline_access)
    logger.info(
        f"Score baseline ({profile.name}) : {baseline_score.composite_score:,.0f} €/an "
        f"(accès moyen = {baseline_access.mean_reachable:.1f} zones, "
        f"Gini = {baseline_access.gini:.2f})"
    )

    # 1. Génération de candidats
    pre = _generate_proposals(
        network, od, baseline_ue,
        max_proposals=max_proposals, max_fw_evals=max_fw_evals,
        obstacle_index=_obs, soft_index=soft_index,
        periphery_margin_m=periphery_margin_m,
    )
    if not pre:
        logger.warning("Aucun candidat trouvé.")
        return [], baseline_score, None
    counts = {t: sum(p.proposal_type == t for p in pre)
              for t in ("corridor", "upgrade", "new_route")}
    logger.info(
        f"NDP — {len(pre)} candidats générés : "
        f"{counts['corridor']} corridors + {counts['upgrade']} upgrades + "
        f"{counts['new_route']} nouvelles routes "
        f"({_time.perf_counter() - _t0:.1f}s)"
    )

    # 2. Évaluation marginale INSTANTANÉE (pas de FW !)
    _t1 = _time.perf_counter()
    evals: list[NewArcEvaluation] = []
    for i, prop in enumerate(pre, 1):
        ev = _marginal_evaluate(network, prop, baseline_ue, baseline_score, profile)
        evals.append(ev)
        logger.info(
            f"  [{i:>2}/{len(pre)}] {prop.proposal_type:>9s} {prop.highway:>9s} "
            f"{prop.length_m:>5.0f}m ΔVHT={ev.delta_vht_h:>+7.1f}h "
            f"benef={ev.annual_benefit_eur:>+12,.0f}€/an BCR={ev.bcr:>5.2f}"
        )
    logger.info(f"  Évaluation marginale terminée en {_time.perf_counter() - _t1:.2f}s")

    # 3. Sélection sous budget (knapsack DP ou greedy)
    if selection_method == "knapsack":
        chosen, spent = knapsack_dp_select(evals, budget_eur)
    elif selection_method == "greedy":
        chosen, spent = _greedy_select(evals, budget_eur)
    else:
        raise ValueError(
            f"selection_method inconnue : {selection_method!r} "
            f"(attendu : 'knapsack' ou 'greedy')"
        )
    cnt = {t: sum(e.proposal.proposal_type == t for e in chosen)
           for t in ("corridor", "upgrade", "new_route")}
    logger.info(
        f"NDP — sélection [{selection_method}] : {len(chosen)} retenues "
        f"({cnt['corridor']} corridors + {cnt['upgrade']} upgrades + "
        f"{cnt['new_route']} routes), coût = {spent:,.0f}€ / {budget_eur:,.0f}€"
    )

    # 4. Optionnel : local-search 2-swap (FW joint à chaque essai)
    joint: JointPlanResult | None
    if enable_2swap and chosen:
        def _joint_fn(plan: list[NewArcEvaluation], plan_spent: float):
            return _joint_re_evaluate(
                network, od, profile, baseline_ue, baseline_score,
                baseline_access.mean_reachable, baseline_access.gini,
                plan, plan_spent,
                fw_max_iter=fw_max_iter, fw_tol=fw_tol,
                accessibility_threshold_s=accessibility_threshold_s,
                enable_induced_demand=enable_induced_demand,
                induced_elasticity=induced_elasticity,
                induced_max_iter=induced_max_iter,
                induced_tol=induced_tol,
            )

        chosen, spent, joint, fw_calls = local_search_2swap(
            chosen, evals, spent, budget_eur,
            joint_eval_fn=_joint_fn,
            max_iter=swap_max_iter,
            top_k_out=swap_top_k_out,
            top_k_in=swap_top_k_in,
            max_fw_calls=swap_max_fw_calls,
        )
        cnt = {t: sum(e.proposal.proposal_type == t for e in chosen)
               for t in ("corridor", "upgrade", "new_route")}
        logger.info(
            f"NDP — après 2-swap ({fw_calls} FW joints) : {len(chosen)} retenues "
            f"({cnt['corridor']} corridors + {cnt['upgrade']} upgrades + "
            f"{cnt['new_route']} routes), coût = {spent:,.0f}€"
        )
    else:
        # FW joint unique (comportement historique)
        joint = _joint_re_evaluate(
            network, od, profile, baseline_ue, baseline_score,
            baseline_access.mean_reachable, baseline_access.gini,
            chosen, spent,
            fw_max_iter=fw_max_iter, fw_tol=fw_tol,
            accessibility_threshold_s=accessibility_threshold_s,
            enable_induced_demand=enable_induced_demand,
            induced_elasticity=induced_elasticity,
            induced_max_iter=induced_max_iter,
            induced_tol=induced_tol,
        )

    if joint is not None:
        msg_induced = (
            f", induced {joint.induced_trip_share*100:+.2f}% en {joint.induced_iter}it."
            if joint.induced_iter > 0 else ""
        )
        logger.info(
            f"NDP — FW joint final : ΔVHT={joint.joint_delta_vht_h:+.1f}h, "
            f"bénéfice = {joint.joint_annual_benefit_eur:+,.0f}€/an, "
            f"BCR = {joint.joint_bcr:.2f}{msg_induced} "
            f"(total {_time.perf_counter() - _t0:.1f}s)"
        )

    return chosen, baseline_score, joint
