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

from urban_optimizer.network.buildings import ObstacleIndex

# Alias de rétrocompatibilité pour les tests qui importent BuildingIndex directement
BuildingIndex = ObstacleIndex
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

    @property
    def redundancy_factor(self) -> float:
        """1.0 = pas de redondance. 0.7 = 30% du bénéfice naïf est illusoire."""
        if self.naive_sum_delta_vht_h <= 0:
            return 1.0
        return self.joint_delta_vht_h / self.naive_sum_delta_vht_h


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

            corridor_length = float(sum(lengths[e] for e in edge_ids))
            if corridor_length < min_length_m or corridor_length > max_length_m:
                continue

            v_node = demand_nodes[j]
            # vpath dérivé du epath (pas de second Dijkstra)
            vpath = _vpath_from_epath(g, u_node, edge_ids)
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
    max_candidates: int = 80,
) -> list[NewArcProposal]:
    """Identifie des petites rues à transformer en axe principal (mise à niveau).

    Contrairement aux corridors (qui élargissent des axes déjà utilisés), cette
    fonction cherche des chemins passant par des rues de **faible capacité et peu
    chargées** (résidentiel, tertiary) qui pourraient devenir un itinéraire
    alternatif si elles étaient mises à niveau.

    Le tracé suit les rues existantes — aucun bâtiment ne peut être traversé.
    La simulation est identique aux corridors (capacité ×2 via _CorridorContext).
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
    # On évite les axes déjà saturés (pas le bon endroit pour une mise à niveau)
    # et les grandes artères (déjà bien dimensionnées)
    route_weights = lengths.copy()
    route_weights[sat_arr > saturation_avoid] *= 8.0
    route_weights[capacity > capacity_threshold] *= 4.0

    demand_nodes = sorted({od.zone_to_node[z] for z in od.zone_ids})
    if len(demand_nodes) < 2:
        return []
    logger.info(f"NDP mise à niveau — {len(demand_nodes)} nœuds OD candidats")

    od_lookup = _zone_demand_lookup(od)
    coords = np.array([nodes_xy[n] for n in demand_nodes])
    path_lengths = _shortest_length(g, demand_nodes, demand_nodes)

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

            # Ne retenir que les corridors qui passent majoritairement par des petites rues
            edge_caps = [capacity[e] for e in edge_ids]
            if float(np.median(edge_caps)) > capacity_threshold:
                continue  # déjà un axe principal — pas une mise à niveau

            corridor_length = float(sum(lengths[e] for e in edge_ids))
            if corridor_length < min_length_m or corridor_length > max_length_m:
                continue

            v_node = demand_nodes[j]
            vpath = _vpath_from_epath(g, u_node, edge_ids)
            corridor_xy = [nodes_xy[v] for v in vpath if v in nodes_xy]

            hw_list = [hw_types[e] for e in edge_ids]
            corridor_hw = max(hw_list, key=lambda h: _HW_RANK.get(h, 0))
            corridor_cap = float(np.min(edge_caps))
            corridor_speed = float(np.min([free_speeds[e] for e in edge_ids]))
            avg_sat = float(np.mean([sat_arr[e] for e in edge_ids]))

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
) -> list[NewArcProposal]:
    """Propose de nouvelles routes via A* sur grille, en zone périphérique uniquement.

    Obstacles durs (bâtiments, parcs) : contournés.
    Obstacles doux (voies ferrées, cours d'eau) : traversés par pont, coût majoré.

    Seules les paires OD dont au moins un nœud est hors du noyau intérieur de la
    ville (convex hull érodé de periphery_margin_m) sont étudiées. Le centre-ville
    dense est laissé aux corridors et mises à niveau (plus adaptés).
    """
    if obstacle_index is None:
        return []

    g = network.graph
    nodes_xy = network.nodes_xy

    hard_geoms = list(obstacle_index._tree.geometries)
    soft_geoms = list(soft_index._tree.geometries) if soft_index else []

    demand_nodes = sorted({od.zone_to_node[z] for z in od.zone_ids})
    if len(demand_nodes) < 2:
        return []

    # Noyau intérieur : paires dont les DEUX nœuds sont dedans → ignorées
    core = _build_periphery_core(nodes_xy, periphery_margin_m)
    if core is not None:
        logger.info(
            f"NDP nouvelles routes — filtre périphérie activé "
            f"(marge {periphery_margin_m:.0f}m, {core.area / 1e6:.1f} km² de noyau exclu)"
        )
    logger.info(f"NDP nouvelles routes (A*) — {len(demand_nodes)} nœuds OD candidats")

    od_lookup = _zone_demand_lookup(od)
    coords = np.array([nodes_xy[n] for n in demand_nodes])
    path_lengths = _shortest_length(g, demand_nodes, demand_nodes)

    # Pré-construction des STRtrees — une seule fois pour toute la boucle
    hard_tree_pre = _STRtree(hard_geoms) if hard_geoms else None
    soft_tree_pre = _STRtree(soft_geoms) if soft_geoms else None

    # Pré-calcul du filtre périphérie pour chaque nœud OD (évite O(n²) contains)
    if core is not None:
        in_core = [core.contains(_Point(*nodes_xy[n])) for n in demand_nodes]
    else:
        in_core = [False] * len(demand_nodes)

    proposals: list[tuple[float, NewArcProposal]] = []
    n = len(demand_nodes)
    n_skipped_core = 0

    for i in range(n):
        for j in range(i + 1, n):
            # Filtre périphérie pré-calculé : au moins un nœud doit être hors du noyau
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

            u_node = demand_nodes[i]
            v_node = demand_nodes[j]

            # Optimisation : si la ligne droite ne croise aucun obstacle dur,
            # on l'utilise directement (pas besoin d'A*)
            straight = LineString([(ux, uy), (vx, vy)])
            if hard_tree_pre is not None:
                buf_seg = straight.buffer(7.0)
                hits = hard_tree_pre.query(buf_seg)
                needs_astar = any(hard_geoms[h].intersects(buf_seg) for h in hits)
            else:
                needs_astar = False

            if needs_astar:
                route_xy, bridge_len = _route_avoiding_obstacles(
                    (ux, uy), (vx, vy), hard_geoms, soft_geoms,
                    _hard_tree=hard_tree_pre, _soft_tree=soft_tree_pre,
                )
                if route_xy is None:
                    continue
            else:
                route_xy = [(ux, uy), (vx, vy)]
                bridge_len = 0.0
                if soft_tree_pre is not None:
                    hits = soft_tree_pre.query(straight)
                    if any(soft_geoms[h].intersects(straight) for h in hits):
                        bridge_len = euclid

            route_length = float(sum(
                hypot(route_xy[k+1][0] - route_xy[k][0], route_xy[k+1][1] - route_xy[k][1])
                for k in range(len(route_xy) - 1)
            ))
            if route_length < min_length_m or route_length > max_length_m:
                continue

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
                    edge_ids=[],           # → _NewArcContext pour la simulation
                    corridor_xy=route_xy,
                    proposal_type="new_route",
                ),
            ))

    if n_skipped_core > 0:
        logger.info(f"NDP nouvelles routes — {n_skipped_core} paires ignorées (centre-ville)")
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
        # 1. Élargissements : on note la capacité d'origine puis on multiplie
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
        # Restaurer les capacités
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


# ── Évaluation Frank-Wolfe ────────────────────────────────────────────────────

def _warm_start_flows(baseline_flows: np.ndarray, n_now: int) -> np.ndarray | None:
    """Étend les flux baseline si de nouveaux arcs ont été ajoutés (zéros).

    - Corridor/upgrade : ecount inchangé → flows réutilisables directement.
    - Nouvel arc : ecount augmenté → on complète par des zéros (les nouveaux arcs
      n'ont pas encore de trafic).
    """
    n_base = len(baseline_flows)
    if n_base == n_now:
        return baseline_flows
    if n_base < n_now:
        return np.concatenate([baseline_flows, np.zeros(n_now - n_base, dtype=float)])
    return None  # ecount diminué : warm-start impossible


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
        warm = _warm_start_flows(baseline_ue.flows, g.ecount())
        new_ue = solve_user_equilibrium(
            network, od, max_iter=max_iter, tol=tol, initial_flows=warm,
        )

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


# ── Helpers internes (réutilisés par propose_urban_plan + compute_pareto_frontier) ──

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
) -> list[NewArcProposal]:
    """Génère + remplit (corridors, upgrades, new routes) → liste `pre`."""
    n_corr = max(1, int(max_fw_evals * 0.50))
    n_upgr = max(1, int(max_fw_evals * 0.25))
    n_new = max_fw_evals - n_corr - n_upgr

    corridors = generate_corridor_candidates(
        network, od, ue=baseline_ue, max_candidates=max_proposals,
    )
    upgrades = generate_upgrade_candidates(
        network, od, ue=baseline_ue, max_candidates=max_proposals,
    )
    new_routes: list[NewArcProposal] = []
    if obstacle_index is not None and n_new > 0:
        new_routes = generate_new_route_candidates(
            network, od, ue=baseline_ue,
            obstacle_index=obstacle_index, soft_index=soft_index,
            max_candidates=max_proposals,
            periphery_margin_m=periphery_margin_m,
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


def _evaluate_proposals(
    network: UrbanNetwork,
    od: ODMatrix,
    profile: MayorProfile,
    baseline_ue: AssignmentResult,
    baseline_score: CityScore,
    proposals: list[NewArcProposal],
    *,
    fw_max_iter: int,
    fw_tol: float,
) -> list[NewArcEvaluation]:
    """Évalue chaque proposition individuellement (1 FW par candidat)."""
    evals: list[NewArcEvaluation] = []
    for i, prop in enumerate(proposals, start=1):
        ev = _evaluate(
            network, od, prop, baseline_ue, baseline_score, profile,
            max_iter=fw_max_iter, tol=fw_tol,
        )
        evals.append(ev)
        logger.info(
            f"  [{i:>2}/{len(proposals)}] {prop.highway:>9s} {prop.length_m:>5.0f}m "
            f"u={prop.u_node} v={prop.v_node} "
            f"ΔVHT={ev.delta_vht_h:>+7.1f}h benef={ev.annual_benefit_eur:>+12,.0f}€/an "
            f"BCR={ev.bcr:>5.2f} payback={ev.payback_years:>5.1f}y"
        )
    return evals


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
) -> JointPlanResult | None:
    """Évaluation jointe : FW unique avec toutes les interventions appliquées."""
    if not chosen:
        return None
    chosen_props = [e.proposal for e in chosen]
    naive_sum_dvht = sum(e.delta_vht_h for e in chosen)
    naive_sum_benef = sum(e.annual_benefit_eur for e in chosen)
    n_existing_pre = network.graph.ecount()
    with _JointContext(network.graph, chosen_props):
        warm = _warm_start_flows(baseline_ue.flows, network.graph.ecount())
        joint_ue = solve_user_equilibrium(
            network, od, max_iter=fw_max_iter, tol=fw_tol, initial_flows=warm,
        )
        caps_after = np.asarray(network.graph.es["capacity"], dtype=float)
        joint_access = compute_accessibility(
            network, od, joint_ue, threshold_seconds=accessibility_threshold_s,
        )
    sat_existing_after = (
        joint_ue.flows[:n_existing_pre]
        / np.maximum(caps_after[:n_existing_pre], 1.0)
    )
    joint_score = score_network(joint_ue, profile, access=joint_access)
    joint_dvht = baseline_ue.vht - joint_ue.vht
    joint_benef = baseline_score.total_annual_cost_eur - joint_score.total_annual_cost_eur
    joint_bcr = joint_benef / spent if spent > 0 else 0.0
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
    fw_max_iter: int = 25,
    fw_tol: float = 5e-3,
    building_index: ObstacleIndex | None = None,    # rétrocompat (= obstacle_index)
    obstacle_index: ObstacleIndex | None = None,    # obstacles durs (bâtiments, parcs)
    soft_index: ObstacleIndex | None = None,        # ponts (voies ferrées, eau)
    periphery_margin_m: float = 600.0,              # érosion du hull pour filtre périphérie
    accessibility_threshold_s: float = 15 * 60,     # seuil isochrone pour l'accessibilité
    _baseline_access=None,                          # pré-calculé (évite double compute)
) -> tuple[list[NewArcEvaluation], CityScore, JointPlanResult | None]:
    """Pipeline complet : trois types d'interventions sur le réseau.

    1. **Corridors** (bleu) : élargissement (×2 cap.) d'axes existants saturés.
    2. **Mises à niveau** (rose) : transformation de petites rues en axes de transit.
    3. **Nouvelles routes** (vert) : A* contournant bâtiments/parcs, surcoût pont si
       traversée de voie ferrée ou cours d'eau.

    Split FW : ~50 % corridors, ~25 % mises à niveau, ~25 % nouvelles routes.
    """
    _obs = obstacle_index or building_index

    # Accessibilité baseline — réutiliser si déjà calculée (évite un double FW de distances)
    baseline_access = _baseline_access or compute_accessibility(
        network, od, baseline_ue, threshold_seconds=accessibility_threshold_s,
    )
    baseline_score = score_network(baseline_ue, profile, access=baseline_access)
    logger.info(
        f"Score baseline ({profile.name}) : {baseline_score.composite_score:,.0f} €/an "
        f"(accès moyen = {baseline_access.mean_reachable:.1f} zones, Gini = {baseline_access.gini:.2f})"
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
        f"NDP — {len(pre)} candidats pour FW : "
        f"{counts['corridor']} corridors + {counts['upgrade']} mises à niveau + "
        f"{counts['new_route']} nouvelles routes"
    )

    # 2. Évaluation individuelle (FW par candidat)
    evals = _evaluate_proposals(
        network, od, profile, baseline_ue, baseline_score, pre,
        fw_max_iter=fw_max_iter, fw_tol=fw_tol,
    )

    # 3. Sélection gloutonne sous budget
    chosen, spent = _greedy_select(evals, budget_eur)
    cnt = {t: sum(e.proposal.proposal_type == t for e in chosen)
           for t in ("corridor", "upgrade", "new_route")}
    logger.info(
        f"NDP — {len(chosen)} retenues "
        f"({cnt['corridor']} corridors + {cnt['upgrade']} mises à niveau + "
        f"{cnt['new_route']} nouvelles routes), coût = {spent:,.0f}€ / {budget_eur:,.0f}€"
    )

    # 4. Re-évaluation JOINTE (FW unique avec tout appliqué)
    joint = _joint_re_evaluate(
        network, od, profile, baseline_ue, baseline_score,
        baseline_access.mean_reachable, baseline_access.gini,
        chosen, spent,
        fw_max_iter=fw_max_iter, fw_tol=fw_tol,
        accessibility_threshold_s=accessibility_threshold_s,
    )
    if joint is not None:
        logger.info(
            f"NDP — re-évaluation jointe : ΔVHT={joint.joint_delta_vht_h:+.1f}h "
            f"(somme naïve = {joint.naive_sum_delta_vht_h:+.1f}h, "
            f"redondance = {(1-joint.redundancy_factor)*100:.0f}%) "
            f"bénéfice joint = {joint.joint_annual_benefit_eur:+,.0f}€/an, "
            f"BCR joint = {joint.joint_bcr:.2f}"
        )

    return chosen, baseline_score, joint
