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
from shapely.strtree import STRtree as _STRtree
from shapely.geometry import Point as _Point

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

            epaths = g.get_shortest_paths(
                u_node, v_node,
                weights=route_weights.tolist(),
                output="epath",
            )
            if not epaths or not epaths[0]:
                continue
            edge_ids = epaths[0]

            # Ne retenir que les corridors qui passent majoritairement par des petites rues
            edge_caps = [capacity[e] for e in edge_ids]
            if float(np.median(edge_caps)) > capacity_threshold:
                continue  # déjà un axe principal — pas une mise à niveau

            corridor_length = float(sum(lengths[e] for e in edge_ids))
            if corridor_length < min_length_m or corridor_length > max_length_m:
                continue

            vpaths = g.get_shortest_paths(
                u_node, v_node,
                weights=route_weights.tolist(),
                output="vpath",
            )
            vpath = vpaths[0] if vpaths else []
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


def _route_avoiding_obstacles(
    u_xy: tuple[float, float],
    v_xy: tuple[float, float],
    hard_geoms: list,
    soft_geoms: list,
    *,
    margin_m: float = 350.0,
    grid_n: int = 22,
    bridge_factor: float = 5.0,
    buf_m: float = 7.0,
) -> tuple[list[tuple[float, float]] | None, float]:
    """A* sur grille de waypoints entre u et v.

    Contourne les obstacles durs (bâtiments, parcs).
    Traverse les obstacles doux (voies ferrées, eau) avec surcoût bridge_factor.

    Returns:
        (chemin, longueur_pont_m) ou (None, 0) si aucun chemin trouvé.
    """
    from heapq import heappush, heappop

    x_min = min(u_xy[0], v_xy[0]) - margin_m
    x_max = max(u_xy[0], v_xy[0]) + margin_m
    y_min = min(u_xy[1], v_xy[1]) - margin_m
    y_max = max(u_xy[1], v_xy[1]) + margin_m

    W, H = x_max - x_min, y_max - y_min
    cell = max(60.0, max(W, H) / grid_n)
    nx_ = max(3, int(W / cell) + 1)
    ny_ = max(3, int(H / cell) + 1)

    hard_tree = _STRtree(hard_geoms) if hard_geoms else None
    soft_tree = _STRtree(soft_geoms) if soft_geoms else None

    # ── Grille de waypoints (on exclut les cellules dans un obstacle dur) ──
    wpts: list[tuple[float, float]] = []
    gmap: dict[tuple[int, int], int] = {}

    for ix in range(nx_):
        for iy in range(ny_):
            cx = x_min + (ix + 0.5) * cell
            cy = y_min + (iy + 0.5) * cell
            if hard_tree:
                pt_buf = _Point(cx, cy).buffer(buf_m)
                hits = hard_tree.query(pt_buf)
                if any(hard_geoms[h].intersects(pt_buf) for h in hits):
                    continue
            gmap[(ix, iy)] = len(wpts)
            wpts.append((cx, cy))

    start_i = len(wpts); wpts.append(u_xy)
    end_i = len(wpts);   wpts.append(v_xy)
    adj: list[list[tuple[int, float]]] = [[] for _ in range(len(wpts))]

    def _add_edge(ia: int, ib: int) -> None:
        xa, ya = wpts[ia]
        xb, yb = wpts[ib]
        seg = LineString([(xa, ya), (xb, yb)])
        seg_len = seg.length
        if hard_tree:
            buf_seg = seg.buffer(buf_m)
            hits = hard_tree.query(buf_seg)
            if any(hard_geoms[h].intersects(buf_seg) for h in hits):
                return
        bridge_len = 0.0
        if soft_tree:
            hits = soft_tree.query(seg)
            if any(soft_geoms[h].intersects(seg) for h in hits):
                bridge_len = seg_len
        cost = seg_len + bridge_len * (bridge_factor - 1.0)
        adj[ia].append((ib, cost))
        adj[ib].append((ia, cost))

    # 8-connectivité de la grille
    for ix in range(nx_):
        for iy in range(ny_):
            ia = gmap.get((ix, iy))
            if ia is None:
                continue
            for dx, dy in [(1, 0), (0, 1), (1, 1), (1, -1)]:
                ib = gmap.get((ix + dx, iy + dy))
                if ib is None:
                    continue
                _add_edge(ia, ib)

    # Connecter départ/arrivée à la grille
    for sp_i, (sx, sy) in [(start_i, u_xy), (end_i, v_xy)]:
        ix0 = int((sx - x_min) / cell)
        iy0 = int((sy - y_min) / cell)
        for dix in range(-2, 3):
            for diy in range(-2, 3):
                nb = gmap.get((ix0 + dix, iy0 + diy))
                if nb is not None:
                    _add_edge(sp_i, nb)

    # A*
    ex, ey = v_xy
    def _h(i: int) -> float:
        return hypot(wpts[i][0] - ex, wpts[i][1] - ey)

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
                path.append(wpts[node])
                node = parent[node]
            path.reverse()
            # Longueur des segments en pont
            bridge_m = 0.0
            if soft_tree and len(path) > 1:
                for k in range(len(path) - 1):
                    seg = LineString([path[k], path[k + 1]])
                    hits = soft_tree.query(seg)
                    if any(soft_geoms[h].intersects(seg) for h in hits):
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
) -> list[NewArcProposal]:
    """Propose de nouvelles routes via A* sur grille, contournant les obstacles.

    Obstacles durs (bâtiments, parcs) : contournés.
    Obstacles doux (voies ferrées, cours d'eau) : traversés par pont, coût majoré.
    Le tracé retourné suit de vrais segments courts — aucun bâtiment n'est traversé.
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
    logger.info(f"NDP nouvelles routes (A*) — {len(demand_nodes)} nœuds OD candidats")

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

            # Optimisation : si la ligne droite ne croise aucun obstacle dur,
            # on l'utilise directement (pas besoin d'A*)
            straight = LineString([(ux, uy), (vx, vy)])
            hard_tree = _STRtree(hard_geoms) if hard_geoms else None
            if hard_tree:
                buf_seg = straight.buffer(7.0)
                hits = hard_tree.query(buf_seg)
                needs_astar = any(hard_geoms[h].intersects(buf_seg) for h in hits)
            else:
                needs_astar = False

            if needs_astar:
                route_xy, bridge_len = _route_avoiding_obstacles(
                    (ux, uy), (vx, vy), hard_geoms, soft_geoms,
                )
                if route_xy is None:
                    continue
            else:
                route_xy = [(ux, uy), (vx, vy)]
                bridge_len = 0.0
                if soft_geoms:
                    soft_t = _STRtree(soft_geoms)
                    hits = soft_t.query(straight)
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

    proposals.sort(key=lambda x: -x[0])
    top = [p for _, p in proposals[:max_candidates]]
    n_bridge = sum(1 for _, p in proposals[:max_candidates] if any(
        hypot(p.corridor_xy[k+1][0]-p.corridor_xy[k][0],
              p.corridor_xy[k+1][1]-p.corridor_xy[k][1]) > 0
        for k in range(len(p.corridor_xy)-1)
    ))
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
    building_index: ObstacleIndex | None = None,    # rétrocompat (= obstacle_index)
    obstacle_index: ObstacleIndex | None = None,    # obstacles durs (bâtiments, parcs)
    soft_index: ObstacleIndex | None = None,        # ponts (voies ferrées, eau)
) -> tuple[list[NewArcEvaluation], CityScore]:
    """Pipeline complet : trois types d'interventions sur le réseau.

    1. **Corridors** (bleu) : élargissement (×2 cap.) d'axes existants saturés.
    2. **Mises à niveau** (rose) : transformation de petites rues en axes de transit.
    3. **Nouvelles routes** (vert) : A* contournant bâtiments/parcs, surcoût pont si
       traversée de voie ferrée ou cours d'eau.

    Split FW : ~50 % corridors, ~25 % mises à niveau, ~25 % nouvelles routes.
    """
    _obs = obstacle_index or building_index

    baseline_score = score_network(baseline_ue, profile)
    logger.info(f"Score baseline ({profile.name}) : {baseline_score.composite_score:,.0f} €/an")

    # Slots par type
    n_corr  = max(1, int(max_fw_evals * 0.50))
    n_upgr  = max(1, int(max_fw_evals * 0.25))
    n_new   = max_fw_evals - n_corr - n_upgr

    corridors = generate_corridor_candidates(
        network, od, ue=baseline_ue, max_candidates=max_proposals,
    )
    upgrades = generate_upgrade_candidates(
        network, od, ue=baseline_ue, max_candidates=max_proposals,
    )
    new_routes: list[NewArcProposal] = []
    if _obs is not None and n_new > 0:
        new_routes = generate_new_route_candidates(
            network, od, ue=baseline_ue,
            obstacle_index=_obs, soft_index=soft_index,
            max_candidates=max_proposals,
        )
        logger.info(f"NDP nouvelles routes — {len(new_routes)} candidats A*")

    def _fill(primary, secondary, n):
        picked = list(primary[:n])
        if len(picked) < n:
            picked += list(secondary[: n - len(picked)])
        return picked

    pre: list[NewArcProposal] = (
        _fill(corridors,   upgrades,    n_corr)
        + _fill(upgrades,  corridors,   n_upgr)
        + _fill(new_routes, corridors,  n_new)
    )

    if not pre:
        logger.warning("Aucun candidat trouvé.")
        return [], baseline_score

    counts = {t: sum(p.proposal_type == t for p in pre)
              for t in ("corridor", "upgrade", "new_route")}
    logger.info(
        f"NDP — {len(pre)} candidats pour FW : "
        f"{counts['corridor']} corridors + {counts['upgrade']} mises à niveau + "
        f"{counts['new_route']} nouvelles routes"
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

    cnt = {t: sum(e.proposal.proposal_type == t for e in chosen)
           for t in ("corridor", "upgrade", "new_route")}
    logger.info(
        f"NDP — {len(chosen)} retenues "
        f"({cnt['corridor']} corridors + {cnt['upgrade']} mises à niveau + "
        f"{cnt['new_route']} nouvelles routes), coût = {spent:,.0f}€ / {budget_eur:,.0f}€"
    )
    return chosen, baseline_score
