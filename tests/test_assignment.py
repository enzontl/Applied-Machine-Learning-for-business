"""Tests de la brique 3 : affectation AoN / UE / SO."""

import geopandas as gpd
import igraph as ig
import numpy as np
import pytest
from shapely.geometry import LineString

from urban_optimizer.assignment import (
    AssignmentResult,
    assign_aon,
    beckmann_objective,
    bpr_marginal_time,
    bpr_time,
    price_of_anarchy,
    solve_all_or_nothing,
    solve_system_optimum,
    solve_user_equilibrium,
)
from urban_optimizer.demand.od_matrix import ODMatrix
from urban_optimizer.network.urban_network import UrbanNetwork


# ── helpers ──────────────────────────────────────────────────────────────────

def _two_route_network(cap1: float = 1000.0, cap2: float = 1000.0) -> UrbanNetwork:
    """Réseau classique pour tester UE/SO : 1 → 2 par deux arcs parallèles.

    Arc 0 : direct, court (t0=10 s), capacité cap1
    Arc 1 : direct, plus long (t0=20 s), capacité cap2
    """
    nodes_xy = {0: (0.0, 0.0), 1: (1000.0, 0.0)}
    edges = [(0, 1), (0, 1)]

    g = ig.Graph(n=2, edges=edges, directed=True)
    g.es["t0_s"] = [10.0, 20.0]
    g.es["capacity"] = [cap1, cap2]
    g.es["bpr_alpha"] = [0.15, 0.15]
    g.es["bpr_beta"] = [4.0, 4.0]
    g.es["length_m"] = [1000.0, 1000.0]
    g.es["free_speed_kmh"] = [50.0, 50.0]
    g.es["highway"] = ["primary", "primary"]
    g.es["source"] = ["osm", "osm"]

    edges_gdf = gpd.GeoDataFrame(
        {"highway": ["primary", "primary"], "source": ["osm", "osm"]},
        geometry=[LineString([(0, 0), (1000, 0)])] * 2,
        crs="EPSG:2154",
    )
    return UrbanNetwork(graph=g, nodes_xy=nodes_xy, edges_gdf=edges_gdf, crs=2154)


def _simple_od(trips: float = 800.0) -> ODMatrix:
    return ODMatrix(
        matrix=np.array([[0.0, trips], [0.0, 0.0]]),
        zone_ids=["A", "B"],
        zone_to_node={"A": 0, "B": 1},
        hour=8,
        scenario="weekday",
    )


def _grid_network(n: int = 4, step: float = 200.0) -> UrbanNetwork:
    """Grille n×n bidirectionnelle, capacités modestes pour induire de la congestion."""
    nodes_xy = {}
    idx = 0
    for i in range(n):
        for j in range(n):
            nodes_xy[idx] = (i * step, j * step)
            idx += 1

    edges = []
    highways = []
    for i in range(n):
        for j in range(n):
            u = i * n + j
            if i + 1 < n:
                v = (i + 1) * n + j
                edges.extend([(u, v), (v, u)])
                highways.extend(["primary", "primary"])
            if j + 1 < n:
                v = i * n + (j + 1)
                edges.extend([(u, v), (v, u)])
                highways.extend(["residential", "residential"])

    g = ig.Graph(n=len(nodes_xy), edges=edges, directed=True)
    g.es["t0_s"] = [step / (50 * 1000 / 3600)] * len(edges)
    g.es["capacity"] = [800.0] * len(edges)
    g.es["bpr_alpha"] = [0.15] * len(edges)
    g.es["bpr_beta"] = [4.0] * len(edges)
    g.es["length_m"] = [step] * len(edges)
    g.es["free_speed_kmh"] = [50.0] * len(edges)
    g.es["highway"] = highways
    g.es["source"] = ["osm"] * len(edges)

    edges_gdf = gpd.GeoDataFrame(
        {"highway": highways, "source": ["osm"] * len(edges)},
        geometry=[LineString([(0, 0), (1, 1)])] * len(edges),
        crs="EPSG:2154",
    )
    return UrbanNetwork(graph=g, nodes_xy=nodes_xy, edges_gdf=edges_gdf, crs=2154)


# ── BPR ──────────────────────────────────────────────────────────────────────

class TestBPR:
    def test_free_flow_recovers_t0(self):
        t0 = np.array([10.0])
        result = bpr_time(np.array([0.0]), t0, np.array([1000.0]), np.array([0.15]), np.array([4.0]))
        assert np.isclose(result[0], 10.0)

    def test_at_capacity_alpha_overhead(self):
        # v = c → t = t0 * (1 + alpha)
        result = bpr_time(
            np.array([1000.0]), np.array([10.0]), np.array([1000.0]),
            np.array([0.15]), np.array([4.0]),
        )
        assert np.isclose(result[0], 10.0 * 1.15)

    def test_marginal_greater_than_average(self):
        f = np.array([800.0])
        avg = bpr_time(f, np.array([10.0]), np.array([1000.0]), np.array([0.15]), np.array([4.0]))
        marg = bpr_marginal_time(
            f, np.array([10.0]), np.array([1000.0]), np.array([0.15]), np.array([4.0]),
        )
        assert marg[0] > avg[0]

    def test_beckmann_increasing_in_flow(self):
        t0 = np.array([10.0])
        cap = np.array([1000.0])
        alpha = np.array([0.15])
        beta = np.array([4.0])
        z1 = beckmann_objective(np.array([100.0]), t0, cap, alpha, beta)
        z2 = beckmann_objective(np.array([200.0]), t0, cap, alpha, beta)
        assert z2 > z1


# ── All-or-Nothing ───────────────────────────────────────────────────────────

class TestAoN:
    def test_charges_shortest_path(self):
        net = _two_route_network()
        od = _simple_od(trips=500.0)
        flows = assign_aon(net.graph, od, weights=np.array(net.graph.es["t0_s"]))
        # Tout passe par l'arc 0 (t0=10 < 20)
        assert flows[0] == 500.0
        assert flows[1] == 0.0

    def test_returns_zero_when_unreachable(self):
        net = _two_route_network()
        # Inversement : zone_to_node pointe vers un nœud isolé
        od = ODMatrix(
            matrix=np.array([[0.0, 1.0], [0.0, 0.0]]),
            zone_ids=["A", "B"],
            zone_to_node={"A": 1, "B": 0},   # B → A : pas d'arc dans ce sens
            hour=8, scenario="w",
        )
        flows = assign_aon(net.graph, od, weights=np.array(net.graph.es["t0_s"]))
        assert flows.sum() == 0.0


# ── Affectation classique sur deux arcs parallèles ───────────────────────────

class TestTwoRouteUE:
    def test_ue_splits_when_demand_high(self):
        # Demande élevée : les deux arcs doivent être utilisés (Wardrop)
        net = _two_route_network(cap1=500.0, cap2=500.0)
        od = _simple_od(trips=2000.0)
        res = solve_user_equilibrium(net, od, max_iter=300, tol=1e-7)
        assert res.flows[0] > 0
        assert res.flows[1] > 0
        # Total = demande
        assert pytest.approx(res.flows.sum(), rel=1e-3) == 2000.0
        # À l'équilibre Wardrop, les temps des deux arcs utilisés sont égaux
        t = res.travel_times
        assert pytest.approx(t[0], rel=1e-2) == t[1]

    def test_ue_converges(self):
        net = _two_route_network(cap1=500.0, cap2=500.0)
        od = _simple_od(trips=2000.0)
        res = solve_user_equilibrium(net, od, max_iter=300, tol=1e-6)
        assert res.converged

    def test_so_total_time_le_ue_total_time(self):
        net = _two_route_network(cap1=500.0, cap2=500.0)
        od = _simple_od(trips=2000.0)
        ue = solve_user_equilibrium(net, od, max_iter=300, tol=1e-7)
        so = solve_system_optimum(net, od, max_iter=300, tol=1e-7)
        # SO minimise le coût social → VHT(SO) ≤ VHT(UE)
        assert so.vht <= ue.vht * (1.0 + 1e-3)

    def test_price_of_anarchy_ge_one(self):
        net = _two_route_network(cap1=500.0, cap2=500.0)
        od = _simple_od(trips=2000.0)
        ue = solve_user_equilibrium(net, od, max_iter=300, tol=1e-7)
        so = solve_system_optimum(net, od, max_iter=300, tol=1e-7)
        assert price_of_anarchy(ue, so) >= 1.0 - 1e-3

    def test_aon_routes_all_through_cheaper(self):
        # Demande faible (< capa de l'arc rapide) → AoN charge tout dessus
        net = _two_route_network()
        od = _simple_od(trips=200.0)
        res = solve_all_or_nothing(net, od)
        assert res.flows[0] == 200.0
        assert res.flows[1] == 0.0


# ── Sanity sur réseau plus large ─────────────────────────────────────────────

class TestGridConvergence:
    def test_ue_converges_on_grid(self):
        net = _grid_network(n=4, step=200.0)
        od = ODMatrix(
            matrix=np.array([
                [0, 0, 0, 100],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [100, 0, 0, 0],
            ], dtype=float),
            zone_ids=["A", "B", "C", "D"],
            zone_to_node={"A": 0, "B": 3, "C": 12, "D": 15},
            hour=8, scenario="w",
        )
        res = solve_user_equilibrium(net, od, max_iter=50, tol=1e-4)
        assert isinstance(res, AssignmentResult)
        assert res.vht > 0
        assert (res.flows > 0).any()
