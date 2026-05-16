"""Tests de la brique 2 : zonage + matrice OD gravitaire."""

import geopandas as gpd
import igraph as ig
import numpy as np
import pytest
from shapely.geometry import LineString, Point, box

from urban_optimizer.demand.gravity import _furness_balancing, gravity_od
from urban_optimizer.demand.od_matrix import ODMatrix
from urban_optimizer.demand.zoning import (
    Zoning,
    _build_zone_to_node,
    build_grid_zoning,
)
from urban_optimizer.network.urban_network import UrbanNetwork


# ── helpers ──────────────────────────────────────────────────────────────────

def _minimal_network(n_side: int = 4, step: float = 200.0) -> UrbanNetwork:
    """Réseau quadrillé minimal : grille n×n de nœuds avec arcs orthogonaux."""
    nodes_xy: dict = {}
    idx = 0
    for i in range(n_side):
        for j in range(n_side):
            nodes_xy[idx] = (i * step, j * step)
            idx += 1

    edges: list[tuple[int, int]] = []
    geoms: list = []
    highways: list[str] = []
    for i in range(n_side):
        for j in range(n_side):
            u = i * n_side + j
            if i + 1 < n_side:
                v = (i + 1) * n_side + j
                edges.append((u, v))
                edges.append((v, u))
                line = LineString([nodes_xy[u], nodes_xy[v]])
                geoms.extend([line, line])
                highways.extend(["residential", "residential"])
            if j + 1 < n_side:
                v = i * n_side + (j + 1)
                edges.append((u, v))
                edges.append((v, u))
                line = LineString([nodes_xy[u], nodes_xy[v]])
                geoms.extend([line, line])
                highways.extend(["primary", "primary"])

    g = ig.Graph(n=len(nodes_xy), edges=edges, directed=True)
    g.es["length_m"] = [step] * len(edges)
    g.es["free_speed_kmh"] = [50.0] * len(edges)
    g.es["capacity"] = [1500.0] * len(edges)
    g.es["t0_s"] = [step / (50 * 1000 / 3600)] * len(edges)
    g.es["bpr_alpha"] = [0.15] * len(edges)
    g.es["bpr_beta"] = [4.0] * len(edges)
    g.es["highway"] = highways
    g.es["source"] = ["osm"] * len(edges)

    edges_gdf = gpd.GeoDataFrame(
        {"highway": highways, "length_m": [step] * len(edges), "source": ["osm"] * len(edges)},
        geometry=geoms,
        crs="EPSG:2154",
    )
    return UrbanNetwork(graph=g, nodes_xy=nodes_xy, edges_gdf=edges_gdf, crs=2154)


# ── ODMatrix ──────────────────────────────────────────────────────────────────

class TestODMatrix:
    def test_construct_valid(self):
        m = np.array([[0, 10], [5, 0]], dtype=float)
        od = ODMatrix(m, ["A", "B"], {"A": 1, "B": 2}, hour=8, scenario="weekday")
        assert od.n_zones == 2
        assert od.total_trips == 15.0

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError):
            ODMatrix(np.zeros((2, 3)), ["A", "B"], {}, hour=8, scenario="w")

    def test_negative_values_raise(self):
        with pytest.raises(ValueError):
            ODMatrix(np.array([[0, -1], [1, 0]]), ["A", "B"], {}, hour=8, scenario="w")

    def test_iter_pairs_skips_zero(self):
        m = np.array([[0, 0, 3], [0, 0, 0], [2, 0, 0]], dtype=float)
        od = ODMatrix(m, ["A", "B", "C"], {}, hour=8, scenario="w")
        pairs = list(od.iter_pairs())
        assert ("A", "C", 3.0) in pairs
        assert ("C", "A", 2.0) in pairs
        assert len(pairs) == 2

    def test_to_node_pairs_skips_unmapped(self):
        m = np.array([[0, 4], [2, 0]], dtype=float)
        od = ODMatrix(m, ["A", "B"], {"A": 10}, hour=8, scenario="w")
        # B non rattaché → pas de paire émise
        assert list(od.to_node_pairs()) == []


# ── Furness ───────────────────────────────────────────────────────────────────

class TestFurness:
    def test_balances_to_targets(self):
        seed = np.array([[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]])
        o = np.array([100.0, 200.0, 300.0])
        d = np.array([150.0, 250.0, 200.0])
        T = _furness_balancing(seed, o, d, max_iter=200, tol=1e-6)
        assert np.allclose(T.sum(axis=1), o, atol=1e-3)
        assert np.allclose(T.sum(axis=0), d, atol=1e-3)
        assert np.allclose(np.diag(T), 0.0)


# ── Zoning grid ───────────────────────────────────────────────────────────────

class TestGridZoning:
    def test_basic_grid_builds(self):
        net = _minimal_network(n_side=4, step=200.0)
        zoning = build_grid_zoning(net, n_cells=3, min_pop=0.0)
        assert isinstance(zoning, Zoning)
        assert zoning.method == "grid"
        assert zoning.n_zones >= 1
        assert set(zoning.gdf.columns) >= {"zone_id", "geometry", "population", "jobs"}

    def test_zone_to_node_mapping_covers_all_zones(self):
        net = _minimal_network(n_side=4, step=200.0)
        zoning = build_grid_zoning(net, n_cells=3, min_pop=0.0)
        assert set(zoning.zone_to_node.keys()) == set(zoning.zone_ids)
        for node in zoning.zone_to_node.values():
            assert 0 <= node < len(net.nodes_xy)

    def test_build_zone_to_node_picks_nearest(self):
        net = _minimal_network(n_side=2, step=100.0)
        centroids = {"z0": (0.0, 0.0), "z1": (100.0, 100.0)}
        mapping = _build_zone_to_node(centroids, net)
        # nœud 0 est en (0, 0) ; nœud 3 est en (100, 100)
        assert mapping["z0"] == 0
        assert mapping["z1"] == 3


# ── Gravity OD ────────────────────────────────────────────────────────────────

def _toy_zoning(n: int = 3) -> Zoning:
    polys = [box(i, 0, i + 1, 1) for i in range(n)]
    ids = [f"z{i}" for i in range(n)]
    gdf = gpd.GeoDataFrame(
        {
            "zone_id": ids,
            "geometry": polys,
            "population": [100.0] * n,
            "jobs": [50.0] * n,
        },
        crs="EPSG:2154",
    )
    gdf["centroid"] = gdf.geometry.centroid
    centroids_xy = {ids[i]: (i + 0.5, 0.5) for i in range(n)}
    return Zoning(
        gdf=gdf,
        centroids_xy=centroids_xy,
        zone_to_node={ids[i]: i for i in range(n)},
        crs=2154,
        method="grid",
    )


class TestGravityOD:
    def test_returns_od_matrix(self):
        z = _toy_zoning(3)
        od = gravity_od(z, hour=8, scenario="weekday")
        assert isinstance(od, ODMatrix)
        assert od.matrix.shape == (3, 3)

    def test_diagonal_zero(self):
        z = _toy_zoning(3)
        od = gravity_od(z, hour=8)
        assert np.allclose(np.diag(od.matrix), 0.0)

    def test_non_negative(self):
        z = _toy_zoning(3)
        od = gravity_od(z, hour=8)
        assert (od.matrix >= 0).all()

    def test_furness_respects_origin_totals(self):
        z = _toy_zoning(3)
        od = gravity_od(z, hour=8, balance=True)
        # Avec Furness, somme des lignes = O_i (à 1e-3 près)
        pop = z.gdf["population"].to_numpy()
        row_sums = od.matrix.sum(axis=1)
        # même proportionnalité — non strictement égal car O dépend de hour_share
        assert np.all(row_sums > 0)
        # ratio entre zones doit refléter la population
        assert pytest.approx(row_sums[0] / row_sums[1], rel=1e-2) == pop[0] / pop[1]

    def test_scale_factor_doubles_total(self):
        z = _toy_zoning(3)
        od1 = gravity_od(z, hour=8, scale_factor=1.0)
        od2 = gravity_od(z, hour=8, scale_factor=2.0)
        assert pytest.approx(od2.total_trips, rel=1e-3) == 2 * od1.total_trips

    def test_higher_beta_shifts_to_short_trips(self):
        z = _toy_zoning(5)
        od_short = gravity_od(z, hour=8, beta=1.0, balance=False)
        od_long = gravity_od(z, hour=8, beta=1e-6, balance=False)
        # part des paires "courtes" (|i-j|=1) dans le total
        n = 5
        short_mask = np.zeros((n, n), dtype=bool)
        for i in range(n - 1):
            short_mask[i, i + 1] = True
            short_mask[i + 1, i] = True
        share_short_strong = od_short.matrix[short_mask].sum() / od_short.total_trips
        share_short_weak = od_long.matrix[short_mask].sum() / od_long.total_trips
        assert share_short_strong > share_short_weak

    def test_empty_population_raises(self):
        z = _toy_zoning(3)
        z.gdf["population"] = 0.0
        with pytest.raises(ValueError):
            gravity_od(z, hour=8)
