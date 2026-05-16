"""Tests de la brique 4 : diagnostic du réseau."""

import geopandas as gpd
import igraph as ig
import numpy as np
import pytest
from shapely.geometry import LineString

from urban_optimizer.assignment.result import AssignmentResult
from urban_optimizer.diagnosis import (
    NetworkDiagnosis,
    diagnose,
    rank_by_congestion_delay,
    rank_by_saturation,
)
from urban_optimizer.network.urban_network import UrbanNetwork


def _three_edge_network() -> UrbanNetwork:
    """Réseau jouet : 3 arcs, capacités différenciées."""
    nodes_xy = {0: (0, 0), 1: (1, 0), 2: (2, 0)}
    edges = [(0, 1), (1, 2), (0, 2)]
    g = ig.Graph(n=3, edges=edges, directed=True)
    g.es["t0_s"] = [10.0, 10.0, 20.0]
    g.es["capacity"] = [1000.0, 500.0, 1000.0]
    g.es["bpr_alpha"] = [0.15, 0.15, 0.15]
    g.es["bpr_beta"] = [4.0, 4.0, 4.0]
    g.es["length_m"] = [1000.0, 1000.0, 2000.0]
    g.es["free_speed_kmh"] = [50.0, 50.0, 50.0]
    g.es["highway"] = ["primary", "tertiary", "secondary"]
    g.es["source"] = ["osm", "osm", "osm"]

    edges_gdf = gpd.GeoDataFrame(
        {"highway": ["primary", "tertiary", "secondary"], "source": ["osm"] * 3},
        geometry=[LineString([(0, 0), (1, 0)])] * 3,
        crs="EPSG:2154",
    )
    return UrbanNetwork(graph=g, nodes_xy=nodes_xy, edges_gdf=edges_gdf, crs=2154)


def _result_from_flows(flows: list[float]) -> AssignmentResult:
    arr = np.asarray(flows, dtype=float)
    # On simule un t actuel = t0 * 1.2 partout pour un test minimal
    t0 = np.array([10.0, 10.0, 20.0])
    return AssignmentResult(
        flows=arr,
        travel_times=t0 * 1.2,
        free_flow_times=t0,
        vht=float((arr * t0 * 1.2).sum() / 3600.0),
        iterations=10,
        converged=True,
        final_gap=1e-5,
        method="ue",
    )


# ── diagnose ─────────────────────────────────────────────────────────────────

class TestDiagnose:
    def test_basic_metrics(self):
        net = _three_edge_network()
        res = _result_from_flows([400.0, 450.0, 100.0])
        d = diagnose(net, res, saturation_threshold=0.8)
        assert isinstance(d, NetworkDiagnosis)
        assert d.n_arcs == 3
        # capacités = [1000, 500, 1000] → sat = [0.4, 0.9, 0.1] ; ≥ 0.8 → 1 arc
        assert d.n_saturated == 1
        assert d.vht > 0
        assert d.congestion_overhead == pytest.approx(0.2, rel=1e-6)

    def test_no_so_returns_none_poa(self):
        net = _three_edge_network()
        res = _result_from_flows([100.0, 100.0, 100.0])
        d = diagnose(net, res)
        assert d.price_of_anarchy is None

    def test_poa_computed(self):
        net = _three_edge_network()
        ue = _result_from_flows([400.0, 400.0, 100.0])
        # ue.vht / 0.9 = poa
        so = AssignmentResult(
            flows=ue.flows, travel_times=ue.travel_times, free_flow_times=ue.free_flow_times,
            vht=ue.vht / 1.2, iterations=10, converged=True, final_gap=1e-5, method="so",
        )
        d = diagnose(net, ue, so_result=so)
        assert pytest.approx(d.price_of_anarchy, rel=1e-6) == 1.2

    def test_mean_saturation_zero_when_no_flow(self):
        net = _three_edge_network()
        res = _result_from_flows([0.0, 0.0, 0.0])
        d = diagnose(net, res)
        assert d.mean_saturation == 0.0


# ── critical arcs ────────────────────────────────────────────────────────────

class TestCriticalArcs:
    def test_rank_by_delay_returns_top_n(self):
        net = _three_edge_network()
        res = _result_from_flows([100.0, 800.0, 200.0])
        df = rank_by_congestion_delay(net, res, top_n=2)
        assert len(df) == 2
        # L'arc 1 a le plus de flux × délai (delay/user = 2s, flow=800)
        # vs arc 0 (delay=2s, flow=100), arc 2 (delay=4s, flow=200=800sh)
        # delay_total : arc0=200, arc1=1600, arc2=800 → ordre : arc1 > arc2 > arc0
        assert df.iloc[0]["edge_id"] == 1
        assert df.iloc[1]["edge_id"] == 2

    def test_rank_by_saturation_orders_correctly(self):
        net = _three_edge_network()
        # capacités [1000, 500, 1000], flux [200, 400, 50]
        # saturations [0.2, 0.8, 0.05] → ordre 1, 0, 2
        res = _result_from_flows([200.0, 400.0, 50.0])
        df = rank_by_saturation(net, res, top_n=3)
        assert df.iloc[0]["edge_id"] == 1
        assert df.iloc[1]["edge_id"] == 0
        assert df.iloc[2]["edge_id"] == 2

    def test_share_of_total_delay_sums_to_one(self):
        net = _three_edge_network()
        res = _result_from_flows([100.0, 200.0, 300.0])
        df = rank_by_congestion_delay(net, res, top_n=3)
        assert df["share_of_total_delay"].sum() == pytest.approx(1.0, rel=1e-9)
