"""Tests de la brique 1 : construction du réseau (OSM + ROUTE500)."""

import geopandas as gpd
import igraph as ig
import numpy as np
import pytest
from shapely.geometry import LineString, Point

from urban_optimizer.network.merger import build_unified_network
from urban_optimizer.network.osm_loader import (
    _clean_lanes,
    _normalize_highway,
    _parse_maxspeed,
)
from urban_optimizer.network.route500_loader import filter_route500_outside_urban
from urban_optimizer.network.urban_network import UrbanNetwork


# ── helpers ──────────────────────────────────────────────────────────────────

def _make_node_gdf(coords: dict) -> gpd.GeoDataFrame:
    """Construit un GeoDataFrame de nœuds OSM à partir de {osmid: (x, y)}."""
    return gpd.GeoDataFrame(
        {"geometry": [Point(xy) for xy in coords.values()]},
        index=list(coords.keys()),
        crs="EPSG:2154",
    )


def _make_edge_gdf(rows: list[dict], u_coords: dict, v_coords: dict) -> gpd.GeoDataFrame:
    """Construit un GeoDataFrame d'arcs OSM (minimal)."""
    geoms = [LineString([u_coords[r["u"]], v_coords[r["v"]]]) for r in rows]
    df = gpd.GeoDataFrame(rows, geometry=geoms, crs="EPSG:2154")
    df = df.set_index(["u", "v", "key"])
    return df


# ── tests normalisation OSM ───────────────────────────────────────────────────

class TestNormalizeHighway:
    def test_string_unchanged(self):
        assert _normalize_highway("residential") == "residential"

    def test_list_returns_first(self):
        assert _normalize_highway(["primary", "residential"]) == "primary"


class TestCleanLanes:
    def test_none_returns_default(self):
        assert _clean_lanes(None) == 1

    def test_string_int(self):
        assert _clean_lanes("2") == 2

    def test_list_returns_max(self):
        assert _clean_lanes(["1", "3"]) == 3

    def test_float_string(self):
        assert _clean_lanes("1.0") == 1


class TestParseMaxspeed:
    def test_none(self):
        assert _parse_maxspeed(None) is None

    def test_kmh_string(self):
        assert _parse_maxspeed("50") == 50

    def test_mph_converted(self):
        result = _parse_maxspeed("30 mph")
        assert 48 <= result <= 49

    def test_invalid_string(self):
        assert _parse_maxspeed("variable") is None


# ── test filter_route500_outside_urban ───────────────────────────────────────

def test_filter_route500_outside_urban():
    from shapely.geometry import box

    urban = box(0, 0, 100, 100)

    lines = gpd.GeoDataFrame(
        {
            "highway_clean": ["motorway", "primary"],
            "lanes_clean": [4, 2],
            "length_m": [500.0, 300.0],
            "free_speed_kmh": [130.0, 90.0],
            "capacity": [8000.0, 2700.0],
            "t0_s": [13.8, 12.0],
            "bpr_alpha": [0.15, 0.15],
            "bpr_beta": [4.0, 4.0],
            "source": ["route500", "route500"],
            "r500_id": ["A", "B"],
            "oneway": [False, False],
        },
        geometry=[
            LineString([(50, 50), (60, 60)]),   # centroïde dans la zone → retiré
            LineString([(200, 200), (300, 300)]),  # centroïde hors zone → conservé
        ],
        crs="EPSG:2154",
    )

    outside = filter_route500_outside_urban(lines, urban)
    assert len(outside) == 1
    assert outside.iloc[0]["r500_id"] == "B"


# ── test build_unified_network ────────────────────────────────────────────────

def _minimal_osm_network():
    """Réseau OSM minimal : 3 nœuds, 2 arcs."""
    coords = {1: (0.0, 0.0), 2: (100.0, 0.0), 3: (200.0, 0.0)}
    nodes = _make_node_gdf(coords)
    rows = [
        {"u": 1, "v": 2, "key": 0, "highway_clean": "residential", "lanes_clean": 1,
         "length_m": 100.0, "free_speed_kmh": 30.0, "capacity": 600.0,
         "t0_s": 12.0, "bpr_alpha": 0.15, "bpr_beta": 4.0, "source": "osm"},
        {"u": 2, "v": 3, "key": 0, "highway_clean": "primary", "lanes_clean": 2,
         "length_m": 100.0, "free_speed_kmh": 50.0, "capacity": 3000.0,
         "t0_s": 7.2, "bpr_alpha": 0.15, "bpr_beta": 4.0, "source": "osm"},
    ]
    edges = _make_edge_gdf(rows, coords, coords)
    return edges, nodes


class TestBuildUnifiedNetwork:
    def test_osm_only(self):
        edges, nodes = _minimal_osm_network()
        net = build_unified_network(edges, nodes, edges_r500=None)
        assert isinstance(net, UrbanNetwork)
        assert isinstance(net.graph, ig.Graph)
        assert net.graph.vcount() == 3
        assert net.graph.ecount() == 2

    def test_with_route500(self):
        edges_osm, nodes_osm = _minimal_osm_network()

        r500 = gpd.GeoDataFrame(
            {
                "highway_clean": ["motorway"],
                "lanes_clean": [4],
                "length_m": [500.0],
                "free_speed_kmh": [130.0],
                "capacity": [8000.0],
                "t0_s": [13.8],
                "bpr_alpha": [0.15],
                "bpr_beta": [4.0],
                "source": ["route500"],
                "r500_id": ["X"],
                "oneway": [False],
            },
            geometry=[LineString([(500.0, 0.0), (600.0, 0.0)])],
            crs="EPSG:2154",
        )

        net = build_unified_network(edges_osm, nodes_osm, edges_r500=r500)
        # arcs OSM (2) + arcs route500 bidirectionnels (2)
        assert net.graph.ecount() == 4
        assert set(net.edges_gdf["source"]) == {"osm", "route500"}

    def test_edge_attributes_present(self):
        edges, nodes = _minimal_osm_network()
        net = build_unified_network(edges, nodes)
        for attr in ["length_m", "free_speed_kmh", "capacity", "t0_s", "bpr_alpha", "bpr_beta"]:
            assert attr in net.graph.edge_attributes(), f"attribut manquant : {attr}"

    def test_no_negative_values(self):
        edges, nodes = _minimal_osm_network()
        net = build_unified_network(edges, nodes)
        g = net.graph
        assert all(v > 0 for v in g.es["length_m"])
        assert all(v > 0 for v in g.es["capacity"])
        assert all(v > 0 for v in g.es["t0_s"])

    def test_nodes_xy_coverage(self):
        edges, nodes = _minimal_osm_network()
        net = build_unified_network(edges, nodes)
        assert len(net.nodes_xy) == net.graph.vcount()

    def test_crs(self):
        edges, nodes = _minimal_osm_network()
        net = build_unified_network(edges, nodes)
        assert net.crs == 2154
        assert net.edges_gdf.crs.to_epsg() == 2154
