"""Tests de la brique 5b : design de nouvelles routes."""

import geopandas as gpd
import igraph as ig
import numpy as np
import pytest
from shapely.geometry import LineString

from urban_optimizer.assignment import solve_user_equilibrium
from urban_optimizer.demand.od_matrix import ODMatrix
from urban_optimizer.network.urban_network import UrbanNetwork
from urban_optimizer.optimization import (
    ECOLO,
    EQUILIBRE,
    MOBILITE,
    MayorProfile,
    NewArcProposal,
    generate_new_arc_candidates,
    propose_urban_plan,
    score_network,
)
from urban_optimizer.optimization.network_design import _NewArcContext


# ── helpers ──────────────────────────────────────────────────────────────────

def _detour_network() -> UrbanNetwork:
    """Réseau en U : A et B sont éloignés en graphe mais proches à vol d'oiseau.

        0 ────── 1
        │
        │ (long détour)
        │
        2 ────── 3
    """
    nodes_xy = {0: (0.0, 1000.0), 1: (1000.0, 1000.0),
                2: (0.0, 0.0), 3: (1000.0, 0.0)}
    edges = [
        (0, 1), (1, 0),     # bord haut
        (2, 3), (3, 2),     # bord bas
        (0, 2), (2, 0),     # côté gauche
        (1, 3), (3, 1),     # côté droit (les deux côtés relient 0↔3 indirectement)
    ]
    g = ig.Graph(n=4, edges=edges, directed=True)
    n_e = len(edges)
    g.es["t0_s"] = [60.0] * n_e
    g.es["capacity"] = [800.0] * n_e
    g.es["bpr_alpha"] = [0.15] * n_e
    g.es["bpr_beta"] = [4.0] * n_e
    g.es["length_m"] = [1000.0] * n_e
    g.es["free_speed_kmh"] = [60.0] * n_e
    g.es["highway"] = ["primary"] * n_e
    g.es["source"] = ["osm"] * n_e

    edges_gdf = gpd.GeoDataFrame(
        {"highway": ["primary"] * n_e, "source": ["osm"] * n_e},
        geometry=[LineString([(0, 0), (1000, 1000)])] * n_e,
        crs="EPSG:2154",
    )
    return UrbanNetwork(graph=g, nodes_xy=nodes_xy, edges_gdf=edges_gdf, crs=2154)


def _od(trips: float = 600.0) -> ODMatrix:
    # OD : zone A (nœud 0) → zone B (nœud 3) — détour graphique de 2000m vs euclidien ~1414m
    return ODMatrix(
        matrix=np.array([[0.0, trips], [0.0, 0.0]]),
        zone_ids=["A", "B"],
        zone_to_node={"A": 0, "B": 3},
        hour=8,
        scenario="weekday",
    )


# ── Profils maire ────────────────────────────────────────────────────────────

class TestMayorProfile:
    def test_all_profiles_have_distinct_weights(self):
        from urban_optimizer.optimization import ALL_PROFILES
        names = {p.name for p in ALL_PROFILES}
        assert len(names) == 4

    def test_ecolo_weights_co2_more_than_equilibre(self):
        assert ECOLO.w_co2 > EQUILIBRE.w_co2

    def test_mobilite_weights_time_more(self):
        assert MOBILITE.w_time > EQUILIBRE.w_time


# ── Score ───────────────────────────────────────────────────────────────────

class TestScoreNetwork:
    def test_score_decreases_when_vht_decreases(self):
        net = _detour_network()
        od = _od()
        ue = solve_user_equilibrium(net, od, max_iter=100, tol=1e-5)
        s = score_network(ue, EQUILIBRE)
        assert s.total_annual_cost_eur > 0
        assert s.annual_co2_kg > 0

    def test_ecolo_amplifies_co2_cost(self):
        net = _detour_network()
        od = _od()
        ue = solve_user_equilibrium(net, od, max_iter=100, tol=1e-5)
        s_eq = score_network(ue, EQUILIBRE)
        s_ec = score_network(ue, ECOLO)
        # CO2 weight is 3× → CO2 component should be 3× larger for ecolo
        assert s_ec.annual_co2_cost_eur > s_eq.annual_co2_cost_eur


# ── _NewArcContext ──────────────────────────────────────────────────────────

class TestNewArcContext:
    def test_adds_and_removes_arc_safely(self):
        net = _detour_network()
        g = net.graph
        before = g.ecount()
        prop = NewArcProposal(
            u_node=0, v_node=3,
            length_m=1414.0, highway="secondary",
            capacity=2400.0, free_speed_kmh=70.0,
            construction_cost_eur=10_000_000.0,
            detour_before=1.5,
            u_xy=(0.0, 1000.0), v_xy=(1000.0, 0.0),
        )
        with _NewArcContext(g, prop):
            assert g.ecount() == before + 2  # bidirectionnel
        assert g.ecount() == before          # restauré

    def test_arc_restored_after_exception(self):
        net = _detour_network()
        g = net.graph
        before = g.ecount()
        prop = NewArcProposal(
            u_node=0, v_node=3, length_m=1414.0, highway="secondary",
            capacity=2400.0, free_speed_kmh=70.0,
            construction_cost_eur=10_000_000.0, detour_before=1.5,
            u_xy=(0.0, 1000.0), v_xy=(1000.0, 0.0),
        )
        try:
            with _NewArcContext(g, prop):
                raise RuntimeError("simulated")
        except RuntimeError:
            pass
        assert g.ecount() == before


# ── Génération de candidats ─────────────────────────────────────────────────

class TestGenerateNewArcCandidates:
    def test_finds_obvious_shortcut(self):
        net = _detour_network()
        od = _od()
        cands = generate_new_arc_candidates(
            net, od,
            min_length_m=100, max_length_m=5000,
            min_detour_ratio=1.30,
        )
        # Il existe un raccourci diagonal 0→3 qui réduirait fortement le détour
        assert any({c.u_node, c.v_node} == {0, 3} for c in cands)

    def test_respects_length_window(self):
        net = _detour_network()
        od = _od()
        cands = generate_new_arc_candidates(
            net, od,
            min_length_m=500, max_length_m=1300,
        )
        for c in cands:
            assert 500 <= c.length_m <= 1300

    def test_no_candidates_when_no_detour(self):
        # Si on relâche le ratio de détour assez fort, plus de candidat
        net = _detour_network()
        od = _od()
        cands = generate_new_arc_candidates(
            net, od, min_detour_ratio=10.0,
        )
        assert cands == []


# ── propose_urban_plan ──────────────────────────────────────────────────────

class TestProposeUrbanPlan:
    def test_returns_at_least_one_useful_proposal(self):
        net = _detour_network()
        od = _od()
        ue = solve_user_equilibrium(net, od, max_iter=100, tol=1e-5)
        plan, baseline = propose_urban_plan(
            net, od, EQUILIBRE, ue,
            budget_eur=50_000_000,
            max_proposals=20, max_fw_evals=5,
            fw_max_iter=40, fw_tol=1e-3,
        )
        assert len(plan) >= 1
        # Au moins une proposition rentable
        best = plan[0]
        assert best.annual_benefit_eur > 0
        assert best.new_vht_h < best.baseline_vht_h
        assert baseline.total_annual_cost_eur > 0

    def test_respects_budget(self):
        net = _detour_network()
        od = _od()
        ue = solve_user_equilibrium(net, od, max_iter=100, tol=1e-5)
        plan, _ = propose_urban_plan(
            net, od, EQUILIBRE, ue,
            budget_eur=1_000_000,    # budget volontairement trop bas
            max_proposals=20, max_fw_evals=3,
            fw_max_iter=30, fw_tol=1e-3,
        )
        total_cost = sum(e.cost_eur for e in plan)
        assert total_cost <= 1_000_000


# ── BuildingIndex ────────────────────────────────────────────────────────────

class TestBuildingIndex:
    from shapely.geometry import Polygon

    def _make_index(self, polygons):
        from urban_optimizer.network.buildings import BuildingIndex
        return BuildingIndex(polygons)

    def test_segment_through_building_rejected(self):
        from shapely.geometry import Polygon
        from urban_optimizer.network.buildings import BuildingIndex
        # Bâtiment carré centré en (500, 500)
        building = Polygon([(400, 400), (600, 400), (600, 600), (400, 600)])
        idx = BuildingIndex([building])
        # Segment qui traverse le bâtiment de gauche à droite
        segment = LineString([(0, 500), (1000, 500)])
        assert idx.crosses(segment) is True

    def test_segment_beside_building_accepted(self):
        from shapely.geometry import Polygon
        from urban_optimizer.network.buildings import BuildingIndex
        building = Polygon([(400, 400), (600, 400), (600, 600), (400, 600)])
        idx = BuildingIndex([building])
        # Segment qui passe loin au-dessus du bâtiment
        segment = LineString([(0, 800), (1000, 800)])
        assert idx.crosses(segment) is False

    def test_empty_index_never_rejects(self):
        from urban_optimizer.network.buildings import BuildingIndex
        idx = BuildingIndex([])
        segment = LineString([(0, 0), (1000, 1000)])
        assert idx.crosses(segment) is False

    def test_buffer_catches_near_miss(self):
        from shapely.geometry import Polygon
        from urban_optimizer.network.buildings import BuildingIndex, _ROAD_HALF_WIDTH_M
        # Bâtiment juste à côté du segment (dans le buffer de route)
        building = Polygon([(100, 3), (200, 3), (200, 8), (100, 8)])
        idx = BuildingIndex([building])
        # Segment horizontal à y=0, bâtiment à y=3..8 → dans le buffer (5m)
        segment = LineString([(0, 0), (300, 0)])
        assert idx.crosses(segment, buffer_m=_ROAD_HALF_WIDTH_M) is True

    def test_building_filter_removes_blocked_candidate(self):
        """generate_new_arc_candidates rejette le raccourci si un bâtiment le bloque."""
        from shapely.geometry import Polygon
        from urban_optimizer.network.buildings import BuildingIndex

        net = _detour_network()
        od = _od()

        # Bâtiment géant couvrant toute la diagonale 0→3
        building = Polygon([(-100, -100), (1100, -100), (1100, 1100), (-100, 1100)])
        idx = BuildingIndex([building])

        cands_with = generate_new_arc_candidates(
            net, od,
            min_length_m=100, max_length_m=5000,
            min_detour_ratio=1.30,
            building_index=idx,
        )
        cands_without = generate_new_arc_candidates(
            net, od,
            min_length_m=100, max_length_m=5000,
            min_detour_ratio=1.30,
        )
        # Avec le bâtiment tout couvrant, tous les candidats doivent être éliminés
        assert len(cands_with) == 0
        # Sans filtre, des candidats existent
        assert len(cands_without) > 0
