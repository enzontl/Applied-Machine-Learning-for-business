"""Tests de la brique 5 : génération + ranking de candidats."""

import geopandas as gpd
import igraph as ig
import numpy as np
import pytest
from shapely.geometry import LineString

from urban_optimizer.assignment import solve_user_equilibrium
from urban_optimizer.demand.od_matrix import ODMatrix
from urban_optimizer.network.urban_network import UrbanNetwork
from urban_optimizer.optimization import (
    Candidate,
    generate_candidates,
    rank_candidates,
    select_under_budget,
)


def _two_route_net() -> UrbanNetwork:
    """Petit réseau A→B sur deux arcs parallèles (un congestionné)."""
    nodes_xy = {0: (0.0, 0.0), 1: (1000.0, 0.0)}
    edges = [(0, 1), (0, 1)]
    g = ig.Graph(n=2, edges=edges, directed=True)
    g.es["t0_s"] = [10.0, 20.0]
    g.es["capacity"] = [500.0, 500.0]
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


def _od(trips: float = 2000.0) -> ODMatrix:
    return ODMatrix(
        matrix=np.array([[0.0, trips], [0.0, 0.0]]),
        zone_ids=["A", "B"],
        zone_to_node={"A": 0, "B": 1},
        hour=8,
        scenario="weekday",
    )


# ── candidates ───────────────────────────────────────────────────────────────

class TestGenerateCandidates:
    def test_returns_at_least_one_candidate_per_arc(self):
        net = _two_route_net()
        ue = solve_user_equilibrium(net, _od(), max_iter=100, tol=1e-5)
        cands = generate_candidates(net, ue, top_n=2)
        # 2 arcs × 4 actions par défaut = 8 candidats
        assert len(cands) == 8
        ids = {c.edge_id for c in cands}
        assert ids == {0, 1}

    def test_capacity_boost_cost_increases_with_magnitude(self):
        net = _two_route_net()
        ue = solve_user_equilibrium(net, _od(), max_iter=100, tol=1e-5)
        c20, c50 = None, None
        for c in generate_candidates(net, ue, top_n=2):
            if c.edge_id == 0 and c.action == "capacity_boost" and c.magnitude == 1.2:
                c20 = c
            if c.edge_id == 0 and c.action == "capacity_boost" and c.magnitude == 1.5:
                c50 = c
        assert c20 is not None and c50 is not None
        assert c50.cost_eur > c20.cost_eur

    def test_remove_uses_remove_cost(self):
        net = _two_route_net()
        ue = solve_user_equilibrium(net, _od(), max_iter=100, tol=1e-5)
        rem = next(c for c in generate_candidates(net, ue, top_n=2) if c.action == "remove")
        # Coût retrait = 50 € / m × 1000 m
        assert rem.cost_eur == pytest.approx(50_000.0, rel=1e-6)


# ── ranking ──────────────────────────────────────────────────────────────────

class TestRankCandidates:
    def test_capacity_boost_reduces_vht(self):
        net = _two_route_net()
        od = _od()
        ue = solve_user_equilibrium(net, od, max_iter=200, tol=1e-7)

        # Ne tester qu'une action ciblée pour rester rapide
        cand = Candidate(
            edge_id=0, action="capacity_boost", magnitude=1.5,
            cost_eur=300_000.0, length_m=1000.0, highway="primary",
        )
        evals = rank_candidates(net, od, [cand], ue, max_iter=80, tol=1e-5)
        ev = evals[0]
        # Doubler quasi-50% la capacité de l'arc rapide doit faire baisser VHT
        assert ev.delta_vht_h > 0
        assert ev.new_vht < ue.vht
        assert ev.annual_benefit_eur > 0

    def test_remove_useful_arc_increases_vht(self):
        net = _two_route_net()
        od = _od()
        ue = solve_user_equilibrium(net, od, max_iter=200, tol=1e-7)
        cand = Candidate(
            edge_id=0, action="remove", magnitude=0.0,
            cost_eur=50_000.0, length_m=1000.0, highway="primary",
        )
        evals = rank_candidates(net, od, [cand], ue, max_iter=80, tol=1e-5)
        ev = evals[0]
        # Retirer l'arc le plus utilisé dégrade le VHT → ΔVHT < 0, pas un Braess
        assert ev.delta_vht_h < 0
        assert ev.is_braess is False


# ── selection ────────────────────────────────────────────────────────────────

class TestSelectUnderBudget:
    def _mk_eval(self, edge_id, benefit, cost) -> "CandidateEvaluation":
        from urban_optimizer.optimization.ranking import CandidateEvaluation
        c = Candidate(
            edge_id=edge_id, action="capacity_boost", magnitude=1.2,
            cost_eur=cost, length_m=100.0, highway="primary",
        )
        return CandidateEvaluation(
            candidate=c,
            delta_vht_h=benefit / 100,
            delta_vht_share=0.05,
            new_vht=1000.0,
            annual_time_value_eur=benefit * 0.7,
            annual_fuel_value_eur=benefit * 0.3,
            annual_benefit_eur=benefit,
            cost_eur=cost,
            score=benefit - cost,
            bcr=benefit / cost if cost > 0 else 0.0,
            is_braess=False,
        )

    def test_picks_top_bcr_under_budget(self):
        evals = [
            self._mk_eval(1, benefit=10_000, cost=5_000),    # BCR = 2.0
            self._mk_eval(2, benefit=15_000, cost=20_000),   # BCR = 0.75
            self._mk_eval(3, benefit=8_000, cost=4_000),     # BCR = 2.0
        ]
        chosen = select_under_budget(evals, budget_eur=10_000)
        ids = {e.candidate.edge_id for e in chosen}
        # Budget 10k → on peut prendre les deux BCR=2.0 (4+5=9k)
        assert ids == {1, 3}

    def test_skips_negative_benefit(self):
        evals = [self._mk_eval(1, benefit=-100, cost=1_000)]
        chosen = select_under_budget(evals, budget_eur=10_000)
        assert chosen == []

    def test_one_per_edge(self):
        evals = [
            self._mk_eval(1, benefit=10_000, cost=2_000),   # BCR = 5
            self._mk_eval(1, benefit=20_000, cost=8_000),   # BCR = 2.5, même arc
        ]
        chosen = select_under_budget(evals, budget_eur=10_000, one_per_edge=True)
        assert len(chosen) == 1
        assert chosen[0].candidate.edge_id == 1
