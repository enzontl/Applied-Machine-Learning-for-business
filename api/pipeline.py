"""Wrapper API autour de l'orchestration urban_optimizer.

Lance le pipeline complet (build → OD → UE → plan), sérialise en GeoJSON pour
le front, et publie l'état dans le JobRegistry.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from math import isfinite
from pathlib import Path
from typing import Any

import numpy as np

from urban_optimizer.assignment import solve_user_equilibrium
from urban_optimizer.config import CRS_WGS84
from urban_optimizer.demand import generate_od_matrix
from urban_optimizer.diagnosis import compute_accessibility
from urban_optimizer.network import build_network, load_bridge_triggers, load_obstacles, purge_old_caches

# Purger les anciens caches réseau au chargement du module (1 seule fois)
# pour forcer la reconstruction avec composante connexe.
purge_old_caches()
from urban_optimizer.optimization import (
    ALL_PROFILES,
    PROFILE_BY_NAME,
    compute_pareto_frontier,
    evaluate_plan_robustness,
    generate_candidates,
    propose_urban_plan,
    rank_candidates,
    score_network,
)

from .forecast_integration import compute_budget, project_od_if_eligible
from .jobs import JobState

# Trouver l'insee de la ville depuis PRESET_CITIES (lazy import pour éviter cycle)
def _lookup_insee_population(city_osm: str) -> tuple[str | None, float | None]:
    from .main import PRESET_CITIES
    for c in PRESET_CITIES:
        if c["osm"] == city_osm:
            return c.get("insee"), float(c.get("population", 0)) or None
    return None, None


# Chemin par défaut du modèle forecast (résolu dans api/main.py)
FORECAST_MODEL_PATH = str(
    Path(__file__).resolve().parent.parent / "models" / "forecast" / "idf"
)

logger = logging.getLogger(__name__)

# Profils proposés au front (ordre = ordre d'affichage)
PROFILE_ORDER = ["equilibre", "ecolo", "mobilite", "economique"]


def _format_eur(value: float) -> str:
    """Format compact pour montants en €."""
    if not isfinite(value):
        return "—"
    abs_v = abs(value)
    sign = "−" if value < 0 else ""
    if abs_v >= 1e9:
        return f"{sign}{abs_v / 1e9:.2f} G€"
    if abs_v >= 1e6:
        return f"{sign}{abs_v / 1e6:.1f} M€"
    if abs_v >= 1e3:
        return f"{sign}{abs_v / 1e3:.0f} k€"
    return f"{sign}{abs_v:.0f} €"


def _format_int(value: float) -> str:
    if not isfinite(value):
        return "—"
    return f"{value:,.0f}".replace(",", " ")


def _action_label(proposal_type: str) -> str:
    return {
        "corridor": "Élargissement",
        "upgrade": "Mise à niveau",
        "new_route": "Nouvelle route",
    }.get(proposal_type, proposal_type)


def _profile_label(profile_name: str) -> str:
    p = PROFILE_BY_NAME.get(profile_name)
    return p.label if p else profile_name


def _interventions_payload(plan: list) -> list[dict]:
    rows = []
    for i, ev in enumerate(plan, start=1):
        p = ev.proposal
        rows.append({
            "rank": i,
            "type": p.proposal_type,  # "corridor" | "upgrade" | "new_route"
            "action": _action_label(p.proposal_type),
            "highway": p.highway,
            "length_m": float(p.length_m),
            "detour_before": float(p.detour_before),
            "delta_vht_h": float(ev.delta_vht_h),
            "annual_benefit_eur": float(ev.annual_benefit_eur),
            "construction_cost_eur": float(p.construction_cost_eur),
            "payback_years": float(ev.payback_years) if isfinite(ev.payback_years) else None,
            "bcr": float(ev.bcr),
        })
    return rows


def _baseline_payload(baseline_score, baseline_access, ue, net, od) -> dict:
    """Décomposition du baseline pour le bloc 'Score actuel' du dashboard."""
    from urban_optimizer.diagnosis import diagnose
    diag = diagnose(net, ue)
    return {
        "annual_total_eur": float(baseline_score.total_annual_cost_eur),
        "annual_time_eur": float(baseline_score.annual_time_cost_eur),
        "annual_fuel_eur": float(baseline_score.annual_fuel_cost_eur),
        "annual_co2_eur": float(baseline_score.annual_co2_cost_eur),
        "annual_co2_kg": float(baseline_score.annual_co2_kg),
        "accessibility_mean": float(baseline_access.mean_reachable),
        "accessibility_gini": float(baseline_access.gini),
        "accessibility_value_eur": float(baseline_score.accessibility_value_eur),
        "equity_penalty_eur": float(baseline_score.equity_penalty_eur),
        "vht_h": float(ue.vht),
        # Diagnostic technique du réseau
        "n_nodes": int(net.graph.vcount()),
        "n_edges": int(net.graph.ecount()),
        "n_zones": int(od.n_zones),
        "total_trips": float(od.total_trips),
        "congestion_overhead_pct": float(diag.congestion_overhead * 100),
        "n_saturated_arcs": int(diag.n_saturated),
        "fw_gap": float(ue.final_gap),
        "fw_iterations": int(ue.iterations),
    }


def _joint_payload(joint) -> dict | None:
    """Détail re-évaluation jointe pour le bloc dashboard correspondant."""
    if joint is None:
        return None
    redundancy_pct = (1.0 - joint.redundancy_factor) * 100
    return {
        "n_interventions": int(joint.n_interventions),
        "joint_delta_vht_h": float(joint.joint_delta_vht_h),
        "naive_sum_delta_vht_h": float(joint.naive_sum_delta_vht_h),
        "redundancy_pct": float(redundancy_pct),
        "joint_bcr": float(joint.joint_bcr),
        "joint_annual_benefit_eur": float(joint.joint_annual_benefit_eur),
        "naive_sum_annual_benefit_eur": float(joint.naive_sum_annual_benefit_eur),
        "total_cost_eur": float(joint.total_cost_eur),
        "accessibility_before": float(joint.accessibility_before),
        "accessibility_after": float(joint.accessibility_after),
        "gini_before": float(joint.gini_before),
        "gini_after": float(joint.gini_after),
        "joint_vht_h": float(joint.joint_vht_h),
        "baseline_vht_h": float(joint.baseline_vht_h),
    }


def _kpis_for_profile(
    profile_label: str,
    baseline_score,
    joint,
    plan_count: int,
    capex_total: float,
) -> list[dict]:
    """Construit la liste de KPIs à afficher en cards en haut du dashboard."""
    if joint is not None:
        joint_score_total = baseline_score.total_annual_cost_eur - joint.joint_annual_benefit_eur
        delta_score = joint.joint_annual_benefit_eur
        delta_vht = joint.joint_delta_vht_h
    else:
        joint_score_total = baseline_score.total_annual_cost_eur
        delta_score = 0.0
        delta_vht = 0.0

    return [
        {
            "label": "Coût annuel après plan",
            "value": _format_eur(joint_score_total),
            "delta": f"gain annuel {_format_eur(delta_score)}/an" if delta_score > 0 else None,
            "accent": "good" if delta_score > 0 else "neutral",
        },
        {
            "label": "Gain annuel estimé",
            "value": _format_eur(delta_score),
            "delta": "bénéfice social net",
            "accent": "good" if delta_score > 0 else "neutral",
        },
        {
            "label": "ΔVHT (joint)",
            "value": f"−{_format_int(delta_vht)} h" if delta_vht > 0 else f"{_format_int(delta_vht)} h",
            "accent": "good" if delta_vht > 0 else "neutral",
        },
        {
            "label": "Interventions retenues",
            "value": str(plan_count),
            "delta": f"{_format_eur(capex_total)} CAPEX" if capex_total > 0 else None,
            "accent": "neutral",
        },
    ]


def _network_geojson(
    net,
    sat_before: np.ndarray,
    sat_after: np.ndarray,
) -> dict:
    """Convertit le réseau en GeoJSON FeatureCollection (WGS84).

    Chaque arc devient une feature avec ses 2 saturations (avant/après).
    Le front choisit laquelle afficher selon le toggle.
    """
    edges_wgs = net.edges_gdf.to_crs(CRS_WGS84)

    # Échantillonnage intelligent : tous les arcs saturés (≥ 0.5), 1500 autres aléatoires
    n_edges = len(edges_wgs)
    sat_union = np.maximum(sat_before, sat_after)
    important = np.where(sat_union >= 0.5)[0]
    rest = np.where(sat_union < 0.5)[0]
    rest_sample = min(1500, len(rest))
    rng = np.random.default_rng(0)
    if len(rest) > rest_sample:
        rest_idx = rng.choice(rest, size=rest_sample, replace=False)
    else:
        rest_idx = rest
    selected = np.concatenate([important, rest_idx])

    features = []
    geoms = edges_wgs.geometry
    for i in selected:
        geom = geoms.iloc[int(i)]
        if geom is None or geom.geom_type != "LineString":
            continue
        # Coordonnées GeoJSON : [lng, lat] (pas lat,lng)
        coords = [[float(x), float(y)] for x, y in geom.coords]
        features.append({
            "type": "Feature",
            "geometry": {"type": "LineString", "coordinates": coords},
            "properties": {
                "sat_before": float(sat_before[int(i)]),
                "sat_after": float(sat_after[int(i)]),
            },
        })
    return {"type": "FeatureCollection", "features": features}


def _serialize_robustness(rob) -> dict:
    """Test de robustesse → payload JSON."""
    points = []
    for p in rob.points:
        points.append({
            "demand_scale": float(p.demand_scale),
            "baseline_vht_h": float(p.baseline_vht_h),
            "plan_vht_h": float(p.plan_vht_h),
            "delta_vht_h": float(p.delta_vht_h),
            "annual_benefit_eur": float(p.annual_benefit_eur),
            "is_beneficial": bool(p.is_beneficial),
        })
    worst = rob.worst_point
    best = rob.best_point
    return {
        "is_robust": bool(rob.is_robust),
        "points": points,
        "worst_scale": float(worst.demand_scale) if worst else None,
        "worst_benefit_eur": float(worst.annual_benefit_eur) if worst else None,
        "best_scale": float(best.demand_scale) if best else None,
        "best_benefit_eur": float(best.annual_benefit_eur) if best else None,
        "n_failing": sum(1 for p in points if not p["is_beneficial"]),
    }


def _serialize_pareto(par) -> dict:
    """Courbe Pareto → payload JSON."""
    points = []
    for p in par.points:
        points.append({
            "budget_eur": float(p.budget_eur),
            "capex_used_eur": float(p.capex_used_eur),
            "n_interventions": int(p.n_interventions),
            "joint_annual_benefit_eur": float(p.joint_annual_benefit_eur),
            "joint_vht_h": float(p.joint_vht_h),
            "delta_vht_h": float(p.delta_vht_h),
            "bcr": float(p.bcr),
            "marginal_efficiency_eur_per_meur": float(p.marginal_efficiency_eur_per_meur),
        })
    sweet = par.best_marginal_point
    return {
        "points": points,
        "sweet_spot_budget_eur": float(sweet.budget_eur) if sweet else None,
        "sweet_spot_benefit_eur": float(sweet.joint_annual_benefit_eur) if sweet else None,
        "sweet_spot_marginal": float(sweet.marginal_efficiency_eur_per_meur) if sweet else None,
    }


def _braess_payload(braess_evals: list) -> list[dict]:
    """Liste des arcs Braess (suppressions bénéfiques)."""
    rows = []
    for i, ev in enumerate(braess_evals, start=1):
        c = ev.candidate
        rows.append({
            "rank": i,
            "edge_id": int(c.edge_id),
            "highway": c.highway,
            "length_m": float(c.length_m),
            "delta_vht_h": float(ev.delta_vht_h),
            "annual_benefit_eur": float(ev.annual_benefit_eur),
        })
    return rows


def _braess_geojson(braess_evals: list, net) -> dict:
    """GeoJSON des arcs à supprimer (Braess) — projetés en WGS84."""
    edges_wgs = net.edges_gdf.to_crs(CRS_WGS84)
    features = []
    for i, ev in enumerate(braess_evals, start=1):
        eid = int(ev.candidate.edge_id)
        if eid >= len(edges_wgs):
            continue
        geom = edges_wgs.geometry.iloc[eid]
        if geom is None or geom.geom_type != "LineString":
            continue
        coords = [[float(x), float(y)] for x, y in geom.coords]
        features.append({
            "type": "Feature",
            "geometry": {"type": "LineString", "coordinates": coords},
            "properties": {
                "rank": i,
                "edge_id": eid,
                "highway": ev.candidate.highway,
                "length_m": float(ev.candidate.length_m),
                "delta_vht_h": float(ev.delta_vht_h),
                "annual_benefit_eur": float(ev.annual_benefit_eur),
            },
        })
    return {"type": "FeatureCollection", "features": features}


def _plan_geojson(plan: list, net) -> dict:
    """GeoJSON des interventions retenues."""
    import geopandas as gpd
    from shapely.geometry import LineString

    features = []
    for i, ev in enumerate(plan, start=1):
        p = ev.proposal
        raw_coords = p.corridor_xy if len(p.corridor_xy) >= 2 else [p.u_xy, p.v_xy]
        line = LineString(raw_coords)
        line_wgs = gpd.GeoSeries([line], crs=net.crs).to_crs(CRS_WGS84).iloc[0]
        coords = [[float(x), float(y)] for x, y in line_wgs.coords]
        features.append({
            "type": "Feature",
            "geometry": {"type": "LineString", "coordinates": coords},
            "properties": {
                "rank": i,
                "type": p.proposal_type,
                "action": _action_label(p.proposal_type),
                "highway": p.highway,
                "length_m": float(p.length_m),
                "construction_cost_eur": float(p.construction_cost_eur),
                "annual_benefit_eur": float(ev.annual_benefit_eur),
                "bcr": float(ev.bcr),
            },
        })
    return {"type": "FeatureCollection", "features": features}


def _adaptive_params(n_edges: int, req_dict: dict[str, Any]) -> dict[str, Any]:
    """Ajuste les hyper-paramètres FW selon la taille du graphe."""
    # (cap_ue, fw_iter, fw_evals_max, cand_max, rob_iter, rob_tol, label)
    if n_edges > 15_000:
        cap_ue, fw_iter, fw_max, cand_max, rob_i, rob_tol, label = 30, 10, 5, 12, 8, 1e-2, "GRANDE"
    elif n_edges > 5_000:
        cap_ue, fw_iter, fw_max, cand_max, rob_i, rob_tol, label = 40, 12, 7, 15, 10, 8e-3, "MOYENNE"
    else:
        cap_ue, fw_iter, fw_max, cand_max, rob_i, rob_tol, label = 999, 15, 999, 999, 15, 5e-3, "PETITE"
    logger.info(f"Adaptive params: ville {label} ({n_edges} arcs)")
    return {
        "max_iter_ue": min(int(req_dict.get("max_iter_ue", 50)), cap_ue),
        "fw_max_iter": fw_iter,
        "fw_tol": 5e-3,
        "max_fw_evals": min(int(req_dict.get("max_fw_evals", 8)), fw_max),
        "max_candidates": min(int(req_dict.get("max_candidates", 20)), cand_max),
        "rob_max_iter": rob_i,
        "rob_tol": rob_tol,
    }


def run_pipeline(job: JobState, req_dict: dict[str, Any]) -> None:
    """Lance le pipeline complet et stocke le résultat sérialisable dans job.result.

    Appelé en background thread. Met à jour job.progress + job.step au fil de l'eau.
    """
    try:
        import time as _time
        _t0_pipeline = _time.perf_counter()

        def _elapsed() -> str:
            return f"{_time.perf_counter() - _t0_pipeline:.1f}s"

        def _step_timer(label: str) -> float:
            t = _time.perf_counter()
            return t

        job.update(status="running", progress=0.02, step="Construction du réseau…")
        city = req_dict["city"]
        include_route500 = req_dict["include_route500"]
        simplified_highway = req_dict.get("simplified_highway", False)

        _t = _step_timer("build_network")
        net = build_network(
            city,
            include_route500=include_route500,
            simplified_highway=simplified_highway,
        )
        logger.info(f"⏱ build_network : {_time.perf_counter() - _t:.1f}s")

        # ── Auto-adaptation des paramètres selon la taille du graphe ──
        n_edges = net.graph.ecount()
        ap = _adaptive_params(n_edges, req_dict)
        logger.info(
            f"Réseau construit : {net.graph.vcount()} nœuds, {n_edges} arcs "
            f"(simplified={simplified_highway})"
        )

        # ── Lancer le chargement des obstacles en arrière-plan ──────────
        # Les obstacles sont indépendants du réseau OD / UE / accessibilité.
        # On les télécharge pendant que le FW tourne → gain ~60-180 s au premier run.
        _obs_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="obs")
        _fut_obstacles = _obs_pool.submit(load_obstacles, city)
        _fut_bridges   = _obs_pool.submit(load_bridge_triggers, city)
        _t_obs_start   = _time.perf_counter()

        job.update(progress=0.10, step="Génération de la demande OD…")
        _t = _step_timer("generate_od")
        od = generate_od_matrix(
            net,
            hour=req_dict["hour"],
            method="grid",
            n_cells=req_dict["n_cells"],
            scale_factor=req_dict["scale_factor"],
        )
        logger.info(f"⏱ generate_od : {_time.perf_counter() - _t:.1f}s")

        # ── Projection ML à l'horizon (si éligible) ──────────────────────
        insee_code, city_pop = _lookup_insee_population(city)
        horizon_years = int(req_dict.get("horizon_years", 0))
        projection_payload = None
        if horizon_years > 0:
            job.update(
                progress=0.15,
                step=f"Projection demande horizon H+{horizon_years}…",
            )
            _t = _step_timer("project_od")
            od, projection_payload = project_od_if_eligible(
                od, insee_code=insee_code, horizon_years=horizon_years,
                model_path=FORECAST_MODEL_PATH,
            )
            logger.info(f"⏱ project_od : {_time.perf_counter() - _t:.1f}s "
                        f"(projected={projection_payload is not None})")

        # ── Budget automatique (OFGL si IDF, sinon heuristique pop) ──────
        budget_payload = compute_budget(
            insee_code=insee_code, population=city_pop,
            horizon_years=horizon_years,
        )
        budget_eur = budget_payload["total_eur"]
        logger.info(f"Budget {budget_payload['source']} = {budget_eur/1e6:.1f} M€ "
                    f"sur {budget_payload['horizon_years']} ans")

        max_iter_ue = ap["max_iter_ue"]
        job.update(progress=0.20, step=f"Affectation Frank-Wolfe ({max_iter_ue} itérations)…")
        _t = _step_timer("UE baseline")
        ue = solve_user_equilibrium(net, od, max_iter=max_iter_ue, tol=1e-4)
        logger.info(f"⏱ UE baseline : {_time.perf_counter() - _t:.1f}s ({ue.iterations} iter, gap={ue.final_gap:.2e})")

        job.update(progress=0.40, step="Accessibilité + score de base…")
        _t = _step_timer("accessibility")
        access_thresh_s = float(req_dict["access_threshold_min"]) * 60.0
        baseline_access = compute_accessibility(
            net, od, ue, threshold_seconds=access_thresh_s,
        )
        logger.info(f"⏱ accessibility : {_time.perf_counter() - _t:.1f}s")

        # ── Récupérer les obstacles (déjà chargés en arrière-plan) ─────
        job.update(progress=0.45, step="Obstacles OSM…")
        obstacles = _fut_obstacles.result()
        bridges   = _fut_bridges.result()
        _obs_pool.shutdown(wait=False)
        logger.info(f"⏱ load_obstacles+bridges : {_time.perf_counter() - _t_obs_start:.1f}s (parallèle avec OD+UE+access)")

        # Profils à évaluer
        if req_dict["multi_profile"]:
            profile_names = PROFILE_ORDER
        else:
            profile_names = [req_dict["profile"]]

        # Saturation v/c avant plan
        capacity = np.asarray(net.graph.es["capacity"], dtype=float)
        sat_before = ue.flows / np.maximum(capacity, 1.0)

        results: list[dict] = []
        network_geojson = None
        plan_geojson = None
        main_plan = None       # liste des NewArcEvaluation du profil principal (pour robustesse)
        main_profile = None    # MayorProfile principal
        main_baseline_payload = None
        main_joint_payload = None

        # Réserve la dernière tranche de progress pour les analyses optionnelles
        wants_robust = req_dict.get("include_robustness", False)
        wants_pareto = req_dict.get("include_pareto", False)
        wants_braess = req_dict.get("include_braess", False)
        n_extra = int(wants_robust) + int(wants_pareto) + int(wants_braess)
        # base : 0.50 → 0.50 + main_span ; reste pour les extras
        main_span = (0.45 if n_extra == 0 else 0.30)
        for idx, prof_name in enumerate(profile_names):
            profile = PROFILE_BY_NAME[prof_name]
            base_prog = 0.50 + idx * (main_span / len(profile_names))
            step_prog = main_span / len(profile_names)

            job.update(
                progress=base_prog,
                step=f"Optimisation profil {profile.label}…",
            )

            _t = _step_timer(f"propose_urban_plan[{prof_name}]")
            plan, baseline_score, joint = propose_urban_plan(
                net, od, profile, ue,
                budget_eur=budget_eur,
                max_proposals=ap["max_candidates"],
                max_fw_evals=ap["max_fw_evals"],
                fw_max_iter=ap["fw_max_iter"], fw_tol=ap["fw_tol"],
                obstacle_index=obstacles,
                soft_index=bridges,
                periphery_margin_m=float(req_dict["periphery_margin_m"]),
                accessibility_threshold_s=access_thresh_s,
                enable_induced_demand=(horizon_years > 0),  # rebond activé si projection
                _baseline_access=baseline_access,
            )
            logger.info(f"⏱ propose_urban_plan[{prof_name}] : {_time.perf_counter() - _t:.1f}s → {len(plan)} interventions")

            # Sat après = depuis le joint (sinon = avant)
            if joint is not None and joint.existing_sat_after.size == len(capacity):
                sat_after = joint.existing_sat_after
            else:
                sat_after = sat_before

            capex_total = sum(ev.proposal.construction_cost_eur for ev in plan)

            results.append({
                "profile_name": profile.name,
                "profile_label": profile.label,
                "kpis": _kpis_for_profile(
                    profile.label, baseline_score, joint, len(plan),
                    capex_total,
                ),
                "interventions": _interventions_payload(plan),
                "baseline_vht_h": float(ue.vht),
                "joint_vht_h": float(joint.joint_vht_h) if joint else float(ue.vht),
                "baseline_score_eur": float(baseline_score.total_annual_cost_eur),
                "joint_score_eur": float(
                    baseline_score.total_annual_cost_eur - (joint.joint_annual_benefit_eur if joint else 0.0)
                ),
                "accessibility_before": float(baseline_access.mean_reachable),
                "accessibility_after": float(joint.accessibility_after) if joint else float(baseline_access.mean_reachable),
                "gini_before": float(baseline_access.gini),
                "gini_after": float(joint.gini_after) if joint else float(baseline_access.gini),
            })

            # GeoJSON + plan principal : on les capture sur le PREMIER profil
            if idx == 0:
                network_geojson = _network_geojson(net, sat_before, sat_after)
                plan_geojson = _plan_geojson(plan, net)
                main_plan = plan
                main_profile = profile
                main_baseline_payload = _baseline_payload(baseline_score, baseline_access, ue, net, od)
                main_joint_payload = _joint_payload(joint)

            job.update(progress=base_prog + step_prog * 0.95)

        # ── Analyses optionnelles (sur le profil principal uniquement) ────
        robustness_payload = None
        pareto_payload = None
        braess_payload = None
        braess_geojson = None
        analyses_base = 0.50 + main_span
        analyses_step = (1.0 - analyses_base - 0.02) / max(1, n_extra)

        if wants_braess and main_plan is not None:
            job.update(
                progress=analyses_base + 0.01,
                step="Recherche d'arcs Braess (suppressions bénéfiques)…",
            )
            removal_cands = [
                c for c in generate_candidates(net, ue, top_n=15)
                if c.action == "remove"
            ]
            evals_rem = rank_candidates(net, od, removal_cands, ue, max_iter=ap["fw_max_iter"], tol=ap["fw_tol"])
            braess_evals = [e for e in evals_rem if e.is_braess][:3]
            braess_payload = _braess_payload(braess_evals)
            braess_geojson = _braess_geojson(braess_evals, net)
            analyses_base += analyses_step

        if wants_robust and main_plan and main_profile is not None:
            job.update(
                progress=analyses_base + 0.01,
                step="Test de robustesse (4 scénarios de demande)…",
            )
            rob = evaluate_plan_robustness(
                net, od, main_plan, main_profile,
                demand_scales=(0.8, 1.0, 1.2, 1.5),
                fw_max_iter=ap["rob_max_iter"], fw_tol=ap["rob_tol"],
            )
            robustness_payload = _serialize_robustness(rob)
            analyses_base += analyses_step

        if wants_pareto:
            job.update(
                progress=analyses_base + 0.01,
                step="Courbe Pareto budget → bénéfice (6 niveaux)…",
            )
            # 6 niveaux centrés autour du budget calculé (×0.2, ×0.5, ×1, ×2, ×4, ×8)
            b = budget_eur
            pareto_budgets = tuple(b * f for f in (0.2, 0.5, 1.0, 2.0, 4.0, 8.0))
            par = compute_pareto_frontier(
                net, od, main_profile or PROFILE_BY_NAME[profile_names[0]], ue,
                budgets_eur=pareto_budgets,
                max_proposals=ap["max_candidates"],
                max_fw_evals=ap["max_fw_evals"],
                fw_max_iter=ap["fw_max_iter"], fw_tol=ap["fw_tol"],
                obstacle_index=obstacles,
                soft_index=bridges,
                periphery_margin_m=float(req_dict["periphery_margin_m"]),
                accessibility_threshold_s=access_thresh_s,
            )
            pareto_payload = _serialize_pareto(par)

        logger.info(f"⏱⏱ PIPELINE TOTAL : {_elapsed()} pour {city}")

        job.update(
            progress=1.0,
            step=f"Terminé en {_elapsed()}.",
            status="done",
            result={
                "city": city,
                "horizon_years": horizon_years,
                "budget": budget_payload,
                "projection": projection_payload,
                "profiles": results,
                "baseline": main_baseline_payload,
                "joint": main_joint_payload,
                "network_geojson": network_geojson,
                "plan_geojson": plan_geojson,
                "robustness": robustness_payload,
                "pareto": pareto_payload,
                "braess": braess_payload,
                "braess_geojson": braess_geojson,
            },
        )

    except Exception as exc:
        logger.exception("Pipeline failed")
        job.update(status="error", error=str(exc), step=f"Erreur : {exc}")
