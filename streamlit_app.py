"""Démo interactive Urban Optimizer.

Pipeline :
1. Construction du réseau (OSM, ROUTE500 optionnel)
2. Génération de la demande OD gravitaire
3. Affectation User Equilibrium (Frank-Wolfe)
4. **Score actuel** de la ville selon le profil maire choisi (€/an)
5. **Plan urbain** : nouvelles routes proposées (bleu) + arcs Braess à supprimer (rouge)
6. Score après plan + coût de construction + payback

Lancement : ``streamlit run streamlit_app.py``
"""

from __future__ import annotations

import logging
import time

import folium
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

from urban_optimizer.assignment import (
    price_of_anarchy,
    solve_system_optimum,
    solve_user_equilibrium,
)
from urban_optimizer.config import CRS_WGS84
from urban_optimizer.demand import generate_od_matrix
from urban_optimizer.diagnosis import compute_accessibility, diagnose
from urban_optimizer.network import build_network, load_bridge_triggers, load_obstacles
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

logging.getLogger("urban_optimizer").setLevel(logging.WARNING)

st.set_page_config(page_title="Urban Optimizer", layout="wide", page_icon="🛣️")


# ────────────────────────────────────────────────────────────────────────────
# Caches
# ────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def cached_network(city: str, include_route500: bool, simplified_highway: bool = False):
    return build_network(
        city, include_route500=include_route500,
        simplified_highway=simplified_highway, use_disk_cache=True,
    )


@st.cache_resource(show_spinner=False)
def cached_obstacles(city: str):
    return load_obstacles(city)


@st.cache_resource(show_spinner=False)
def cached_bridge_triggers(city: str):
    return load_bridge_triggers(city)


@st.cache_data(show_spinner=False)
def cached_od(city: str, hour: int, n_cells: int, scale_factor: float,
              include_route500: bool, simplified_highway: bool):
    net = cached_network(city, include_route500, simplified_highway)
    return generate_od_matrix(
        net, hour=hour, method="grid", n_cells=n_cells, scale_factor=scale_factor,
    )


@st.cache_data(show_spinner=False)
def cached_ue(city: str, hour: int, n_cells: int, scale_factor: float,
              include_route500: bool, simplified_highway: bool, max_iter: int):
    net = cached_network(city, include_route500, simplified_highway)
    od = cached_od(city, hour, n_cells, scale_factor, include_route500, simplified_highway)
    return solve_user_equilibrium(net, od, max_iter=max_iter, tol=1e-4)


@st.cache_resource(show_spinner=False)
def cached_edges_wgs_and_coords(city: str, include_route500: bool, simplified_highway: bool):
    """Pré-calcule edges_gdf en WGS84 + liste de coords [(lat, lon), ...] par arc.

    Évite (1) la reprojection CRS Lambert→WGS84 à chaque run streamlit,
    (2) le pandas .iloc[] dans la boucle de rendu des cartes.
    """
    net = cached_network(city, include_route500, simplified_highway)
    edges_wgs = net.edges_gdf.to_crs(CRS_WGS84)
    coords_per_edge: list = []
    for geom in edges_wgs.geometry:
        if geom is None or geom.geom_type != "LineString":
            coords_per_edge.append(None)
        else:
            coords_per_edge.append([(y, x) for x, y in geom.coords])
    return edges_wgs, coords_per_edge


# ────────────────────────────────────────────────────────────────────────────
# Sidebar — paramètres
# ────────────────────────────────────────────────────────────────────────────

st.sidebar.title("🛣️ Urban Optimizer")
st.sidebar.markdown("**Démo d'optimisation urbaine**")

st.sidebar.markdown("### Ville")
city = st.sidebar.text_input("Place OSM", value="Villeurbanne, France")
include_route500 = st.sidebar.checkbox(
    "Inclure ROUTE500 (interurbain, requiert data/raw)", value=False,
)
simplified_highway = st.sidebar.checkbox(
    "Réseau simplifié (primary/secondary uniquement)",
    value=False,
    help="Garde uniquement les axes primary, secondary, trunk et motorway (~30 % des arcs). "
         "Dijkstra 3-4× plus rapide. Peut manquer des rues locales dans l'OD.",
)

st.sidebar.markdown("### Profil du décideur")
profile_label = st.sidebar.selectbox(
    "Profil maire",
    options=[p.label for p in ALL_PROFILES],
    index=0,
    help="Le profil amplifie certaines composantes du coût social. "
         "Un écolo pèsera fort le CO2 ; un maire mobilité priorisera le gain de temps.",
)
profile = next(p for p in ALL_PROFILES if p.label == profile_label)

st.sidebar.markdown("### Demande")
hour = st.sidebar.slider("Heure de pointe", 0, 23, 8)
n_cells = st.sidebar.slider("Cellules de zonage", 4, 20, 10)
scale_factor = st.sidebar.slider("Échelle de demande", 0.1, 2.0, 0.3, step=0.1)

st.sidebar.markdown("### Solveur Frank-Wolfe")
max_iter = st.sidebar.slider("Itérations UE", 30, 300, 100, step=10)

st.sidebar.markdown("### Plan urbain")
budget_meur = st.sidebar.slider("Budget construction (M€)", 5, 500, 50, step=5)
budget_eur = budget_meur * 1_000_000
max_candidates = st.sidebar.slider("Candidats à explorer", 10, 100, 30)
max_fw_evals = st.sidebar.slider("Évaluations Frank-Wolfe", 5, 25, 10)
include_braess = st.sidebar.checkbox(
    "Aussi suggérer des suppressions (Braess)",
    value=True,
    help="Quelques arcs existants dont le retrait améliorerait le réseau.",
)
periphery_margin_m = st.sidebar.slider(
    "Marge périphérie nouvelles routes (m)", 0, 2000, 600, step=100,
    help="Distance d'érosion du convex hull : les nouvelles routes ne sont proposées "
         "que si au moins un nœud OD est à l'extérieur de ce noyau. "
         "0 = pas de filtre (tout le territoire). 1000+ = périphérie uniquement.",
)
access_threshold_min = st.sidebar.slider(
    "Seuil isochrone accessibilité (min)", 5, 45, 15, step=5,
    help="Pour chaque zone, on compte les autres zones joignables sous ce seuil "
         "(en utilisant les temps de trajet congestionnés). Sert au calcul "
         "d'accessibilité moyenne + indice d'équité (Gini).",
)
include_robustness = st.sidebar.checkbox(
    "Test de robustesse (×0.8/×1.0/×1.2/×1.5 demande)",
    value=False,
    help="Re-lance le FW avec et sans plan pour 4 niveaux de demande. "
         "Permet de juger si le plan reste bénéfique en cas de croissance "
         "démographique ou de baisse imprévue. Coût : +8 FW (~2× le temps de calcul).",
)
include_pareto = st.sidebar.checkbox(
    "Courbe Pareto budget → bénéfice",
    value=False,
    help="Évalue le plan pour 6 niveaux de budget (5/15/30/60/120/250 M€) "
         "et trace la courbe des rendements marginaux. "
         "Coût : ~max_fw_evals + 6 FW supplémentaires.",
)

run_btn = st.sidebar.button("▶ Lancer le pipeline", type="primary", use_container_width=True)


# ────────────────────────────────────────────────────────────────────────────
# En-tête
# ────────────────────────────────────────────────────────────────────────────

st.title("Plan urbain optimisé")
st.caption(
    "Construit le réseau, génère la demande, calcule l'équilibre, score la ville selon "
    "le profil maire, puis propose un plan de construction sous budget."
)


# ────────────────────────────────────────────────────────────────────────────
# Pipeline
# ────────────────────────────────────────────────────────────────────────────

if run_btn or st.session_state.get("ran_once"):
    st.session_state["ran_once"] = True

    with st.spinner(f"Construction du réseau « {city} »…"):
        net = cached_network(city, include_route500, simplified_highway)
    with st.spinner("Génération de la matrice OD gravitaire…"):
        od = cached_od(city, hour, n_cells, scale_factor, include_route500, simplified_highway)
    with st.spinner(f"Affectation Frank-Wolfe ({max_iter} itérations)…"):
        ue = cached_ue(city, hour, n_cells, scale_factor, include_route500, simplified_highway, max_iter)

    baseline_access = compute_accessibility(
        net, od, ue, threshold_seconds=float(access_threshold_min) * 60.0,
    )
    baseline_score = score_network(ue, profile, access=baseline_access)

    # ── Score actuel ──────────────────────────────────────────────────────
    st.markdown("### Score actuel de la ville")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric(
        "Coût annuel total",
        f"{baseline_score.total_annual_cost_eur / 1e6:.1f} M€",
        help=f"Pondéré par le profil {profile.name} "
             f"(temps + carburant + CO2 − bénéfice accessibilité + pénalité Gini)",
    )
    k2.metric("Coût temps", f"{baseline_score.annual_time_cost_eur / 1e6:.1f} M€")
    k3.metric("Coût carburant", f"{baseline_score.annual_fuel_cost_eur / 1e6:.1f} M€")
    k4.metric(
        "Émissions CO2",
        f"{baseline_score.annual_co2_kg / 1000:.0f} t/an",
        delta=f"{baseline_score.annual_co2_cost_eur / 1e3:.0f} k€ externalité",
        delta_color="off",
    )

    a1, a2, a3 = st.columns(3)
    a1.metric(
        f"Accessibilité (≤ {access_threshold_min} min)",
        f"{baseline_access.mean_reachable:.1f} zones",
        help=f"Nombre moyen de zones joignables sous {access_threshold_min} minutes "
             "à l'heure de pointe, depuis chaque zone OD.",
    )
    a2.metric(
        "Indice d'équité (Gini)",
        f"{baseline_access.gini:.3f}",
        delta="0 = égalité parfaite, 1 = inégalité totale",
        delta_color="off",
        help="Coefficient de Gini sur l'accessibilité par zone — mesure si certaines "
             "zones sont desservies de manière très inégale.",
    )
    a3.metric(
        "Pénalité Gini intégrée au score",
        f"{baseline_score.equity_penalty_eur / 1e6:.2f} M€/an",
        delta=f"bénéfice accès = −{baseline_score.accessibility_value_eur / 1e6:.2f} M€/an",
        delta_color="off",
    )

    # ── Plan urbain : nouvelles routes ────────────────────────────────────
    st.markdown(f"### Plan urbain proposé — profil {profile.label}")
    with st.spinner(f"Génération du plan urbain (jusqu'à {max_fw_evals} arcs évalués)…"):
        t = time.time()
        obstacle_index = cached_obstacles(city)
        soft_index = cached_bridge_triggers(city)
        plan, baseline_enriched, joint = propose_urban_plan(
            net, od, profile, ue,
            budget_eur=budget_eur,
            max_proposals=max_candidates,
            max_fw_evals=max_fw_evals,
            fw_max_iter=25, fw_tol=5e-3,
            obstacle_index=obstacle_index,
            soft_index=soft_index,
            periphery_margin_m=float(periphery_margin_m),
            accessibility_threshold_s=float(access_threshold_min) * 60.0,
            _baseline_access=baseline_access,
        )
        t_plan = time.time() - t
    st.caption(f"Plan généré en {t_plan:.1f}s")

    if plan:
        rows = []
        for i, ev in enumerate(plan, start=1):
            p = ev.proposal
            rows.append({
                "rang": i,
                "action": {"corridor": "élargissement", "upgrade": "mise à niveau",
                           "new_route": "nouvelle route"}.get(p.proposal_type, p.proposal_type),
                "type": p.highway,
                "longueur": f"{p.length_m:.0f} m",
                "détour avant": f"{p.detour_before:.2f}×",
                "ΔVHT": f"{ev.delta_vht_h:+.1f} h/pointe",
                "bénéfice annuel": f"{ev.annual_benefit_eur:,.0f} €",
                "construction": f"{p.construction_cost_eur:,.0f} €",
                "payback": (f"{ev.payback_years:.1f} ans"
                            if np.isfinite(ev.payback_years) else "∞"),
                "BCR": f"{ev.bcr:.2f}",
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        total_cost = sum(ev.proposal.construction_cost_eur for ev in plan)
        naive_benefit = sum(ev.annual_benefit_eur for ev in plan)

        # Score effectif issu de la re-éval jointe (FW unique avec tout appliqué)
        if joint is not None:
            joint_score_eur = baseline_score.total_annual_cost_eur - joint.joint_annual_benefit_eur
            real_benefit = joint.joint_annual_benefit_eur
            real_payback = total_cost / real_benefit if real_benefit > 0 else float("inf")
        else:
            joint_score_eur = baseline_score.total_annual_cost_eur
            real_benefit = naive_benefit
            real_payback = total_cost / max(naive_benefit, 1)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Constructions retenues", f"{len(plan)}")
        c2.metric("Coût total CAPEX", f"{total_cost / 1e6:.1f} M€",
                  delta=f"sur {budget_meur} M€ de budget", delta_color="off")
        c3.metric(
            "Score après plan (joint)",
            f"{joint_score_eur / 1e6:.1f} M€",
            delta=f"−{real_benefit / 1e6:.2f} M€/an",
            delta_color="inverse",
        )
        c4.metric(
            "Payback combiné",
            f"{real_payback:.1f} ans" if np.isfinite(real_payback) else "∞",
        )

        # ── Bloc re-évaluation jointe ─────────────────────────────────────
        if joint is not None:
            redundancy_pct = (1.0 - joint.redundancy_factor) * 100
            st.markdown("#### Re-évaluation jointe du plan")
            j1, j2, j3, j4 = st.columns(4)
            j1.metric(
                "ΔVHT joint (réel)",
                f"−{joint.joint_delta_vht_h:.1f} h",
                help="Frank-Wolfe unique avec toutes les interventions actives. "
                     "C'est le gain RÉEL du plan, à comparer avec la somme naïve.",
            )
            j2.metric(
                "ΔVHT somme naïve",
                f"−{joint.naive_sum_delta_vht_h:.1f} h",
                help="Somme des ΔVHT évalués individuellement (overestimation).",
            )
            j3.metric(
                "Redondance",
                f"{redundancy_pct:.0f} %",
                delta="part du gain individuel cannibalisée par l'effet conjoint",
                delta_color="off",
                help="0% = aucune redondance, les interventions sont totalement complémentaires. "
                     "30% = un tiers du bénéfice individuel disparaît lorsqu'on combine les plans.",
            )
            j4.metric(
                "BCR joint",
                f"{joint.joint_bcr:.2f}",
                help="Bénéfice annuel joint ÷ CAPEX cumulé.",
            )
            if redundancy_pct > 25:
                st.warning(
                    f"⚠️ Redondance élevée ({redundancy_pct:.0f}%) : plusieurs interventions "
                    f"se cannibalisent. Envisager d'en retirer pour libérer du budget."
                )

            # Comparaison accessibilité / équité avant ↔ après
            st.markdown("#### Accessibilité et équité — avant ↔ après plan")
            e1, e2, e3 = st.columns(3)
            d_access = joint.accessibility_after - joint.accessibility_before
            d_gini = joint.gini_after - joint.gini_before
            e1.metric(
                f"Zones joignables ≤ {access_threshold_min} min",
                f"{joint.accessibility_after:.1f}",
                delta=f"{d_access:+.2f} vs {joint.accessibility_before:.1f}",
                delta_color="normal",  # plus haut = mieux
                help="Moyenne sur toutes les zones OD. Le plan élargit-il le périmètre "
                     "joignable depuis chaque quartier ?",
            )
            e2.metric(
                "Indice d'équité (Gini)",
                f"{joint.gini_after:.3f}",
                delta=f"{d_gini:+.3f} vs {joint.gini_before:.3f}",
                delta_color="inverse",  # plus bas = mieux
                help="Coefficient de Gini sur l'accessibilité par zone : plus bas = "
                     "réseau plus équitable. Le plan profite-t-il aux zones les plus mal desservies ?",
            )
            if d_gini < -0.01:
                e3.success("✓ Plan plutôt égalitaire — l'équité s'améliore.")
            elif d_gini > 0.01:
                e3.warning("⚠ Plan inégalitaire — les zones déjà bien desservies gagnent plus.")
            else:
                e3.info("≈ Effet neutre sur l'équité.")

        # ── Test de robustesse (optionnel) ────────────────────────────────
        if include_robustness:
            st.markdown("#### Robustesse face aux variations de demande")
            with st.spinner("Re-évaluation du plan sous demande × {0.8, 1.0, 1.2, 1.5}…"):
                rob = evaluate_plan_robustness(
                    net, od, plan, profile,
                    demand_scales=(0.8, 1.0, 1.2, 1.5),
                    fw_max_iter=25, fw_tol=5e-3,
                )

            if rob.points:
                rob_rows = []
                for p in rob.points:
                    rob_rows.append({
                        "demande": f"×{p.demand_scale:.2f}",
                        "VHT baseline": f"{p.baseline_vht_h:,.0f} h",
                        "VHT plan": f"{p.plan_vht_h:,.0f} h",
                        "ΔVHT": f"{p.delta_vht_h:+,.1f} h",
                        "bénéfice annuel": f"{p.annual_benefit_eur:+,.0f} €",
                        "statut": "✓ rentable" if p.is_beneficial else "✗ déficitaire",
                    })
                st.dataframe(
                    pd.DataFrame(rob_rows),
                    use_container_width=True, hide_index=True,
                )

                # Synthèse verdict
                worst = rob.worst_point
                best = rob.best_point
                if rob.is_robust:
                    st.success(
                        f"✓ Plan **robuste** : reste bénéfique sur les 4 scénarios. "
                        f"Pire cas = demande ×{worst.demand_scale:.2f} "
                        f"(+{worst.annual_benefit_eur / 1e3:,.0f} k€/an)."
                    )
                else:
                    n_fail = sum(1 for p in rob.points if not p.is_beneficial)
                    st.error(
                        f"✗ Plan **fragile** : déficitaire dans {n_fail}/{len(rob.points)} scénario(s). "
                        f"Pire cas = demande ×{worst.demand_scale:.2f} "
                        f"({worst.annual_benefit_eur / 1e3:,.0f} k€/an)."
                    )
                st.caption(
                    f"Meilleur cas : demande ×{best.demand_scale:.2f} "
                    f"→ {best.annual_benefit_eur / 1e6:.2f} M€/an de bénéfice. "
                    f"Lecture : si la demande dévie de ±20%, le plan tient-il ? "
                    f"Si elle grimpe à +50% (forte croissance), reste-t-il efficace ?"
                )

        # ── Courbe Pareto budget → bénéfice (optionnel) ───────────────────
        if include_pareto:
            st.markdown("#### Courbe Pareto budget → bénéfice")
            with st.spinner("Évaluation Pareto sur 6 niveaux de budget…"):
                pareto = compute_pareto_frontier(
                    net, od, profile, ue,
                    max_proposals=max_candidates,
                    max_fw_evals=max_fw_evals,
                    fw_max_iter=25, fw_tol=5e-3,
                    obstacle_index=obstacle_index,
                    soft_index=soft_index,
                    periphery_margin_m=float(periphery_margin_m),
                    accessibility_threshold_s=float(access_threshold_min) * 60.0,
                )

            if pareto.points:
                pareto_rows = []
                for pt in pareto.points:
                    pareto_rows.append({
                        "budget (M€)": pt.budget_eur / 1e6,
                        "CAPEX utilisé (M€)": pt.capex_used_eur / 1e6,
                        "interventions": pt.n_interventions,
                        "bénéfice annuel (M€/an)": pt.joint_annual_benefit_eur / 1e6,
                        "ΔVHT (h)": pt.delta_vht_h,
                        "BCR": pt.bcr,
                        "efficience (€/an par M€)": pt.marginal_efficiency_eur_per_meur,
                    })
                df_pareto = pd.DataFrame(pareto_rows)

                # Graphique : CAPEX utilisé vs bénéfice annuel
                chart_df = df_pareto[["CAPEX utilisé (M€)", "bénéfice annuel (M€/an)"]].set_index(
                    "CAPEX utilisé (M€)"
                )
                st.line_chart(chart_df, height=300)

                # Tableau détaillé
                st.dataframe(
                    df_pareto.style.format({
                        "budget (M€)": "{:.0f}",
                        "CAPEX utilisé (M€)": "{:.1f}",
                        "bénéfice annuel (M€/an)": "{:+.2f}",
                        "ΔVHT (h)": "{:+.1f}",
                        "BCR": "{:.2f}",
                        "efficience (€/an par M€)": "{:,.0f}",
                    }),
                    use_container_width=True, hide_index=True,
                )

                sweet = pareto.best_marginal_point
                if sweet and sweet.capex_used_eur > 0:
                    st.success(
                        f"🎯 **Sweet spot** : budget ~{sweet.budget_eur / 1e6:.0f} M€ → "
                        f"{sweet.joint_annual_benefit_eur / 1e6:+.2f} M€/an de bénéfice "
                        f"({sweet.marginal_efficiency_eur_per_meur:,.0f} €/an par M€ investi). "
                        f"Au-delà, les rendements marginaux décroissent."
                    )
                st.caption(
                    "Lecture : courbe concave = rendements décroissants — chaque euro "
                    "supplémentaire rapporte moins. Le point d'inflexion est le budget "
                    "« optimal » en investissement public."
                )
    else:
        st.warning(
            "Aucune construction rentable identifiée — augmente la demande, "
            "le budget, ou le nombre de candidats."
        )

    # ── Optionnel : Braess (arcs à supprimer) ─────────────────────────────
    braess: list = []
    if include_braess and plan is not None:
        with st.spinner("Recherche d'arcs Braess (suppression bénéfique)…"):
            removal_cands = [
                c for c in generate_candidates(net, ue, top_n=15)
                if c.action == "remove"
            ]
            evals_rem = rank_candidates(net, od, removal_cands, ue,
                                        max_iter=50, tol=1e-3)
            braess = [e for e in evals_rem if e.is_braess][:3]

        if braess:
            st.markdown("### Arcs à supprimer (paradoxe de Braess)")
            rows = []
            for i, ev in enumerate(braess, start=1):
                c = ev.candidate
                rows.append({
                    "rang": i,
                    "edge_id": c.edge_id,
                    "type": c.highway,
                    "longueur": f"{c.length_m:.0f} m",
                    "ΔVHT en retirant": f"{ev.delta_vht_h:+.1f} h",
                    "économie annuelle": f"{ev.annual_benefit_eur:,.0f} €",
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # ── Carte avant/après ─────────────────────────────────────────────────
    st.markdown("### Carte du plan urbain")
    edges_wgs, edges_coords = cached_edges_wgs_and_coords(city, include_route500, simplified_highway)

    capacity = np.asarray(net.graph.es["capacity"], dtype=float)
    sat_before = ue.flows / np.maximum(capacity, 1.0)
    sat_after = (
        joint.existing_sat_after
        if (joint is not None and joint.existing_sat_after.size == len(capacity))
        else sat_before
    )

    # Sample partagé entre cartes avant/après pour cohérence visuelle.
    # Stratégie : on garde TOUS les arcs saturés (≥ 0.5) et on échantillonne
    # les arcs fluides pour la base visuelle. Ça donne une carte aussi lisible
    # avec ~2× moins de polylines à dessiner.
    n_edges = len(edges_coords)
    sat_union = np.maximum(sat_before, sat_after) if sat_after is not sat_before else sat_before
    important = np.where(sat_union >= 0.5)[0]
    rest = np.where(sat_union < 0.5)[0]
    sample_n_rest = min(1500, len(rest))
    rng = np.random.default_rng(0)
    if len(rest) > sample_n_rest:
        rest_sample = rng.choice(rest, size=sample_n_rest, replace=False)
    else:
        rest_sample = rest
    sample_idx = np.concatenate([important, rest_sample])

    # Reprojection helper
    import geopandas as gpd
    from shapely.geometry import LineString as _LineString

    def to_wgs(coords_lambert):
        gs = gpd.GeoSeries([_LineString(coords_lambert)], crs=net.crs)
        return gs.to_crs(CRS_WGS84).iloc[0]

    def sat_style(s):
        """Couleur, épaisseur, opacité d'un arc selon sa saturation v/c."""
        if s < 0.5:
            return "#9ec5a5", 1, 0.35      # vert pâle — fluide
        if s < 0.8:
            return "#ffbf00", 2, 0.70      # ambre — chargé
        if s < 0.95:
            return "#ff6a00", 3, 0.85      # orange — saturé
        return "#c92020", 4, 0.95          # rouge — bouchon

    _PLAN_COLORS = {"corridor": "#1f6feb", "upgrade": "#d63384", "new_route": "#1a9b4b"}
    _PLAN_LABELS = {"corridor": "ÉLARGISSEMENT", "upgrade": "MISE À NIVEAU", "new_route": "NOUVELLE ROUTE"}

    def build_map(sat_array, *, with_plan_overlay: bool):
        center = edges_wgs.geometry.union_all().centroid
        m = folium.Map(location=[center.y, center.x], zoom_start=13, tiles="CartoDB positron")

        # Arcs colorés par saturation — coords pré-calculées (pas de pandas .iloc[])
        for i in sample_idx:
            coords = edges_coords[int(i)]
            if coords is None:
                continue
            s_val = float(sat_array[i])
            color, weight, opacity = sat_style(s_val)
            folium.PolyLine(
                coords, color=color, weight=weight, opacity=opacity,
                tooltip=f"v/c = {s_val:.2f}" if s_val > 0.5 else None,
            ).add_to(m)

        # Overlay du plan (toujours visible pour situer les interventions)
        for idx, ev in enumerate(plan, start=1):
            p = ev.proposal
            color = _PLAN_COLORS.get(p.proposal_type, "#888888")
            label = _PLAN_LABELS.get(p.proposal_type, p.proposal_type)
            raw_coords = p.corridor_xy if len(p.corridor_xy) >= 2 else [p.u_xy, p.v_xy]
            ll = to_wgs(raw_coords)
            map_coords = [(y, x) for x, y in ll.coords]
            # En mode "avant", on grise l'overlay pour montrer "voici où on VA agir"
            ov_opacity = 0.95 if with_plan_overlay else 0.55
            ov_weight = 6 if with_plan_overlay else 4
            folium.PolyLine(
                map_coords, color=color, weight=ov_weight, opacity=ov_opacity,
                tooltip=(f"{label} #{idx} • {p.highway} {p.length_m:.0f}m • "
                         f"coût {p.construction_cost_eur:,.0f}€ • "
                         f"bénéf {ev.annual_benefit_eur:+,.0f}€/an"),
            ).add_to(m)
            mid = ll.interpolate(0.5, normalized=True)
            folium.CircleMarker(
                [mid.y, mid.x],
                radius=11, color=color, fill=True, fill_opacity=ov_opacity,
            ).add_to(m)
            folium.map.Marker(
                [mid.y, mid.x],
                icon=folium.DivIcon(
                    icon_size=(24, 24), icon_anchor=(7, 11),
                    html=f'<div style="font-size:13px;color:white;'
                         f'font-weight:bold;text-align:center;">{idx}</div>',
                ),
            ).add_to(m)

        # Braess — uniquement sur la carte "après" (suppressions effectivement faites)
        if with_plan_overlay:
            for idx, ev in enumerate(braess, start=1):
                eid = int(ev.candidate.edge_id)
                coords = edges_coords[eid] if eid < len(edges_coords) else None
                if coords is None:
                    continue
                folium.PolyLine(
                    coords, color="#d22b2b", weight=6, opacity=0.95, dash_array="10",
                    tooltip=(f"SUPPRESSION #{idx} • {ev.candidate.highway} "
                             f"{ev.candidate.length_m:.0f}m"),
                ).add_to(m)

        return m

    # Légende
    st.markdown(
        f"""
        <div style="background:#fafafa; padding:10px 14px; border:1px solid #ddd;
                    border-radius:6px; font-size:13px; line-height:1.9;
                    color:#111; margin-bottom:8px;">
            <b>Plan urbain — {profile.label}</b><br>
            <b>Saturation v/c :</b>
            <span style="color:#9ec5a5;">━</span> fluide (&lt;0.5) &nbsp;·&nbsp;
            <span style="color:#ffbf00;">━</span> chargé (0.5–0.8) &nbsp;·&nbsp;
            <span style="color:#ff6a00;">━</span> saturé (0.8–0.95) &nbsp;·&nbsp;
            <span style="color:#c92020;">━</span> bouchon (≥0.95)<br>
            <b>Plan :</b>
            <span style="color:#1f6feb;font-weight:bold;">━ ÉLARGISSEMENT</span> &nbsp;·&nbsp;
            <span style="color:#d63384;font-weight:bold;">━ MISE À NIVEAU</span> &nbsp;·&nbsp;
            <span style="color:#1a9b4b;font-weight:bold;">━ NOUVELLE ROUTE</span> &nbsp;·&nbsp;
            <span style="color:#d22b2b;font-weight:bold;">┄ À SUPPRIMER</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    view_mode = st.radio(
        "Mode d'affichage",
        options=["Avant / Après côte à côte", "Avant le plan", "Après le plan"],
        index=0, horizontal=True, label_visibility="collapsed",
    )

    if view_mode == "Avant / Après côte à côte":
        col_a, col_b = st.columns(2)
        with col_a:
            st.caption(f"**Avant** — VHT = {ue.vht:,.0f} h")
            st_folium(
                build_map(sat_before, with_plan_overlay=False),
                width=None, height=560, returned_objects=[], key="map_before",
            )
        with col_b:
            joint_vht_disp = joint.joint_vht_h if joint else ue.vht
            st.caption(f"**Après** — VHT = {joint_vht_disp:,.0f} h")
            st_folium(
                build_map(sat_after, with_plan_overlay=True),
                width=None, height=560, returned_objects=[], key="map_after",
            )
    elif view_mode == "Avant le plan":
        st.caption(f"VHT = {ue.vht:,.0f} h")
        st_folium(
            build_map(sat_before, with_plan_overlay=False),
            width=None, height=640, returned_objects=[], key="map_before_solo",
        )
    else:
        joint_vht_disp = joint.joint_vht_h if joint else ue.vht
        st.caption(f"VHT = {joint_vht_disp:,.0f} h")
        st_folium(
            build_map(sat_after, with_plan_overlay=True),
            width=None, height=640, returned_objects=[], key="map_after_solo",
        )

    # ── Pied de page ──────────────────────────────────────────────────────
    diag = diagnose(net, ue)
    with st.expander("État technique du réseau (UE Frank-Wolfe)"):
        st.text(
            f"Réseau : {net.graph.vcount():,} nœuds, {net.graph.ecount():,} arcs\n"
            f"Demande : {od.n_zones} zones, {od.total_trips:,.0f} véh/h\n"
            f"VHT (h pointe) : {diag.vht:,.1f}\n"
            f"Surcoût congestion : +{diag.congestion_overhead * 100:.1f}%\n"
            f"Arcs saturés (v/c≥0.9) : {diag.n_saturated} / {diag.n_arcs}\n"
            f"FW gap : {ue.final_gap:.1e} (iter {ue.iterations})\n"
        )

else:
    st.info("Configure les paramètres dans la barre latérale puis clique sur **Lancer le pipeline**.")
    st.markdown(
        """
        **Comment lire les résultats** :
        - **Score actuel** : coût social total du réseau actuel, exprimé en €/an et **pondéré
          selon le profil du maire** (un maire écolo pèse fort le CO2, etc.).
        - **Plan urbain** : nouvelles routes proposées (lignes droites entre nœuds existants),
          coût de construction, gain annuel, payback. Sélection gloutonne sous budget par BCR.
        - **Carte** : 🟦 **bleu** = à construire, 🟥 **rouge tireté** = à supprimer (Braess).

        Commence par **Villeurbanne, France** + 100 itérations + 10 évaluations FW
        (~1-2 min). Pour Lyon : compter 5-10 min.
        """
    )
