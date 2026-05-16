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
from urban_optimizer.diagnosis import diagnose
from urban_optimizer.network import build_network, load_buildings
from urban_optimizer.optimization import (
    ALL_PROFILES,
    PROFILE_BY_NAME,
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
def cached_network(city: str, include_route500: bool):
    return build_network(city, include_route500=include_route500)


@st.cache_resource(show_spinner=False)
def cached_buildings(city: str):
    return load_buildings(city)


@st.cache_data(show_spinner=False)
def cached_od(city: str, hour: int, n_cells: int, scale_factor: float, include_route500: bool):
    net = cached_network(city, include_route500)
    return generate_od_matrix(
        net, hour=hour, method="grid", n_cells=n_cells, scale_factor=scale_factor,
    )


@st.cache_data(show_spinner=False)
def cached_ue(city: str, hour: int, n_cells: int, scale_factor: float,
              include_route500: bool, max_iter: int):
    net = cached_network(city, include_route500)
    od = cached_od(city, hour, n_cells, scale_factor, include_route500)
    return solve_user_equilibrium(net, od, max_iter=max_iter, tol=1e-4)


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
max_iter = st.sidebar.slider("Itérations UE", 20, 200, 60, step=10)

st.sidebar.markdown("### Plan urbain")
budget_meur = st.sidebar.slider("Budget construction (M€)", 5, 500, 50, step=5)
budget_eur = budget_meur * 1_000_000
max_candidates = st.sidebar.slider("Candidats à explorer", 5, 60, 20)
max_fw_evals = st.sidebar.slider(
    "Évaluations Frank-Wolfe",
    3, 15, 8,
    help="Réduit pour accélérer : chaque évaluation relance un FW complet.",
)
include_braess = st.sidebar.checkbox(
    "Aussi suggérer des suppressions (Braess)",
    value=True,
    help="Quelques arcs existants dont le retrait améliorerait le réseau.",
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
        net = cached_network(city, include_route500)
    with st.spinner("Génération de la matrice OD gravitaire…"):
        od = cached_od(city, hour, n_cells, scale_factor, include_route500)
    with st.spinner(f"Affectation Frank-Wolfe ({max_iter} itérations)…"):
        ue = cached_ue(city, hour, n_cells, scale_factor, include_route500, max_iter)

    baseline_score = score_network(ue, profile)

    # ── Score actuel ──────────────────────────────────────────────────────
    st.markdown("### Score actuel de la ville")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric(
        "Coût annuel total",
        f"{baseline_score.total_annual_cost_eur / 1e6:.1f} M€",
        help=f"Pondéré par le profil {profile.name}",
    )
    k2.metric("Coût temps", f"{baseline_score.annual_time_cost_eur / 1e6:.1f} M€")
    k3.metric("Coût carburant", f"{baseline_score.annual_fuel_cost_eur / 1e6:.1f} M€")
    k4.metric(
        "Émissions CO2",
        f"{baseline_score.annual_co2_kg / 1000:.0f} t/an",
        delta=f"{baseline_score.annual_co2_cost_eur / 1e3:.0f} k€ externalité",
        delta_color="off",
    )

    # ── Plan urbain : nouvelles routes ────────────────────────────────────
    st.markdown(f"### Plan urbain proposé — profil {profile.label}")
    with st.spinner(f"Génération du plan urbain (jusqu'à {max_fw_evals} arcs évalués)…"):
        t = time.time()
        building_index = cached_buildings(city)
        plan, _ = propose_urban_plan(
            net, od, profile, ue,
            budget_eur=budget_eur,
            max_proposals=max_candidates,
            max_fw_evals=max_fw_evals,
            fw_max_iter=25, fw_tol=5e-3,
            building_index=building_index,
        )
        t_plan = time.time() - t
    st.caption(f"Plan généré en {t_plan:.1f}s")

    if plan:
        rows = []
        for i, ev in enumerate(plan, start=1):
            p = ev.proposal
            rows.append({
                "rang": i,
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
        total_benefit = sum(ev.annual_benefit_eur for ev in plan)

        new_vht = ue.vht - sum(ev.delta_vht_h for ev in plan)
        # Score projeté approximatif : on conserve les ratios baseline et on rabote
        # proportionnellement au VHT — c'est une estimation, l'équilibre exact
        # nécessite de recalculer le FW avec tous les arcs ajoutés simultanément.
        projected_share = max(0.0, new_vht / max(ue.vht, 1e-9))
        projected_score = baseline_score.total_annual_cost_eur * projected_share

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Constructions retenues", f"{len(plan)}")
        c2.metric("Coût total CAPEX", f"{total_cost / 1e6:.1f} M€",
                  delta=f"sur {budget_meur} M€ de budget", delta_color="off")
        c3.metric(
            "Score projeté",
            f"{projected_score / 1e6:.1f} M€",
            delta=f"−{(baseline_score.total_annual_cost_eur - projected_score) / 1e6:.2f} M€/an",
            delta_color="inverse",
        )
        c4.metric(
            "Payback combiné",
            f"{total_cost / max(total_benefit, 1):.1f} ans"
            if total_benefit > 0 else "∞",
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

    # ── Carte ─────────────────────────────────────────────────────────────
    st.markdown("### Carte du plan urbain")
    edges_wgs = net.edges_gdf.to_crs(CRS_WGS84)
    nodes_xy = net.nodes_xy

    capacity = np.asarray(net.graph.es["capacity"], dtype=float)
    sat = ue.flows / np.maximum(capacity, 1.0)

    center = edges_wgs.geometry.union_all().centroid
    m = folium.Map(location=[center.y, center.x], zoom_start=13, tiles="CartoDB positron")

    # Réseau existant — gris
    for _, row in edges_wgs.sample(min(2500, len(edges_wgs)), random_state=0).iterrows():
        if row.geometry is None or row.geometry.geom_type != "LineString":
            continue
        coords = [(y, x) for x, y in row.geometry.coords]
        folium.PolyLine(coords, color="#bbbbbb", weight=1, opacity=0.5).add_to(m)

    # Arcs très saturés — orange pâle
    top_sat = np.argsort(-sat)[:60]
    for eid in top_sat:
        if sat[eid] < 0.8:
            continue
        geom = edges_wgs.iloc[int(eid)].geometry
        if geom is None or geom.geom_type != "LineString":
            continue
        coords = [(y, x) for x, y in geom.coords]
        folium.PolyLine(
            coords, color="#ff8c00", weight=3, opacity=0.5,
            tooltip=f"saturé v/c={sat[eid]:.2f}",
        ).add_to(m)

    # Reprojection vers WGS84 pour les nouveaux arcs (lignes droites)
    import geopandas as gpd
    from shapely.geometry import LineString as _LineString

    def to_wgs(xy_pair):
        gs = gpd.GeoSeries([_LineString([xy_pair[0], xy_pair[1]])], crs=net.crs)
        return gs.to_crs(CRS_WGS84).iloc[0]

    # 🟦 NOUVELLES ROUTES (top du plan)
    for i, ev in enumerate(plan, start=1):
        p = ev.proposal
        ll = to_wgs((p.u_xy, p.v_xy))
        coords = [(y, x) for x, y in ll.coords]
        folium.PolyLine(
            coords, color="#1f6feb", weight=6, opacity=0.95,
            tooltip=(f"NOUVELLE ROUTE #{i} • {p.highway} {p.length_m:.0f}m • "
                     f"coût {p.construction_cost_eur:,.0f}€ • "
                     f"bénéf {ev.annual_benefit_eur:+,.0f}€/an"),
        ).add_to(m)

        midpoint = ll.interpolate(0.5, normalized=True)
        folium.CircleMarker(
            [midpoint.y, midpoint.x],
            radius=11, color="#1f6feb", fill=True, fill_opacity=0.95,
        ).add_to(m)
        folium.map.Marker(
            [midpoint.y, midpoint.x],
            icon=folium.DivIcon(
                icon_size=(24, 24), icon_anchor=(7, 11),
                html=f'<div style="font-size:13px;color:white;'
                     f'font-weight:bold;text-align:center;">{i}</div>',
            ),
        ).add_to(m)

    # 🟥 SUPPRESSIONS BRAESS
    for i, ev in enumerate(braess, start=1):
        eid = ev.candidate.edge_id
        geom = edges_wgs.iloc[int(eid)].geometry
        if geom is None or geom.geom_type != "LineString":
            continue
        coords = [(y, x) for x, y in geom.coords]
        folium.PolyLine(
            coords, color="#d22b2b", weight=6, opacity=0.95, dash_array="10",
            tooltip=(f"SUPPRESSION #{i} • {ev.candidate.highway} "
                     f"{ev.candidate.length_m:.0f}m • "
                     f"bénéf {ev.annual_benefit_eur:+,.0f}€/an"),
        ).add_to(m)

    # Légende
    legend_html = f"""
    <div style="position:fixed; bottom:20px; left:20px; z-index:9999;
                background:white; padding:10px 14px; border:1px solid #444;
                border-radius:6px; font-size:13px; line-height:1.6;">
        <b>Plan urbain — {profile.label}</b><br>
        <span style="color:#bbbbbb">━</span> réseau existant<br>
        <span style="color:#ff8c00">━</span> arcs très saturés<br>
        <span style="color:#1f6feb;font-weight:bold">━ NOUVELLE ROUTE</span><br>
        <span style="color:#d22b2b;font-weight:bold">┄ À SUPPRIMER (Braess)</span>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    st_folium(m, width=None, height=640, returned_objects=[])

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
