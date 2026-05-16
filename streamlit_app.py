"""Démo interactive Urban Optimizer.

Pipeline complet sur une ville quelconque :
1. Construction du réseau (OSM, ROUTE500 optionnel)
2. Génération de la demande OD gravitaire
3. Affectation à l'équilibre (UE Frank-Wolfe) + System Optimum
4. Diagnostic : VHT, saturation, prix de l'anarchie
5. Top-3 recommandations sous budget : améliorations en BLEU, retraits Braess en ROUGE.

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
from urban_optimizer.diagnosis import diagnose, rank_by_congestion_delay
from urban_optimizer.network import build_network
from urban_optimizer.optimization import (
    generate_candidates,
    rank_candidates,
    select_under_budget,
)

logging.getLogger("urban_optimizer").setLevel(logging.WARNING)

st.set_page_config(page_title="Urban Optimizer", layout="wide", page_icon="🛣️")


# ────────────────────────────────────────────────────────────────────────────
# Caches pour ne pas recalculer entre les interactions
# ────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def _cached_network(city: str, include_route500: bool):
    return build_network(city, include_route500=include_route500)


@st.cache_data(show_spinner=False)
def _cached_od(city: str, hour: int, n_cells: int, scale_factor: float, _net_id: int):
    # _net_id est un sentinel pour invalider quand le réseau change
    net = _cached_network(*st.session_state["_net_args"])
    od = generate_od_matrix(
        net, hour=hour, method="grid", n_cells=n_cells, scale_factor=scale_factor,
    )
    return od


@st.cache_data(show_spinner=False)
def _cached_assignment(_net_id: int, _od_id: int, max_iter: int):
    net = _cached_network(*st.session_state["_net_args"])
    od = st.session_state["od"]
    ue = solve_user_equilibrium(net, od, max_iter=max_iter, tol=1e-4)
    so = solve_system_optimum(net, od, max_iter=max_iter, tol=1e-4)
    return ue, so


# ────────────────────────────────────────────────────────────────────────────
# Sidebar — paramètres
# ────────────────────────────────────────────────────────────────────────────

st.sidebar.title("🛣️ Urban Optimizer")
st.sidebar.markdown("**Démo : optimisation des réseaux routiers urbains**")

st.sidebar.markdown("### Ville")
city = st.sidebar.text_input("Place OSM", value="Villeurbanne, France")
include_route500 = st.sidebar.checkbox(
    "Inclure ROUTE500 (liaisons interurbaines, requiert data/raw)", value=False,
)

st.sidebar.markdown("### Demande")
hour = st.sidebar.slider("Heure", 0, 23, 8)
n_cells = st.sidebar.slider("Cellules de zonage (grille n×n)", 4, 20, 10)
scale_factor = st.sidebar.slider("Échelle de demande", 0.1, 2.0, 0.3, step=0.1)

st.sidebar.markdown("### Affectation Frank-Wolfe")
max_iter = st.sidebar.slider("Itérations max", 30, 300, 100, step=10)

st.sidebar.markdown("### Optimisation")
top_n_arcs = st.sidebar.slider("Top arcs candidats", 5, 50, 15)
budget_meur = st.sidebar.slider("Budget total (M€)", 1, 100, 10)
budget_eur = budget_meur * 1_000_000

run_btn = st.sidebar.button("▶ Lancer le pipeline", type="primary", use_container_width=True)


# ────────────────────────────────────────────────────────────────────────────
# En-tête
# ────────────────────────────────────────────────────────────────────────────

st.title("Optimisation du réseau routier urbain")
st.caption(
    "Affectation à l'équilibre (Wardrop) + System Optimum + diagnostic + "
    "ranking marginal sous contrainte budgétaire."
)


# ────────────────────────────────────────────────────────────────────────────
# Bouton principal
# ────────────────────────────────────────────────────────────────────────────

if run_btn or "ue" in st.session_state:
    st.session_state["_net_args"] = (city, include_route500)
    net_id = hash((city, include_route500))
    od_id = hash((city, hour, n_cells, scale_factor))

    with st.spinner(f"Construction du réseau « {city} »…"):
        net = _cached_network(city, include_route500)

    with st.spinner("Génération de la matrice OD gravitaire…"):
        od = _cached_od(city, hour, n_cells, scale_factor, net_id)
        st.session_state["od"] = od

    with st.spinner(f"Affectation Frank-Wolfe ({max_iter} itérations)…"):
        ue, so = _cached_assignment(net_id, od_id, max_iter)
        st.session_state["ue"] = ue
        st.session_state["so"] = so

    diag = diagnose(net, ue, so_result=so, saturation_threshold=0.9)

    # ── Bandeau de KPI ────────────────────────────────────────────────────
    st.markdown("### État actuel du réseau (User Equilibrium)")
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Nœuds / Arcs", f"{net.graph.vcount():,} / {net.graph.ecount():,}")
    k2.metric("VHT total (h)", f"{diag.vht:,.0f}")
    k3.metric("Surcoût congestion", f"+{diag.congestion_overhead * 100:.1f}%")
    k4.metric("Arcs saturés (v/c≥0.9)", f"{diag.n_saturated} / {diag.n_arcs}")
    k5.metric("Prix de l'anarchie", f"×{diag.price_of_anarchy:.3f}"
              if diag.price_of_anarchy else "n/a")

    # ── Top arcs critiques ────────────────────────────────────────────────
    st.markdown("### Top 10 arcs critiques (où se perd le temps)")
    top_delay = rank_by_congestion_delay(net, ue, top_n=10)
    st.dataframe(
        top_delay[[
            "edge_id", "highway", "flow", "t_actual_s",
            "delay_per_user_s", "total_delay_h", "share_of_total_delay",
        ]].rename(columns={
            "flow": "flux (véh/h)",
            "t_actual_s": "t actuel (s)",
            "delay_per_user_s": "délai/usager (s)",
            "total_delay_h": "délai total (h)",
            "share_of_total_delay": "part du délai global",
        }),
        use_container_width=True,
        hide_index=True,
    )

    # ── Optimisation : ranking + sélection sous budget ────────────────────
    st.markdown("### Recommandations d'amélioration")
    with st.spinner(f"Évaluation marginale de {top_n_arcs * 4} candidats…"):
        t_opt = time.time()
        candidates = generate_candidates(net, ue, top_n=top_n_arcs)
        evals = rank_candidates(net, od, candidates, ue, max_iter=60, tol=1e-3)
        chosen = select_under_budget(evals, budget_eur=budget_eur)
        t_opt = time.time() - t_opt
    st.caption(f"Évaluation : {len(candidates)} candidats en {t_opt:.1f}s.")

    # Tableau du top 3
    rows = []
    for i, ev in enumerate(chosen[:3], start=1):
        rows.append({
            "rang": i,
            "edge_id": ev.candidate.edge_id,
            "action": ev.candidate.label,
            "type d'arc": ev.candidate.highway,
            "longueur (m)": int(ev.candidate.length_m),
            "ΔVHT (h)": round(ev.delta_vht_h, 1),
            "ΔVHT (%)": f"{ev.delta_vht_share * 100:.2f}%",
            "bénéfice annuel (€)": int(ev.annual_benefit_eur),
            "coût (€)": int(ev.cost_eur),
            "BCR (annuel)": round(ev.bcr, 2),
        })
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        total_gain = sum(ev.delta_vht_h for ev in chosen[:3])
        total_cost = sum(ev.cost_eur for ev in chosen[:3])
        st.success(
            f"**Top 3 sélectionné** : gain combiné ≈ {total_gain:.1f} h/h-de-pointe "
            f"({total_gain / diag.vht * 100:.2f}% du VHT actuel), coût ≈ {total_cost:,.0f} €"
            f" (budget {budget_eur:,.0f} €)"
        )
    else:
        st.warning("Aucune action rentable sous le budget — relâche le budget ou augmente la demande.")

    # ── Carte folium ──────────────────────────────────────────────────────
    st.markdown("### Carte : arcs saturés + recommandations (3 premières)")
    edges_wgs = net.edges_gdf.to_crs(CRS_WGS84)
    capacity = np.asarray(net.graph.es["capacity"], dtype=float)
    sat = ue.flows / np.maximum(capacity, 1.0)

    center = edges_wgs.geometry.union_all().centroid
    m = folium.Map(location=[center.y, center.x], zoom_start=12, tiles="CartoDB positron")

    # Fond : tous les arcs en gris fin, échantillonné pour rester léger
    sample = edges_wgs.sample(min(2000, len(edges_wgs)), random_state=0)
    for _, row in sample.iterrows():
        if row.geometry is None or row.geometry.geom_type != "LineString":
            continue
        coords = [(y, x) for x, y in row.geometry.coords]
        folium.PolyLine(coords, color="#bbbbbb", weight=1, opacity=0.5).add_to(m)

    # Arcs saturés en orange (uniquement les vraiment chargés pour lisibilité)
    sat_indices = np.argsort(-sat)[:80]
    for eid in sat_indices:
        if sat[eid] < 0.7:
            continue
        geom = edges_wgs.iloc[int(eid)].geometry
        if geom is None or geom.geom_type != "LineString":
            continue
        coords = [(y, x) for x, y in geom.coords]
        folium.PolyLine(
            coords, color="#ff8c00", weight=3, opacity=0.7,
            tooltip=f"arc {eid} • v/c = {sat[eid]:.2f}",
        ).add_to(m)

    # Top 3 recommandations
    colors = {True: "#1f6feb", False: "#d22b2b"}   # True = améliorer (bleu), False = retrait (rouge)
    for i, ev in enumerate(chosen[:3], start=1):
        eid = ev.candidate.edge_id
        geom = edges_wgs.iloc[int(eid)].geometry
        if geom is None or geom.geom_type != "LineString":
            continue
        coords = [(y, x) for x, y in geom.coords]
        is_improve = ev.candidate.action != "remove"
        folium.PolyLine(
            coords,
            color=colors[is_improve],
            weight=8, opacity=0.95,
            tooltip=(
                f"#{i} • {ev.candidate.label} • arc {eid} • "
                f"ΔVHT={ev.delta_vht_h:+.1f}h • BCR={ev.bcr:.2f}"
            ),
        ).add_to(m)

        # Étiquette numérotée au milieu
        midpoint = geom.interpolate(0.5, normalized=True)
        # midpoint est en Lambert-93 ; reprojette via le GeoSeries one-element
        from geopandas import GeoSeries
        midpoint_ll = GeoSeries([midpoint], crs=edges_wgs.crs).iloc[0]
        # Note : la GeoSeries hérite du crs du parent (déjà 4326)
        folium.CircleMarker(
            [midpoint_ll.y, midpoint_ll.x],
            radius=12, color=colors[is_improve], fill=True, fill_opacity=0.9,
            tooltip=f"Recommandation #{i}",
        ).add_to(m)
        folium.map.Marker(
            [midpoint_ll.y, midpoint_ll.x],
            icon=folium.DivIcon(
                icon_size=(24, 24), icon_anchor=(8, 12),
                html=f'<div style="font-size:14px;color:white;'
                     f'font-weight:bold;text-align:center;">{i}</div>',
            ),
        ).add_to(m)

    # Légende
    legend_html = """
    <div style="position:fixed; bottom:20px; left:20px; z-index:9999;
                background:white; padding:10px 14px; border:1px solid #444;
                border-radius:6px; font-size:13px; line-height:1.6;">
        <b>Légende</b><br>
        <span style="color:#bbbbbb">━</span> réseau (échantillon)<br>
        <span style="color:#ff8c00">━</span> arcs saturés (v/c ≥ 0.7)<br>
        <span style="color:#1f6feb">━</span> améliorer (top-3)<br>
        <span style="color:#d22b2b">━</span> retirer (Braess)
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    st_folium(m, width=None, height=620, returned_objects=[])

    st.markdown("---")
    st.caption(
        f"Pipeline : {net.graph.vcount():,} nœuds • {net.graph.ecount():,} arcs • "
        f"{od.n_zones} zones • {od.total_trips:,.0f} véh/h • "
        f"UE gap={ue.final_gap:.1e} (iter {ue.iterations}) • "
        f"SO gap={so.final_gap:.1e} (iter {so.iterations})"
    )
else:
    st.info("Configure les paramètres dans la barre latérale puis clique sur **Lancer le pipeline**.")
    st.markdown(
        "**Astuce** : commence sur `Villeurbanne, France` avec ~100 itérations. "
        "Pour `Lyon, France`, le réseau est ~8× plus grand → compter quelques minutes."
    )
