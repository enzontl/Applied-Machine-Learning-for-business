"""Fusion des arcs OSM et ROUTE500 dans un graphe igraph orienté."""

import geopandas as gpd
import igraph as ig

from urban_optimizer.config import CRS_LAMBERT93
from urban_optimizer.utils.logging import get_logger

from .urban_network import UrbanNetwork

logger = get_logger(__name__)

_EDGE_ATTRS = ["length_m", "free_speed_kmh", "capacity", "t0_s", "bpr_alpha", "bpr_beta"]


def _endpoints(geom) -> tuple:
    coords = list(geom.coords)
    return (
        (round(coords[0][0], 1), round(coords[0][1], 1)),
        (round(coords[-1][0], 1), round(coords[-1][1], 1)),
    )


def build_unified_network(
    edges_osm: gpd.GeoDataFrame,
    nodes_osm: gpd.GeoDataFrame,
    edges_r500: gpd.GeoDataFrame | None = None,
    crs: int = CRS_LAMBERT93,
) -> UrbanNetwork:
    """Construit le graphe orienté igraph à partir des sources enrichies.

    Args:
        edges_osm: arcs OSM enrichis (index MultiIndex u/v/key, colonnes standard).
        nodes_osm: nœuds OSM projetés (index = osmid, geometry = Point Lambert-93).
        edges_r500: arcs ROUTE500 filtrés et enrichis (optionnel).
        crs: code EPSG du CRS de travail.

    Returns:
        UrbanNetwork avec graphe igraph, dictionnaire des coordonnées nœuds, et GeoDataFrame des arcs.
    """
    osm_id_to_coord = {
        nid: (round(row.geometry.x, 1), round(row.geometry.y, 1))
        for nid, row in nodes_osm.iterrows()
    }
    all_coords: set = set(osm_id_to_coord.values())

    r500 = None
    if edges_r500 is not None and len(edges_r500) > 0:
        r500 = edges_r500.copy()
        eps = r500.geometry.apply(_endpoints)
        r500["u_xy"] = eps.apply(lambda x: x[0])
        r500["v_xy"] = eps.apply(lambda x: x[1])
        all_coords.update(r500["u_xy"])
        all_coords.update(r500["v_xy"])

    coord_to_idx = {c: i for i, c in enumerate(sorted(all_coords))}
    logger.info(f"Nœuds totaux : {len(coord_to_idx):,}")

    edges_list: list[tuple[int, int]] = []
    attrs: dict[str, list] = {k: [] for k in _EDGE_ATTRS}
    attrs["highway"] = []
    attrs["source"] = []
    geoms: list = []

    skipped = 0
    for _, row in edges_osm.reset_index().iterrows():
        u_c = osm_id_to_coord.get(row["u"])
        v_c = osm_id_to_coord.get(row["v"])
        if u_c is None or v_c is None:
            skipped += 1
            continue
        edges_list.append((coord_to_idx[u_c], coord_to_idx[v_c]))
        for k in _EDGE_ATTRS:
            attrs[k].append(float(row[k]))
        attrs["highway"].append(str(row["highway_clean"]))
        attrs["source"].append("osm")
        geoms.append(row.geometry)

    if skipped:
        logger.warning(f"Arcs OSM ignorés (nœud introuvable) : {skipped}")

    if r500 is not None:
        for _, row in r500.iterrows():
            u_idx = coord_to_idx[row["u_xy"]]
            v_idx = coord_to_idx[row["v_xy"]]
            directions = (
                [(u_idx, v_idx)]
                if row.get("oneway", False)
                else [(u_idx, v_idx), (v_idx, u_idx)]
            )
            for src_n, tgt_n in directions:
                edges_list.append((src_n, tgt_n))
                for k in _EDGE_ATTRS:
                    attrs[k].append(float(row[k]))
                attrs["highway"].append(str(row["highway_clean"]))
                attrs["source"].append("route500")
                geoms.append(row.geometry)

    g = ig.Graph(n=len(coord_to_idx), edges=edges_list, directed=True)
    for k, vals in attrs.items():
        g.es[k] = vals

    nodes_xy = {i: c for c, i in coord_to_idx.items()}
    edges_gdf_out = gpd.GeoDataFrame(
        {k: attrs[k] for k in attrs},
        geometry=geoms,
        crs=f"EPSG:{crs}",
    )

    source_counts = {s: attrs["source"].count(s) for s in set(attrs["source"])}
    logger.info(f"Graphe brut — nœuds: {g.vcount():,}, arcs: {g.ecount():,} {source_counts}")

    # ── Extraire la plus grande composante faiblement connexe ──────────
    # Sans ça, Dijkstra spamme des milliers de warnings "Couldn't reach
    # some vertices" et le pipeline prend 10× plus de temps.
    components = g.connected_components(mode="weak")
    if len(components) > 1:
        giant_id = max(range(len(components)), key=lambda i: len(components[i]))
        keep_vids = sorted(components[giant_id])
        n_removed = g.vcount() - len(keep_vids)
        logger.info(
            f"Graphe déconnecté : {len(components)} composantes. "
            f"On garde la principale ({len(keep_vids):,} nœuds), "
            f"suppression de {n_removed:,} nœuds isolés."
        )
        # Sous-graphe induit
        g_sub = g.induced_subgraph(keep_vids)

        # Re-mapper nodes_xy et edges_gdf
        old_to_new = {old: new for new, old in enumerate(keep_vids)}
        nodes_xy = {old_to_new[old]: nodes_xy[old] for old in keep_vids}

        # Filtrer edges_gdf : garder uniquement les arcs du sous-graphe
        kept_eids = set()
        for e in g.es:
            if e.source in old_to_new and e.target in old_to_new:
                kept_eids.add(e.index)
        edges_gdf_out = edges_gdf_out.iloc[sorted(kept_eids)].reset_index(drop=True)

        g = g_sub
        logger.info(f"Graphe connexe — nœuds: {g.vcount():,}, arcs: {g.ecount():,}")

    return UrbanNetwork(graph=g, nodes_xy=nodes_xy, edges_gdf=edges_gdf_out, crs=crs)
