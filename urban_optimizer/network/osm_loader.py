"""Chargement et enrichissement du réseau routier via OSMnx."""

from pathlib import Path

import numpy as np
import osmnx as ox
import pandas as pd

from urban_optimizer.config import (
    BPR_ALPHA,
    BPR_BETA,
    CAPACITY_BY_TYPE,
    CRS_LAMBERT93,
    FREE_SPEED_BY_TYPE,
    RAW_DIR,
)
from urban_optimizer.utils.logging import get_logger

logger = get_logger(__name__)


def _setup_cache(cache_dir: Path | None = None) -> None:
    folder = cache_dir or RAW_DIR / "osmnx_cache"
    folder.mkdir(parents=True, exist_ok=True)
    ox.settings.use_cache = True
    ox.settings.cache_folder = str(folder)
    ox.settings.log_console = False


def _normalize_highway(hw) -> str:
    if isinstance(hw, list):
        return hw[0]
    return str(hw)


def _clean_lanes(val, default: int = 1) -> int:
    if val is None:
        return default
    try:
        if pd.isna(val):
            return default
    except (ValueError, TypeError):
        pass
    if isinstance(val, (list, np.ndarray)):
        try:
            return max(int(v) for v in val)
        except (ValueError, TypeError):
            return default
    try:
        return int(float(val))
    except (ValueError, TypeError):
        return default


def _parse_maxspeed(ms) -> int | None:
    if ms is None:
        return None
    try:
        if pd.isna(ms):
            return None
    except (ValueError, TypeError):
        pass
    if isinstance(ms, (list, np.ndarray)):
        ms = ms[0]
    s = str(ms).lower().strip()
    if "mph" in s:
        try:
            return int(float(s.replace("mph", "").strip()) * 1.609)
        except ValueError:
            return None
    try:
        return int(float(s.replace("km/h", "").strip()))
    except ValueError:
        return None


def _enrich_edges(edges):
    """Ajoute les colonnes nécessaires à l'affectation : capacité, t0, BPR."""
    edges = edges.copy()
    edges["highway_clean"] = edges["highway"].apply(_normalize_highway)
    edges["lanes_clean"] = edges["lanes"].apply(_clean_lanes) if "lanes" in edges else 1
    edges["length_m"] = edges.geometry.length

    edges["free_speed_kmh"] = edges["highway_clean"].map(FREE_SPEED_BY_TYPE).fillna(50.0)
    if "maxspeed" in edges.columns:
        observed = edges["maxspeed"].apply(_parse_maxspeed)
        edges["free_speed_kmh"] = observed.fillna(edges["free_speed_kmh"])

    cap_per_lane = edges["highway_clean"].map(CAPACITY_BY_TYPE).fillna(600.0)
    edges["capacity"] = cap_per_lane * edges["lanes_clean"]

    edges["t0_s"] = edges["length_m"] / (edges["free_speed_kmh"] * 1000.0 / 3600.0)
    edges["bpr_alpha"] = BPR_ALPHA
    edges["bpr_beta"] = BPR_BETA
    edges["source"] = "osm"
    return edges


_SIMPLIFIED_HIGHWAY_TYPES = frozenset({
    "motorway", "trunk", "primary", "secondary",
    "motorway_link", "trunk_link", "primary_link", "secondary_link",
})


def load_osm(
    city: str,
    network_type: str = "drive",
    simplify: bool = True,
    highway_filter: frozenset[str] | None = None,
):
    """Télécharge le réseau OSM pour *city* et renvoie (edges_gdf, nodes_gdf) en Lambert-93.

    Args:
        city: nom de la ville passé à OSMnx (ex. "Lyon, France").
        network_type: type de réseau OSM (default "drive").
        simplify: simplification topologique du graphe OSM.
        highway_filter: si fourni, garde uniquement les arcs dont highway_clean
            est dans cet ensemble (ex. _SIMPLIFIED_HIGHWAY_TYPES). Réduit
            fortement la taille du graphe pour accélérer les calculs.

    Returns:
        Tuple (edges_gdf, nodes_gdf) reprojetés en EPSG:2154, avec colonnes enrichies.
    """
    _setup_cache()
    logger.info(f"Téléchargement OSM : {city}")
    G = ox.graph_from_place(city, network_type=network_type, simplify=simplify)
    logger.info(f"  OSM brut — nœuds: {G.number_of_nodes():,}, arcs: {G.number_of_edges():,}")

    G_proj = ox.project_graph(G, to_crs=f"EPSG:{CRS_LAMBERT93}")
    edges = ox.graph_to_gdfs(G_proj, nodes=False)
    nodes = ox.graph_to_gdfs(G_proj, edges=False)

    edges = _enrich_edges(edges)

    if highway_filter:
        before = len(edges)
        edges = edges[edges["highway_clean"].isin(highway_filter)]
        logger.info(f"  Filtre highway — {before:,} → {len(edges):,} arcs")

    logger.info(f"  OSM enrichi — arcs: {len(edges):,}")
    return edges, nodes
