"""Zonage spatial d'une ville pour la génération de demande.

Trois méthodes :
- ``grid`` : grille régulière sur la bbox du réseau (aucune donnée externe).
- ``iris``  : polygones IRIS (IGN) enrichis des données INSEE pop/emplois.
- ``h3``    : hexagones H3 (optionnel, nécessite le paquet ``h3``).

Toutes renvoient un ``Zoning`` homogène prêt à entrer dans le modèle gravitaire.
"""

from __future__ import annotations

from dataclasses import dataclass

import geopandas as gpd
import numpy as np
from shapely.geometry import box

from urban_optimizer.config import CRS_LAMBERT93
from urban_optimizer.network.urban_network import UrbanNetwork
from urban_optimizer.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class Zoning:
    """Découpage spatial enrichi d'attracteurs de demande.

    Attributes:
        gdf: GeoDataFrame avec colonnes ``zone_id``, ``geometry``, ``population``,
            ``jobs`` et ``centroid`` (Point).
        centroids_xy: zone_id → (x, y) en Lambert-93.
        zone_to_node: zone_id → index du nœud du réseau le plus proche.
        crs: code EPSG (Lambert-93 par défaut).
        method: méthode de zonage utilisée.
    """

    gdf: gpd.GeoDataFrame
    centroids_xy: dict
    zone_to_node: dict
    crs: int
    method: str

    @property
    def n_zones(self) -> int:
        return len(self.gdf)

    @property
    def zone_ids(self) -> list:
        return self.gdf["zone_id"].tolist()

    def summary(self) -> None:
        print(f"Zonage {self.method!r} — {self.n_zones} zones")
        print(f"  Population totale : {self.gdf['population'].sum():,.0f}")
        print(f"  Emplois totaux    : {self.gdf['jobs'].sum():,.0f}")
        print(f"  Zones rattachées  : {len(self.zone_to_node)} / {self.n_zones}")


# ────────────────────────────────────────────────────────────────────────────
# Rattachement des zones au réseau
# ────────────────────────────────────────────────────────────────────────────

def _build_zone_to_node(centroids_xy: dict, network: UrbanNetwork) -> dict:
    """Pour chaque centroïde de zone, trouve l'index du nœud le plus proche.

    Utilise une recherche brute O(n_zones × n_nodes) — suffisant pour les tailles
    typiques (< 10⁴ zones, < 10⁵ nœuds). Si le réseau grandit, passer à un KDTree.
    """
    node_ids = list(network.nodes_xy.keys())
    node_xy = np.array([network.nodes_xy[n] for n in node_ids])

    mapping: dict = {}
    for zone_id, (cx, cy) in centroids_xy.items():
        d2 = (node_xy[:, 0] - cx) ** 2 + (node_xy[:, 1] - cy) ** 2
        mapping[zone_id] = int(node_ids[int(np.argmin(d2))])
    return mapping


# ────────────────────────────────────────────────────────────────────────────
# Méthode "grid" — autonome, sans données externes
# ────────────────────────────────────────────────────────────────────────────

# Types OSM utilisés comme proxy "résidentiel" (origines) et "attracteur" (destinations)
RESIDENTIAL_HIGHWAYS = {"residential", "living_street", "unclassified", "tertiary"}
ATTRACTOR_HIGHWAYS = {"primary", "secondary", "trunk", "motorway"}


def _attractor_weights(
    edges_gdf: gpd.GeoDataFrame,
    zones_gdf: gpd.GeoDataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """Calcule population et emplois proxy par zone à partir des arcs OSM.

    Hypothèses (proxy simple, calibrable ensuite) :
      - population ∝ longueur cumulée des arcs résidentiels
      - emplois    ∝ longueur cumulée des arcs structurants

    Plus tard : remplacer par des vraies données INSEE via insee_loader.
    """
    edges = edges_gdf[edges_gdf["source"] == "osm"].copy()

    res = edges[edges["highway"].isin(RESIDENTIAL_HIGHWAYS)]
    attr = edges[edges["highway"].isin(ATTRACTOR_HIGHWAYS)]

    pop = np.zeros(len(zones_gdf), dtype=float)
    jobs = np.zeros(len(zones_gdf), dtype=float)

    if len(res):
        joined = gpd.sjoin(
            res[["geometry", "length_m"]],
            zones_gdf[["zone_id", "geometry"]],
            how="inner",
            predicate="intersects",
        )
        agg = joined.groupby("zone_id")["length_m"].sum()
        for zid, val in agg.items():
            i = int(zones_gdf.index[zones_gdf["zone_id"] == zid][0])
            pop[i] = val

    if len(attr):
        joined = gpd.sjoin(
            attr[["geometry", "length_m"]],
            zones_gdf[["zone_id", "geometry"]],
            how="inner",
            predicate="intersects",
        )
        agg = joined.groupby("zone_id")["length_m"].sum()
        for zid, val in agg.items():
            i = int(zones_gdf.index[zones_gdf["zone_id"] == zid][0])
            jobs[i] = val

    return pop, jobs


def build_grid_zoning(
    network: UrbanNetwork,
    n_cells: int = 20,
    min_pop: float = 1.0,
) -> Zoning:
    """Construit un zonage en grille régulière sur la bbox du réseau.

    Aucune donnée externe requise — utile pour démarrer et pour les tests.

    Args:
        network: réseau urbain construit par la brique 1.
        n_cells: nombre de cellules par côté (la grille est carrée n×n).
        min_pop: seuil pour conserver une cellule (filtre les cellules vides).

    Returns:
        Zoning de méthode "grid".
    """
    minx, miny, maxx, maxy = network.edges_gdf.total_bounds
    dx = (maxx - minx) / n_cells
    dy = (maxy - miny) / n_cells

    polys, ids = [], []
    k = 0
    for i in range(n_cells):
        for j in range(n_cells):
            x0 = minx + i * dx
            y0 = miny + j * dy
            polys.append(box(x0, y0, x0 + dx, y0 + dy))
            ids.append(f"grid_{i:02d}_{j:02d}")
            k += 1

    zones = gpd.GeoDataFrame(
        {"zone_id": ids, "geometry": polys},
        crs=f"EPSG:{network.crs}",
    )

    pop, jobs = _attractor_weights(network.edges_gdf, zones)
    zones["population"] = pop
    zones["jobs"] = jobs

    keep = zones["population"] >= min_pop
    zones = zones[keep].reset_index(drop=True)
    logger.info(f"Grille {n_cells}×{n_cells} — zones conservées : {len(zones):,}")

    zones["centroid"] = zones.geometry.centroid
    centroids_xy = {
        row["zone_id"]: (row["centroid"].x, row["centroid"].y)
        for _, row in zones.iterrows()
    }
    zone_to_node = _build_zone_to_node(centroids_xy, network)

    return Zoning(
        gdf=zones,
        centroids_xy=centroids_xy,
        zone_to_node=zone_to_node,
        crs=network.crs,
        method="grid",
    )


# ────────────────────────────────────────────────────────────────────────────
# Méthode "iris" — pop/emplois réels (nécessite données INSEE/IGN)
# ────────────────────────────────────────────────────────────────────────────

def build_iris_zoning(
    network: UrbanNetwork,
    insee_codes_dept: list[str] | None = None,
) -> Zoning:
    """Construit un zonage IRIS enrichi des données INSEE pop/emplois.

    Les vraies fonctions de chargement sont dans ``insee_loader``. Cette fonction
    ne fait que la composition.

    Args:
        network: réseau urbain.
        insee_codes_dept: liste des codes département à charger (ex. ["69"]).

    Returns:
        Zoning de méthode "iris".

    Raises:
        FileNotFoundError: si les données IGN/INSEE ne sont pas dans data/external.
    """
    from .insee_loader import load_insee_pop_jobs, load_iris_contours

    bounds = network.edges_gdf.total_bounds
    iris = load_iris_contours(bounds=bounds, dept_codes=insee_codes_dept)
    pop_jobs = load_insee_pop_jobs(dept_codes=insee_codes_dept)

    zones = iris.merge(pop_jobs, on="zone_id", how="left")
    zones["population"] = zones["population"].fillna(0.0)
    zones["jobs"] = zones["jobs"].fillna(0.0)

    zones["centroid"] = zones.geometry.centroid
    centroids_xy = {
        row["zone_id"]: (row["centroid"].x, row["centroid"].y)
        for _, row in zones.iterrows()
    }
    zone_to_node = _build_zone_to_node(centroids_xy, network)

    logger.info(f"IRIS — {len(zones)} zones, pop totale = {zones['population'].sum():,.0f}")

    return Zoning(
        gdf=zones,
        centroids_xy=centroids_xy,
        zone_to_node=zone_to_node,
        crs=network.crs,
        method="iris",
    )


# ────────────────────────────────────────────────────────────────────────────
# Méthode "h3" — hexagones uniformes (optionnel)
# ────────────────────────────────────────────────────────────────────────────

def build_h3_zoning(
    network: UrbanNetwork,
    resolution: int = 8,
    min_pop: float = 1.0,
) -> Zoning:
    """Construit un zonage en hexagones H3 sur la bbox du réseau.

    Nécessite le paquet ``h3`` (non listé dans les dépendances de base).
    """
    try:
        import h3
    except ImportError as exc:
        raise ImportError(
            "build_h3_zoning nécessite `pip install h3`."
        ) from exc

    from shapely.geometry import Polygon

    bounds_wgs84 = (
        network.edges_gdf.to_crs(4326).total_bounds
    )
    minx, miny, maxx, maxy = bounds_wgs84
    bbox_polygon = {
        "type": "Polygon",
        "coordinates": [[
            [minx, miny], [maxx, miny], [maxx, maxy], [minx, maxy], [minx, miny],
        ]],
    }
    cells = list(h3.polyfill(bbox_polygon, resolution, geo_json_conformant=True))

    polys, ids = [], []
    for c in cells:
        boundary = h3.h3_to_geo_boundary(c, geo_json=True)
        polys.append(Polygon(boundary))
        ids.append(c)

    zones = gpd.GeoDataFrame({"zone_id": ids, "geometry": polys}, crs="EPSG:4326")
    zones = zones.to_crs(f"EPSG:{network.crs}")

    pop, jobs = _attractor_weights(network.edges_gdf, zones)
    zones["population"] = pop
    zones["jobs"] = jobs
    zones = zones[zones["population"] >= min_pop].reset_index(drop=True)

    zones["centroid"] = zones.geometry.centroid
    centroids_xy = {
        row["zone_id"]: (row["centroid"].x, row["centroid"].y)
        for _, row in zones.iterrows()
    }
    zone_to_node = _build_zone_to_node(centroids_xy, network)

    return Zoning(
        gdf=zones,
        centroids_xy=centroids_xy,
        zone_to_node=zone_to_node,
        crs=CRS_LAMBERT93,
        method="h3",
    )
