"""Téléchargement et indexation des obstacles physiques OSM.

Deux catégories :
- **Obstacles durs** (``load_obstacles``) : bâtiments, parcs, forêts, friches —
  une nouvelle route ne peut pas les traverser.
- **Déclencheurs de pont** (``load_bridge_triggers``) : voies ferrées et cours d'eau —
  traversables avec un pont, à coût majoré.
"""

from __future__ import annotations

import geopandas as gpd
import osmnx as ox
from shapely import STRtree
from shapely.geometry import LineString

from urban_optimizer.config import CRS_LAMBERT93
from urban_optimizer.utils.logging import get_logger

logger = get_logger(__name__)

_ROAD_HALF_WIDTH_M = 5.0

# Obstacles qu'on ne peut PAS traverser
_HARD_OBSTACLE_LAYERS: list[tuple[str, dict, float]] = [
    ("bâtiments",         {"building": True},                                          0.0),
    ("parcs",             {"leisure": ["park", "nature_reserve", "garden"]},            0.0),
    ("forêts/cimetières", {"landuse": ["forest", "cemetery", "recreation_ground"]},    0.0),
    ("friches",           {"landuse": ["brownfield", "greenfield", "grass", "meadow"]}, 0.0),
]

# Obstacles traversables par un pont (coût majoré, pas bloquants)
_BRIDGE_TRIGGER_LAYERS: list[tuple[str, dict, float]] = [
    ("eau",           {"natural": ["water", "wetland"]},                       0.0),
    ("bassins",       {"landuse": ["reservoir", "basin"]},                     0.0),
    ("voies ferrées", {"railway": ["rail", "subway", "light_rail", "tram"]}, 12.0),
]


class ObstacleIndex:
    """Index spatial STRtree sur un ensemble d'obstacles, en Lambert-93.

    Usage::

        idx = load_obstacles("Villeurbanne, France")
        if idx.crosses(LineString([u_xy, v_xy])):
            # rejeter ce candidat
    """

    def __init__(self, geometries: list, label: str = "obstacles"):
        self._tree = STRtree(geometries)
        self._label = label
        logger.info(f"Index {label} construit — {len(geometries):,} géométries")

    def crosses(self, segment: LineString, buffer_m: float = _ROAD_HALF_WIDTH_M) -> bool:
        """Renvoie True si *segment* (buffé de *buffer_m*) touche un obstacle."""
        corridor = segment.buffer(buffer_m)
        candidates = self._tree.query(corridor)
        if len(candidates) == 0:
            return False
        geoms = self._tree.geometries
        return any(corridor.intersects(geoms[i]) for i in candidates)

    @classmethod
    def from_geodataframe(cls, gdf: gpd.GeoDataFrame, label: str = "obstacles") -> "ObstacleIndex":
        return cls(list(gdf.geometry), label=label)


# Alias de rétrocompatibilité
BuildingIndex = ObstacleIndex


def _load_layers(
    city: str,
    layers: list[tuple[str, dict, float]],
    crs: int,
    label: str,
) -> ObstacleIndex:
    all_geoms: list = []
    for lbl, tags, buf in layers:
        try:
            gdf = ox.features_from_place(city, tags=tags)
            gdf = gdf.to_crs(epsg=crs)
            if buf > 0:
                lines = gdf[gdf.geometry.geom_type.isin(["LineString", "MultiLineString"])]
                polys = [g.buffer(buf) for g in lines.geometry if g is not None]
                all_geoms.extend(polys)
                logger.info(f"  {lbl} : {len(polys):,} lignes buffées {buf:.0f}m")
            else:
                polys = gdf[gdf.geometry.geom_type.isin(["Polygon", "MultiPolygon"])]
                all_geoms.extend(list(polys.geometry))
                logger.info(f"  {lbl} : {len(polys):,} polygones")
        except Exception as exc:
            logger.warning(f"  {lbl} non chargé ({exc})")
    return ObstacleIndex(all_geoms, label=label)


def load_obstacles(city: str, crs: int = CRS_LAMBERT93) -> ObstacleIndex:
    """Obstacles durs : bâtiments, parcs, forêts, friches.

    Une nouvelle route ne peut pas traverser ces zones.
    Pour les voies ferrées et cours d'eau (traversables par pont), utiliser
    ``load_bridge_triggers``.
    """
    logger.info(f"Chargement obstacles durs OSM : {city}")
    return _load_layers(city, _HARD_OBSTACLE_LAYERS, crs, "obstacles durs")


def load_bridge_triggers(city: str, crs: int = CRS_LAMBERT93) -> ObstacleIndex:
    """Obstacles doux : voies ferrées et cours d'eau.

    Traversables avec un pont — coût majoré mais pas bloquants.
    """
    logger.info(f"Chargement déclencheurs de pont OSM : {city}")
    return _load_layers(city, _BRIDGE_TRIGGER_LAYERS, crs, "ponts")


def load_buildings(city: str, crs: int = CRS_LAMBERT93) -> ObstacleIndex:
    """Charge uniquement les bâtiments OSM (rétrocompatibilité)."""
    logger.info(f"Téléchargement des bâtiments OSM : {city}")
    try:
        gdf = ox.features_from_place(city, tags={"building": True})
        gdf = gdf[gdf.geometry.geom_type.isin(["Polygon", "MultiPolygon"])].copy()
        gdf = gdf.to_crs(epsg=crs)
        return ObstacleIndex(list(gdf.geometry), label="bâtiments")
    except Exception as exc:
        logger.warning(f"Impossible de charger les bâtiments ({exc})")
        return ObstacleIndex([], label="bâtiments (vide)")
