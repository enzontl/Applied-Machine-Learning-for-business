"""Téléchargement et indexation des obstacles physiques OSM.

Utilisé par le NDP pour rejeter les candidats dont le tracé rectiligne
traverse un obstacle (bâtiment, plan d'eau, parc, voie ferrée).
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

# Tags OSM et comportement associé (buffer_m > 0 pour les géométries linéaires)
_OBSTACLE_LAYERS: list[tuple[str, dict, float]] = [
    ("bâtiments",      {"building": True},                                        0.0),
    ("eau",            {"natural": ["water", "wetland"]},                          0.0),
    ("bassins",        {"landuse": ["reservoir", "basin"]},                        0.0),
    ("parcs",          {"leisure": ["park", "nature_reserve", "garden"]},          0.0),
    ("forêts/cimetières", {"landuse": ["forest", "cemetery", "recreation_ground"]}, 0.0),
    ("voies ferrées",  {"railway": ["rail", "subway", "light_rail", "tram"]},     12.0),
]


class ObstacleIndex:
    """Index spatial STRtree sur l'ensemble des obstacles physiques, en Lambert-93.

    Interface identique à l'ancien ``BuildingIndex`` — rétrocompatible.

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


def load_obstacles(city: str, crs: int = CRS_LAMBERT93) -> ObstacleIndex:
    """Télécharge tous les obstacles physiques OSM pour *city*.

    Couches chargées :
    - bâtiments (``building=*``)
    - plans d'eau et zones humides (``natural=water/wetland``)
    - bassins / réservoirs (``landuse=reservoir/basin``)
    - parcs et espaces verts (``leisure=park/nature_reserve/garden``)
    - forêts et cimetières (``landuse=forest/cemetery/recreation_ground``)
    - voies ferrées — buffées à 12 m (``railway=rail/subway/…``)

    Returns:
        ObstacleIndex prêt à l'emploi (vide si tous les téléchargements échouent).
    """
    logger.info(f"Chargement des obstacles OSM : {city}")
    all_geoms: list = []

    for label, tags, buf in _OBSTACLE_LAYERS:
        try:
            gdf = ox.features_from_place(city, tags=tags)
            gdf = gdf.to_crs(epsg=crs)

            if buf > 0:
                # Géométries linéaires (voies ferrées) → on crée une zone tampon
                lines = gdf[gdf.geometry.geom_type.isin(["LineString", "MultiLineString"])]
                polys = [g.buffer(buf) for g in lines.geometry if g is not None]
                all_geoms.extend(polys)
                logger.info(f"  {label} : {len(polys):,} lignes buffées à {buf:.0f}m")
            else:
                polys = gdf[gdf.geometry.geom_type.isin(["Polygon", "MultiPolygon"])]
                all_geoms.extend(list(polys.geometry))
                logger.info(f"  {label} : {len(polys):,} polygones")
        except Exception as exc:
            logger.warning(f"  {label} non chargé ({exc})")

    return ObstacleIndex(all_geoms, label="obstacles")


def load_buildings(city: str, crs: int = CRS_LAMBERT93) -> ObstacleIndex:
    """Charge uniquement les bâtiments OSM (rétrocompatibilité).

    Préférer ``load_obstacles`` qui inclut aussi l'eau, les parcs et les voies ferrées.
    """
    logger.info(f"Téléchargement des bâtiments OSM : {city}")
    try:
        gdf = ox.features_from_place(city, tags={"building": True})
        gdf = gdf[gdf.geometry.geom_type.isin(["Polygon", "MultiPolygon"])].copy()
        gdf = gdf.to_crs(epsg=crs)
        logger.info(f"  Bâtiments : {len(gdf):,} emprises")
        return ObstacleIndex(list(gdf.geometry), label="bâtiments")
    except Exception as exc:
        logger.warning(f"Impossible de charger les bâtiments ({exc}) — filtre désactivé")
        return ObstacleIndex([], label="bâtiments (vide)")
