"""Téléchargement et indexation des emprises bâties OSM.

Utilisé par le NDP pour rejeter les candidats dont le tracé rectiligne
traverse un bâtiment existant.
"""

from __future__ import annotations

import geopandas as gpd
import osmnx as ox
from shapely import STRtree
from shapely.geometry import LineString

from urban_optimizer.config import CRS_LAMBERT93
from urban_optimizer.utils.logging import get_logger

logger = get_logger(__name__)

# Demi-largeur de route utilisée pour le test d'intersection (mètres).
# Une route tertiaire fait ~3.5 m/voie ; on prend 5 m de marge.
_ROAD_HALF_WIDTH_M = 5.0


class BuildingIndex:
    """Index spatial STRtree sur les emprises bâties, en Lambert-93.

    Usage::

        idx = BuildingIndex.from_city("Villeurbanne, France")
        if idx.crosses(LineString([u_xy, v_xy])):
            # rejeter ce candidat
    """

    def __init__(self, geometries: list):
        self._tree = STRtree(geometries)

    def crosses(self, segment: LineString, buffer_m: float = _ROAD_HALF_WIDTH_M) -> bool:
        """Renvoie True si *segment* (buffé de *buffer_m*) touche un bâtiment."""
        corridor = segment.buffer(buffer_m)
        candidates = self._tree.query(corridor)
        if len(candidates) == 0:
            return False
        # query retourne les indices des géométries candidates → vérification exacte
        geoms = self._tree.geometries
        return any(corridor.intersects(geoms[i]) for i in candidates)

    @classmethod
    def from_geodataframe(cls, gdf: gpd.GeoDataFrame) -> "BuildingIndex":
        geoms = list(gdf.geometry)
        logger.info(f"Index bâtiments construit — {len(geoms):,} emprises")
        return cls(geoms)


def load_buildings(city: str, crs: int = CRS_LAMBERT93) -> BuildingIndex:
    """Télécharge les emprises bâties OSM pour *city* et renvoie un BuildingIndex.

    Les bâtiments sont reprojetés en Lambert-93 (même CRS que le réseau routier)
    pour que les intersections soient calculées en mètres.

    Args:
        city: nom de la ville passé à OSMnx (ex. "Villeurbanne, France").
        crs: code EPSG cible (default Lambert-93).

    Returns:
        BuildingIndex prêt à l'emploi, ou un index vide si le téléchargement échoue.
    """
    logger.info(f"Téléchargement des bâtiments OSM : {city}")
    try:
        gdf = ox.features_from_place(city, tags={"building": True})
        # On ne garde que les polygones (pas les points/lignes)
        gdf = gdf[gdf.geometry.geom_type.isin(["Polygon", "MultiPolygon"])].copy()
        gdf = gdf.to_crs(epsg=crs)
        logger.info(f"  Bâtiments téléchargés : {len(gdf):,} emprises")
        return BuildingIndex.from_geodataframe(gdf)
    except Exception as exc:
        logger.warning(f"Impossible de charger les bâtiments ({exc}) — filtre désactivé")
        return BuildingIndex([])
