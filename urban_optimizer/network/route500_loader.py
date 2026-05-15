"""Chargement et conversion des données ROUTE500 (IGN) en schéma standard."""

from pathlib import Path

import geopandas as gpd
from shapely.geometry import box

from urban_optimizer.config import (
    BPR_ALPHA,
    BPR_BETA,
    CAPACITY_BY_TYPE,
    FREE_SPEED_BY_TYPE,
    RAW_DIR,
)
from urban_optimizer.utils.logging import get_logger

logger = get_logger(__name__)

# Chemin par défaut (structure d'archive IGN)
DEFAULT_ROUTE500_PATH = (
    RAW_DIR
    / "ROUTE500_3-0__SHP_LAMB93_FXX_2021-11-03"
    / "ROUTE500"
    / "1_DONNEES_LIVRAISON_2022-01-00175"
    / "R500_3-0_SHP_LAMB93_FXX-ED211"
    / "RESEAU_ROUTIER"
    / "TRONCON_ROUTE.shp"
)

VOCATION_TO_HIGHWAY = {
    "Type autoroutier": "motorway",
    "Liaison principale": "trunk",
    "Liaison régionale": "primary",
    "Liaison locale": "secondary",
    "Bretelle": "motorway_link",
}

VOCATION_LANES = {
    "Type autoroutier": 2,
    "Liaison principale": 1,
    "Liaison régionale": 1,
    "Liaison locale": 1,
    "Bretelle": 1,
}


def _r500_to_standard(r500_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Convertit un GeoDataFrame ROUTE500 vers le schéma commun (OSM-like)."""
    out = gpd.GeoDataFrame(geometry=r500_gdf.geometry, crs=r500_gdf.crs)
    out["highway_clean"] = r500_gdf["VOCATION"].map(VOCATION_TO_HIGHWAY).fillna("secondary")
    nb_chaussees = r500_gdf["NB_CHAUSSE"].apply(
        lambda x: 2 if isinstance(x, str) and "2" in x else 1
    )
    out["lanes_clean"] = nb_chaussees * r500_gdf["VOCATION"].map(VOCATION_LANES).fillna(1)
    out["length_m"] = out.geometry.length
    out["free_speed_kmh"] = out["highway_clean"].map(FREE_SPEED_BY_TYPE).fillna(70.0)
    cap_per_lane = out["highway_clean"].map(CAPACITY_BY_TYPE).fillna(900.0)
    out["capacity"] = cap_per_lane * out["lanes_clean"]
    out["t0_s"] = out["length_m"] / (out["free_speed_kmh"] * 1000.0 / 3600.0)
    out["bpr_alpha"] = BPR_ALPHA
    out["bpr_beta"] = BPR_BETA
    out["source"] = "route500"
    out["r500_id"] = r500_gdf["ID_RTE500"].values
    # True si circulation en sens unique (sens direct ou inverse)
    out["oneway"] = r500_gdf["SENS"].apply(
        lambda s: isinstance(s, str) and ("direct" in s.lower() or "inverse" in s.lower())
    )
    return out


def load_route500(
    urban_bounds,
    buffer_m: float = 10_000,
    shp_path: Path | None = None,
) -> gpd.GeoDataFrame:
    """Charge ROUTE500 dans la zone tampon autour de l'emprise urbaine.

    Args:
        urban_bounds: tuple (minx, miny, maxx, maxy) en Lambert-93 du réseau OSM.
        buffer_m: distance tampon en mètres autour de l'emprise (default 10 km).
        shp_path: chemin vers TRONCON_ROUTE.shp ; utilise DEFAULT_ROUTE500_PATH si None.

    Returns:
        GeoDataFrame converti au schéma standard, en Lambert-93.

    Raises:
        FileNotFoundError: si le fichier SHP est introuvable.
    """
    path = Path(shp_path) if shp_path else DEFAULT_ROUTE500_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"ROUTE500 introuvable : {path}\n"
            "Téléchargez l'archive IGN et décompressez-la dans data/raw/."
        )

    minx, miny, maxx, maxy = urban_bounds
    bbox = box(minx - buffer_m, miny - buffer_m, maxx + buffer_m, maxy + buffer_m)
    logger.info(f"Chargement ROUTE500 dans la zone tampon ({buffer_m/1000:.0f} km)")

    r500 = gpd.read_file(path, bbox=bbox, encoding="latin1")
    logger.info(f"  ROUTE500 brut — tronçons: {len(r500):,}")

    r500_clean = _r500_to_standard(r500)
    logger.info(f"  ROUTE500 converti — tronçons: {len(r500_clean):,}")
    return r500_clean


def filter_route500_outside_urban(r500_gdf: gpd.GeoDataFrame, urban_envelope) -> gpd.GeoDataFrame:
    """Retire les tronçons ROUTE500 dont le centroïde est dans l'enveloppe urbaine OSM.

    Évite le double comptage : l'intérieur de la ville est déjà couvert par OSM.

    Args:
        r500_gdf: GeoDataFrame ROUTE500 complet.
        urban_envelope: géométrie shapely de l'enveloppe urbaine OSM.

    Returns:
        Sous-ensemble de r500_gdf conservé (hors zone urbaine).
    """
    in_urban = r500_gdf.geometry.centroid.within(urban_envelope)
    outside = r500_gdf[~in_urban].copy()
    logger.info(
        f"ROUTE500 — conservé: {len(outside):,}, retiré (doublon OSM): {in_urban.sum():,}"
    )
    return outside
