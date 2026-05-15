"""Façade publique : build_network(city) fait tout le pipeline brique 1."""

from pathlib import Path

from urban_optimizer.utils.logging import get_logger

from .merger import build_unified_network
from .osm_loader import load_osm
from .route500_loader import filter_route500_outside_urban, load_route500
from .urban_network import UrbanNetwork

logger = get_logger(__name__)


def build_network(
    city: str,
    buffer_km: float = 10.0,
    route500_path: Path | None = None,
    include_route500: bool = True,
    network_type: str = "drive",
    simplify: bool = True,
) -> UrbanNetwork:
    """Pipeline complet de construction du réseau routier.

    1. Télécharge le réseau OSM pour *city* et l'enrichit.
    2. Charge ROUTE500 (si disponible et *include_route500* est True).
    3. Supprime les doublons à l'interface OSM/ROUTE500.
    4. Fusionne tout dans un graphe igraph orienté.

    Args:
        city: nom de la ville (ex. "Lyon, France").
        buffer_km: zone tampon autour de la bbox OSM pour ROUTE500.
        route500_path: chemin vers TRONCON_ROUTE.shp ; utilise le chemin par défaut si None.
        include_route500: si False, seul OSM est utilisé (utile pour les tests).
        network_type: type de réseau OSM.
        simplify: simplification topologique OSM.

    Returns:
        UrbanNetwork prêt pour les briques suivantes.
    """
    logger.info(f"=== Construction du réseau : {city} ===")

    edges_osm, nodes_osm = load_osm(city, network_type=network_type, simplify=simplify)

    r500_filtered = None
    if include_route500:
        try:
            r500 = load_route500(
                urban_bounds=edges_osm.total_bounds,
                buffer_m=buffer_km * 1000,
                shp_path=route500_path,
            )
            urban_envelope = edges_osm.union_all().buffer(200).convex_hull
            r500_filtered = filter_route500_outside_urban(r500, urban_envelope)
        except FileNotFoundError as exc:
            logger.warning(f"ROUTE500 non disponible — réseau OSM seul. ({exc})")

    net = build_unified_network(edges_osm, nodes_osm, r500_filtered)
    logger.info("=== Réseau construit ===")
    return net
