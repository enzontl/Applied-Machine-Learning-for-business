"""Façade publique : build_network(city) fait tout le pipeline brique 1."""

import hashlib
import pickle
from pathlib import Path

from urban_optimizer.config import RAW_DIR
from urban_optimizer.utils.logging import get_logger

from .merger import build_unified_network
from .osm_loader import _SIMPLIFIED_HIGHWAY_TYPES, load_osm
from .route500_loader import filter_route500_outside_urban, load_route500
from .urban_network import UrbanNetwork

logger = get_logger(__name__)

_CACHE_DIR = RAW_DIR / "network_cache"

# Incrémenter à chaque changement structurel du graphe (ex: ajout composante connexe)
_CACHE_VERSION = 2


def _cache_key(city: str, include_route500: bool, simplified: bool) -> Path:
    slug = hashlib.md5(f"v{_CACHE_VERSION}|{city}|{include_route500}|{simplified}".encode()).hexdigest()[:16]
    return _CACHE_DIR / f"{slug}.pkl"


def purge_old_caches() -> int:
    """Supprime tous les .pkl dans le cache réseau.

    Utile après un changement de version structurelle (ex: composante connexe).
    Retourne le nombre de fichiers supprimés.
    """
    if not _CACHE_DIR.exists():
        return 0
    removed = 0
    for f in _CACHE_DIR.glob("*.pkl"):
        f.unlink()
        removed += 1
    if removed:
        logger.info(f"Cache réseau purgé : {removed} fichier(s) supprimé(s)")
    return removed


def build_network(
    city: str,
    buffer_km: float = 10.0,
    route500_path: Path | None = None,
    include_route500: bool = True,
    network_type: str = "drive",
    simplify: bool = True,
    simplified_highway: bool = False,
    use_disk_cache: bool = True,
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
        simplified_highway: si True, garde uniquement primary/secondary/trunk/motorway
            (et leurs links). Réduit ~70 % des arcs → Dijkstra 3-4× plus rapide.
        use_disk_cache: si True, sauvegarde/charge le réseau sur disque (pickle).
            Évite de re-télécharger et re-traiter OSM à chaque redémarrage.

    Returns:
        UrbanNetwork prêt pour les briques suivantes.
    """
    if use_disk_cache:
        cp = _cache_key(city, include_route500, simplified_highway)
        if cp.exists():
            logger.info(f"Réseau chargé depuis cache disque : {cp.name}")
            with open(cp, "rb") as f:
                return pickle.load(f)

    logger.info(f"=== Construction du réseau : {city} ===")
    hw_filter = _SIMPLIFIED_HIGHWAY_TYPES if simplified_highway else None
    edges_osm, nodes_osm = load_osm(
        city, network_type=network_type, simplify=simplify, highway_filter=hw_filter,
    )

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

    if use_disk_cache:
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        with open(cp, "wb") as f:
            pickle.dump(net, f)
        logger.info(f"Réseau mis en cache disque : {cp.name}")

    return net
