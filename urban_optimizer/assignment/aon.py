"""All-or-Nothing : tous les utilisateurs prennent le plus court chemin.

Sert (1) de baseline (sans congestion endogène) et (2) de direction de descente
dans Frank-Wolfe pour UE / SO.

Implémentation : Dijkstra depuis chaque origine vers toutes ses destinations
via ``igraph.get_shortest_paths(..., output="epath")`` qui renvoie directement
les arcs traversés. C'est l'opération critique du solveur — les autres briques
ne tournent pas si elle est lente.
"""

from __future__ import annotations

import warnings
from collections import defaultdict

import igraph as ig
import numpy as np

from urban_optimizer.demand.od_matrix import ODMatrix


# Cache module-level : pour chaque OD, pré-calcule source → (targets_list, trips_list).
# Évite à la fois la reconstruction de by_source ET la construction de targets[] par AoN.
_BY_SOURCE_CACHE: dict[int, dict[int, tuple[list[int], list[float]]]] = {}


def _get_by_source(od: ODMatrix) -> dict[int, tuple[list[int], list[float]]]:
    """Construit (ou récupère du cache) la map source → (targets_list, trips_list).

    Pré-décomposer en deux listes (au lieu de [(dest, trips), ...]) permet à
    `assign_aon` de passer directement `targets_list` à igraph sans reconstruire
    la liste à chaque appel.
    """
    key = id(od)
    cached = _BY_SOURCE_CACHE.get(key)
    if cached is not None:
        return cached
    temp: dict[int, list[tuple[int, float]]] = defaultdict(list)
    for o, d, trips in od.to_node_pairs():
        if trips <= 0:
            continue
        temp[o].append((d, trips))
    bs: dict[int, tuple[list[int], list[float]]] = {
        source: ([d for d, _ in dests], [t for _, t in dests])
        for source, dests in temp.items()
    }
    _BY_SOURCE_CACHE[key] = bs
    return bs


def assign_aon(
    graph: ig.Graph,
    od: ODMatrix,
    weights: np.ndarray | list | str = "t0_s",
) -> np.ndarray:
    """Charge les flux par All-or-Nothing.

    Args:
        graph: graphe igraph orienté du réseau.
        od: matrice OD horaire.
        weights: poids d'arc utilisés pour Dijkstra. Soit un nom d'attribut
            d'arc (``"t0_s"``), soit un tableau (n_edges,) ou une liste de
            poids explicites (typiquement les temps actuels dans une itération
            Frank-Wolfe). Si c'est une liste, on l'utilise directement.

    Returns:
        Tableau (n_edges,) des flux véh/h.
    """
    flows = np.zeros(graph.ecount(), dtype=float)

    by_source = _get_by_source(od)

    if isinstance(weights, np.ndarray):
        weights_arg = weights.tolist()
    else:
        # str (attribut) ou list déjà — on passe tel quel
        weights_arg = weights

    # igraph émet un RuntimeWarning si certaines destinations sont injoignables.
    # On l'ignore : la paire OD orpheline est simplement non chargée (epath vide).
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning, message="Couldn't reach")
        for source, (targets, trips_list) in by_source.items():
            epaths = graph.get_shortest_paths(
                source, to=targets, weights=weights_arg, mode="out", output="epath"
            )
            for epath, trips in zip(epaths, trips_list):
                if not epath:
                    continue
                flows[epath] += trips

    return flows
