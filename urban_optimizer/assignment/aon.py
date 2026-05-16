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


def assign_aon(
    graph: ig.Graph,
    od: ODMatrix,
    weights: np.ndarray | str = "t0_s",
) -> np.ndarray:
    """Charge les flux par All-or-Nothing.

    Args:
        graph: graphe igraph orienté du réseau.
        od: matrice OD horaire.
        weights: poids d'arc utilisés pour Dijkstra. Soit un nom d'attribut
            d'arc (``"t0_s"``), soit un tableau (n_edges,) de poids explicites
            (typiquement les temps actuels dans une itération Frank-Wolfe).

    Returns:
        Tableau (n_edges,) des flux véh/h.
    """
    flows = np.zeros(graph.ecount(), dtype=float)

    by_source: dict[int, list[tuple[int, float]]] = defaultdict(list)
    for o, d, trips in od.to_node_pairs():
        if trips <= 0:
            continue
        by_source[o].append((d, trips))

    weights_arg = list(weights) if isinstance(weights, np.ndarray) else weights

    # igraph émet un RuntimeWarning si certaines destinations sont injoignables.
    # On l'ignore : la paire OD orpheline est simplement non chargée (epath vide).
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning, message="Couldn't reach")
        for source, dests in by_source.items():
            targets = [d for d, _ in dests]
            epaths = graph.get_shortest_paths(
                source, to=targets, weights=weights_arg, mode="out", output="epath"
            )
            for epath, (_, trips) in zip(epaths, dests, strict=True):
                if not epath:
                    continue
                flows[epath] += trips

    return flows
