"""Identification des arcs critiques du réseau.

Trois angles complémentaires :

1. **Délai de congestion** — arcs où le plus de temps est *perdu* par rapport
   au temps libre : ``flow · (t − t0)``. C'est le pire endroit pour la perte
   sociale ; les améliorer rendrait le plus de temps aux usagers.

2. **Saturation** — arcs proches ou au-dessus de leur capacité (``v/c``).
   Indicateur de fragilité opérationnelle plus que de perte stricte.

3. **Centralité pondérée par les flux** — arcs structuralement importants :
   beaucoup de chemins de plus courts passent par eux.

Les trois ne pointent pas toujours vers les mêmes arcs. Le ranking
"délai" est celui qu'on utilisera pour la brique 5.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from urban_optimizer.assignment.result import AssignmentResult
from urban_optimizer.network.urban_network import UrbanNetwork


def rank_by_congestion_delay(
    network: UrbanNetwork,
    result: AssignmentResult,
    top_n: int = 20,
) -> pd.DataFrame:
    """Classe les arcs par délai de congestion total ``flow · (t − t0)``.

    Cette métrique est additive sur le réseau et égale ``VHT − VHT_free_flow``.
    Top-N = les arcs qui pèsent le plus dans la perte de temps globale.
    """
    flows = result.flows
    times = result.travel_times
    t0 = result.free_flow_times
    delay = flows * (times - t0)

    df = pd.DataFrame({
        "edge_id": np.arange(len(flows)),
        "flow": flows,
        "t_actual_s": times,
        "t0_s": t0,
        "delay_per_user_s": times - t0,
        "total_delay_h": delay / 3600.0,
        "highway": network.edges_gdf["highway"].to_numpy()
            if "highway" in network.edges_gdf else "?",
        "source": network.edges_gdf["source"].to_numpy()
            if "source" in network.edges_gdf else "?",
    })
    df["share_of_total_delay"] = df["total_delay_h"] / max(df["total_delay_h"].sum(), 1e-9)
    df = df.sort_values("total_delay_h", ascending=False).head(top_n).reset_index(drop=True)
    return df


def rank_by_saturation(
    network: UrbanNetwork,
    result: AssignmentResult,
    top_n: int = 20,
) -> pd.DataFrame:
    """Classe les arcs par taux de saturation v/c (descendant)."""
    flows = result.flows
    capacity = np.asarray(network.graph.es["capacity"], dtype=float)
    sat = flows / np.maximum(capacity, 1.0)

    df = pd.DataFrame({
        "edge_id": np.arange(len(flows)),
        "flow": flows,
        "capacity": capacity,
        "saturation": sat,
        "highway": network.edges_gdf["highway"].to_numpy()
            if "highway" in network.edges_gdf else "?",
    })
    df = df.sort_values("saturation", ascending=False).head(top_n).reset_index(drop=True)
    return df


def flow_weighted_betweenness(
    network: UrbanNetwork,
    result: AssignmentResult,
) -> np.ndarray:
    """Approximation : on remonte la centralité par les flux d'affectation.

    Plutôt que la betweenness de Newman (très coûteuse en O(V·E)), on utilise
    directement ``result.flows`` : un arc qui porte beaucoup de flux est, par
    construction, un arc traversé par beaucoup de paires OD. C'est exactement
    une betweenness "demande-pondérée" sur les plus courts chemins du graphe
    pondéré par les temps d'équilibre.

    Returns:
        Tableau (n_arcs,) — pour usage en pondération de visualisation ou
        en argument d'autres calculs.
    """
    return np.asarray(result.flows, dtype=float)
