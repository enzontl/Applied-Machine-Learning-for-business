"""Accessibilité et équité — mesures par zone.

Pour chaque zone OD, on calcule **combien de destinations** sont atteignables
sous un seuil de temps (typiquement 15 min) en utilisant les temps de trajet
**congestionnés** issus du Frank-Wolfe.

Deux indicateurs sortent de là :
- **Accessibilité moyenne** : moyenne sur les zones du nombre de destinations
  joignables sous T min. Plus haut = mieux.
- **Coefficient de Gini** sur l'accessibilité par zone : 0 = égalité parfaite
  (toutes les zones également bien desservies), 1 = inégalité extrême. Plus
  bas = mieux.

Un maire écolo / social pèse fortement l'équité ; un maire mobilité pèse
l'accessibilité (gain de temps moyen).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from urban_optimizer.assignment.bpr import bpr_time
from urban_optimizer.assignment.result import AssignmentResult
from urban_optimizer.demand.od_matrix import ODMatrix
from urban_optimizer.network.urban_network import UrbanNetwork


@dataclass
class AccessibilityReport:
    """Sortie du calcul d'accessibilité.

    per_zone_reachable: nombre de destinations joignables sous le seuil, par zone.
    mean_reachable:     moyenne (≃ "à combien de zones je peux accéder en T min ?").
    gini:               coefficient de Gini ∈ [0, 1] sur per_zone_reachable.
    threshold_seconds:  seuil utilisé (rappelé pour affichage).
    n_zones:            nombre de zones évaluées.
    """

    per_zone_reachable: np.ndarray
    mean_reachable: float
    gini: float
    threshold_seconds: float
    n_zones: int


def edge_travel_times(network: UrbanNetwork, result: AssignmentResult) -> np.ndarray:
    """Temps de parcours BPR (secondes) par arc, sous les flots du UE/SO."""
    g = network.graph
    capacity = np.asarray(g.es["capacity"], dtype=float)
    t0 = np.asarray(g.es["t0_s"], dtype=float)
    alpha = np.asarray(g.es["bpr_alpha"], dtype=float)
    beta = np.asarray(g.es["bpr_beta"], dtype=float)
    return bpr_time(result.flows, t0, capacity, alpha, beta)


def gini(values: np.ndarray) -> float:
    """Coefficient de Gini ∈ [0, 1]. 0 = égalité parfaite.

    Formule discrète sur valeurs triées : G = (n + 1 - 2·Σ((n+1-i)·x_i) / Σx) / n.
    """
    v = np.asarray(values, dtype=float)
    if v.size < 2:
        return 0.0
    total = v.sum()
    if total <= 0:
        return 0.0
    sorted_v = np.sort(v)
    n = sorted_v.size
    cum = np.cumsum(sorted_v)
    # G = 1 - 2·(somme cumulée des parts) / n + 1/n
    return float((n + 1 - 2 * cum.sum() / cum[-1]) / n)


def compute_accessibility(
    network: UrbanNetwork,
    od: ODMatrix,
    result: AssignmentResult,
    *,
    threshold_seconds: float = 15 * 60,
) -> AccessibilityReport:
    """Pour chaque zone, compte les destinations atteignables sous le seuil.

    Utilise les temps de trajet **congestionnés** (BPR évalués sur les flots
    d'équilibre).

    Args:
        network: réseau urbain (graphe igraph).
        od: matrice OD — on utilise ses zones comme origines et destinations.
        result: résultat d'affectation (UE ou SO) — pour les flots/temps.
        threshold_seconds: seuil d'isochrone (par défaut 15 min).

    Returns:
        AccessibilityReport.
    """
    g = network.graph
    tt = edge_travel_times(network, result)

    # Liste ordonnée des zones — celles non rattachées à un nœud sont ignorées
    zone_nodes: list[int] = []
    for z in od.zone_ids:
        n = od.zone_to_node.get(z)
        if n is not None:
            zone_nodes.append(n)
    n_zones = len(zone_nodes)
    if n_zones < 2:
        return AccessibilityReport(
            per_zone_reachable=np.zeros(n_zones, dtype=float),
            mean_reachable=0.0,
            gini=0.0,
            threshold_seconds=threshold_seconds,
            n_zones=n_zones,
        )

    # Temps minimal (secondes) de chaque zone vers chaque autre zone
    # Note: igraph accepte aussi des listes Python pour weights (plus rapide que ndarray)
    tt_list = tt.tolist()
    times = np.asarray(
        g.distances(source=zone_nodes, target=zone_nodes, weights=tt_list),
        dtype=float,
    )
    np.fill_diagonal(times, np.inf)  # une zone n'accède pas à elle-même
    reachable = (times <= threshold_seconds).sum(axis=1).astype(float)

    return AccessibilityReport(
        per_zone_reachable=reachable,
        mean_reachable=float(reachable.mean()),
        gini=gini(reachable),
        threshold_seconds=threshold_seconds,
        n_zones=n_zones,
    )
