"""Demande induite (induced demand / "rebound effect").

Référence : SACTRA 1994 ; Goodwin 1996. Quand une intervention réduit le temps
de parcours d'une paire OD, une partie de la demande latente (TC, vélo,
trajets différés, étalement urbain) bascule vers la voiture. L'élasticité de
la demande au temps de parcours est l'outil standard pour quantifier cet
effet :

    ΔT / T = ε · Δt / t      avec ε ∈ [-1.5, -0.3]

Valeur par défaut ε = -0.6 (long-terme urbain, médiane de la méta-analyse
Goodwin). Avec Δt/t = -10 % (gain) → ΔT/T = +6 % de trafic induit.

Ce module fournit un ``apply_induced_demand`` qui retourne une **nouvelle**
ODMatrix ajustée, sans muter l'entrée. La logique d'itération
demande↔congestion (boucle FW-induced) est faite côté pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass

import igraph as ig
import numpy as np

from urban_optimizer.assignment.result import AssignmentResult
from urban_optimizer.demand.od_matrix import ODMatrix
from urban_optimizer.network.urban_network import UrbanNetwork
from urban_optimizer.utils.logging import get_logger

logger = get_logger(__name__)


# Élasticité par défaut : long-terme urbain (Goodwin 1996, SACTRA 1994)
DEFAULT_ELASTICITY = -0.6

# Garde-fou : on n'autorise pas plus de ×2.0 ni moins de ×0.5 de variation
# d'une paire OD (évite les explosions sur paires marginales aux temps quasi nuls)
INDUCED_CLIP_MIN = 0.5
INDUCED_CLIP_MAX = 2.0


@dataclass
class InducedDemandStats:
    """Diagnostic d'une ronde d'ajustement."""

    elasticity: float
    n_pairs_affected: int
    mean_time_delta_pct: float          # Δt/t moyen pondéré par trips (en %)
    mean_trip_delta_pct: float          # ΔT/T moyen pondéré (en %)
    total_trips_before: float
    total_trips_after: float
    induced_trips: float                # > 0 si demande globale augmente
    induced_share: float                # induced_trips / total_trips_before

    def summary(self) -> str:
        return (
            f"induced demand (ε={self.elasticity:+.2f}) — "
            f"{self.n_pairs_affected} paires impactées, "
            f"Δt moyen = {self.mean_time_delta_pct:+.2f}%, "
            f"ΔT moyen = {self.mean_trip_delta_pct:+.2f}%, "
            f"trafic induit = {self.induced_trips:+,.0f} véh/h "
            f"({self.induced_share * 100:+.2f}%)"
        )


def _shortest_path_times(
    g: ig.Graph,
    travel_times: np.ndarray,
    sources: list[int],
    targets: list[int],
) -> np.ndarray:
    """Matrice (|sources|, |targets|) des temps de plus court chemin (s).

    Note : utilise un seul appel batch igraph (Dijkstra), beaucoup plus
    rapide qu'une boucle Python sur les paires.
    """
    w = travel_times.tolist()
    mat = g.distances(source=sources, target=targets, weights=w)
    return np.asarray(mat, dtype=float)


def apply_induced_demand(
    od: ODMatrix,
    network: UrbanNetwork,
    ue_before: AssignmentResult,
    ue_after: AssignmentResult,
    *,
    elasticity: float = DEFAULT_ELASTICITY,
    min_time_s: float = 30.0,
) -> tuple[ODMatrix, InducedDemandStats]:
    """Ajuste l'OD selon l'élasticité au temps de parcours.

    Pour chaque paire (zo, zd) avec trips > 0 :
        t_before = shortest_path(ue_before.travel_times)
        t_after  = shortest_path(ue_after.travel_times)
        Δt/t     = (t_after - t_before) / t_before          (< 0 si gain)
        trips'   = trips · (1 + ε · Δt/t)                    (> trips si gain)

    Garde-fous :
    - Paires avec t_before < ``min_time_s`` ignorées (bruit numérique).
    - Multiplicateur clippé sur [INDUCED_CLIP_MIN, INDUCED_CLIP_MAX].
    - Trips négatifs interdits (np.maximum à 0).

    L'OD retournée est une copie ; l'entrée n'est pas modifiée.
    """
    g = network.graph

    # Mapping zone → node (les zones non rattachées sont sautées)
    zone_to_node = od.zone_to_node
    zones = od.zone_ids
    nodes = [zone_to_node.get(z) for z in zones]
    valid_idx = [i for i, n in enumerate(nodes) if n is not None]
    valid_nodes = [nodes[i] for i in valid_idx]
    if len(valid_nodes) < 2:
        logger.warning("induced demand : <2 zones rattachées au graphe, skip")
        return od, InducedDemandStats(
            elasticity=elasticity, n_pairs_affected=0,
            mean_time_delta_pct=0.0, mean_trip_delta_pct=0.0,
            total_trips_before=od.total_trips, total_trips_after=od.total_trips,
            induced_trips=0.0, induced_share=0.0,
        )

    # Matrices de temps avant / après (sur les nœuds valides uniquement)
    t_before = _shortest_path_times(g, ue_before.travel_times, valid_nodes, valid_nodes)
    t_after = _shortest_path_times(g, ue_after.travel_times, valid_nodes, valid_nodes)

    # On replie sur la grille (n_zones, n_zones) via valid_idx
    n_zones = len(zones)
    full_t_before = np.full((n_zones, n_zones), np.nan, dtype=float)
    full_t_after = np.full((n_zones, n_zones), np.nan, dtype=float)
    valid_arr = np.array(valid_idx, dtype=int)
    rows = valid_arr[:, None]
    cols = valid_arr[None, :]
    full_t_before[rows, cols] = t_before
    full_t_after[rows, cols] = t_after

    # Masques : paires avec demande, t_before valide et > min_time_s
    matrix = od.matrix
    valid_mask = (
        (matrix > 0)
        & np.isfinite(full_t_before)
        & np.isfinite(full_t_after)
        & (full_t_before >= min_time_s)
    )

    # Δt/t et multiplicateur
    dt_ratio = np.zeros_like(matrix, dtype=float)
    dt_ratio[valid_mask] = (
        (full_t_after[valid_mask] - full_t_before[valid_mask])
        / full_t_before[valid_mask]
    )
    multiplier = np.ones_like(matrix, dtype=float)
    multiplier[valid_mask] = np.clip(
        1.0 + elasticity * dt_ratio[valid_mask],
        INDUCED_CLIP_MIN, INDUCED_CLIP_MAX,
    )
    new_matrix = np.maximum(matrix * multiplier, 0.0)

    # Stats (pondérées par trips d'origine pour rester comparables)
    n_pairs = int(valid_mask.sum())
    trips_before_sum = float(matrix[valid_mask].sum())
    if trips_before_sum > 0:
        mean_dt = float(
            (matrix[valid_mask] * dt_ratio[valid_mask]).sum() / trips_before_sum
        )
        trip_delta_ratio = multiplier[valid_mask] - 1.0
        mean_trip = float(
            (matrix[valid_mask] * trip_delta_ratio).sum() / trips_before_sum
        )
    else:
        mean_dt = 0.0
        mean_trip = 0.0
    total_before = float(matrix.sum())
    total_after = float(new_matrix.sum())
    stats = InducedDemandStats(
        elasticity=elasticity,
        n_pairs_affected=n_pairs,
        mean_time_delta_pct=mean_dt * 100.0,
        mean_trip_delta_pct=mean_trip * 100.0,
        total_trips_before=total_before,
        total_trips_after=total_after,
        induced_trips=total_after - total_before,
        induced_share=(total_after - total_before) / total_before
        if total_before > 0 else 0.0,
    )

    new_od = ODMatrix(
        matrix=new_matrix,
        zone_ids=list(zones),
        zone_to_node=dict(zone_to_node),
        hour=od.hour,
        scenario=f"{od.scenario}+induced(ε={elasticity:+.2f})",
    )
    return new_od, stats
