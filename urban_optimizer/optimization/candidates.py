"""Génération de candidats d'amélioration du réseau.

Au lieu de tester tous les arcs (intractable : 2k+ × ré-affectation FW), on
restreint aux arcs **les plus contributeurs au délai** issus de la brique 4.
Pour chaque arc retenu, on propose plusieurs scénarios (boosts de capacité,
de vitesse, ou retrait pour la détection de Braess).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from urban_optimizer.assignment.result import AssignmentResult
from urban_optimizer.diagnosis.critical_arcs import rank_by_congestion_delay
from urban_optimizer.network.urban_network import UrbanNetwork

ActionType = Literal["capacity_boost", "speed_boost", "remove"]


@dataclass
class Candidate:
    """Une action atomique sur un arc.

    Attributes:
        edge_id: index dans le graphe igraph.
        action: type d'action.
        magnitude: facteur multiplicatif (capacity_boost / speed_boost) ou 0 pour remove.
        cost_eur: coût estimé (€).
        length_m: longueur de l'arc (m), pour la viz.
        highway: type d'arc, pour la viz et le tarif.
    """

    edge_id: int
    action: ActionType
    magnitude: float
    cost_eur: float
    length_m: float
    highway: str

    @property
    def label(self) -> str:
        if self.action == "capacity_boost":
            return f"+{int((self.magnitude - 1) * 100)}% capacité"
        if self.action == "speed_boost":
            return f"+{int((self.magnitude - 1) * 100)}% vitesse"
        return "retrait"


# Tarifs unitaires par type d'action et type d'arc (€ / m).
# Ordres de grandeur indicatifs — élargissement de chaussée, mise à 2×2 voies,
# resurface qualifiante. À calibrer pour chaque projet réel.
COST_PER_M_CAPACITY = {
    "motorway": 1200, "trunk": 800, "primary": 600, "secondary": 400,
    "tertiary": 250, "residential": 150, "unclassified": 150, "service": 100,
    "motorway_link": 800, "primary_link": 500, "secondary_link": 350, "trunk_link": 700,
}
COST_PER_M_SPEED = {
    "motorway": 400, "trunk": 300, "primary": 250, "secondary": 180,
    "tertiary": 120, "residential": 80, "unclassified": 80, "service": 60,
}
# Le retrait n'est pas gratuit (signalisation, remise en état) mais reste
# très inférieur à une création.
COST_PER_M_REMOVE = 50


def _arc_cost(action: ActionType, highway: str, length_m: float, magnitude: float) -> float:
    if action == "capacity_boost":
        base = COST_PER_M_CAPACITY.get(highway, 200)
        # coût croît avec l'ampleur du boost : 1.2x → 1.0, 1.5x → 2.0, 2.0x → 4.0
        return base * length_m * max(0.0, magnitude - 1.0) * 5.0
    if action == "speed_boost":
        base = COST_PER_M_SPEED.get(highway, 150)
        return base * length_m * max(0.0, magnitude - 1.0) * 5.0
    return COST_PER_M_REMOVE * length_m


def generate_candidates(
    network: UrbanNetwork,
    result: AssignmentResult,
    top_n: int = 30,
    actions: tuple[tuple[ActionType, float], ...] = (
        ("capacity_boost", 1.20),
        ("capacity_boost", 1.50),
        ("speed_boost", 1.10),
        ("remove", 0.0),
    ),
) -> list[Candidate]:
    """Crée la liste des actions à évaluer.

    Args:
        network: réseau.
        result: affectation de référence (typiquement UE actuelle).
        top_n: nombre d'arcs candidats (les plus pénalisants en délai).
        actions: tuples (type d'action, magnitude). Magnitude=0 pour ``remove``.

    Returns:
        Liste des candidats à passer au ranker.
    """
    delay_df = rank_by_congestion_delay(network, result, top_n=top_n)
    lengths = np.asarray(network.graph.es["length_m"], dtype=float)
    highways = list(network.graph.es["highway"])

    candidates: list[Candidate] = []
    for _, row in delay_df.iterrows():
        eid = int(row["edge_id"])
        L = float(lengths[eid])
        hw = highways[eid]
        for action, mag in actions:
            cost = _arc_cost(action, hw, L, mag)
            candidates.append(Candidate(
                edge_id=eid, action=action, magnitude=mag,
                cost_eur=cost, length_m=L, highway=hw,
            ))
    return candidates
