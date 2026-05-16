"""Dataclass représentant une matrice Origine-Destination."""

from dataclasses import dataclass

import numpy as np


@dataclass
class ODMatrix:
    """Matrice OD horaire prête pour l'affectation.

    Attributes:
        matrix: tableau (n_zones, n_zones) — flux véh/h, diagonale nulle.
        zone_ids: identifiants des zones (alignés sur les lignes/colonnes).
        zone_to_node: zone_id → index du nœud du réseau le plus proche du centroïde.
        hour: heure de référence pour laquelle la matrice est calibrée (0-23).
        scenario: scénario (ex. "weekday", "weekend").
    """

    matrix: np.ndarray
    zone_ids: list
    zone_to_node: dict
    hour: int
    scenario: str

    def __post_init__(self) -> None:
        n = len(self.zone_ids)
        if self.matrix.shape != (n, n):
            raise ValueError(
                f"matrix shape {self.matrix.shape} ≠ (n_zones, n_zones) = ({n}, {n})"
            )
        if np.any(self.matrix < 0):
            raise ValueError("La matrice OD contient des valeurs négatives.")

    @property
    def total_trips(self) -> float:
        return float(self.matrix.sum())

    @property
    def n_zones(self) -> int:
        return len(self.zone_ids)

    def iter_pairs(self):
        """Itère sur les triplets (zone_o, zone_d, trips) avec trips > 0."""
        nz = np.nonzero(self.matrix)
        for i, j in zip(*nz, strict=False):
            yield self.zone_ids[i], self.zone_ids[j], float(self.matrix[i, j])

    def to_node_pairs(self):
        """Itère sur (node_o, node_d, trips) — pour passage direct à l'affectation.

        Les zones non rattachées à un nœud (zone_to_node manquant) sont ignorées.
        """
        for zo, zd, trips in self.iter_pairs():
            no = self.zone_to_node.get(zo)
            nd = self.zone_to_node.get(zd)
            if no is None or nd is None or no == nd:
                continue
            yield no, nd, trips

    def summary(self) -> None:
        print(f"Matrice OD : {self.n_zones} zones, scénario={self.scenario}, h={self.hour}")
        print(f"  Total trips : {self.total_trips:,.0f} véh/h")
        print(f"  Paires non nulles : {int((self.matrix > 0).sum()):,}")
        print(f"  Max OD : {self.matrix.max():.1f} véh/h")
