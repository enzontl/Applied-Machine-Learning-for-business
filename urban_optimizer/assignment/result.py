"""Résultat d'une affectation : flux, temps, VHT, convergence."""

from dataclasses import dataclass

import numpy as np


@dataclass
class AssignmentResult:
    """Sortie standardisée d'un algorithme d'affectation.

    Attributes:
        flows: tableau (n_edges,) — flux véh/h par arc.
        travel_times: tableau (n_edges,) — temps de parcours actuel par arc (s).
        free_flow_times: tableau (n_edges,) — t0 (s), pour les métriques.
        vht: vehicle-hours-traveled total (h).
        iterations: nombre d'itérations effectuées.
        converged: True si le gap est sous la tolérance.
        final_gap: gap relatif final (sans dimension).
        method: étiquette "aon" / "ue" / "so".
    """

    flows: np.ndarray
    travel_times: np.ndarray
    free_flow_times: np.ndarray
    vht: float
    iterations: int
    converged: bool
    final_gap: float
    method: str

    @property
    def total_travel_time_s(self) -> float:
        return float((self.flows * self.travel_times).sum())

    @property
    def total_free_flow_time_s(self) -> float:
        return float((self.flows * self.free_flow_times).sum())

    @property
    def congestion_index(self) -> float:
        """Ratio temps actuel / temps libre, pondéré par les flux."""
        denom = self.total_free_flow_time_s
        return float(self.total_travel_time_s / denom) if denom > 0 else 1.0

    def saturated_mask(self, capacity: np.ndarray, threshold: float = 0.9) -> np.ndarray:
        """Booléen par arc : True si v/c ≥ threshold."""
        return (self.flows / np.maximum(capacity, 1.0)) >= threshold

    def summary(self) -> None:
        print(f"Affectation '{self.method}' — {self.iterations} itérations "
              f"(converged={self.converged}, gap={self.final_gap:.2e})")
        print(f"  VHT total         : {self.vht:,.1f} véh·h")
        print(f"  Indice congestion : ×{self.congestion_index:.3f}")
        print(f"  Flux max sur arc  : {self.flows.max():,.0f} véh/h")
