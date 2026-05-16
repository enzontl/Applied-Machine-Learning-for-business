"""Métriques agrégées d'un état d'équilibre du réseau."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from urban_optimizer.assignment.result import AssignmentResult
from urban_optimizer.network.urban_network import UrbanNetwork


@dataclass
class NetworkDiagnosis:
    """Vue d'ensemble du fonctionnement du réseau à un horaire donné.

    Tous les temps sont en secondes, tous les VHT en véhicules·heures.

    Attributes:
        vht: temps total passé par tous les véhicules (h).
        vht_free_flow: ce que ce serait sans congestion (somme des x_a · t0_a).
        congestion_overhead: vht / vht_free_flow − 1, surcoût relatif.
        n_arcs: nombre total d'arcs.
        n_saturated: nombre d'arcs au-dessus du seuil de saturation.
        saturation_threshold: seuil v/c utilisé.
        saturation_rates: tableau (n_arcs,) des v/c.
        mean_saturation: moyenne pondérée par les flux des v/c.
        price_of_anarchy: ratio VHT(UE) / VHT(SO) si SO fourni, sinon None.
    """

    vht: float
    vht_free_flow: float
    congestion_overhead: float
    n_arcs: int
    n_saturated: int
    saturation_threshold: float
    saturation_rates: np.ndarray
    mean_saturation: float
    price_of_anarchy: float | None = None

    def summary(self) -> None:
        print(f"VHT total          : {self.vht:>12,.1f} h")
        print(f"VHT temps libre    : {self.vht_free_flow:>12,.1f} h")
        print(f"Surcoût congestion : ×{1 + self.congestion_overhead:.3f} "
              f"(+{self.congestion_overhead * 100:.1f}%)")
        print(f"Arcs saturés       : {self.n_saturated:>5} / {self.n_arcs} "
              f"({self.n_saturated / self.n_arcs * 100:.1f}%, seuil v/c≥{self.saturation_threshold})")
        print(f"Saturation moyenne : {self.mean_saturation:.3f} (pondérée par les flux)")
        if self.price_of_anarchy is not None:
            print(f"Prix de l'anarchie : ×{self.price_of_anarchy:.4f}")


def diagnose(
    network: UrbanNetwork,
    result: AssignmentResult,
    so_result: AssignmentResult | None = None,
    saturation_threshold: float = 0.9,
) -> NetworkDiagnosis:
    """Calcule les métriques agrégées de l'affectation.

    Args:
        network: réseau associé.
        result: affectation à diagnostiquer (typiquement UE).
        so_result: affectation SO pour calculer le prix de l'anarchie (optionnel).
        saturation_threshold: seuil v/c pour qualifier un arc de saturé.

    Returns:
        NetworkDiagnosis.
    """
    flows = result.flows
    times = result.travel_times
    t0 = result.free_flow_times

    capacity = np.asarray(network.graph.es["capacity"], dtype=float)
    sat = flows / np.maximum(capacity, 1.0)

    vht_h = float((flows * times).sum() / 3600.0)
    vht_ff = float((flows * t0).sum() / 3600.0)
    overhead = (vht_h / vht_ff - 1.0) if vht_ff > 0 else 0.0

    # Saturation moyenne pondérée par les flux : reflète l'expérience-véhicule
    total_flow = flows.sum()
    mean_sat = float((sat * flows).sum() / total_flow) if total_flow > 0 else 0.0

    poa: float | None = None
    if so_result is not None and so_result.vht > 0:
        poa = float(result.vht / so_result.vht)

    return NetworkDiagnosis(
        vht=vht_h,
        vht_free_flow=vht_ff,
        congestion_overhead=overhead,
        n_arcs=len(flows),
        n_saturated=int((sat >= saturation_threshold).sum()),
        saturation_threshold=saturation_threshold,
        saturation_rates=sat,
        mean_saturation=mean_sat,
        price_of_anarchy=poa,
    )
