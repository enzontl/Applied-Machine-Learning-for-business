"""Profils du décideur : poids relatifs sur les composantes du coût social.

Idée : un même réseau a un coût annuel différent selon ce qu'on valorise.
Un maire écolo pèsera fort le CO2 et le carburant ; un maire mobilité
maximisera l'accessibilité ; un maire économe sera frileux sur la
construction. Chaque profil multiplie le coût unitaire de chaque composante.

Les profils sont volontairement extrêmes pour que les recommandations
varient visiblement entre scénarios.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MayorProfile:
    """Vecteur de poids appliqué au score.

    Tous les poids sont ≥ 0. Un poids de 1.0 = pas d'inflexion. Un poids > 1
    rend la composante plus pénalisante (donc plus prioritaire à réduire).
    """

    name: str
    label: str
    w_time: float = 1.0          # coût du temps (VHT × valeur du temps)
    w_fuel: float = 1.0          # coût du carburant
    w_co2: float = 1.0           # externalité CO2
    w_construction: float = 1.0  # frein à la construction (× sur les coûts CAPEX)


ECOLO = MayorProfile(
    name="ecolo",
    label="🌿 Écolo — minimiser CO2 et carburant",
    w_time=0.8, w_fuel=2.0, w_co2=3.0, w_construction=1.5,
)
MOBILITE = MayorProfile(
    name="mobilite",
    label="🚗 Mobilité — minimiser le temps perdu",
    w_time=3.0, w_fuel=1.0, w_co2=0.5, w_construction=0.8,
)
ECONOMIQUE = MayorProfile(
    name="economique",
    label="💰 Économique — éviter les gros travaux",
    w_time=1.0, w_fuel=1.0, w_co2=0.5, w_construction=2.5,
)
EQUILIBRE = MayorProfile(
    name="equilibre",
    label="⚖️ Équilibré",
    w_time=1.0, w_fuel=1.0, w_co2=1.0, w_construction=1.0,
)

ALL_PROFILES: tuple[MayorProfile, ...] = (EQUILIBRE, ECOLO, MOBILITE, ECONOMIQUE)
PROFILE_BY_NAME: dict[str, MayorProfile] = {p.name: p for p in ALL_PROFILES}
