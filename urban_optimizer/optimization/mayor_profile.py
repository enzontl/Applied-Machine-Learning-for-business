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
    rend la composante plus pénalisante (donc plus prioritaire à réduire pour
    les coûts, ou à augmenter pour l'accessibilité).
    """

    name: str
    label: str
    w_time: float = 1.0           # coût du temps (VHT × valeur du temps)
    w_fuel: float = 1.0           # coût du carburant
    w_co2: float = 1.0            # externalité CO2
    w_construction: float = 1.0   # frein à la construction (× sur les coûts CAPEX)
    w_accessibility: float = 1.0  # valeur attribuée au gain d'accessibilité
    w_equity: float = 1.0         # pénalisation de l'inégalité (Gini)


ECOLO = MayorProfile(
    name="ecolo",
    label="🌿 Écolo — minimiser CO2 et carburant",
    w_time=0.8, w_fuel=2.0, w_co2=3.0, w_construction=1.5,
    w_accessibility=1.2, w_equity=2.5,         # forte sensibilité à l'équité sociale
)
MOBILITE = MayorProfile(
    name="mobilite",
    label="🚗 Mobilité — minimiser le temps perdu",
    w_time=3.0, w_fuel=1.0, w_co2=0.5, w_construction=0.8,
    w_accessibility=2.5, w_equity=0.8,         # valorise fort l'accès, peu l'équité
)
ECONOMIQUE = MayorProfile(
    name="economique",
    label="💰 Économique — éviter les gros travaux",
    w_time=1.0, w_fuel=1.0, w_co2=0.5, w_construction=2.5,
    w_accessibility=0.7, w_equity=0.7,         # peu disposé à payer pour ces externalités
)
EQUILIBRE = MayorProfile(
    name="equilibre",
    label="⚖️ Équilibré",
    w_time=1.0, w_fuel=1.0, w_co2=1.0, w_construction=1.0,
    w_accessibility=1.0, w_equity=1.0,
)

ALL_PROFILES: tuple[MayorProfile, ...] = (EQUILIBRE, ECOLO, MOBILITE, ECONOMIQUE)
PROFILE_BY_NAME: dict[str, MayorProfile] = {p.name: p for p in ALL_PROFILES}
