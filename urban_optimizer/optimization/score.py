"""Score d'une ville : coût annuel total exprimé en €.

Toutes les composantes sont tirées du résultat d'affectation :

- **Temps** : VHT (véh·h à l'heure de pointe) × heures-pointe/an × valeur du temps.
- **Carburant** : VHT × L/h moyen × prix du litre.
- **CO2** : L de carburant × kg CO2/L × prix social du carbone.

Les poids du profil maire amplifient ou atténuent chaque composante. Plus le
score est *bas*, mieux c'est (c'est un coût social).
"""

from __future__ import annotations

from dataclasses import dataclass

from urban_optimizer.assignment.result import AssignmentResult

from .mayor_profile import MayorProfile


# ── Hypothèses économiques (réutilisables, exposables en config) ──────────
ANNUAL_PEAK_HOURS = 220          # ~ jours ouvrés × 1 heure de pointe / jour
VALUE_OF_TIME_EUR_H = 12.0       # valeur du temps social moyen en France
FUEL_LITERS_PER_VEH_HOUR = 4.5   # consommation horaire moyenne en urbain
FUEL_PRICE_EUR_L = 1.85          # prix moyen 2025
KG_CO2_PER_LITER = 2.31          # essence + diesel mix
SOCIAL_COST_CO2_EUR_KG = 0.080   # ~80 €/tonne — coût social du carbone


@dataclass
class CityScore:
    """Décomposition du coût annuel d'un état du réseau (€ par an).

    Tous les chiffres sont annuels et pondérés par le profil maire.
    """

    profile_name: str
    vht_peak_h: float
    annual_time_cost_eur: float
    annual_fuel_cost_eur: float
    annual_co2_cost_eur: float
    annual_co2_kg: float
    total_annual_cost_eur: float       # pondéré par le profil

    @property
    def composite_score(self) -> float:
        """Plus bas = meilleur réseau."""
        return self.total_annual_cost_eur

    def summary(self) -> str:
        return (
            f"Score [{self.profile_name}] : {self.total_annual_cost_eur:,.0f} €/an\n"
            f"  ↳ temps    : {self.annual_time_cost_eur:,.0f} €\n"
            f"  ↳ essence  : {self.annual_fuel_cost_eur:,.0f} €\n"
            f"  ↳ CO2      : {self.annual_co2_cost_eur:,.0f} € "
            f"({self.annual_co2_kg / 1000:,.1f} t/an)"
        )


def score_network(
    result: AssignmentResult,
    profile: MayorProfile,
) -> CityScore:
    """Calcule le coût annuel pondéré du réseau pour un profil donné."""
    vht_peak = result.vht
    annual_vht = vht_peak * ANNUAL_PEAK_HOURS

    time_raw = annual_vht * VALUE_OF_TIME_EUR_H
    fuel_l = annual_vht * FUEL_LITERS_PER_VEH_HOUR
    fuel_raw = fuel_l * FUEL_PRICE_EUR_L
    co2_kg = fuel_l * KG_CO2_PER_LITER
    co2_raw = co2_kg * SOCIAL_COST_CO2_EUR_KG

    time_w = time_raw * profile.w_time
    fuel_w = fuel_raw * profile.w_fuel
    co2_w = co2_raw * profile.w_co2

    return CityScore(
        profile_name=profile.name,
        vht_peak_h=vht_peak,
        annual_time_cost_eur=time_w,
        annual_fuel_cost_eur=fuel_w,
        annual_co2_cost_eur=co2_w,
        annual_co2_kg=co2_kg,
        total_annual_cost_eur=time_w + fuel_w + co2_w,
    )
