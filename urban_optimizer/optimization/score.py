"""Score d'une ville : coût annuel total exprimé en €.

Toutes les composantes sont tirées du résultat d'affectation :

- **Temps** : VHT (véh·h à l'heure de pointe) × heures-pointe/an × valeur du temps.
- **Carburant** : VHT × L/h moyen × prix du litre.
- **CO2** : L de carburant × kg CO2/L × prix social du carbone.
- **Accessibilité** (optionnel) : zones joignables en T min → bénéfice social.
- **Équité** (optionnel) : Gini sur accessibilité → pénalité si forte inégalité.

Les poids du profil maire amplifient ou atténuent chaque composante. Plus le
score est *bas*, mieux c'est (c'est un coût social).
"""

from __future__ import annotations

from dataclasses import dataclass

from urban_optimizer.assignment.result import AssignmentResult
from urban_optimizer.diagnosis.accessibility import AccessibilityReport

from .mayor_profile import MayorProfile


# ── Hypothèses économiques (réutilisables, exposables en config) ──────────
ANNUAL_PEAK_HOURS = 220          # ~ jours ouvrés × 1 heure de pointe / jour
VALUE_OF_TIME_EUR_H = 12.0       # valeur du temps social moyen en France
FUEL_LITERS_PER_VEH_HOUR = 4.5   # consommation horaire moyenne en urbain
FUEL_PRICE_EUR_L = 1.85          # prix moyen 2025
KG_CO2_PER_LITER = 2.31          # essence + diesel mix
SOCIAL_COST_CO2_EUR_KG = 0.080   # ~80 €/tonne — coût social du carbone

# Valorisation accessibilité : "1 zone supplémentaire joignable en 15 min" vaut X €/an
# par habitant moyen. On prend une valeur prudente fondée sur la valeur du temps :
# accéder à un emploi/service de plus = ~50 €/an d'utilité par zone par tranche d'habitants.
ACCESSIBILITY_VALUE_EUR_PER_ZONE_YEAR = 50_000.0
# Pénalité d'iniquité : Gini = 0.5 → ~5% de surcoût social ajouté au temps annuel.
EQUITY_PENALTY_RATE = 0.10       # 10% du coût temps par point de Gini complet (=1.0)


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
    total_annual_cost_eur: float        # pondéré par le profil
    # Accessibilité / équité (optionnels — 0 si pas calculé)
    accessibility_mean: float = 0.0     # zones moyennes joignables sous le seuil
    accessibility_gini: float = 0.0     # 0 = parfaite équité
    accessibility_value_eur: float = 0.0  # bénéfice social annuel (= soustrait du coût total)
    equity_penalty_eur: float = 0.0       # pénalité Gini (= ajoutée au coût total)

    @property
    def composite_score(self) -> float:
        """Plus bas = meilleur réseau."""
        return self.total_annual_cost_eur

    def summary(self) -> str:
        s = (
            f"Score [{self.profile_name}] : {self.total_annual_cost_eur:,.0f} €/an\n"
            f"  ↳ temps    : {self.annual_time_cost_eur:,.0f} €\n"
            f"  ↳ essence  : {self.annual_fuel_cost_eur:,.0f} €\n"
            f"  ↳ CO2      : {self.annual_co2_cost_eur:,.0f} € "
            f"({self.annual_co2_kg / 1000:,.1f} t/an)\n"
        )
        if self.accessibility_mean > 0:
            s += (
                f"  ↳ accès    : {self.accessibility_mean:.1f} zones (-{self.accessibility_value_eur:,.0f} €)\n"
                f"  ↳ équité   : Gini={self.accessibility_gini:.2f} (+{self.equity_penalty_eur:,.0f} €)\n"
            )
        return s


def score_network(
    result: AssignmentResult,
    profile: MayorProfile,
    access: AccessibilityReport | None = None,
) -> CityScore:
    """Calcule le coût annuel pondéré du réseau pour un profil donné.

    Args:
        result: résultat d'affectation (UE ou SO).
        profile: profil maire pour pondérer les composantes.
        access: rapport d'accessibilité (optionnel). Si fourni, ajoute
            une composante "bénéfice accessibilité" (soustraite) et une
            "pénalité d'inéquité Gini" (ajoutée).
    """
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

    # Accessibilité (bénéfice — soustrait du coût) et équité (pénalité — ajoutée)
    access_value = 0.0
    equity_penalty = 0.0
    access_mean = 0.0
    access_gini = 0.0
    if access is not None and access.n_zones > 0:
        access_mean = access.mean_reachable
        access_gini = access.gini
        # Bénéfice = nb moyen × valeur × poids du profil
        access_value = access_mean * ACCESSIBILITY_VALUE_EUR_PER_ZONE_YEAR * profile.w_accessibility
        # Pénalité = Gini × taux × coût temps brut × poids du profil
        equity_penalty = access_gini * EQUITY_PENALTY_RATE * time_raw * profile.w_equity

    total = time_w + fuel_w + co2_w - access_value + equity_penalty

    return CityScore(
        profile_name=profile.name,
        vht_peak_h=vht_peak,
        annual_time_cost_eur=time_w,
        annual_fuel_cost_eur=fuel_w,
        annual_co2_cost_eur=co2_w,
        annual_co2_kg=co2_kg,
        total_annual_cost_eur=total,
        accessibility_mean=access_mean,
        accessibility_gini=access_gini,
        accessibility_value_eur=access_value,
        equity_penalty_eur=equity_penalty,
    )
