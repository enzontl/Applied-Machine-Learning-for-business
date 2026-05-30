"""Prévision de la demande à horizon 5-10 ans.

Le module charge et entraîne un modèle cross-section sur des données MOBPRO
INSEE pour prédire les **émissions** et **attractions** de trajets voiture
par commune à partir d’indicateurs socio-économiques.

L’architecture retenue est volontairement simple :
- **émissions** : régresseur proportionnel à la population ;
- **attractions** : régression linéaire standardisée sur population + emplois.

À l’horizon H (5 ou 10 ans), on injecte les projections INSEE OMPHALE
(population et emplois par commune), puis on projette l’OD baseline via IPF.

Limites assumées :
- MOBPRO fournit un snapshot cross-section, pas une série temporelle ;
- la prédiction H+10 est une extrapolation paramétrique ;
- les changements modaux futurs ne sont pas modélisés.
"""

from .data import (
    COMMUNES_CENTROIDS_CSV,
    INSEE_EMP_LT_CSV,
    INSEE_POP_IRIS_CSV,
    MOBPRO_DEFAULT_CSV,
    OMPHALE_DEFAULT_CSV,
    build_training_dataset,
    generate_synthetic_mobpro,
    load_commune_centroids,
    load_emp_lt_by_commune,
    load_mobpro_aggregated,
    load_omphale_projections,
    load_pop_by_commune,
)
from .model import (
    FlowForecastModel,
    ModelMetrics,
    train_forecast_model,
)
from .projection import (
    ProjectionStats,
    project_od_future,
)

__all__ = [
    "COMMUNES_CENTROIDS_CSV",
    "INSEE_EMP_LT_CSV",
    "INSEE_POP_IRIS_CSV",
    "MOBPRO_DEFAULT_CSV",
    "OMPHALE_DEFAULT_CSV",
    "build_training_dataset",
    "generate_synthetic_mobpro",
    "load_commune_centroids",
    "load_emp_lt_by_commune",
    "load_mobpro_aggregated",
    "load_omphale_projections",
    "load_pop_by_commune",
    "FlowForecastModel",
    "ModelMetrics",
    "train_forecast_model",
    "ProjectionStats",
    "project_od_future",
]
