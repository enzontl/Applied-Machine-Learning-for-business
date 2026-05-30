"""Intégration forecast ML + budget OFGL dans le pipeline API.

Centralise le chargement (cached) du modèle, des features par commune et
des comptes OFGL, et fournit deux helpers :

- ``project_od_if_eligible`` : projette l'OD à l'horizon si la ville est
  éligible (IDF avec modèle entraîné). Sinon retourne l'OD inchangée.
- ``compute_budget`` : calcule le budget voirie cumulé sur l'horizon, soit
  via OFGL (IDF), soit via heuristique pop × per-capita (hors IDF).

Le cache module-level évite de relire les CSV à chaque requête.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path

import pandas as pd

from urban_optimizer.budget import (
    DEFAULT_VOIRIE_SHARE,
    FALLBACK_PER_CAPITA_EUR,
    get_city_budget,
)
from urban_optimizer.demand.od_matrix import ODMatrix
from urban_optimizer.forecast import (
    FlowForecastModel,
    build_training_dataset,
    load_commune_centroids,
    load_emp_lt_by_commune,
    load_mobpro_aggregated,
    load_pop_by_commune,
    project_od_future,
)

logger = logging.getLogger(__name__)

IDF_DEPT_PREFIXES = ("75", "92", "93", "94")


@lru_cache(maxsize=1)
def _load_model(model_path: str) -> FlowForecastModel | None:
    p = Path(model_path)
    if not (p / "meta.json").exists():
        logger.warning(f"Modèle forecast absent : {p} → projection désactivée")
        return None
    logger.info(f"Chargement du modèle forecast : {p}")
    return FlowForecastModel.load(p)


@lru_cache(maxsize=1)
def _load_idf_features() -> tuple[pd.DataFrame, pd.DataFrame] | None:
    """(features_par_commune, pop_jobs) ou None si données manquantes."""
    try:
        pop = load_pop_by_commune(dept_codes=list(IDF_DEPT_PREFIXES))
        jobs = load_emp_lt_by_commune(dept_codes=list(IDF_DEPT_PREFIXES))
        centroids = load_commune_centroids(dept_codes=list(IDF_DEPT_PREFIXES))
    except FileNotFoundError as exc:
        logger.warning(f"Features IDF indisponibles : {exc}")
        return None
    pop_jobs = (
        pop.merge(jobs, on="commune_code", how="inner")
           .pipe(lambda d: d[d["commune_code"].isin(centroids)])
           .reset_index(drop=True)
    )
    if pop_jobs.empty:
        return None
    mobpro = load_mobpro_aggregated(keep_communes=set(pop_jobs["commune_code"]))
    import numpy as np
    xy = np.array([centroids[c] for c in pop_jobs["commune_code"]])
    pops = pop_jobs["population"].values
    center = (
        float((xy[:, 0] * pops).sum() / pops.sum()),
        float((xy[:, 1] * pops).sum() / pops.sum()),
    )
    dataset = build_training_dataset(
        mobpro, pop_jobs, centroids, city_center_xy=center,
    )
    return dataset.features, pop_jobs


def _is_idf(insee_code: str | None) -> bool:
    return bool(insee_code) and str(insee_code).startswith(IDF_DEPT_PREFIXES)


def project_od_if_eligible(
    od: ODMatrix,
    *,
    insee_code: str | None,
    horizon_years: int,
    model_path: str,
) -> tuple[ODMatrix, dict | None]:
    """Si éligible (IDF + horizon>0 + modèle dispo) : projette à H+H.

    Retourne ``(od_potentiellement_projetée, payload_stats|None)``.
    """
    if horizon_years <= 0 or not _is_idf(insee_code):
        return od, None
    model = _load_model(model_path)
    feats_pop = _load_idf_features()
    if model is None or feats_pop is None:
        return od, None
    features, pop_jobs = feats_pop

    # OMPHALE fallback : +1 %/an sur pop et emplois
    growth_pop = (1.01) ** horizon_years
    growth_jobs = (1.008) ** horizon_years
    omphale = pd.DataFrame({
        "commune_code": pop_jobs["commune_code"],
        "pop_proj": pop_jobs["population"] * growth_pop,
        "emploi_proj": pop_jobs["jobs"] * growth_jobs,
    })

    # Mapping zone → commune : faute de shapefile IRIS, on rattache toutes les
    # zones à la commune sélectionnée (suffit car les croissances sont quasi
    # uniformes ; pour un vrai mapping il faudrait un spatial-join).
    iris_to_commune = {z: str(insee_code) for z in od.zone_ids}

    new_od, stats = project_od_future(
        od, model, features, omphale, iris_to_commune,
        horizon_years=horizon_years,
    )
    payload = {
        "horizon_years": stats.horizon_years,
        "total_trips_before": stats.total_trips_before,
        "total_trips_after": stats.total_trips_after,
        "growth_pct": stats.growth_pct,
        "n_communes_projected": stats.n_communes_projected,
        "ipf_iterations": stats.ipf_iterations,
    }
    logger.info(f"Projection OD : {stats.summary()}")
    return new_od, payload


def compute_budget(
    *,
    insee_code: str | None,
    population: float | None,
    horizon_years: int,
    voirie_share: float = DEFAULT_VOIRIE_SHARE,
) -> dict:
    """Calcule le budget voirie horizon, via OFGL ou heuristique pop.

    Retourne payload : ``{ source, total_eur, annual_voirie_eur, horizon_years, ... }``.
    """
    horizon_for_budget = max(1, horizon_years)  # même si projection=0
    if _is_idf(insee_code):
        try:
            total, bdgs = get_city_budget(
                [str(insee_code)], horizon_years=horizon_for_budget,
                voirie_share=voirie_share,
            )
            if bdgs:
                b = bdgs[0]
                return {
                    "source": "OFGL",
                    "total_eur": float(total),
                    "annual_capex_eur": float(b.annual_capex_eur),
                    "annual_voirie_eur": float(b.annual_voirie_eur),
                    "voirie_share": float(voirie_share),
                    "horizon_years": int(horizon_for_budget),
                    "commune_code": b.commune_code,
                }
        except FileNotFoundError:
            logger.warning("OFGL CSV manquant → fallback heuristique")
    # Fallback : pop × per-capita
    pop = float(population or 50_000)
    annual_voirie = pop * FALLBACK_PER_CAPITA_EUR
    return {
        "source": "heuristic_per_capita",
        "total_eur": annual_voirie * horizon_for_budget,
        "annual_capex_eur": annual_voirie / voirie_share,
        "annual_voirie_eur": annual_voirie,
        "voirie_share": float(voirie_share),
        "horizon_years": int(horizon_for_budget),
        "commune_code": str(insee_code) if insee_code else None,
    }
