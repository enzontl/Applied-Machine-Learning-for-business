"""Projection d'une OD baseline vers un horizon 5-10 ans.

Pipeline :
1. Prédire les émissions/attractions **actuelles** par commune (modèle ML
   sur features actuelles).
2. Construire les features **projetées** en remplaçant population et emplois
   par les valeurs OMPHALE à l'horizon H. Prédire les émissions/attractions
   futures.
3. Calculer un ratio de croissance par commune :
       g_em[c]  = em_future[c]  / em_now[c]
       g_att[c] = att_future[c] / att_now[c]
4. Construire les marges cibles par zone IRIS (chaque IRIS hérite des
   ratios de sa commune).
5. Furness / IPF (Iterative Proportional Fitting) : ajuster la matrice OD
   pour matcher simultanément les marges émission et attraction futures.

IPF est la méthode de référence en planning transport (Beckmann, Wilson) :
elle préserve la structure spatiale de l'OD baseline tout en alignant les
totaux. Convergence garantie en quelques itérations sur ce volume.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from urban_optimizer.demand.od_matrix import ODMatrix
from urban_optimizer.utils.logging import get_logger

from .data import TrainingDataset
from .model import FlowForecastModel

logger = get_logger(__name__)


# Plancher pour les ratios (évite la division par 0)
_GROWTH_FLOOR = 0.5
_GROWTH_CEIL = 3.0


@dataclass
class ProjectionStats:
    horizon_years: int
    total_trips_before: float
    total_trips_after: float
    growth_pct: float
    n_communes_projected: int
    mean_growth_em: float
    mean_growth_att: float
    ipf_iterations: int
    ipf_max_residual: float

    def summary(self) -> str:
        return (
            f"projection H+{self.horizon_years} : "
            f"{self.total_trips_before:,.0f} → {self.total_trips_after:,.0f} "
            f"({self.growth_pct:+.1f}%), "
            f"{self.n_communes_projected} communes (g_em={self.mean_growth_em:.2f}, "
            f"g_att={self.mean_growth_att:.2f}), "
            f"IPF {self.ipf_iterations} iter (residu={self.ipf_max_residual:.1e})"
        )


def _ipf(
    matrix: np.ndarray,
    target_row: np.ndarray,
    target_col: np.ndarray,
    *,
    max_iter: int = 50,
    tol: float = 1e-3,
) -> tuple[np.ndarray, int, float]:
    """Iterative Proportional Fitting : ajuste matrix pour matcher les marges.

    Préserve les zéros de l'entrée (cellule = 0 reste 0). Convergence :
    max|row_sum - target_row| / sum(target_row) < tol.
    """
    M = matrix.astype(np.float64).copy()
    tr = target_row.astype(np.float64)
    tc = target_col.astype(np.float64)
    # Marges nulles → cellules associées forcées à 0 (cohérence)
    if (tr.sum() == 0) or (tc.sum() == 0):
        return np.zeros_like(M), 0, 0.0
    # Re-balance : les marges doivent avoir la même somme
    total = 0.5 * (tr.sum() + tc.sum())
    tr *= total / max(tr.sum(), 1e-12)
    tc *= total / max(tc.sum(), 1e-12)

    last_res = float("inf")
    for it in range(1, max_iter + 1):
        row_sum = M.sum(axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            scale_r = np.where(row_sum > 1e-12, tr / np.where(row_sum > 0, row_sum, 1), 1.0)
        M *= scale_r[:, None]
        col_sum = M.sum(axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            scale_c = np.where(col_sum > 1e-12, tc / np.where(col_sum > 0, col_sum, 1), 1.0)
        M *= scale_c[None, :]

        residual = max(
            np.abs(M.sum(axis=1) - tr).max() / max(tr.sum(), 1.0),
            np.abs(M.sum(axis=0) - tc).max() / max(tc.sum(), 1.0),
        )
        if residual < tol:
            return M, it, residual
        last_res = residual
    return M, max_iter, last_res


def project_od_future(
    od: ODMatrix,
    model: FlowForecastModel,
    features_now: pd.DataFrame,           # index = commune_code
    pop_jobs_proj: pd.DataFrame,          # cols : commune_code, pop_proj, emploi_proj
    iris_to_commune: dict[str, str],      # zone_id IRIS → code commune
    *,
    horizon_years: int = 10,
    fallback_annual_growth: float = 0.01,
) -> tuple[ODMatrix, ProjectionStats]:
    """Renvoie une OD future obtenue par IPF sur marges prédites par le modèle.

    Args:
        od: OD baseline (zones IRIS).
        model: FlowForecastModel entraîné.
        features_now: features par commune (telles que dans le training).
        pop_jobs_proj: projections OMPHALE — colonnes ``commune_code``,
            ``pop_proj``, ``emploi_proj``. Si vide → fallback ×(1+g)^H.
        iris_to_commune: mapping IRIS → commune INSEE (5 premiers chars du
            code IRIS en général).
        horizon_years: horizon de projection (5 ou 10 typiquement).
    """
    # ── Étape 1 : émissions / attractions actuelles
    pred_now = model.predict(features_now)
    em_now = pred_now["emission_pred"]
    at_now = pred_now["attraction_pred"]

    # ── Étape 2 : features projetées (remplacer pop/jobs)
    feats_future = features_now.copy()
    if not pop_jobs_proj.empty:
        proj = pop_jobs_proj.set_index("commune_code")
        common = feats_future.index.intersection(proj.index)
        feats_future.loc[common, "population"] = proj.loc[common, "pop_proj"].astype(float)
        feats_future.loc[common, "jobs"] = proj.loc[common, "emploi_proj"].astype(float)
        feats_future["job_balance"] = (
            feats_future["jobs"] / np.maximum(feats_future["population"], 1.0)
        )
        logger.info(
            f"OMPHALE : {len(common)}/{len(feats_future)} communes projetées"
        )
    else:
        # Fallback : croissance uniforme +g %/an
        factor = (1.0 + fallback_annual_growth) ** horizon_years
        feats_future["population"] *= factor
        feats_future["jobs"] *= factor
        logger.info(
            f"Fallback projection ×{factor:.3f} sur pop et jobs"
        )

    pred_future = model.predict(feats_future)
    em_fut = pred_future["emission_pred"]
    at_fut = pred_future["attraction_pred"]

    # ── Étape 3 : ratios de croissance par commune (clip pour stabilité)
    g_em = np.clip(em_fut / np.maximum(em_now, 1.0), _GROWTH_FLOOR, _GROWTH_CEIL)
    g_at = np.clip(at_fut / np.maximum(at_now, 1.0), _GROWTH_FLOOR, _GROWTH_CEIL)

    # ── Étape 4 : marges cibles par zone IRIS
    zones = od.zone_ids
    n_z = len(zones)
    em_base = od.matrix.sum(axis=1)  # émissions baseline par IRIS
    at_base = od.matrix.sum(axis=0)  # attractions baseline par IRIS
    target_em = em_base.copy()
    target_at = at_base.copy()
    n_projected = 0
    for i, z in enumerate(zones):
        commune = iris_to_commune.get(str(z))
        if commune in g_em.index:
            target_em[i] = em_base[i] * float(g_em.loc[commune])
            target_at[i] = at_base[i] * float(g_at.loc[commune])
            n_projected += 1

    # ── Étape 5 : IPF
    new_matrix, it, residual = _ipf(od.matrix, target_em, target_at)

    new_od = ODMatrix(
        matrix=new_matrix,
        zone_ids=list(zones),
        zone_to_node=dict(od.zone_to_node),
        hour=od.hour,
        scenario=f"{od.scenario}+H{horizon_years}",
    )
    total_before = float(od.matrix.sum())
    total_after = float(new_matrix.sum())
    stats = ProjectionStats(
        horizon_years=horizon_years,
        total_trips_before=total_before,
        total_trips_after=total_after,
        growth_pct=(total_after - total_before) / max(total_before, 1.0) * 100,
        n_communes_projected=n_projected,
        mean_growth_em=float(g_em.mean()),
        mean_growth_att=float(g_at.mean()),
        ipf_iterations=it,
        ipf_max_residual=residual,
    )
    return new_od, stats
