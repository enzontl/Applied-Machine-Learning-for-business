"""Stress-test du modèle de prévision de flux.

Trois axes :
1. **Baselines** — moyenne simple et OLS avec intercept sur les mêmes features.
2. **Validation spatiale** — leave-one-département-out pour mesurer la
   généralisation réelle hors autocorrélation locale.
3. **Stabilité** — multi-seed CV pour estimer la variance du modèle.

Plus un *sanity-check* sur les prédictions (top-10 communes).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold

from urban_optimizer.utils.logging import get_logger

from .data import TrainingDataset
from .model import _build_pipeline

logger = get_logger(__name__)


@dataclass
class StressTestReport:
    """Rapport consolidé d'un stress-test sur une cible (emission/attraction).

    Toutes les MAE sont évaluées **en leave-one-département-out** spatial pour
    une comparaison juste de la généralisation entre dépts.

    Alternatives comparées :
    - ``mean`` : prédire la moyenne (variance intrinsèque).
    - ``ols_intercept`` : régression linéaire avec intercept + scaling, sur les
      mêmes features que le modèle. C'est l'alternative naturelle au choix
      d'architecture.
    - ``model`` : le modèle final (`PopProportionalRegressor` pour emission,
      `LinearRegression` standardisée pour attraction — cf. model.py).
    """

    target: str
    n_samples: int
    mean_spatial_mae: float
    ols_intercept_spatial_mae: float
    model_spatial_mae: float
    model_spatial_r2: float
    model_random_mae_mean: float
    model_random_mae_std: float
    model_random_r2: float
    spatial_folds: dict[str, tuple[float, float]] = field(default_factory=dict)
    top10_table: pd.DataFrame = field(default_factory=pd.DataFrame)

    @property
    def gain_vs_mean_pct(self) -> float:
        return _gain(self.mean_spatial_mae, self.model_spatial_mae)

    @property
    def gain_vs_ols_pct(self) -> float:
        return _gain(self.ols_intercept_spatial_mae, self.model_spatial_mae)

    def summary(self) -> str:
        return (
            f"\n=== [{self.target}] stress-test (n={self.n_samples}, "
            f"leave-one-dept-out) ===\n"
            f"  Baseline mean        : MAE = {self.mean_spatial_mae:8.1f}\n"
            f"  Baseline OLS+inter.  : MAE = {self.ols_intercept_spatial_mae:8.1f}\n"
            f"  Modèle (final)       : MAE = {self.model_spatial_mae:8.1f}, "
            f"R² = {self.model_spatial_r2:.3f}\n"
            f"  Modèle random CV     : MAE = {self.model_random_mae_mean:8.1f} "
            f"± {self.model_random_mae_std:.1f}, R² = {self.model_random_r2:.3f}\n"
            f"  Gain vs mean         : {self.gain_vs_mean_pct:+.1f}%\n"
            f"  Gain vs OLS+intercept: {self.gain_vs_ols_pct:+.1f}%"
        )


def _gain(baseline_mae: float, model_mae: float) -> float:
    """Pourcentage de gain en MAE (positif si modèle bat baseline)."""
    if baseline_mae <= 0:
        return 0.0
    return (baseline_mae - model_mae) / baseline_mae * 100


def _train_predict(X_tr, y_tr, X_te, *, kind: str):
    p = _build_pipeline(kind).fit(X_tr, y_tr)
    return np.maximum(p.predict(X_te), 0.0)


def _baseline_mean_spatial(y: np.ndarray, fold_ids: np.ndarray) -> float:
    """Prédit la moyenne du train sur chaque fold spatial."""
    preds = np.zeros(len(y))
    for fold in sorted(set(fold_ids)):
        te = fold_ids == fold
        tr = ~te
        if te.sum() < 2 or tr.sum() < 5:
            continue
        preds[te] = y[tr].mean()
    return float(mean_absolute_error(y, preds))


def _baseline_ols_intercept_spatial(
    X: np.ndarray, y: np.ndarray, fold_ids: np.ndarray,
) -> float:
    """OLS + intercept + scaling sur chaque fold spatial — alternative
    naturelle à comparer au modèle `PopProportionalRegressor` (émission) ou
    à la régression linéaire standardisée (attraction)."""
    from sklearn.linear_model import LinearRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    preds = np.zeros(len(y))
    for fold in sorted(set(fold_ids)):
        te = fold_ids == fold
        tr = ~te
        if te.sum() < 2 or tr.sum() < 5:
            continue
        p = Pipeline([("s", StandardScaler()), ("l", LinearRegression())]).fit(X[tr], y[tr])
        preds[te] = np.maximum(p.predict(X[te]), 0.0)
    return float(mean_absolute_error(y, preds))


def _multi_seed_cv(X, y, *, kind: str, k=5, seeds=(0, 1, 2, 3, 4)):
    maes, r2s = [], []
    for seed in seeds:
        preds = np.zeros(len(y))
        for tr, te in KFold(k, shuffle=True, random_state=seed).split(X):
            preds[te] = _train_predict(X[tr], y[tr], X[te], kind=kind)
        maes.append(mean_absolute_error(y, preds))
        r2s.append(r2_score(y, preds) if np.var(y) > 0 else 0.0)
    return float(np.mean(maes)), float(np.std(maes)), float(np.mean(r2s))


def _spatial_cv(
    X, y, fold_ids: np.ndarray, *, kind: str,
) -> tuple[float, float, dict[str, tuple[float, float]]]:
    """Leave-one-group-out (typiquement par département)."""
    preds = np.zeros(len(y))
    fold_metrics = {}
    for fold in sorted(set(fold_ids)):
        mask_te = fold_ids == fold
        mask_tr = ~mask_te
        if mask_te.sum() < 2 or mask_tr.sum() < 5:
            continue
        preds[mask_te] = _train_predict(
            X[mask_tr], y[mask_tr], X[mask_te], kind=kind,
        )
        fold_mae = mean_absolute_error(y[mask_te], preds[mask_te])
        fold_r2 = (
            r2_score(y[mask_te], preds[mask_te])
            if np.var(y[mask_te]) > 0 else float("nan")
        )
        fold_metrics[str(fold)] = (float(fold_mae), float(fold_r2))
    overall_mae = mean_absolute_error(y, preds)
    overall_r2 = r2_score(y, preds) if np.var(y) > 0 else 0.0
    return float(overall_mae), float(overall_r2), fold_metrics


def stress_test_dataset(
    dataset: TrainingDataset,
    *,
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4),
    spatial_group_fn=None,
    top_n_preview: int = 10,
    **_legacy_kwargs,
) -> dict[str, StressTestReport]:
    """Évalue baselines + modèle final (random + spatial CV) sur les 2 cibles.

    ``spatial_group_fn``: fonction ``commune_code -> group_id`` (typiquement le
    code département = 2 premiers chars). Si None : groupe = 2 premiers chars.
    """
    commune_codes = list(dataset.features.index)
    if spatial_group_fn is None:
        spatial_group_fn = lambda c: str(c)[:2]
    fold_ids = np.array([spatial_group_fn(c) for c in commune_codes])

    reports: dict[str, StressTestReport] = {}
    for target_name, y_series in [
        ("emission", dataset.y_emission), ("attraction", dataset.y_attraction),
    ]:
        # Features par cible (cf. TrainingDataset)
        X = dataset.X(target_name).values
        feat_names = (
            dataset.emission_features if target_name == "emission"
            else dataset.attraction_features
        )
        y = y_series.values

        # 1. Baselines en spatial CV (même protocole que le modèle)
        mean_mae = _baseline_mean_spatial(y, fold_ids)
        ols_mae = _baseline_ols_intercept_spatial(X, y, fold_ids)

        # 2. Modèle final — spatial CV (leave-one-département-out)
        spa_mae, spa_r2, fold_dict = _spatial_cv(X, y, fold_ids, kind=target_name)

        # 3. Modèle final — random CV multi-seed (diagnostic complémentaire)
        rnd_mae, rnd_std, rnd_r2 = _multi_seed_cv(
            X, y, kind=target_name, k=5, seeds=seeds,
        )

        # 4. Sanity-check : top-N observés vs prédits in-sample
        pipe = _build_pipeline(target_name).fit(X, y)
        preds_all = np.maximum(pipe.predict(X), 0.0)
        order = np.argsort(-y)[:top_n_preview]
        top_df = pd.DataFrame({
            "commune": [commune_codes[i] for i in order],
            "observed": y[order].astype(int),
            "predicted": preds_all[order].astype(int),
            "abs_err": np.abs(y[order] - preds_all[order]).astype(int),
            "rel_err_%": np.where(
                y[order] > 0,
                (preds_all[order] - y[order]) / y[order] * 100,
                0.0,
            ).round(1),
        })

        reports[target_name] = StressTestReport(
            target=target_name,
            n_samples=len(y),
            mean_spatial_mae=mean_mae,
            ols_intercept_spatial_mae=ols_mae,
            model_spatial_mae=spa_mae,
            model_spatial_r2=spa_r2,
            model_random_mae_mean=rnd_mae,
            model_random_mae_std=rnd_std,
            model_random_r2=rnd_r2,
            spatial_folds=fold_dict,
            top10_table=top_df,
        )
    return reports
