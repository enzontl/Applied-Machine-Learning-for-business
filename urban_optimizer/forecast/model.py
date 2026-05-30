"""Modèle de prévision des flux.

Le module met en œuvre deux têtes distinctes :

- **émissions** : régresseur proportionnel à la population, robuste et simple ;
- **attractions** : pipeline `StandardScaler → LinearRegression`.

Le choix d’architecture est guidé par le stress-test spatial du module
`forecast.evaluation` : sur les données utilisées pour le projet, un modèle
linéaire simple généralise mieux que des variantes plus complexes.

Les pipelines sont persistés par pickle avec leurs métadonnées.
"""

from __future__ import annotations

import json
import pickle
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

# pickle est requis pour sérialiser les Pipeline sklearn (StandardScaler.mean_ etc.)

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from urban_optimizer.utils.logging import get_logger

from .data import TrainingDataset

logger = get_logger(__name__)


class PopProportionalRegressor(BaseEstimator, RegressorMixin):
    """Régresseur proportionnel : y_pred = (Σy / Σpop) × pop.

    Équivalent à WLS avec poids = 1/pop. Plus stable que OLS sans intercept
    en présence d'outliers (Paris 75056 fausse l'OLS sinon).
    Suppose ``X[:, 0]`` = population.
    """

    def fit(self, X, y, sample_weight=None):
        pop = np.asarray(X)[:, 0]
        self.coef_ = np.array([float(np.sum(y) / max(np.sum(pop), 1e-9))])
        self.intercept_ = 0.0
        return self

    def predict(self, X):
        return float(self.coef_[0]) * np.asarray(X)[:, 0]


@dataclass
class ModelMetrics:
    target: Literal["emission", "attraction"]
    n_samples: int
    mae: float            # cross-val MAE
    r2: float             # cross-val R²
    mae_train: float      # in-sample MAE
    r2_train: float       # in-sample R²
    coefficients: dict[str, float] = field(default_factory=dict)

    def summary(self) -> str:
        top3 = sorted(self.coefficients.items(), key=lambda x: -abs(x[1]))[:3]
        top3_str = ", ".join(f"{k}={v:+.2f}" for k, v in top3)
        return (
            f"[{self.target}] n={self.n_samples}, "
            f"CV MAE={self.mae:.1f}, CV R²={self.r2:.3f} "
            f"(train: MAE={self.mae_train:.1f}, R²={self.r2_train:.3f}) "
            f"| top coefs: {top3_str}"
        )


def _build_pipeline(kind: str) -> Pipeline:
    """Construit le pipeline adapté à la cible.

    - ``"emission"`` : `PopProportionalRegressor` (y = α·pop)
    - ``"attraction"`` : `StandardScaler` → `LinearRegression`
    """
    if kind == "emission":
        return Pipeline([("linear", PopProportionalRegressor())])
    if kind == "attraction":
        return Pipeline([
            ("scale", StandardScaler()),
            ("linear", LinearRegression(fit_intercept=True)),
        ])
    raise ValueError(f"kind inconnu : {kind!r}")


@dataclass
class FlowForecastModel:
    """Paire de pipelines sklearn et métadonnées associées.

    Les deux têtes peuvent utiliser des features différentes : l’émission
    repose sur la population, tandis que l’attraction utilise population
    et emplois.
    """

    emission_pipeline: Pipeline
    attraction_pipeline: Pipeline
    emission_features: list[str]
    attraction_features: list[str]
    metrics_emission: ModelMetrics
    metrics_attraction: ModelMetrics
    training_target_total: dict[str, float]

    @property
    def feature_names(self) -> list[str]:
        """Ensemble des features nécessaires en entrée."""
        return list(dict.fromkeys(self.emission_features + self.attraction_features))

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        X_em = features[self.emission_features].values
        X_at = features[self.attraction_features].values
        return pd.DataFrame({
            "emission_pred": np.maximum(self.emission_pipeline.predict(X_em), 0.0),
            "attraction_pred": np.maximum(self.attraction_pipeline.predict(X_at), 0.0),
        }, index=features.index)

    def save(self, path: Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        with (path / "emission.pkl").open("wb") as f:
            pickle.dump(self.emission_pipeline, f)
        with (path / "attraction.pkl").open("wb") as f:
            pickle.dump(self.attraction_pipeline, f)
        (path / "meta.json").write_text(json.dumps({
            "emission_features": self.emission_features,
            "attraction_features": self.attraction_features,
            "metrics_emission": asdict(self.metrics_emission),
            "metrics_attraction": asdict(self.metrics_attraction),
            "training_target_total": self.training_target_total,
        }, indent=2))
        logger.info(f"Modèle persisté dans {path}")

    @classmethod
    def load(cls, path: Path) -> "FlowForecastModel":
        path = Path(path)
        with (path / "emission.pkl").open("rb") as f:
            em = pickle.load(f)
        with (path / "attraction.pkl").open("rb") as f:
            at = pickle.load(f)
        meta = json.loads((path / "meta.json").read_text())
        return cls(
            emission_pipeline=em,
            attraction_pipeline=at,
            emission_features=meta["emission_features"],
            attraction_features=meta["attraction_features"],
            metrics_emission=ModelMetrics(**meta["metrics_emission"]),
            metrics_attraction=ModelMetrics(**meta["metrics_attraction"]),
            training_target_total=meta["training_target_total"],
        )


def _train_one(
    X: np.ndarray,
    y: np.ndarray,
    feat_names: list[str],
    target_name: str,
    *,
    cv_folds: int,
    seed: int,
) -> tuple[Pipeline, ModelMetrics]:
    n = len(y)
    cv_folds = min(cv_folds, n)
    if cv_folds >= 2:
        preds_cv = np.zeros(n)
        for tr, te in KFold(cv_folds, shuffle=True, random_state=seed).split(X):
            p = _build_pipeline(target_name).fit(X[tr], y[tr])
            preds_cv[te] = np.maximum(p.predict(X[te]), 0.0)
        cv_mae = float(mean_absolute_error(y, preds_cv))
        cv_r2 = float(r2_score(y, preds_cv)) if np.var(y) > 0 else 0.0
    else:
        cv_mae, cv_r2 = float("nan"), float("nan")

    final = _build_pipeline(target_name).fit(X, y)
    preds_tr = np.maximum(final.predict(X), 0.0)
    coef_map = dict(zip(feat_names, np.atleast_1d(final.named_steps["linear"].coef_).tolist()))

    metrics = ModelMetrics(
        target=target_name,  # type: ignore[arg-type]
        n_samples=n, mae=cv_mae, r2=cv_r2,
        mae_train=float(mean_absolute_error(y, preds_tr)),
        r2_train=float(r2_score(y, preds_tr)) if np.var(y) > 0 else 0.0,
        coefficients=coef_map,
    )
    logger.info(metrics.summary())
    return final, metrics


def train_forecast_model(
    dataset: TrainingDataset,
    *,
    cv_folds: int = 5,
    seed: int = 0,
    **_legacy_kwargs,   # tolérance pour anciens `alpha=`, `n_estimators=`
) -> FlowForecastModel:
    """Entraîne les deux têtes du modèle de prévision.

    Le choix d’architecture est séparé par cible :
    - **émission** : `PopProportionalRegressor`
    - **attraction** : `LinearRegression` standardisée sur `(population, jobs)`
    """
    em, em_metrics = _train_one(
        dataset.X("emission").values, dataset.y_emission.values,
        dataset.emission_features, "emission",
        cv_folds=cv_folds, seed=seed,
    )
    at, at_metrics = _train_one(
        dataset.X("attraction").values, dataset.y_attraction.values,
        dataset.attraction_features, "attraction",
        cv_folds=cv_folds, seed=seed,
    )
    return FlowForecastModel(
        emission_pipeline=em, attraction_pipeline=at,
        emission_features=dataset.emission_features,
        attraction_features=dataset.attraction_features,
        metrics_emission=em_metrics, metrics_attraction=at_metrics,
        training_target_total={
            "emission": float(dataset.y_emission.sum()),
            "attraction": float(dataset.y_attraction.sum()),
        },
    )
