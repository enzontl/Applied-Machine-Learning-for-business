"""Loaders MOBPRO + OMPHALE + générateur de dataset synthétique.

MOBPRO INSEE — fichier agrégé "Mobilités professionnelles" :
    https://www.insee.fr/fr/statistiques/8403349
    Déposer ``base-flux-mobpro-XXXX.csv`` dans ``data/external/``.
    Format CSV ";" avec colonnes typiques :
        COMMUNE  (code INSEE résidence)
        DCLT     (code INSEE travail)
        TRANS    (mode 1-6 ; 5 = voiture)
        IPONDI   (poids individuel = nb de navetteurs)

OMPHALE INSEE — projections de population/emplois par commune :
    https://www.insee.fr/fr/statistiques/5894083
    Fichier ``omphale-pop-emploi-commune.csv`` (à constituer à partir des
    extractions) avec colonnes : code_commune, horizon, pop_proj, emploi_proj.

Si MOBPRO est absent, ``generate_synthetic_mobpro`` produit un dataset
*MOBPRO-like* à partir d'un modèle gravitaire sur les pop/emplois IRIS.
C'est un fallback utile pour la démo et les tests CI ; le modèle entraîné
dessus reflète seulement la loi gravitaire — il faut MOBPRO réel pour
capturer la signature comportementale.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from urban_optimizer.config import EXTERNAL_DIR
from urban_optimizer.utils.logging import get_logger

logger = get_logger(__name__)


MOBPRO_DEFAULT_CSV = EXTERNAL_DIR / "base-flux-mobpro.csv"
OMPHALE_DEFAULT_CSV = EXTERNAL_DIR / "omphale-pop-emploi-commune.csv"
COMMUNES_CENTROIDS_CSV = EXTERNAL_DIR / "communes.csv"
INSEE_POP_IRIS_CSV = EXTERNAL_DIR / "insee_pop_iris.csv"
INSEE_EMP_LT_CSV = EXTERNAL_DIR / "insee_emp_lt_commune.csv"

# Code INSEE des modes MOBPRO (présent uniquement dans le fichier détail —
# la base agrégée 2021 ne distingue plus le mode, voir docstring loader)
MOBPRO_MODE_CAR = "5"

# Quand MOBPRO réel est manquant, on génère un dataset synthétique via gravitaire
SYNTHETIC_GRAVITY_BETA = 0.0015     # 1 / km en exp(-β·d)
SYNTHETIC_EMISSION_RATE = 0.30      # ~30 % de la pop fait un trajet voiture/j


@dataclass
class TrainingDataset:
    """Table commune-level prête pour l'entraînement.

    - ``features``: DataFrame (n_communes, n_features) ; colonnes indexées
      par ``commune_code`` (str).
    - ``y_emission`` / ``y_attraction``: flux voiture émis / attirés par
      commune (véh-trajets/jour). Target des deux modèles.
    - ``model_feature_names``: sous-ensemble des colonnes effectivement
      utilisées par le modèle. Stress-test a montré que toute feature
      au-delà de (population, jobs) dégrade la qualité spatiale CV à n≈100.
    """

    features: pd.DataFrame
    y_emission: pd.Series
    y_attraction: pd.Series
    # Features par cible — le stress-test a montré que :
    # - les émissions sont dominées par la population résidente (jobs colinéaires bruyants en spatial CV)
    # - les attractions dépendent de pop + jobs (employeurs locaux)
    emission_features: list[str] = field(default_factory=lambda: ["population"])
    attraction_features: list[str] = field(
        default_factory=lambda: ["population", "jobs"]
    )

    @property
    def feature_names(self) -> list[str]:
        return self.features.columns.tolist()

    def X(self, target: str) -> pd.DataFrame:
        feats = self.emission_features if target == "emission" else self.attraction_features
        return self.features[feats]


# ── 1. MOBPRO loader (réel) ──────────────────────────────────────────────────

def load_mobpro_aggregated(
    path: Path | None = None,
    *,
    only_car: bool = True,
    keep_communes: set[str] | None = None,
) -> pd.DataFrame:
    """Charge MOBPRO et agrège par paire (commune_o, commune_d).

    Supporte deux schémas :
    - **Base agrégée 2021** (INSEE 8201899) : ``CODGEO`` / ``DCLT`` /
      ``NBFLUX_C21_ACTOCC15P``. **Pas de colonne mode** — tous modes
      confondus (``only_car`` est ignoré, un avertissement est tracé).
    - **Fichier détail** (INSEE 8205896) : ``COMMUNE`` / ``DCLT`` / ``TRANS``
      / ``IPONDI``. ``only_car`` filtre alors sur mode voiture.

    Retourne un DataFrame avec colonnes :
        - commune_o, commune_d : codes INSEE (str, zéros à gauche)
        - n_trips : nb de navetteurs
    """
    path = Path(path) if path else MOBPRO_DEFAULT_CSV
    if not path.exists():
        raise FileNotFoundError(
            f"MOBPRO introuvable : {path}\n"
            "Télécharger sur https://www.insee.fr/fr/statistiques/8201899 "
            f"(base agrégée 2021, ~12 MB) et déposer dans {EXTERNAL_DIR}."
        )
    logger.info(f"Chargement MOBPRO : {path}")
    df = pd.read_csv(
        path, sep=";", encoding="utf-8", low_memory=False,
        dtype=str,
    )

    # Tolérance sur les noms de colonnes (base agrégée vs fichier détail)
    col_o = next(
        (c for c in ("CODGEO", "COMMUNE", "COMMUNE_O", "commune") if c in df.columns),
        None,
    )
    col_d = next(
        (c for c in ("DCLT", "DCLT_RES", "DCOMD") if c in df.columns),
        None,
    )
    col_w = next(
        (c for c in ("NBFLUX_C21_ACTOCC15P", "NBFLUX", "IPONDI", "NPERSO", "EFFECTIF")
         if c in df.columns),
        None,
    )
    col_t = next((c for c in ("TRANS", "TRANS_RES") if c in df.columns), None)
    if not all([col_o, col_d, col_w]):
        raise ValueError(
            f"Colonnes MOBPRO manquantes — détectées : {df.columns.tolist()[:15]}"
        )

    if only_car:
        if col_t is not None:
            df = df[df[col_t].astype(str) == MOBPRO_MODE_CAR]
        else:
            logger.warning(
                "Base MOBPRO agrégée — pas de colonne TRANS, tous modes confondus. "
                "Pour filtrer mode voiture, utiliser le fichier détail "
                "(INSEE 8205896, ~1 GB)."
            )

    df[col_w] = pd.to_numeric(df[col_w], errors="coerce")
    if keep_communes is not None:
        df = df[df[col_o].isin(keep_communes) & df[col_d].isin(keep_communes)]

    agg = (
        df.groupby([col_o, col_d], as_index=False)[col_w]
          .sum()
          .rename(columns={col_o: "commune_o", col_d: "commune_d", col_w: "n_trips"})
    )
    agg["commune_o"] = agg["commune_o"].astype(str).str.zfill(5)
    agg["commune_d"] = agg["commune_d"].astype(str).str.zfill(5)
    agg = agg[agg["n_trips"] > 0].reset_index(drop=True)
    logger.info(
        f"MOBPRO agrégé : {len(agg):,} paires, "
        f"{agg['n_trips'].sum():,.0f} trajets"
    )
    return agg


# ── 1bis. Loaders commune-level (pop + emplois LT + centroïdes) ─────────────

def load_pop_by_commune(
    pop_iris_csv: Path | None = None,
    *,
    dept_codes: list[str] | None = None,
) -> pd.DataFrame:
    """Agrège la pop IRIS au niveau commune.

    Retourne DataFrame : commune_code (str, zfill 5), population (float).
    """
    path = Path(pop_iris_csv) if pop_iris_csv else INSEE_POP_IRIS_CSV
    if not path.exists():
        raise FileNotFoundError(
            f"Base pop IRIS manquante : {path}\n"
            "https://www.insee.fr/fr/statistiques/8268806 (~25 MB CSV)"
        )
    df = pd.read_csv(
        path, sep=";", encoding="utf-8", low_memory=False,
        dtype={"IRIS": str, "COM": str},
        usecols=lambda c: c in ("IRIS", "COM", "P21_POP"),
    )
    df["COM"] = df["COM"].str.zfill(5)
    if dept_codes:
        prefixes = tuple(d.zfill(2) for d in dept_codes)
        df = df[df["COM"].str.startswith(prefixes)]
    agg = (
        df.groupby("COM", as_index=False)["P21_POP"].sum()
          .rename(columns={"COM": "commune_code", "P21_POP": "population"})
    )
    logger.info(
        f"Pop par commune : {len(agg):,} communes, "
        f"{agg['population'].sum():,.0f} habitants"
    )
    return agg


def load_emp_lt_by_commune(
    emp_csv: Path | None = None,
    *,
    dept_codes: list[str] | None = None,
) -> pd.DataFrame:
    """Charge TD_EMP1_2021 (INSEE) et agrège les emplois LT par commune.

    Format INSEE : NIVGEO, CODGEO, ..., NB. On filtre NIVGEO=COM et somme NB
    par CODGEO (= emplois totaux au lieu de travail).
    """
    path = Path(emp_csv) if emp_csv else INSEE_EMP_LT_CSV
    if not path.exists():
        raise FileNotFoundError(
            f"Base emplois LT manquante : {path}\n"
            "https://www.insee.fr/fr/statistiques/8202930 (~8 MB CSV)"
        )
    df = pd.read_csv(
        path, sep=";", encoding="utf-8", low_memory=False,
        dtype={"CODGEO": str, "NIVGEO": str},
        usecols=lambda c: c in ("NIVGEO", "CODGEO", "NB"),
    )
    df = df[df["NIVGEO"] == "COM"]
    df["CODGEO"] = df["CODGEO"].str.zfill(5)
    if dept_codes:
        prefixes = tuple(d.zfill(2) for d in dept_codes)
        df = df[df["CODGEO"].str.startswith(prefixes)]
    df["NB"] = pd.to_numeric(df["NB"], errors="coerce")
    agg = (
        df.groupby("CODGEO", as_index=False)["NB"].sum()
          .rename(columns={"CODGEO": "commune_code", "NB": "jobs"})
    )
    logger.info(
        f"Emplois LT par commune : {len(agg):,} communes, "
        f"{agg['jobs'].sum():,.0f} emplois"
    )
    return agg


def load_commune_centroids(
    communes_csv: Path | None = None,
    *,
    dept_codes: list[str] | None = None,
    target_crs: int = 2154,
) -> dict[str, tuple[float, float]]:
    """Charge les centroïdes des communes (data.gouv.fr).

    Reprojette en Lambert-93 (EPSG:2154) pour cohérence avec
    ``UrbanNetwork``. Retourne ``{commune_code: (x_m, y_m)}``.
    """
    path = Path(communes_csv) if communes_csv else COMMUNES_CENTROIDS_CSV
    if not path.exists():
        raise FileNotFoundError(
            f"Centroïdes communes manquants : {path}\n"
            "https://www.data.gouv.fr/datasets/communes-et-villes-de-france-en-csv"
        )
    df = pd.read_csv(
        path, sep=",", dtype={"code_insee": str, "dep_code": str},
        low_memory=False,
        usecols=["code_insee", "dep_code", "latitude_centre", "longitude_centre"],
    )
    df["code_insee"] = df["code_insee"].str.zfill(5)
    if dept_codes:
        prefixes = tuple(d.zfill(2) for d in dept_codes)
        df = df[df["dep_code"].astype(str).str.zfill(2).isin(prefixes)]
    df = df.dropna(subset=["latitude_centre", "longitude_centre"])

    # Reprojection WGS84 → Lambert-93 (vectorisée pyproj)
    from pyproj import Transformer
    tr = Transformer.from_crs(4326, target_crs, always_xy=True)
    x, y = tr.transform(df["longitude_centre"].values, df["latitude_centre"].values)
    out = {
        c: (float(xi), float(yi))
        for c, xi, yi in zip(df["code_insee"], x, y)
    }
    logger.info(f"Centroïdes communes : {len(out):,} (CRS {target_crs})")
    return out


# ── 2. Générateur synthétique (fallback) ────────────────────────────────────

def generate_synthetic_mobpro(
    pop_jobs_by_commune: pd.DataFrame,
    centroids_xy: dict[str, tuple[float, float]],
    *,
    seed: int = 0,
    beta: float = SYNTHETIC_GRAVITY_BETA,
    emission_rate: float = SYNTHETIC_EMISSION_RATE,
    noise_std: float = 0.15,
) -> pd.DataFrame:
    """Génère un dataset MOBPRO-like via gravitaire bruité.

    Pour chaque paire (c_o, c_d) :
        flux = E_o · A_d · exp(-β·d_km) / Σ_d' A_d' · exp(-β·d_o,d')

    Avec E_o = pop(c_o) × emission_rate et A_d = emploi(c_d).

    Le bruit lognormal (std=noise_std) simule les écarts entre gravitaire
    théorique et observation réelle. Utile pour la démo/test sans MOBPRO.
    """
    rng = np.random.default_rng(seed)
    df = pop_jobs_by_commune.set_index("commune_code")
    communes = [c for c in df.index if c in centroids_xy]
    if len(communes) < 2:
        return pd.DataFrame(columns=["commune_o", "commune_d", "n_trips"])
    df = df.loc[communes]
    xy = np.array([centroids_xy[c] for c in communes])

    # Matrice de distances (km), self-loop = 0.5 km (effet de zone)
    dx = xy[:, 0:1] - xy[:, 0:1].T
    dy = xy[:, 1:2] - xy[:, 1:2].T
    dist_km = np.sqrt(dx * dx + dy * dy) / 1000.0
    np.fill_diagonal(dist_km, 0.5)

    emit = df["population"].values * emission_rate
    attract = np.maximum(df["jobs"].values, 1.0)

    decay = np.exp(-beta * dist_km * 1000.0)  # β en 1/m
    A = attract[None, :] * decay
    norm = A.sum(axis=1, keepdims=True)
    norm = np.where(norm > 0, norm, 1.0)
    flux = emit[:, None] * A / norm

    # Bruit lognormal indépendant
    flux *= rng.lognormal(mean=0.0, sigma=noise_std, size=flux.shape)
    flux = np.maximum(flux, 0.0)

    n = len(communes)
    rows = []
    for i in range(n):
        for j in range(n):
            v = flux[i, j]
            if v < 0.5:   # ignorer le bruit < 1 navetteur
                continue
            rows.append((communes[i], communes[j], float(v)))
    out = pd.DataFrame(rows, columns=["commune_o", "commune_d", "n_trips"])
    logger.info(
        f"MOBPRO synthétique : {len(out):,} paires, "
        f"{out['n_trips'].sum():,.0f} trajets simulés"
    )
    return out


# ── 3. OMPHALE loader (projections futures) ──────────────────────────────────

def load_omphale_projections(
    path: Path | None = None,
    *,
    horizon_years: int = 10,
) -> pd.DataFrame:
    """Charge les projections population/emplois à horizon H.

    Format attendu (CSV ";") :
        commune_code, horizon, pop_proj, emploi_proj

    Si l'horizon exact n'est pas présent, on interpole linéairement entre
    les horizons disponibles encadrant H.

    Fallback si fichier manquant : projection naïve (+1 %/an pop et emplois).
    """
    path = Path(path) if path else OMPHALE_DEFAULT_CSV
    if not path.exists():
        logger.warning(
            f"OMPHALE introuvable ({path}) → fallback projection +1 %/an "
            f"sur {horizon_years} ans (×{(1.01) ** horizon_years:.3f})"
        )
        return pd.DataFrame(
            columns=["commune_code", "horizon", "pop_proj", "emploi_proj"]
        )

    df = pd.read_csv(path, sep=";", dtype={"commune_code": str})
    df["commune_code"] = df["commune_code"].str.zfill(5)
    if horizon_years not in df["horizon"].unique():
        # Interpolation linéaire entre les horizons disponibles
        horizons = sorted(df["horizon"].unique())
        if horizon_years < horizons[0] or horizon_years > horizons[-1]:
            logger.warning(
                f"Horizon {horizon_years} hors borne {horizons} → clip"
            )
            horizon_years = max(horizons[0], min(horizon_years, horizons[-1]))
        lo = max(h for h in horizons if h <= horizon_years)
        hi = min(h for h in horizons if h >= horizon_years)
        if lo == hi:
            return df[df["horizon"] == lo].copy()
        w = (horizon_years - lo) / (hi - lo)
        d_lo = df[df["horizon"] == lo].set_index("commune_code")
        d_hi = df[df["horizon"] == hi].set_index("commune_code")
        common = d_lo.index.intersection(d_hi.index)
        out = pd.DataFrame({
            "commune_code": common,
            "horizon": horizon_years,
            "pop_proj": (1 - w) * d_lo.loc[common, "pop_proj"]
                        + w * d_hi.loc[common, "pop_proj"],
            "emploi_proj": (1 - w) * d_lo.loc[common, "emploi_proj"]
                           + w * d_hi.loc[common, "emploi_proj"],
        })
        return out.reset_index(drop=True)
    return df[df["horizon"] == horizon_years].copy()


# ── 4. Features par commune ──────────────────────────────────────────────────

def build_training_dataset(
    mobpro: pd.DataFrame,
    pop_jobs_by_commune: pd.DataFrame,
    centroids_xy: dict[str, tuple[float, float]],
    *,
    city_center_xy: tuple[float, float] | None = None,
) -> TrainingDataset:
    """Construit (features, y_emission, y_attraction) à partir de MOBPRO + INSEE.

    Features par commune :
        - population, jobs, density (pop/km² proxy via inverse distance moyenne),
        - job_balance = jobs / max(pop, 1)
        - centrality = exp(-distance_centre / 5km) si city_center_xy fourni
        - n_neighbors_3km : nb de communes voisines ≤ 3 km (proxy de densité)
        - mean_dist_to_5_neighbors : km moyen vers 5 voisines (proxy d'isolement)

    Targets :
        - y_emission : Σ_d mobpro[c, d] = trajets sortant de c
        - y_attraction : Σ_o mobpro[o, c] = trajets entrant à c
    """
    pj = pop_jobs_by_commune.set_index("commune_code")
    communes = [c for c in pj.index if c in centroids_xy]
    pj = pj.loc[communes]

    # ── Targets : agrégation des flux par commune
    em = mobpro.groupby("commune_o")["n_trips"].sum().reindex(communes, fill_value=0.0)
    at = mobpro.groupby("commune_d")["n_trips"].sum().reindex(communes, fill_value=0.0)
    em.index.name = "commune_code"
    at.index.name = "commune_code"

    # ── Features
    xy = np.array([centroids_xy[c] for c in communes])
    n = len(communes)

    feats = pd.DataFrame(index=pd.Index(communes, name="commune_code"))
    feats["population"] = pj["population"].astype(float).values
    feats["jobs"] = pj["jobs"].astype(float).values
    feats["job_balance"] = feats["jobs"] / np.maximum(feats["population"], 1.0)

    # n_neighbors_3km + mean dist 5 voisines : vectorisé via matrice distance
    dx = xy[:, 0:1] - xy[:, 0:1].T
    dy = xy[:, 1:2] - xy[:, 1:2].T
    dist_m = np.sqrt(dx * dx + dy * dy)
    np.fill_diagonal(dist_m, np.inf)  # ignorer self
    feats["n_neighbors_3km"] = (dist_m < 3000.0).sum(axis=1).astype(float)
    if n >= 6:
        sorted_d = np.sort(dist_m, axis=1)
        feats["mean_dist_to_5_neighbors_km"] = sorted_d[:, :5].mean(axis=1) / 1000.0
    else:
        feats["mean_dist_to_5_neighbors_km"] = 0.0

    # Densité proxy : 1 / mean_dist_to_5 (forte densité = petites distances)
    feats["density_proxy"] = 1.0 / np.maximum(feats["mean_dist_to_5_neighbors_km"], 0.1)

    # Centralité (si centre fourni)
    if city_center_xy is not None:
        cx, cy = city_center_xy
        d_center_km = np.sqrt(
            (xy[:, 0] - cx) ** 2 + (xy[:, 1] - cy) ** 2
        ) / 1000.0
        feats["dist_to_center_km"] = d_center_km
        feats["centrality"] = np.exp(-d_center_km / 5.0)
    else:
        feats["dist_to_center_km"] = 0.0
        feats["centrality"] = 0.0

    return TrainingDataset(
        features=feats.astype(float),
        y_emission=em.astype(float),
        y_attraction=at.astype(float),
    )
