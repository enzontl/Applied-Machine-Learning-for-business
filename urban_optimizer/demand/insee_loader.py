"""Chargement des données IRIS (IGN) et INSEE (population, emplois).

Ces données ne sont pas embarquées dans le repo. À récupérer :

1. **Contours IRIS** (IGN, ~150 MB une fois décompressé) :
   https://geoservices.ign.fr/contoursiris
   Déposer le dossier décompressé dans ``data/external/CONTOURS-IRIS/``.

2. **Population par IRIS** — base "Population par IRIS" du RP INSEE :
   https://www.insee.fr/fr/statistiques/7704076
   Fichier CSV : ``base-ic-evol-struct-pop-*.CSV`` → ``data/external/insee_pop_iris.csv``.

3. **Emplois par IRIS** — base "Activité résidents" / "Emploi au LT" :
   https://www.insee.fr/fr/statistiques/7704078
   Fichier CSV : ``base-ic-activite-residents-*.CSV`` → ``data/external/insee_jobs_iris.csv``.

Toutes les fonctions sont tolérantes : si un fichier est manquant, elles lèvent
une erreur explicite avec l'URL de téléchargement.
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import pandas as pd

from urban_optimizer.config import EXTERNAL_DIR
from urban_optimizer.utils.logging import get_logger

logger = get_logger(__name__)

# Chemins par défaut (modifiables au cas par cas)
IRIS_CONTOURS_DIR = EXTERNAL_DIR / "CONTOURS-IRIS"
INSEE_POP_CSV = EXTERNAL_DIR / "insee_pop_iris.csv"
INSEE_JOBS_CSV = EXTERNAL_DIR / "insee_jobs_iris.csv"


def _find_iris_shapefile(root: Path) -> Path:
    """Cherche récursivement le shapefile IRIS principal sous ``root``."""
    if not root.exists():
        raise FileNotFoundError(
            f"Dossier IRIS introuvable : {root}\n"
            "Télécharger CONTOURS-IRIS sur https://geoservices.ign.fr/contoursiris "
            "et décompresser dans data/external/CONTOURS-IRIS/."
        )
    candidates = list(root.rglob("CONTOURS-IRIS*.shp")) + list(root.rglob("CONTOURS_IRIS*.shp"))
    if not candidates:
        candidates = list(root.rglob("IRIS*.shp"))
    if not candidates:
        raise FileNotFoundError(f"Aucun shapefile IRIS sous {root}.")
    logger.info(f"Shapefile IRIS détecté : {candidates[0]}")
    return candidates[0]


def load_iris_contours(
    bounds: tuple | None = None,
    dept_codes: list[str] | None = None,
    shp_path: Path | None = None,
    target_crs: int = 2154,
) -> gpd.GeoDataFrame:
    """Charge les contours IRIS, éventuellement filtrés par département ou bbox.

    Args:
        bounds: tuple (minx, miny, maxx, maxy) en CRS de la cible (Lambert-93).
        dept_codes: liste de codes département (deux ou trois caractères).
        shp_path: chemin explicite vers le shapefile, sinon recherche dans EXTERNAL_DIR.
        target_crs: code EPSG cible (Lambert-93 par défaut).

    Returns:
        GeoDataFrame avec colonnes ``zone_id`` (code IRIS) et ``geometry``.
    """
    path = Path(shp_path) if shp_path else _find_iris_shapefile(IRIS_CONTOURS_DIR)

    read_kwargs = {}
    if bounds is not None:
        from shapely.geometry import box

        read_kwargs["bbox"] = box(*bounds)

    iris = gpd.read_file(path, **read_kwargs)
    logger.info(f"IRIS chargés : {len(iris):,}")

    code_col = next(
        (c for c in ("CODE_IRIS", "DCOMIRIS", "code_iris") if c in iris.columns),
        None,
    )
    if code_col is None:
        raise ValueError(f"Aucune colonne code IRIS trouvée dans {iris.columns.tolist()}.")

    iris = iris.rename(columns={code_col: "zone_id"})
    iris["zone_id"] = iris["zone_id"].astype(str)

    if dept_codes is not None:
        prefixes = tuple(d.zfill(2) for d in dept_codes)
        before = len(iris)
        iris = iris[iris["zone_id"].str.startswith(prefixes)].copy()
        logger.info(f"IRIS filtrés par dept {list(prefixes)} : {before} → {len(iris)}")

    iris = iris.to_crs(target_crs)
    return iris[["zone_id", "geometry"]].reset_index(drop=True)


def _read_insee_csv(path: Path) -> pd.DataFrame:
    """Lit un CSV INSEE (séparateur point-virgule, encodage Windows)."""
    if not path.exists():
        raise FileNotFoundError(f"Fichier INSEE manquant : {path}")
    try:
        return pd.read_csv(path, sep=";", encoding="latin1", low_memory=False, dtype={"IRIS": str})
    except UnicodeDecodeError:
        return pd.read_csv(path, sep=";", encoding="utf-8", low_memory=False, dtype={"IRIS": str})


def load_insee_pop_jobs(
    dept_codes: list[str] | None = None,
    pop_csv: Path | None = None,
    jobs_csv: Path | None = None,
) -> pd.DataFrame:
    """Charge population et emplois par IRIS.

    Args:
        dept_codes: filtre par département (ex. ["69"]).
        pop_csv: chemin alternatif au fichier INSEE population.
        jobs_csv: chemin alternatif au fichier INSEE emplois (optionnel).

    Returns:
        DataFrame avec colonnes ``zone_id``, ``population``, ``jobs``.
        Si le fichier emplois est manquant, ``jobs`` vaut 0.
    """
    pop_path = Path(pop_csv) if pop_csv else INSEE_POP_CSV
    pop_df = _read_insee_csv(pop_path)

    iris_col = next(
        (c for c in ("IRIS", "iris", "CODE_IRIS") if c in pop_df.columns),
        None,
    )
    pop_col = next(
        (c for c in ("P20_POP", "P21_POP", "P19_POP", "P18_POP", "POP") if c in pop_df.columns),
        None,
    )
    if iris_col is None or pop_col is None:
        raise ValueError(
            f"Colonnes attendues introuvables ({iris_col=}, {pop_col=}). "
            f"Colonnes : {pop_df.columns.tolist()[:20]}..."
        )

    pop = pop_df[[iris_col, pop_col]].rename(columns={iris_col: "zone_id", pop_col: "population"})
    pop["zone_id"] = pop["zone_id"].astype(str)
    pop["population"] = pd.to_numeric(pop["population"], errors="coerce").fillna(0.0)

    jobs_path = Path(jobs_csv) if jobs_csv else INSEE_JOBS_CSV
    if jobs_path.exists():
        jobs_df = _read_insee_csv(jobs_path)
        jobs_iris_col = next(
            (c for c in ("IRIS", "iris", "CODE_IRIS") if c in jobs_df.columns),
            None,
        )
        jobs_col = next(
            (
                c for c in ("C20_ACT1564", "C21_ACT1564", "EMPLT15", "EMPLT", "P20_EMPLT15")
                if c in jobs_df.columns
            ),
            None,
        )
        if jobs_iris_col and jobs_col:
            jobs = jobs_df[[jobs_iris_col, jobs_col]].rename(
                columns={jobs_iris_col: "zone_id", jobs_col: "jobs"}
            )
            jobs["zone_id"] = jobs["zone_id"].astype(str)
            jobs["jobs"] = pd.to_numeric(jobs["jobs"], errors="coerce").fillna(0.0)
        else:
            logger.warning(
                f"Colonne emplois introuvable dans {jobs_path.name}, jobs = 0."
            )
            jobs = pd.DataFrame({"zone_id": pop["zone_id"], "jobs": 0.0})
    else:
        logger.warning(f"Fichier emplois absent ({jobs_path}), jobs = 0.")
        jobs = pd.DataFrame({"zone_id": pop["zone_id"], "jobs": 0.0})

    merged = pop.merge(jobs, on="zone_id", how="left")
    merged["jobs"] = merged["jobs"].fillna(0.0)

    if dept_codes is not None:
        prefixes = tuple(d.zfill(2) for d in dept_codes)
        merged = merged[merged["zone_id"].str.startswith(prefixes)].copy()

    logger.info(
        f"INSEE — {len(merged):,} IRIS, pop totale = {merged['population'].sum():,.0f}, "
        f"emplois = {merged['jobs'].sum():,.0f}"
    )
    return merged.reset_index(drop=True)
