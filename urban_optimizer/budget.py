"""Budget d'investissement voirie par commune (basé sur OFGL).

Source : Observatoire des Finances et de la Gestion publique Locales (OFGL),
dataset ``ofgl-base-communes`` — agrégat "Dépenses d'équipement" du budget
principal de la commune. C'est le **capex total** annuel : tous postes
(voirie, écoles, sport, eau, etc.).

On applique ensuite un **ratio voirie / transport** par défaut de 20 %,
basé sur les comptes administratifs analytiques agrégés DGFIP (la voirie
représente typiquement 15-25 % du capex communal en milieu urbain dense ;
plus en zones rurales). Le ratio est paramétrable.

URL du téléchargement OFGL (filtré sur 4 dépts IDF, ~24 MB) :

    https://data.ofgl.fr/api/explore/v2.1/catalog/datasets/ofgl-base-communes
      /exports/csv?limit=-1&use_labels=true
      &where=dep_code%20IN%20(%2275%22%2C%2292%22%2C%2293%22%2C%2294%22)

Fallback : si fichier absent, heuristique ``per_capita × population``
(~ 80 €/habitant/an pour voirie investissement urbain dense).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from urban_optimizer.config import EXTERNAL_DIR
from urban_optimizer.utils.logging import get_logger

logger = get_logger(__name__)


OFGL_CSV_DEFAULT = EXTERNAL_DIR / "ofgl_idf.csv"

# Part typique de la voirie dans le capex communal (DGFIP/OFGL agrégés)
DEFAULT_VOIRIE_SHARE = 0.20

# Fallback : capex voirie €/habitant/an (médiane urbaine France)
FALLBACK_PER_CAPITA_EUR = 80.0


@dataclass
class CommuneBudget:
    commune_code: str
    annual_capex_eur: float       # Dépenses d'équipement OFGL (capex total)
    voirie_share: float
    annual_voirie_eur: float      # = annual_capex_eur × voirie_share

    def horizon_budget_eur(self, horizon_years: int) -> float:
        return self.annual_voirie_eur * horizon_years


def load_ofgl_capex(
    ofgl_csv: Path | None = None,
    *,
    exercice: int | None = None,
) -> pd.DataFrame:
    """Charge l'agrégat 'Dépenses d'équipement' du budget principal des communes.

    Retourne DataFrame : commune_code, exercice, annual_capex_eur.
    Si ``exercice=None``, retourne l'année la plus récente disponible.
    """
    path = Path(ofgl_csv) if ofgl_csv else OFGL_CSV_DEFAULT
    if not path.exists():
        raise FileNotFoundError(
            f"OFGL CSV manquant : {path}\n"
            "Télécharger depuis https://data.ofgl.fr/explore/dataset/ofgl-base-communes/"
        )
    df = pd.read_csv(
        path, sep=";", encoding="utf-8", low_memory=False,
        dtype={"Code Insee 2024 Commune": str},
    )
    df = df.rename(columns={
        "Code Insee 2024 Commune": "commune_code",
        "Exercice": "exercice",
        "Agrégat": "agregat",
        "Montant": "montant",
        "Catégorie": "categorie",
        "Type de budget": "type_budget",
    })
    # Filtres : commune (ou Paris statut particulier) + budget principal
    # + dépenses d'équipement
    df = df[
        df["categorie"].isin(["Commune", "PARIS"])
        & (df["type_budget"] == "Budget principal")
        & (df["agregat"] == "Dépenses d'équipement")
    ].copy()
    df["commune_code"] = df["commune_code"].str.zfill(5)
    df["montant"] = pd.to_numeric(df["montant"], errors="coerce")
    if exercice is None:
        exercice = int(df["exercice"].max())
    df = df[df["exercice"] == exercice]
    out = (
        df.groupby("commune_code", as_index=False)["montant"].sum()
          .rename(columns={"montant": "annual_capex_eur"})
    )
    out["exercice"] = exercice
    logger.info(
        f"OFGL capex {exercice} : {len(out)} communes, "
        f"total {out['annual_capex_eur'].sum() / 1e6:,.1f} M€"
    )
    return out


def get_city_budget(
    commune_codes: list[str],
    *,
    horizon_years: int = 10,
    voirie_share: float = DEFAULT_VOIRIE_SHARE,
    pop_by_commune: pd.DataFrame | None = None,
) -> tuple[float, list[CommuneBudget]]:
    """Budget cumulé voirie sur un horizon, pour une liste de communes.

    Tente OFGL d'abord ; si CSV manquant ou une commune absente, fallback
    sur ``pop × FALLBACK_PER_CAPITA_EUR × horizon`` (nécessite
    ``pop_by_commune``).

    Retourne ``(total_eur, [CommuneBudget, ...])``.
    """
    capex = None
    try:
        capex = load_ofgl_capex().set_index("commune_code")
    except FileNotFoundError as exc:
        logger.warning(str(exc))

    pop_lookup = (
        pop_by_commune.set_index("commune_code")["population"]
        if pop_by_commune is not None else None
    )

    budgets: list[CommuneBudget] = []
    for c in commune_codes:
        annual_capex = None
        if capex is not None and c in capex.index:
            annual_capex = float(capex.loc[c, "annual_capex_eur"])
        elif pop_lookup is not None and c in pop_lookup.index:
            annual_capex = (
                float(pop_lookup.loc[c]) * FALLBACK_PER_CAPITA_EUR / voirie_share
            )
        else:
            logger.warning(f"Aucune source de budget pour {c}, skip")
            continue
        voirie = annual_capex * voirie_share
        budgets.append(CommuneBudget(
            commune_code=c,
            annual_capex_eur=annual_capex,
            voirie_share=voirie_share,
            annual_voirie_eur=voirie,
        ))

    total = sum(b.horizon_budget_eur(horizon_years) for b in budgets)
    logger.info(
        f"Budget voirie cumulé H+{horizon_years} : {total / 1e6:,.1f} M€ "
        f"sur {len(budgets)} communes (share voirie = {voirie_share:.0%})"
    )
    return total, budgets
