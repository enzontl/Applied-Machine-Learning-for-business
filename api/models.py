"""Schémas Pydantic pour l'API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class RunRequest(BaseModel):
    """Payload pour lancer un pipeline.

    Le **budget** n'est plus saisi par l'utilisateur : il est calculé
    automatiquement à partir des comptes OFGL de la commune (capex annuel ×
    horizon × part voirie). L'utilisateur choisit seulement l'**horizon**.
    """

    city: str = Field(..., examples=["Boulogne-Billancourt, France"])
    profile: Literal["ecolo", "mobilite", "economique", "equilibre"] = "equilibre"
    hour: int = Field(8, ge=0, le=23)
    n_cells: int = Field(10, ge=4, le=20)
    scale_factor: float = Field(0.3, ge=0.1, le=2.0)
    horizon_years: Literal[5, 10] = Field(
        10,
        description="Horizon de prévision et de budget (5 ou 10 ans).",
    )
    max_candidates: int = Field(20, ge=10, le=100)
    max_fw_evals: int = Field(8, ge=3, le=25)
    periphery_margin_m: float = Field(600.0, ge=0.0, le=2000.0)
    access_threshold_min: int = Field(15, ge=5, le=45)
    include_route500: bool = False
    simplified_highway: bool = False
    max_iter_ue: int = Field(50, ge=20, le=300)
    multi_profile: bool = False
    include_robustness: bool = False
    include_pareto: bool = False
    include_braess: bool = False


class KPI(BaseModel):
    """Un indicateur affiché dans la UI."""

    label: str
    value: str
    unit: str = ""
    delta: str | None = None  # variation (+x, -y%)
    accent: Literal["neutral", "good", "bad", "warning"] = "neutral"


class Intervention(BaseModel):
    """Une intervention retenue dans le plan."""

    rank: int
    action: str  # "élargissement" | "mise à niveau" | "nouvelle route"
    highway: str
    length_m: float
    delta_vht_h: float
    annual_benefit_eur: float
    construction_cost_eur: float
    payback_years: float | None
    bcr: float


class ProfileResult(BaseModel):
    """Résultat pour 1 profil maire."""

    profile_name: str
    profile_label: str
    kpis: list[KPI]
    interventions: list[Intervention]
    baseline_vht_h: float
    joint_vht_h: float
    baseline_score_eur: float
    joint_score_eur: float
    accessibility_before: float
    accessibility_after: float
    gini_before: float
    gini_after: float


class JobStatus(BaseModel):
    """État d'un job en cours / fini."""

    job_id: str
    status: Literal["queued", "running", "done", "error"]
    progress: float = 0.0  # 0.0 - 1.0
    step: str = ""
    elapsed_s: float = 0.0
    error: str | None = None
    # Résultat (seulement quand status == "done")
    city: str | None = None
    profiles: list[ProfileResult] | None = None
    # Carte : GeoJSON FeatureCollection des arcs colorés par saturation avant/après
    network_geojson: dict | None = None
    plan_geojson: dict | None = None  # interventions overlay
