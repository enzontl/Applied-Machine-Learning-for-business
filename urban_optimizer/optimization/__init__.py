"""Optimisation et design du réseau (ranking d'améliorations + nouvelles routes)."""

from .candidates import Candidate, generate_candidates
from .mayor_profile import (
    ALL_PROFILES,
    ECOLO,
    ECONOMIQUE,
    EQUILIBRE,
    MOBILITE,
    PROFILE_BY_NAME,
    MayorProfile,
)
from .network_design import (
    NewArcEvaluation,
    NewArcProposal,
    generate_corridor_candidates,
    generate_new_arc_candidates,
    generate_new_route_candidates,
    generate_upgrade_candidates,
    propose_urban_plan,
)
from .ranking import CandidateEvaluation, rank_candidates, select_under_budget
from .score import CityScore, score_network

__all__ = [
    # Ranking sur les arcs existants (brique 5a)
    "Candidate",
    "generate_candidates",
    "CandidateEvaluation",
    "rank_candidates",
    "select_under_budget",
    # Design réseau (brique 5b — corridors à élargir)
    "NewArcProposal",
    "NewArcEvaluation",
    "generate_corridor_candidates",
    "generate_upgrade_candidates",
    "generate_new_route_candidates",
    "generate_new_arc_candidates",
    "propose_urban_plan",
    # Scoring et profils maire
    "MayorProfile",
    "ECOLO", "MOBILITE", "ECONOMIQUE", "EQUILIBRE",
    "ALL_PROFILES", "PROFILE_BY_NAME",
    "CityScore",
    "score_network",
]
