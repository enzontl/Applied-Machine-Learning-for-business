"""Algorithmes d'affectation : All-or-Nothing, User Equilibrium, System Optimum."""

from .aon import assign_aon
from .bpr import beckmann_objective, bpr_marginal_time, bpr_time
from .frank_wolfe import (
    beckmann,
    price_of_anarchy,
    solve_all_or_nothing,
    solve_system_optimum,
    solve_user_equilibrium,
)
from .result import AssignmentResult

__all__ = [
    "AssignmentResult",
    "assign_aon",
    "bpr_time",
    "bpr_marginal_time",
    "beckmann_objective",
    "beckmann",
    "solve_all_or_nothing",
    "solve_user_equilibrium",
    "solve_system_optimum",
    "price_of_anarchy",
]
