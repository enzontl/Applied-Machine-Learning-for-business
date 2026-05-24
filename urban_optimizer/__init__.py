"""urban_optimizer — modèle générique d'optimisation des réseaux routiers urbains."""

__version__ = "0.1.0"
__author__ = "Enzo Natali"

# ── Supprimer les warnings C igraph ("Couldn't reach some vertices") ────
# Ces RuntimeWarning sont émis par la couche C d'igraph pour chaque sommet
# inatteignable dans Dijkstra. Sur un graphe urbain avec quelques îlots
# déconnectés, ça génère des milliers de lignes sur stderr et ralentit
# massivement le pipeline (pur I/O).
# Le filtre GLOBAL ici est nécessaire car les appels Dijkstra se font
# dans plusieurs modules (aon.py, accessibility, network_design).
import warnings as _warnings

_warnings.filterwarnings(
    "ignore",
    category=RuntimeWarning,
    message=r"Couldn't reach some vertices",
)
