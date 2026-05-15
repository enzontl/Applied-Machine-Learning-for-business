"""Configuration centralisée du projet.

Centralise tous les paramètres pour qu'ils soient cohérents entre les briques
et faciles à ajuster sans toucher au code.
"""

from pathlib import Path

# ============================================================
# CHEMINS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
EXTERNAL_DIR = DATA_DIR / "external"
CONFIGS_DIR = PROJECT_ROOT / "configs"

# ============================================================
# COORDONNÉES DE RÉFÉRENCE
# ============================================================

CRS_LAMBERT93 = 2154   # France métropolitaine, pour les calculs métriques
CRS_WGS84 = 4326       # affichage carto, folium

# ============================================================
# CAPACITÉS PAR TYPE DE ROUTE (véh/h/voie)
# ============================================================
# Valeurs standard issues du Highway Capacity Manual, adaptées au contexte français

CAPACITY_BY_TYPE = {
    "motorway": 2000,
    "trunk": 1800,
    "primary": 1500,
    "secondary": 1200,
    "tertiary": 900,
    "residential": 600,
    "unclassified": 600,
    "service": 300,
}

# Vitesses libres par type de route (km/h)
FREE_SPEED_BY_TYPE = {
    "motorway": 130,
    "trunk": 110,
    "primary": 90,
    "secondary": 70,
    "tertiary": 50,
    "residential": 30,
    "unclassified": 50,
    "service": 20,
}

# ============================================================
# PARAMÈTRES BPR (Bureau of Public Roads)
# ============================================================
# t(v) = t0 * (1 + alpha * (v/c)^beta)

BPR_ALPHA = 0.15
BPR_BETA = 4.0

# ============================================================
# AFFECTATION
# ============================================================

FRANK_WOLFE_MAX_ITER = 100
FRANK_WOLFE_TOLERANCE = 1e-4

# ============================================================
# MODÈLE GRAVITAIRE
# ============================================================

GRAVITY_BETA = 0.1   # paramètre de déclin, à calibrer par ville

# ============================================================
# SCÉNARIOS HORAIRES
# ============================================================
# Part de la demande quotidienne par heure (valeurs indicatives)

HOURLY_DEMAND_SHARE = {
    7: 0.07,  8: 0.10,  9: 0.07,
    12: 0.05, 13: 0.05,
    17: 0.08, 18: 0.10, 19: 0.07,
}
