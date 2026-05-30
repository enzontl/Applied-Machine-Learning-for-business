# urban_optimizer

Outil d’aide à la décision pour la planification des investissements voirie.
Le projet combine réseau routier, demande OD, affectation Frank-Wolfe, prévision
de demande à horizon 5/10 ans, sélection d’interventions sous budget et
dashboard web.

## Ce que fait le projet

- Construit un réseau routier urbain à partir d’OSM, avec option ROUTE500.
- Génère une matrice OD par zonage `grid`, `iris` ou `h3`, puis modèle gravitaire.
- Calcule des équilibres utilisateur et système avec Frank-Wolfe et BPR.
- Diagnostique congestion, accessibilité, inégalités d’accès et arcs critiques.
- Génère des candidats d’aménagement, les sélectionne sous budget, puis les
  re-évalue jointement.
- Compare plusieurs profils de décision : `equilibre`, `ecolo`, `mobilite`,
  `economique`.
- Projette la demande à 5/10 ans pour les communes éligibles, puis ajuste l’OD
  par IPF/Furness.
- Expose une API FastAPI et une interface web vanilla JS avec dashboard.

## Installation

```bash
uv sync
```

Optionnel pour le zonage H3 :

```bash
uv sync --extra h3
```

## Données externes

Déposer les fichiers ci-dessous dans `data/external/` sauf mention contraire.

| Fichier | Source | Usage |
|---|---|---|
| `base-flux-mobpro.csv` | INSEE MOBPRO 2021 | Flux domicile-travail pour l’entraînement |
| `insee_pop_iris.csv` | INSEE 8268806 | Population par IRIS |
| `insee_emp_lt_commune.csv` | INSEE 8202930 (TD_EMP1) | Emplois au lieu de travail par commune |
| `communes.csv` | data.gouv.fr | Centroïdes des communes |
| `omphale-pop-emploi-commune.csv` | INSEE OMPHALE | Projections population/emplois par commune |
| `ofgl_idf.csv` | OFGL | Budget capex communal pour l’Île-de-France |

Pour ROUTE500, déposer l’archive IGN décompressée sous `data/raw/` afin que
`urban_optimizer/network/route500_loader.py` retrouve `TRONCON_ROUTE.shp`.

Le pipeline reste utilisable sans ces fichiers : il bascule sur des fallbacks
heuristiques quand les données manquent.

## Lancer l’application

```bash
uvicorn api.main:app --reload --port 8000
```

Puis ouvrir <http://localhost:8000>.

L’interface propose :

- une sélection de ville préconfigurée ;
- un horizon de 5 ou 10 ans ;
- un profil de décideur ;
- des options pour réseau simplifié, ROUTE500, comparaison multi-profil,
  robustesse, Pareto et suggestions Braess.

Les jobs sont gérés en mémoire et pollés via `GET /api/jobs/{id}` ; le
backend est donc pensé pour une démo locale, pas pour une production multi-
instance.

### Endpoints utiles

- `GET /` — landing page
- `GET /dashboard` — dashboard de résultats
- `GET /api/cities` — villes préconfigurées
- `GET /api/profiles` — profils maire
- `POST /api/jobs` — lancement du pipeline
- `GET /api/jobs/{id}` — suivi du job
- `GET /api/health` — état simple
- `GET /docs` — Swagger OpenAPI

## Chaîne de calcul

```text
OSM / ROUTE500 → réseau unifié
        ↓
Zonage (grid / IRIS / H3)
        ↓
Demande gravitaire → matrice OD
        ↓
Frank-Wolfe UE / SO
        ↓
Diagnostic : VHT, saturation, accessibilité, Gini
        ↓
Génération de candidats d’aménagement
        ↓
Sélection sous budget (knapsack DP + 2-swap)
        ↓
Re-évaluation jointe + analyses optionnelles
        ↓
Dashboard web
```

## Forecast et budget

La brique de prévision repose sur `urban_optimizer/forecast/` :

- `PopProportionalRegressor` pour les émissions ;
- `StandardScaler + LinearRegression` pour les attractions ;
- validation spatiale leave-one-département-out sur les données INSEE.

À l’horizon 5/10 ans, le pipeline :

1. prédit les flux actuels par commune ;
2. remplace population et emplois par les projections OMPHALE ;
3. recalcule les flux futurs ;
4. ajuste l’OD par IPF/Furness.

Pour le budget, `api/forecast_integration.py` applique :

- OFGL pour les communes IDF éligibles ;
- sinon une heuristique population × coût par habitant.

## Structure du dépôt

```text
api/                        FastAPI, jobs et orchestration du pipeline
frontend/                   Landing + dashboard vanilla JS
urban_optimizer/
  network/                  OSM, ROUTE500, fusion réseau, cache
  demand/                   Zonage, gravitaire, OD, projection future
  assignment/               All-or-nothing, UE, SO, BPR, Frank-Wolfe
  diagnosis/                Congestion, accessibilité, arcs critiques
  optimization/             Profils, scoring, sélection, robustesse, Pareto
  forecast/                 Données, modèle, projection, évaluation
  budget.py                 Budget capex voirie via OFGL ou fallback
models/forecast/idf/        Modèle entraîné de référence
data/external/              Données INSEE / OFGL à déposer
```

## Limites assumées

- Le forecast est un modèle cross-section, pas une série temporelle.
- Le jeu d’entraînement forecast est centré sur l’Île-de-France.
- Hors IDF, la projection de demande reste désactivée et le budget passe par un
  fallback heuristique.
- Le mapping IRIS→commune est volontairement simplifié dans le chemin API.
- Les jobs ne survivent pas à un redémarrage du serveur.
