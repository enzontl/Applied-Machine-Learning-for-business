# urban_optimizer

Modèle générique d'**optimisation des réseaux routiers urbains** par affectation dynamique de flux Origine-Destination.

> L'objectif n'est **pas** de calculer un itinéraire optimal pour un trajet, mais d'identifier ce qui peut être amélioré dans un réseau routier urbain pour réduire le temps total passé par l'ensemble des usagers.

## Idée du projet

À partir d'une ville quelconque, le modèle :

1. Construit le **réseau routier** combinant OSM (urbain détaillé) et ROUTE500 (liaisons interurbaines).
2. Génère une **demande de déplacements** (matrice OD) calibrée sur des données INSEE.
3. Calcule l'**équilibre de trafic** par trois méthodes (All-or-Nothing, User Equilibrium, System Optimum).
4. **Diagnostique** les points critiques du réseau (saturation, prix de l'anarchie).
5. **Optimise** : propose un classement des améliorations possibles sous contrainte de budget.

## Architecture

```
urban_optimizer/
├── network/         Construction du graphe (OSM + ROUTE500)
├── demand/          Génération de la demande OD (zonage, gravité)
├── assignment/      Algorithmes d'affectation (AoN, UE, SO)
├── diagnosis/       Métriques diagnostiques (VHT, saturation, PoA)
├── optimization/    Ranking d'améliorations, détection de Braess
├── viz/             Cartes et dashboards
└── utils/           Outils transverses
```

## Installation

```bash
# Création de l'environnement
python -m venv .venv
source .venv/bin/activate  # ou .venv\Scripts\activate sous Windows

# Installation en mode développement
pip install -e ".[dev]"

# Vérification
python -c "import urban_optimizer; print(urban_optimizer.__version__)"
```

## Démo

```python
from urban_optimizer.network import build_network
from urban_optimizer.demand import generate_od_matrix
from urban_optimizer.assignment import solve_user_equilibrium
from urban_optimizer.diagnosis import diagnose
from urban_optimizer.optimization import rank_improvements

# Pipeline complet sur Lyon
G = build_network("Lyon, France")
od = generate_od_matrix(G, hour=8, scenario="weekday")
flows = solve_user_equilibrium(G, od)
diag = diagnose(G, flows)
top_improvements = rank_improvements(G, od, budget=10_000_000)
```

## Roadmap

- [x] Setup global (env, structure)
- [ ] Brique 1 : construction du réseau OSM + ROUTE500
- [ ] Brique 2 : zonage et demande OD via INSEE
- [ ] Brique 3 : algorithmes d'affectation (AoN → UE → SO)
- [ ] Brique 4 : diagnostic du réseau
- [ ] Brique 5 : optimisation et ranking
- [ ] Démo Lyon
- [ ] Application multi-villes

## Sources de données

- **OSMnx** : OpenStreetMap, accès direct via Python
- **ROUTE500** : IGN, à télécharger dans `data/raw/`
- **INSEE** : population et emplois IRIS, à télécharger dans `data/external/`
- **Contours IRIS** : IGN, à télécharger dans `data/external/`