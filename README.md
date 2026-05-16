# urban_optimizer

A framework for urban road network optimization using OD traffic assignment and urban plan evaluation.

The project combines:
1. Urban road network construction from OSM (with optional ROUTE500 integration).
2. OD matrix generation (grid, H3, or IRIS zoning).
3. Traffic assignment (All-or-Nothing, User Equilibrium, System Optimum).
4. Network diagnostics (VHT, congestion, saturation, critical links, price of anarchy).
5. Optimization (candidate ranking and new link proposals under budget constraints).
6. A Streamlit dashboard to run the full pipeline end-to-end.

## Current Project Status

Features currently implemented in the codebase:
- network: build_network, build_unified_network, OSM and ROUTE500 loaders.
- demand: generate_od_matrix, gravity_od, grid/H3/IRIS zoning.
- assignment: solve_all_or_nothing, solve_user_equilibrium, solve_system_optimum, price_of_anarchy.
- diagnosis: diagnose, critical-link ranking.
- optimization: candidate generation/evaluation, city scoring by mayor profile, urban plan proposal.
- streamlit_app.py: interactive simulation and plan comparison interface.

## Structure

```
.
├── streamlit_app.py
├── configs/
├── data/
│   ├── raw/
│   ├── external/
│   └── processed/
├── notebooks/
├── tests/
└── urban_optimizer/
	├── network/
	├── demand/
	├── assignment/
	├── diagnosis/
	├── optimization/
	├── utils/
	└── viz/
```

## Requirements

- Python >= 3.10 (defined in pyproject.toml).
- Recommended: Python 3.11 or 3.12 to avoid Python 3.9 typing incompatibilities.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Quick check:

```bash
python -c "import urban_optimizer; print(urban_optimizer.__version__)"
```

## Run the Dashboard

```bash
source .venv/bin/activate
streamlit run streamlit_app.py
```

The dashboard can:
- Build a city network from OSM.
- Generate OD demand.
- Solve user equilibrium (Frank-Wolfe).
- Display annual city score based on mayor profile.
- Propose new-road plans (with optional Braess removals).

## Minimal API Example

```python
from urban_optimizer.network import build_network
from urban_optimizer.demand import generate_od_matrix
from urban_optimizer.assignment import solve_user_equilibrium
from urban_optimizer.diagnosis import diagnose
from urban_optimizer.optimization import score_network, ECOLO

net = build_network("Villeurbanne, France", include_route500=False)
od = generate_od_matrix(net, hour=8, method="grid", n_cells=10, scale_factor=0.3)
ue = solve_user_equilibrium(net, od, max_iter=100, tol=1e-4)
diag = diagnose(net, ue)
score = score_network(ue, ECOLO)

print(diag.vht, score.total_annual_cost_eur)
```

## Data

- OSM: downloaded automatically via OSMnx.
- ROUTE500 (optional): place source files in data/raw if enabled.
- INSEE/IRIS (depending on zoning method): place source files in data/external.

Note: if external datasets are missing, use grid zoning for a quick start.

## Tests

Run all tests:

```bash
pytest
```

Run a single test module:

```bash
pytest tests/test_network.py -q
```

## Notebooks

The notebooks folder contains exploratory and network-construction notebooks:
- 00_setup_check.ipynb
- 01_network_construction.ipynb
- EDA.ipynb

## Configuration

Example city configuration is available in configs/lyon.yaml (network, demand, assignment, budget).

## Quick Troubleshooting

TypeError unsupported operand type(s) for |: 'type' and 'NoneType':
- Common cause: running the app with Python 3.9.
- Fix: recreate the environment with Python 3.10+ and reinstall dependencies.