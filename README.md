# urban_optimizer

A framework for urban road network optimization using OD traffic assignment and urban plan evaluation.

The project combines:
1. Urban road network construction from OSM (with optional ROUTE500 integration).
2. OD matrix generation (grid, H3, or IRIS zoning).
3. Traffic assignment (All-or-Nothing, User Equilibrium, System Optimum).
4. Network diagnostics (VHT, congestion, saturation, critical links, **accessibility + Gini equity**, price of anarchy).
5. Optimization (candidate ranking, new link proposals under budget constraints, **joint re-evaluation**, **robustness testing**, **Pareto frontier**).
6. A Streamlit dashboard to run the full pipeline end-to-end with **before/after saturation maps**.

## Current Project Status

Features currently implemented in the codebase:
- **network**: build_network, build_unified_network, OSM/ROUTE500 loaders, hard obstacles (buildings, parks) + bridge triggers (rail, water).
- **demand**: generate_od_matrix, gravity_od, grid/H3/IRIS zoning.
- **assignment**: solve_all_or_nothing, solve_user_equilibrium, solve_system_optimum, price_of_anarchy.
- **diagnosis**: diagnose, critical-link ranking, **accessibility report (isochrone + Gini equity)**.
- **optimization**:
  - 3-type interventions: **widening** (corridor), **upgrade** (residential → arterial), **new route** (visibility-graph A* on building corners).
  - **Periphery filter**: new routes only in peripheral areas (avoids failing in dense centers).
  - **Joint plan re-evaluation**: single FW with all interventions applied → real ΔVHT + redundancy detection.
  - **Enriched score**: time + fuel + CO2 + accessibility benefit − equity (Gini) penalty, all weighted by mayor profile.
  - **Robustness testing**: re-FW under demand × {0.8, 1.0, 1.2, 1.5}.
  - **Pareto frontier**: budget → benefit curve for 6 budget levels with sweet-spot detection.
- **streamlit_app.py**: interactive dashboard with before/after maps, KPIs, optional robustness + Pareto.

## Pipeline overview

```
OSM/ROUTE500 → network → OD matrix → Frank-Wolfe UE → diagnosis (VHT + accessibility)
                                                ↓
                                  ┌─────────────┴────────────┐
                                  │   Candidate generation   │
                                  │  (corridors + upgrades   │
                                  │     + new routes)        │
                                  └─────────────┬────────────┘
                                                ↓
                                  Individual FW evaluation (1/candidate)
                                                ↓
                                  Greedy selection under budget
                                                ↓
                                  Joint FW re-evaluation
                                  (real ΔVHT + post-plan accessibility)
                                                ↓
                              ┌─────────────────┼─────────────────┐
                              ↓                 ↓                 ↓
                       Robustness        Pareto curve         Before/after
                       (4 demand          (6 budgets)         saturation maps
                        scenarios)
```

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
- Build a city network from OSM (with hard/soft obstacle layers).
- Generate OD demand.
- Solve user equilibrium (Frank-Wolfe).
- Display annual city score based on mayor profile — including **accessibility** (zones reachable in T min) and **equity** (Gini coefficient).
- Propose urban plans combining **widening / upgrade / new route** under budget — with **joint re-evaluation** (real ΔVHT, redundancy detection).
- Display **before/after saturation maps** side by side (toggle modes available).
- Compare accessibility + Gini before/after the plan, with automatic "egalitarian / inegalitarian / neutral" verdict.
- Optional **robustness test** (4 demand scenarios: ×0.8 / ×1.0 / ×1.2 / ×1.5).
- Optional **Pareto curve** (6 budget levels with sweet-spot detection).
- Optional **Braess removals** (arcs whose deletion improves the network).

## Minimal API Example

```python
from urban_optimizer.network import build_network, load_obstacles, load_bridge_triggers
from urban_optimizer.demand import generate_od_matrix
from urban_optimizer.assignment import solve_user_equilibrium
from urban_optimizer.diagnosis import diagnose, compute_accessibility
from urban_optimizer.optimization import (
    score_network, propose_urban_plan, evaluate_plan_robustness,
    compute_pareto_frontier, ECOLO,
)

# Build network + obstacles
net = build_network("Villeurbanne, France", include_route500=False)
obstacles = load_obstacles("Villeurbanne, France")        # hard: buildings, parks
bridges   = load_bridge_triggers("Villeurbanne, France")  # soft: rail, water

# Demand + equilibrium
od = generate_od_matrix(net, hour=8, method="grid", n_cells=10, scale_factor=0.3)
ue = solve_user_equilibrium(net, od, max_iter=100, tol=1e-4)

# Diagnosis + enriched score
diag    = diagnose(net, ue)
access  = compute_accessibility(net, od, ue, threshold_seconds=15 * 60)
score   = score_network(ue, ECOLO, access=access)
print(diag.vht, score.total_annual_cost_eur, access.mean_reachable, access.gini)

# Propose plan (returns chosen evaluations + baseline score + joint re-evaluation)
plan, baseline, joint = propose_urban_plan(
    net, od, ECOLO, ue,
    budget_eur=50_000_000,
    obstacle_index=obstacles, soft_index=bridges,
    periphery_margin_m=600.0,
)
if joint:
    print(f"Real ΔVHT = {joint.joint_delta_vht_h:+.1f}h (vs naive sum {joint.naive_sum_delta_vht_h:+.1f}h)")
    print(f"Redundancy = {(1 - joint.redundancy_factor)*100:.0f}%")
    print(f"Accessibility: {joint.accessibility_before:.1f} → {joint.accessibility_after:.1f} zones")

# Robustness under demand shocks
rob = evaluate_plan_robustness(net, od, plan, ECOLO)
print(f"Plan robust = {rob.is_robust}")

# Pareto frontier
pareto = compute_pareto_frontier(
    net, od, ECOLO, ue,
    budgets_eur=(5e6, 15e6, 30e6, 60e6, 120e6, 250e6),
    obstacle_index=obstacles, soft_index=bridges,
)
sweet = pareto.best_marginal_point
print(f"Sweet spot: {sweet.budget_eur/1e6:.0f} M€ → {sweet.joint_annual_benefit_eur/1e6:+.2f} M€/an")
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