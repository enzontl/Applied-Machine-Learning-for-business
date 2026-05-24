# urban_optimizer

A framework for optimizing urban road networks using OD traffic assignment and urban plan evaluation.

The project combines:
1. Urban road network construction from OSM (with optional ROUTE500 integration).
2. OD matrix generation (grid, H3, or IRIS zoning).
3. Traffic assignment (All-or-Nothing, User Equilibrium, System Optimum).
4. Network diagnostics (VHT, congestion, saturation, critical links, **accessibility + Gini equity**, price of anarchy).
5. Optimization (candidate ranking, new link proposals under budget constraints, **joint re-evaluation**, **robustness testing**, **Pareto frontier**).
6. **A modern web app** (FastAPI + HTML/JS data-viz front) to run the full pipeline end-to-end with **before/after saturation maps**, multi-profile comparison, Pareto curve, robustness test and Braess removals.
7. A legacy Streamlit dashboard (still available but deprecated in favor of the new front).

---

## Current Project Status

Features currently implemented:

- **network**: `build_network`, `build_unified_network`, OSM/ROUTE500 loaders, hard obstacles (buildings, parks) + bridge triggers (rail, water).
- **demand**: `generate_od_matrix`, `gravity_od`, grid/H3/IRIS zoning.
- **assignment**: `solve_all_or_nothing`, `solve_user_equilibrium`, `solve_system_optimum`, `price_of_anarchy`.
- **diagnosis**: `diagnose`, critical-link ranking, **accessibility report (isochrone + Gini equity)**.
- **optimization**:
  - 3 intervention types: **widening** (corridor), **upgrade** (residential → arterial), **new route** (visibility-graph A* on building corners).
  - **Periphery filter**: new routes restricted to peripheral areas (avoids failures in dense centers).
  - **Joint plan re-evaluation**: single Frank-Wolfe pass with all interventions applied → real ΔVHT + redundancy detection.
  - **Enriched score**: time + fuel + CO2 + accessibility benefit − equity (Gini) penalty, weighted by mayor profile.
  - **Robustness testing**: re-FW under demand × {0.8, 1.0, 1.2, 1.5}.
  - **Pareto frontier**: budget → benefit curve across 6 budget levels with sweet-spot detection.
- **api/ + frontend/**: FastAPI backend + HTML/JS frontend (data-viz pop style) — landing page with city/profile picker, dashboard with KPI cards, Leaflet map (saturation + plan overlay + Braess), Pareto chart (Chart.js), robustness verdict, multi-profile comparison table.
- **streamlit_app.py**: legacy interactive dashboard with before/after maps, KPIs, optional robustness + Pareto.

---

## Pipeline Overview

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

---

## Project Structure

```
.
├── api/                     # FastAPI backend (new web app)
│   ├── main.py              # app + REST endpoints + static frontend serving
│   ├── pipeline.py          # async orchestrator + GeoJSON serialization
│   ├── models.py            # Pydantic schemas
│   └── jobs.py              # in-memory thread-safe job registry
├── frontend/                # HTML/CSS/JS frontend (no framework)
│   ├── index.html           # landing page (city/profile picker + sliders/toggles)
│   ├── dashboard.html       # results view (KPIs + Leaflet map + analyses)
│   └── assets/
│       ├── style.css        # design system (Space Grotesk / Inter / JetBrains Mono)
│       └── app.js           # logic (vanilla JS, fetch API, Leaflet, Chart.js)
├── streamlit_app.py         # legacy dashboard (deprecated)
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

---

## Requirements

- Python >= 3.10 (defined in `pyproject.toml`)
- Recommended: Python 3.11 or 3.12 to avoid typing incompatibilities with Python 3.9

---

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pip install "fastapi" "uvicorn[standard]"   # for the new web app
```

Quick check:

```bash
python -c "import urban_optimizer; print(urban_optimizer.__version__)"
```

---

## Run the Web App (recommended)

```bash
source .venv/bin/activate
uvicorn api.main:app --reload --port 8000
```

Then open **<http://localhost:8000>**.

The web app lets you:
- Pick a city among 6 preset ones (Villeurbanne, Lyon, Lille, Bordeaux, Nantes, Rennes), pick a mayor profile (Équilibré / Écolo / Mobilité / Économique).
- Tune all parameters via sliders: budget, peak hour, zoning cells, demand intensity, FW candidates, periphery margin, isochrone threshold, UE iterations.
- Toggle optional analyses: **simplified network** (3–4× faster), **ROUTE500**, **multi-profile comparison** (4 mayors side by side), **robustness test**, **Pareto curve**, **Braess removals**.
- Launch in a background thread, polled every 1.5 s; the dashboard auto-opens when done.
- Visualize the city's annual score (time / fuel / CO2 / accessibility / equity), the proposed plan with each intervention's detour-before / ΔVHT / BCR / payback, an interactive Leaflet map of saturation v/c colored arcs + plan overlay + numbered markers + Braess overlay, the joint re-evaluation (ΔVHT naive vs joint, redundancy %, BCR), the robustness verdict, the Pareto chart with sweet-spot, the multi-profile comparison table, and a technical footer (n_nodes, n_edges, FW gap, etc.).

### REST endpoints

| Method | URL                       | Purpose                                                |
|--------|---------------------------|--------------------------------------------------------|
| GET    | `/`                       | Landing page                                           |
| GET    | `/dashboard`              | Results dashboard                                      |
| GET    | `/api/health`             | Healthcheck                                            |
| GET    | `/api/cities`             | List of preset cities                                  |
| GET    | `/api/profiles`           | List of mayor profiles with their weights              |
| POST   | `/api/jobs`               | Start a pipeline → returns `{ job_id }`                |
| GET    | `/api/jobs/{job_id}`      | Poll status / progress; full result when `status=done` |
| GET    | `/docs`                   | Swagger UI (auto-generated by FastAPI)                 |

---

## Run the Legacy Streamlit Dashboard

```bash
source .venv/bin/activate
streamlit run streamlit_app.py
```

The dashboard allows you to:
- Build a city network from OSM (with hard/soft obstacle layers).
- Generate OD demand.
- Solve user equilibrium (Frank-Wolfe).
- Display an annual city score based on mayor profile — including **accessibility** (zones reachable in T min) and **equity** (Gini coefficient).
- Propose urban plans combining **widening / upgrade / new route** under budget — with **joint re-evaluation** (real ΔVHT, redundancy detection).
- Display **before/after saturation maps** side by side (toggle modes available).
- Compare accessibility + Gini before/after the plan, with an automatic "egalitarian / inegalitarian / neutral" verdict.
- Run an optional **robustness test** (4 demand scenarios: ×0.8 / ×1.0 / ×1.2 / ×1.5).
- Compute an optional **Pareto curve** (6 budget levels with sweet-spot detection).
- Apply optional **Braess removals** (arcs whose deletion improves the network).

---

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
net       = build_network("Villeurbanne, France", include_route500=False)
obstacles = load_obstacles("Villeurbanne, France")       # hard: buildings, parks
bridges   = load_bridge_triggers("Villeurbanne, France") # soft: rail, water

# Demand + equilibrium
od = generate_od_matrix(net, hour=8, method="grid", n_cells=10, scale_factor=0.3)
ue = solve_user_equilibrium(net, od, max_iter=100, tol=1e-4)

# Diagnosis + enriched score
diag   = diagnose(net, ue)
access = compute_accessibility(net, od, ue, threshold_seconds=15 * 60)
score  = score_network(ue, ECOLO, access=access)
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

---

## Data

- **OSM**: downloaded automatically via OSMnx.
- **ROUTE500** *(optional)*: place source files in `data/raw/` if enabled.
- **INSEE/IRIS** *(optional, depending on zoning method)*: place source files in `data/external/`.

> If external datasets are missing, use grid zoning for a quick start.

---

## Tests

Run all tests:

```bash
pytest
```

Run a single test module:

```bash
pytest tests/test_network.py -q
```

---

## Notebooks

Exploratory and network-construction notebooks available in `notebooks/`:
- `00_setup_check.ipynb`
- `01_network_construction.ipynb`
- `EDA.ipynb`

---

## Configuration

An example city configuration is available in `configs/lyon.yaml` (network, demand, assignment, budget).

---

## Quick Troubleshooting

**`TypeError: unsupported operand type(s) for |: 'type' and 'NoneType'`**
- Cause: running the app with Python 3.9.
- Fix: recreate the environment with Python 3.10+ and reinstall dependencies.

**Web app stuck on "Optimisation en cours…" loader**
- The pipeline runs in a background thread; the loader polls every 1.5 s. Open the browser console (F12) — `[uo]` lines show the polling state. If "Connexion perdue" appears, restart `uvicorn --reload`. You can also inspect directly: `curl http://localhost:8000/api/jobs/{job_id}`.
- The "Optimisation profil…" step is monolithic (1–4 min for Villeurbanne with default params) and intermediate progress per-candidate is not yet reported.

**First run on a new city is very long (5–10 min)**
- OSM + obstacles download + processing. Subsequent runs use the disk cache (`data/raw/network_cache/` + `data/processed/urban_cache/`) and load in a few seconds.
