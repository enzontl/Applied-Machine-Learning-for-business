"""App FastAPI principale : sert le frontend statique + endpoints REST.

Lancement local ::

    uvicorn api.main:app --reload --port 8000

Puis ouvrir http://localhost:8000.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .jobs import REGISTRY
from .models import JobStatus, RunRequest
from .pipeline import PROFILE_ORDER, run_pipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppression des logs verbeux d'urban_optimizer (on garde WARNING+)
logging.getLogger("urban_optimizer").setLevel(logging.WARNING)

app = FastAPI(
    title="Urban Optimizer",
    description="API d'optimisation de plan urbain — backend pour la démo data-viz.",
    version="0.2.0",
)

# ── Frontend statique ─────────────────────────────────────────────────────
FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/assets", StaticFiles(directory=str(FRONTEND_DIR / "assets")), name="assets")


@app.get("/", include_in_schema=False)
def serve_index() -> FileResponse:
    index = FRONTEND_DIR / "index.html"
    if not index.exists():
        raise HTTPException(404, "frontend/index.html absent — front non build")
    return FileResponse(str(index))


@app.get("/dashboard", include_in_schema=False)
def serve_dashboard() -> FileResponse:
    dash = FRONTEND_DIR / "dashboard.html"
    if not dash.exists():
        raise HTTPException(404, "frontend/dashboard.html absent")
    return FileResponse(str(dash))


# ── API ───────────────────────────────────────────────────────────────────

PRESET_CITIES = [
    {"slug": "villeurbanne", "label": "Villeurbanne", "osm": "Villeurbanne, France", "lat": 45.7665, "lng": 4.8795, "population": 152_000, "size": "S"},
    {"slug": "lyon", "label": "Lyon", "osm": "Lyon, France", "lat": 45.7640, "lng": 4.8357, "population": 522_000, "size": "L"},
    {"slug": "lille", "label": "Lille", "osm": "Lille, France", "lat": 50.6292, "lng": 3.0573, "population": 235_000, "size": "M"},
    {"slug": "bordeaux", "label": "Bordeaux", "osm": "Bordeaux, France", "lat": 44.8378, "lng": -0.5792, "population": 260_000, "size": "M"},
    {"slug": "nantes", "label": "Nantes", "osm": "Nantes, France", "lat": 47.2184, "lng": -1.5536, "population": 322_000, "size": "M"},
    {"slug": "rennes", "label": "Rennes", "osm": "Rennes, France", "lat": 48.1173, "lng": -1.6778, "population": 220_000, "size": "M"},
]


@app.get("/api/cities")
def list_cities() -> list[dict]:
    """Villes pré-configurées pour le sélecteur du landing."""
    return PRESET_CITIES


@app.get("/api/profiles")
def list_profiles() -> list[dict]:
    """Profils maire disponibles."""
    from urban_optimizer.optimization import ALL_PROFILES
    by_name = {p.name: p for p in ALL_PROFILES}
    return [
        {
            "name": name,
            "label": by_name[name].label,
            "weights": {
                "time": by_name[name].w_time,
                "fuel": by_name[name].w_fuel,
                "co2": by_name[name].w_co2,
                "accessibility": by_name[name].w_accessibility,
                "equity": by_name[name].w_equity,
            },
        }
        for name in PROFILE_ORDER if name in by_name
    ]


@app.post("/api/jobs", status_code=202)
def create_job(req: RunRequest, background: BackgroundTasks) -> dict:
    """Lance le pipeline en background, renvoie un job_id à poll."""
    job = REGISTRY.create()
    req_dict = req.model_dump()
    # On utilise threading directement plutôt que BackgroundTasks (qui bloque
    # le requeste-response cycle) — utile pour permettre le polling depuis le front.
    thread = threading.Thread(target=run_pipeline, args=(job, req_dict), daemon=True)
    thread.start()
    return {"job_id": job.job_id}


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> dict:
    """Polling : retourne l'état + le résultat si done."""
    job = REGISTRY.get(job_id)
    if job is None:
        raise HTTPException(404, "Job inconnu")

    # On calcule elapsed_s live (sinon il fige entre 2 appels à job.update())
    import time as _t
    live_elapsed = _t.time() - job.started_at if job.status != "done" else job.elapsed_s
    payload = {
        "job_id": job.job_id,
        "status": job.status,
        "progress": round(job.progress, 3),
        "step": job.step,
        "elapsed_s": round(live_elapsed, 1),
        "error": job.error,
    }
    if job.status == "done" and job.result is not None:
        payload.update(job.result)
    return payload


@app.get("/api/health")
def health() -> dict:
    return {"ok": True, "jobs_in_memory": len(REGISTRY.list_ids())}
