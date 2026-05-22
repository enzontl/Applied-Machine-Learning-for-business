"""Gestion en mémoire des jobs en cours (single-process).

Pour la prod, on remplacerait par Redis ou similaire. Pour la démo, un dict
suffit largement.
"""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any


@dataclass
class JobState:
    job_id: str
    status: str = "queued"          # queued | running | done | error
    progress: float = 0.0
    step: str = ""
    started_at: float = field(default_factory=time.time)
    elapsed_s: float = 0.0
    error: str | None = None
    result: dict[str, Any] | None = None

    def update(self, **fields: Any) -> None:
        for k, v in fields.items():
            setattr(self, k, v)
        self.elapsed_s = time.time() - self.started_at


class JobRegistry:
    """Registre thread-safe des jobs."""

    def __init__(self) -> None:
        self._jobs: dict[str, JobState] = {}
        self._lock = threading.Lock()

    def create(self) -> JobState:
        job = JobState(job_id=uuid.uuid4().hex[:12])
        with self._lock:
            self._jobs[job.job_id] = job
        return job

    def get(self, job_id: str) -> JobState | None:
        with self._lock:
            return self._jobs.get(job_id)

    def list_ids(self) -> list[str]:
        with self._lock:
            return list(self._jobs.keys())


# Singleton pour le process
REGISTRY = JobRegistry()
