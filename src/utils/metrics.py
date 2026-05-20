"""
Pipeline metrics tracker — captures latency, counts, and metadata
for every pipeline step. Persists to JSON for dashboard consumption.
"""
import time
import json
import threading
from pathlib import Path
from datetime import datetime, timezone
from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass
class StepMetric:
    """Metrics captured for a single pipeline step execution."""
    step_name: str
    status: str = "pending"          # pending | running | success | failed
    start_time: str | None = None
    end_time: str | None = None
    latency_seconds: float = 0.0
    items_processed: int = 0
    items_failed: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


class MetricsTracker:
    """
    Thread-safe pipeline metrics tracker.
    Accumulates per-step metrics and persists them to a JSON file
    so the dashboard can read them in real-time.
    """

    def __init__(self, run_id: str | None = None):
        self.run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        self.steps: dict[str, StepMetric] = {}
        self._lock = threading.RLock()
        self._persist_dir = (
            Path(__file__).resolve().parent.parent.parent
            / "logs" / "pipeline_runs"
        )
        self._persist_dir.mkdir(parents=True, exist_ok=True)
        self._timers: dict[str, float] = {}

    # ── lifecycle ────────────────────────────────────────────
    def start_step(self, step_name: str, **extra_meta: Any) -> None:
        with self._lock:
            self._timers[step_name] = time.perf_counter()
            self.steps[step_name] = StepMetric(
                step_name=step_name,
                status="running",
                start_time=datetime.now(timezone.utc).isoformat(),
                metadata=extra_meta,
            )
            self._persist()

    def end_step(
        self,
        step_name: str,
        items_processed: int = 0,
        items_failed: int = 0,
        **extra_meta: Any,
    ) -> None:
        with self._lock:
            m = self.steps.get(step_name)
            if m is None:
                return
            m.status = "success"
            m.end_time = datetime.now(timezone.utc).isoformat()
            m.latency_seconds = round(
                time.perf_counter() - self._timers.get(step_name, 0), 3
            )
            m.items_processed = items_processed
            m.items_failed = items_failed
            m.metadata.update(extra_meta)
            self._persist()

    def fail_step(self, step_name: str, error: str) -> None:
        with self._lock:
            m = self.steps.get(step_name)
            if m is None:
                m = StepMetric(step_name=step_name)
                self.steps[step_name] = m
            m.status = "failed"
            m.end_time = datetime.now(timezone.utc).isoformat()
            m.latency_seconds = round(
                time.perf_counter() - self._timers.get(step_name, 0), 3
            )
            m.error = error
            self._persist()

    # ── query ────────────────────────────────────────────────
    def snapshot(self) -> dict:
        with self._lock:
            return {
                "run_id": self.run_id,
                "steps": {k: asdict(v) for k, v in self.steps.items()},
            }

    # ── persistence ──────────────────────────────────────────
    def _persist(self) -> None:
        path = self._persist_dir / f"{self.run_id}.json"
        path.write_text(json.dumps(self.snapshot(), indent=2), encoding="utf-8")

    @classmethod
    def list_runs(cls) -> list[str]:
        d = Path(__file__).resolve().parent.parent.parent / "logs" / "pipeline_runs"
        return sorted([f.stem for f in d.glob("*.json")], reverse=True)

    @classmethod
    def load_run(cls, run_id: str) -> dict:
        path = (
            Path(__file__).resolve().parent.parent.parent
            / "logs" / "pipeline_runs" / f"{run_id}.json"
        )
        return json.loads(path.read_text(encoding="utf-8"))
