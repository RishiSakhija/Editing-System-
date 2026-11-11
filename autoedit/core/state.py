"""
AutoEdit Pro — Pipeline State Management
Persistent run state with resume capability.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

from autoedit.config import get_settings
from autoedit.utils.logging import get_logger

logger = get_logger(__name__)


class RunStatus(str, Enum):
    """Pipeline run status."""
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    AWAITING_REVIEW = "awaiting_review"


class StageStatus(str, Enum):
    """Individual stage status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class StageState:
    """State of a single pipeline stage."""
    name: str
    status: StageStatus = StageStatus.PENDING
    started_at: datetime | None = None
    completed_at: datetime | None = None
    progress: float = 0.0  # 0.0 - 1.0
    input_data: dict[str, Any] = field(default_factory=dict)
    output_data: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    retry_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status.value,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "progress": self.progress,
            "input_data": self.input_data,
            "output_data": self.output_data,
            "error": self.error,
            "retry_count": self.retry_count,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> StageState:
        return cls(
            name=data["name"],
            status=StageStatus(data["status"]),
            started_at=datetime.fromisoformat(data["started_at"]) if data.get("started_at") else None,
            completed_at=datetime.fromisoformat(data["completed_at"]) if data.get("completed_at") else None,
            progress=data.get("progress", 0.0),
            input_data=data.get("input_data", {}),
            output_data=data.get("output_data", {}),
            error=data.get("error"),
            retry_count=data.get("retry_count", 0),
        )


@dataclass
class RunState:
    """Complete pipeline run state."""
    run_id: str
    status: RunStatus = RunStatus.PENDING
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    started_at: datetime | None = None
    completed_at: datetime | None = None
    stages: dict[str, StageState] = field(default_factory=dict)
    input_config: dict[str, Any] = field(default_factory=dict)
    output_paths: dict[str, str] = field(default_factory=dict)
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def duration_seconds(self) -> float | None:
        """Total run duration in seconds."""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        if self.started_at:
            return (datetime.utcnow() - self.started_at).total_seconds()
        return None

    @property
    def current_stage(self) -> StageState | None:
        """Get currently running stage."""
        for stage in self.stages.values():
            if stage.status == StageStatus.RUNNING:
                return stage
        return None

    @property
    def completed_stages(self) -> list[StageState]:
        """Get list of completed stages in order."""
        return [s for s in self.stages.values() if s.status == StageStatus.COMPLETED]

    @property
    def failed_stages(self) -> list[StageState]:
        """Get list of failed stages."""
        return [s for s in self.stages.values() if s.status == StageStatus.FAILED]

    @property
    def progress(self) -> float:
        """Overall progress 0.0 - 1.0."""
        if not self.stages:
            return 0.0
        total = len(self.stages)
        completed = sum(1 for s in self.stages.values() if s.status == StageStatus.COMPLETED)
        running = sum(s.progress for s in self.stages.values() if s.status == StageStatus.RUNNING)
        return (completed + running) / total

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "stages": {k: v.to_dict() for k, v in self.stages.items()},
            "input_config": self.input_config,
            "output_paths": self.output_paths,
            "error": self.error,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RunState:
        return cls(
            run_id=data["run_id"],
            status=RunStatus(data["status"]),
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
            started_at=datetime.fromisoformat(data["started_at"]) if data.get("started_at") else None,
            completed_at=datetime.fromisoformat(data["completed_at"]) if data.get("completed_at") else None,
            stages={k: StageState.from_dict(v) for k, v in data.get("stages", {}).items()},
            input_config=data.get("input_config", {}),
            output_paths=data.get("output_paths", {}),
            error=data.get("error"),
            metadata=data.get("metadata", {}),
        )


class StateStore:
    """
    Persistent state store using SQLite.
    Supports resume, history, and querying.
    """

    def __init__(self, db_path: Path | None = None):
        settings = get_settings()
        self.db_path = db_path or settings.hardware.db_dir / "runs.sqlite"
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        """Initialize database schema."""
        with self._conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    started_at TEXT,
                    completed_at TEXT,
                    input_config TEXT NOT NULL,
                    output_paths TEXT NOT NULL,
                    error TEXT,
                    metadata TEXT NOT NULL,
                    state_json TEXT NOT NULL
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_runs_status ON runs(status)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_runs_created ON runs(created_at)
            """)
            conn.commit()

    @contextmanager
    def _conn(self):
        """Get database connection."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def save(self, state: RunState) -> None:
        """Save run state to database."""
        state.updated_at = datetime.utcnow()
        with self._conn() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO runs
                (run_id, status, created_at, updated_at, started_at, completed_at,
                 input_config, output_paths, error, metadata, state_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                state.run_id,
                state.status.value,
                state.created_at.isoformat(),
                state.updated_at.isoformat(),
                state.started_at.isoformat() if state.started_at else None,
                state.completed_at.isoformat() if state.completed_at else None,
                json.dumps(state.input_config),
                json.dumps(state.output_paths),
                state.error,
                json.dumps(state.metadata),
                json.dumps(state.to_dict()),
            ))
            conn.commit()
        logger.debug("state_saved", run_id=state.run_id, status=state.status.value)

    def load(self, run_id: str) -> RunState | None:
        """Load run state from database."""
        with self._conn() as conn:
            row = conn.execute(
                "SELECT state_json FROM runs WHERE run_id = ?",
                (run_id,)
            ).fetchone()
            if row:
                data = json.loads(row["state_json"])
                return RunState.from_dict(data)
        return None

    def delete(self, run_id: str) -> bool:
        """Delete run state."""
        with self._conn() as conn:
            cursor = conn.execute("DELETE FROM runs WHERE run_id = ?", (run_id,))
            conn.commit()
            return cursor.rowcount > 0

    def list_runs(
        self,
        status: RunStatus | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[RunState]:
        """List runs with optional status filter."""
        query = "SELECT state_json FROM runs"
        params = []
        if status:
            query += " WHERE status = ?"
            params.append(status.value)
        query += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        runs = []
        with self._conn() as conn:
            for row in conn.execute(query, params):
                data = json.loads(row["state_json"])
                runs.append(RunState.from_dict(data))
        return runs

    def get_latest_run(self) -> RunState | None:
        """Get most recent run."""
        runs = self.list_runs(limit=1)
        return runs[0] if runs else None

    def cleanup_old_runs(self, keep: int = 100) -> int:
        """Delete runs beyond keep limit."""
        with self._conn() as conn:
            cursor = conn.execute("""
                DELETE FROM runs
                WHERE run_id NOT IN (
                    SELECT run_id FROM runs ORDER BY created_at DESC LIMIT ?
                )
            """, (keep,))
            conn.commit()
            return cursor.rowcount


# Global state store instance
_state_store: StateStore | None = None


def get_state_store() -> StateStore:
    """Get global state store instance."""
    global _state_store
    if _state_store is None:
        _state_store = StateStore()
    return _state_store


def create_run_state(
    input_config: dict[str, Any],
    stage_names: list[str],
    run_id: str | None = None,
) -> RunState:
    """Create a new run state with initialized stages."""
    run_id = run_id or str(uuid4())[:12]
    stages = {
        name: StageState(name=name)
        for name in stage_names
    }
    return RunState(
        run_id=run_id,
        status=RunStatus.PENDING,
        stages=stages,
        input_config=input_config,
    )