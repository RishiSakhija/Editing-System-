"""
AutoEdit Pro — Pipeline Stage Definitions
Base classes for pipeline stages with retry, timeout, and progress support.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Generic, TypeVar

from autoedit.core.events import EventType, emit_event
from autoedit.core.state import StageState, StageStatus
from autoedit.utils.logging import get_logger

logger = get_logger(__name__)

T = TypeVar("T")


@dataclass
class StageResult(Generic[T]):
    """Result of a pipeline stage execution."""
    success: bool
    data: T | None = None
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    duration_ms: float = 0.0

    @classmethod
    def ok(cls, data: T, **metadata) -> StageResult[T]:
        return cls(success=True, data=data, metadata=metadata)

    @classmethod
    def fail(cls, error: str, **metadata) -> StageResult[T]:
        return cls(success=False, error=error, metadata=metadata)


class PipelineStage(ABC, Generic[T]):
    """
    Base class for pipeline stages.
    Handles retries, timeouts, progress reporting, and state management.
    """

    name: str = "unnamed_stage"
    description: str = ""
    max_retries: int = 2
    timeout_seconds: float = 300.0
    required: bool = True  # If False, failure won't stop pipeline

    def __init__(
        self,
        run_id: str,
        state_store: Any,  # StateStore
        event_bus: Any = None,  # EventBus
    ):
        self.run_id = run_id
        self.state_store = state_store
        self.event_bus = event_bus
        self._stage_state: StageState | None = None

    @property
    def stage_state(self) -> StageState:
        """Get or create stage state."""
        if self._stage_state is None:
            # Load from state store
            run_state = self.state_store.load(self.run_id)
            if run_state and self.name in run_state.stages:
                self._stage_state = run_state.stages[self.name]
            else:
                self._stage_state = StageState(name=self.name)
        return self._stage_state

    def _save_stage_state(self) -> None:
        """Persist stage state to store."""
        run_state = self.state_store.load(self.run_id)
        if run_state:
            run_state.stages[self.name] = self._stage_state
            run_state.updated_at = datetime.utcnow()
            self.state_store.save(run_state)

    async def _emit(
        self,
        event_type: EventType,
        message: str = "",
        progress: float | None = None,
        data: dict[str, Any] | None = None,
    ) -> None:
        """Emit event for this stage."""
        await emit_event(
            event_type=event_type,
            run_id=self.run_id,
            stage=self.name,
            message=message,
            progress=progress,
            data=data or {},
        )

    async def run(self, input_data: dict[str, Any]) -> StageResult[T]:
        """
        Execute the stage with retries and error handling.
        This is the main entry point called by the pipeline.
        """
        self._stage_state.input_data = input_data
        self._stage_state.status = StageStatus.RUNNING
        self._stage_state.started_at = datetime.utcnow()
        self._stage_state.progress = 0.0
        self._save_stage_state()

        await self._emit(EventType.STAGE_STARTED, f"Starting {self.name}", 0.0)

        last_error = None
        for attempt in range(self.max_retries + 1):
            try:
                if attempt > 0:
                    self._stage_state.retry_count = attempt
                    self._save_stage_state()
                    await self._emit(
                        EventType.STAGE_STARTED,
                        f"Retry {attempt}/{self.max_retries} for {self.name}",
                        0.0,
                    )

                result = await asyncio.wait_for(
                    self.execute(input_data),
                    timeout=self.timeout_seconds,
                )

                if result.success:
                    self._stage_state.status = StageStatus.COMPLETED
                    self._stage_state.completed_at = datetime.utcnow()
                    self._stage_state.progress = 1.0
                    self._stage_state.output_data = result.metadata
                    if result.data is not None:
                        self._stage_state.output_data["_result"] = str(result.data)
                    self._save_stage_state()

                    await self._emit(
                        EventType.STAGE_COMPLETED,
                        f"Completed {self.name}",
                        1.0,
                        result.metadata,
                    )
                    return result

                last_error = result.error
                logger.warning(
                    "stage_failed",
                    stage=self.name,
                    attempt=attempt + 1,
                    max_retries=self.max_retries,
                    error=result.error,
                )

            except asyncio.TimeoutError:
                last_error = f"Timeout after {self.timeout_seconds}s"
                logger.error("stage_timeout", stage=self.name, timeout=self.timeout_seconds)

            except Exception as e:
                last_error = f"{type(e).__name__}: {e}"
                logger.exception("stage_exception", stage=self.name)

        # All retries exhausted
        self._stage_state.status = StageStatus.FAILED
        self._stage_state.completed_at = datetime.utcnow()
        self._stage_state.error = last_error
        self._save_stage_state()

        await self._emit(
            EventType.STAGE_FAILED,
            f"Failed {self.name}: {last_error}",
            0.0,
            {"error": last_error},
        )

        if self.required:
            return StageResult.fail(last_error or "Unknown error")
        else:
            # Non-required stage failed, mark as skipped and continue
            self._stage_state.status = StageStatus.SKIPPED
            self._save_stage_state()
            await self._emit(
                EventType.STAGE_SKIPPED,
                f"Skipped {self.name} (non-required): {last_error}",
                1.0,
                {"error": last_error},
            )
            return StageResult.ok(None, skipped=True, error=last_error)

    @abstractmethod
    async def execute(self, input_data: dict[str, Any]) -> StageResult[T]:
        """
        Execute the stage logic.
        Must be implemented by subclasses.

        Args:
            input_data: Data from previous stages

        Returns:
            StageResult with success status and output data
        """
        pass

    def update_progress(self, progress: float, message: str = "") -> None:
        """Update stage progress (0.0 - 1.0)."""
        self._stage_state.progress = max(0.0, min(1.0, progress))
        self._save_stage_state()
        # Note: Can't emit async event from sync method
        # Use async context or call from async method


class Pipeline:
    """
    Pipeline executor that runs stages in sequence with state persistence.
    """

    def __init__(
        self,
        run_id: str,
        stages: list[PipelineStage],
        state_store: Any,
        event_bus: Any = None,
    ):
        self.run_id = run_id
        self.stages = stages
        self.state_store = state_store
        self.event_bus = event_bus

    async def execute(self, initial_data: dict[str, Any]) -> dict[str, Any]:
        """
        Execute all stages in sequence.

        Args:
            initial_data: Initial input data for first stage

        Returns:
            Combined output data from all stages
        """
        current_data = initial_data.copy()
        all_outputs = {}

        for stage in self.stages:
            logger.info("pipeline_stage_start", run_id=self.run_id, stage=stage.name)

            result = await stage.run(current_data)

            if not result.success:
                if stage.required:
                    logger.error("pipeline_stopped", run_id=self.run_id, stage=stage.name, error=result.error)
                    raise PipelineError(f"Stage '{stage.name}' failed: {result.error}")
                else:
                    logger.warning("pipeline_stage_skipped", run_id=self.run_id, stage=stage.name)

            # Merge output data for next stage
            if result.data is not None:
                current_data[f"{stage.name}_result"] = result.data
            current_data.update(result.metadata)
            all_outputs[stage.name] = result.metadata

        return all_outputs

    async def resume(self, run_state: Any) -> dict[str, Any]:
        """
        Resume pipeline from a saved state.
        Skips completed stages, re-runs failed/pending stages.
        """
        current_data = run_state.input_config.copy()
        all_outputs = {}

        for stage in self.stages:
            stage_state = run_state.stages.get(stage.name)
            if not stage_state:
                # New stage not in saved state
                stage_state = StageState(name=stage.name)

            if stage_state.status == StageStatus.COMPLETED:
                logger.info("pipeline_resume_skip", stage=stage.name)
                # Restore output data
                current_data.update(stage_state.output_data)
                all_outputs[stage.name] = stage_state.output_data
                continue

            # Re-run pending/failed stages
            result = await stage.run(current_data)

            if not result.success:
                if stage.required:
                    raise PipelineError(f"Stage '{stage.name}' failed: {result.error}")
            else:
                current_data.update(result.metadata)
                all_outputs[stage.name] = result.metadata

        return all_outputs


class PipelineError(Exception):
    """Pipeline execution error."""
    pass