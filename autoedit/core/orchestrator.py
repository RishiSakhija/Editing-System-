"""
AutoEdit Pro — Main Pipeline Orchestrator
Coordinates all stages, handles review loops, and manages the complete editing workflow.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from autoedit.config import get_settings
from autoedit.core.events import (
    EventType,
    PipelineEvent,
    create_run_id,
    emit_event,
    get_event_bus,
)
from autoedit.core.pipeline import Pipeline, PipelineStage, PipelineError
from autoedit.core.state import (
    RunState,
    RunStatus,
    StageState,
    StageStatus,
    StateStore,
    create_run_state,
    get_state_store,
)
from autoedit.utils.logging import get_logger, LogContext

logger = get_logger(__name__)


# Stage names in execution order
PIPELINE_STAGES = [
    "ingestion",
    "transcription",
    "analysis",
    "decision",
    "asset_resolution",
    "timeline_build",
    "review",
    "export",
    "learning",
    "scoring",
]


@dataclass
class OrchestratorConfig:
    """Orchestrator configuration."""
    auto_review: bool = False  # If True, skip review UI
    max_review_iterations: int = 3
    save_intermediate: bool = True
    cleanup_on_success: bool = False


@dataclass
class OrchestratorResult:
    """Result of a full pipeline run."""
    success: bool
    run_id: str
    run_state: RunState
    output_paths: dict[str, str] = field(default_factory=dict)
    viral_score: dict[str, Any] | None = None
    error: str | None = None
    review_iterations: int = 0


class Orchestrator:
    """
    Main orchestrator for the AutoEdit Pro pipeline.
    Manages the complete workflow from raw footage to finished video.
    """

    def __init__(
        self,
        config: OrchestratorConfig | None = None,
        state_store: StateStore | None = None,
        event_bus: Any = None,
    ):
        self.config = config or OrchestratorConfig()
        self.state_store = state_store or get_state_store()
        self.event_bus = event_bus or get_event_bus()
        self._stages: dict[str, PipelineStage] = {}
        self._review_ui_task: asyncio.Task | None = None

    def register_stage(self, stage: PipelineStage) -> None:
        """Register a pipeline stage."""
        self._stages[stage.name] = stage
        logger.debug("stage_registered", stage=stage.name)

    def get_stage(self, name: str) -> PipelineStage | None:
        """Get registered stage by name."""
        return self._stages.get(name)

    async def run(
        self,
        input_path: Path,
        skill_pack: str = "educational_clean",
        script_path: Path | None = None,
        output_dir: Path | None = None,
        run_id: str | None = None,
        resume: bool = False,
    ) -> OrchestratorResult:
        """
        Execute the complete editing pipeline.

        Args:
            input_path: Path to raw footage (video/audio/images)
            skill_pack: Name of skill pack to use
            script_path: Optional script text file for alignment
            output_dir: Output directory (defaults to input_dir/autoedit_output)
            run_id: Optional run ID (generated if not provided)
            resume: Resume from previous run

        Returns:
            OrchestratorResult with output paths and status
        """
        run_id = run_id or create_run_id()

        # Setup paths
        input_path = Path(input_path).resolve()
        if not input_path.exists():
            return OrchestratorResult(
                success=False,
                run_id=run_id,
                run_state=RunState(run_id=run_id, status=RunStatus.FAILED),
                error=f"Input not found: {input_path}",
            )

        output_dir = output_dir or (input_path.parent / "autoedit_output")
        output_dir.mkdir(parents=True, exist_ok=True)

        # Prepare input config
        input_config = {
            "input_path": str(input_path),
            "skill_pack": skill_pack,
            "script_path": str(script_path) if script_path else None,
            "output_dir": str(output_dir),
            "auto_review": self.config.auto_review,
        }

        logger.info("orchestrator_start", run_id=run_id, input=str(input_path), skill=skill_pack)

        # Create or load run state
        if resume:
            run_state = self.state_store.load(run_id)
            if not run_state:
                return OrchestratorResult(
                    success=False,
                    run_id=run_id,
                    run_state=RunState(run_id=run_id, status=RunStatus.FAILED),
                    error=f"Run not found for resume: {run_id}",
                )
            run_state.status = RunStatus.RUNNING
            run_state.started_at = datetime.utcnow()
        else:
            run_state = create_run_state(input_config, PIPELINE_STAGES, run_id)
            run_state.status = RunStatus.RUNNING
            run_state.started_at = datetime.utcnow()

        self.state_store.save(run_state)

        # Emit pipeline started event
        await emit_event(
            EventType.PIPELINE_STARTED,
            run_id=run_id,
            message=f"Starting pipeline for {input_path.name}",
            data=input_config,
        )

        try:
            # Build pipeline with registered stages
            pipeline_stages = [
                self._stages[name]
                for name in PIPELINE_STAGES
                if name in self._stages
            ]

            if not pipeline_stages:
                raise PipelineError("No stages registered")

            pipeline = Pipeline(run_id, pipeline_stages, self.state_store, self.event_bus)

            # Execute or resume
            if resume:
                outputs = await pipeline.resume(run_state)
            else:
                outputs = await pipeline.execute(input_config)

            # Update run state with outputs
            run_state.output_paths = outputs.get("export", {}).get("output_paths", {})
            run_state.status = RunStatus.COMPLETED
            run_state.completed_at = datetime.utcnow()

            # Extract viral score if available
            viral_score = outputs.get("scoring", {}).get("viral_score")

            self.state_store.save(run_state)

            await emit_event(
                EventType.PIPELINE_COMPLETED,
                run_id=run_id,
                message="Pipeline completed successfully",
                data={"outputs": list(run_state.output_paths.keys())},
            )

            logger.info("orchestrator_complete", run_id=run_id, outputs=list(run_state.output_paths.keys()))

            return OrchestratorResult(
                success=True,
                run_id=run_id,
                run_state=run_state,
                output_paths=run_state.output_paths,
                viral_score=viral_score,
            )

        except PipelineError as e:
            run_state.status = RunStatus.FAILED
            run_state.completed_at = datetime.utcnow()
            run_state.error = str(e)
            self.state_store.save(run_state)

            await emit_event(
                EventType.PIPELINE_FAILED,
                run_id=run_id,
                message=f"Pipeline failed: {e}",
                data={"error": str(e)},
            )

            logger.error("orchestrator_failed", run_id=run_id, error=str(e))

            return OrchestratorResult(
                success=False,
                run_id=run_id,
                run_state=run_state,
                error=str(e),
            )

        except Exception as e:
            run_state.status = RunStatus.FAILED
            run_state.completed_at = datetime.utcnow()
            run_state.error = f"{type(e).__name__}: {e}"
            self.state_store.save(run_state)

            await emit_event(
                EventType.PIPELINE_FAILED,
                run_id=run_id,
                message=f"Pipeline crashed: {e}",
                data={"error": str(e), "type": type(e).__name__},
            )

            logger.exception("orchestrator_crashed", run_id=run_id)
            return OrchestratorResult(
                success=False,
                run_id=run_id,
                run_state=run_state,
                error=f"{type(e).__name__}: {e}",
            )

    async def run_review_loop(
        self,
        run_id: str,
        timeline_path: Path,
        max_iterations: int | None = None,
    ) -> OrchestratorResult:
        """
        Run interactive review loop via Streamlit UI.
        User can approve, tweak, or request changes.
        """
        max_iter = max_iterations or self.config.max_review_iterations
        run_state = self.state_store.load(run_id)

        if not run_state:
            return OrchestratorResult(
                success=False,
                run_id=run_id,
                run_state=RunState(run_id=run_id, status=RunStatus.FAILED),
                error="Run not found",
            )

        for iteration in range(max_iter):
            run_state.status = RunStatus.AWAITING_REVIEW
            run_state.metadata["review_iteration"] = iteration + 1
            self.state_store.save(run_state)

            await emit_event(
                EventType.UI_REVIEW_REQUIRED,
                run_id=run_id,
                message=f"Review iteration {iteration + 1}/{max_iter}",
                data={"iteration": iteration + 1, "timeline": str(timeline_path)},
            )

            # Launch Streamlit review UI (blocking)
            # In practice, this would start the UI and wait for user action
            user_action = await self._wait_for_review_action(run_id)

            if user_action == "approve":
                logger.info("review_approved", run_id=run_id, iteration=iteration + 1)
                break
            elif user_action == "tweak":
                logger.info("review_tweak_requested", run_id=run_id, iteration=iteration + 1)
                # Re-run from decision stage with user tweaks
                # Implementation depends on UI integration
                continue
            elif user_action == "cancel":
                run_state.status = RunStatus.CANCELLED
                self.state_store.save(run_state)
                return OrchestratorResult(
                    success=False,
                    run_id=run_id,
                    run_state=run_state,
                    error="Cancelled by user",
                    review_iterations=iteration + 1,
                )
        else:
            # Max iterations reached
            logger.warning("review_max_iterations", run_id=run_id, max_iter=max_iter)

        run_state.status = RunStatus.RUNNING
        self.state_store.save(run_state)

        return await self.run(
            input_path=Path(run_state.input_config["input_path"]),
            skill_pack=run_state.input_config["skill_pack"],
            script_path=Path(run_state.input_config["script_path"]) if run_state.input_config.get("script_path") else None,
            output_dir=Path(run_state.input_config["output_dir"]),
            run_id=run_id,
            resume=True,
        )

    async def _wait_for_review_action(self, run_id: str) -> str:
        """
        Wait for user action from review UI.
        In practice, this connects to Streamlit via WebSocket or polling.
        """
        # Placeholder - actual implementation uses Streamlit session state
        # or a separate message queue
        await asyncio.sleep(1)
        return "approve"  # Default for headless runs

    def cancel_run(self, run_id: str) -> bool:
        """Cancel a running pipeline."""
        run_state = self.state_store.load(run_id)
        if run_state and run_state.status in (RunStatus.RUNNING, RunStatus.AWAITING_REVIEW):
            run_state.status = RunStatus.CANCELLED
            run_state.completed_at = datetime.utcnow()
            self.state_store.save(run_state)
            return True
        return False

    def get_run_status(self, run_id: str) -> RunState | None:
        """Get current run status."""
        return self.state_store.load(run_id)

    def list_recent_runs(self, limit: int = 20) -> list[RunState]:
        """List recent pipeline runs."""
        return self.state_store.list_runs(limit=limit)


# Global orchestrator instance
_orchestrator: Orchestrator | None = None


def get_orchestrator() -> Orchestrator:
    """Get global orchestrator instance."""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = Orchestrator()
    return _orchestrator


async def run_pipeline(
    input_path: Path,
    skill_pack: str = "educational_clean",
    script_path: Path | None = None,
    output_dir: Path | None = None,
    auto_review: bool = False,
    run_id: str | None = None,
) -> OrchestratorResult:
    """Convenience function to run full pipeline."""
    orchestrator = get_orchestrator()
    orchestrator.config.auto_review = auto_review
    return await orchestrator.run(
        input_path=input_path,
        skill_pack=skill_pack,
        script_path=script_path,
        output_dir=output_dir,
        run_id=run_id,
    )