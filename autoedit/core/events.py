"""
AutoEdit Pro — Event System
Decoupled event bus for pipeline progress, UI updates, and logging.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Coroutine
from uuid import uuid4

from autoedit.utils.logging import get_logger

logger = get_logger(__name__)


class EventType(str, Enum):
    """Pipeline event types."""
    # Pipeline lifecycle
    PIPELINE_STARTED = "pipeline.started"
    PIPELINE_COMPLETED = "pipeline.completed"
    PIPELINE_FAILED = "pipeline.failed"
    PIPELINE_CANCELLED = "pipeline.cancelled"

    # Stage lifecycle
    STAGE_STARTED = "stage.started"
    STAGE_PROGRESS = "stage.progress"
    STAGE_COMPLETED = "stage.completed"
    STAGE_FAILED = "stage.failed"
    STAGE_SKIPPED = "stage.skipped"

    # Specific operations
    INGESTION_STARTED = "ingestion.started"
    TRANSCRIPTION_STARTED = "transcription.started"
    TRANSCRIPTION_COMPLETED = "transcription.completed"
    ANALYSIS_STARTED = "analysis.started"
    ANALYSIS_COMPLETED = "analysis.completed"
    DECISION_STARTED = "decision.started"
    DECISION_COMPLETED = "decision.completed"
    ASSET_SEARCH_STARTED = "asset.search.started"
    ASSET_DOWNLOAD_STARTED = "asset.download.started"
    ASSET_DOWNLOAD_PROGRESS = "asset.download.progress"
    ASSET_DOWNLOAD_COMPLETED = "asset.download.completed"
    TIMELINE_BUILD_STARTED = "timeline.build.started"
    TIMELINE_BUILD_COMPLETED = "timeline.build.completed"
    EXPORT_STARTED = "export.started"
    EXPORT_COMPLETED = "export.completed"
    LEARNING_STARTED = "learning.started"
    LEARNING_COMPLETED = "learning.completed"
    SCORING_COMPLETED = "scoring.completed"

    # UI events
    UI_REVIEW_REQUIRED = "ui.review.required"
    UI_USER_ACTION = "ui.user.action"

    # Errors
    ERROR = "error"
    WARNING = "warning"


@dataclass
class PipelineEvent:
    """Event emitted during pipeline execution."""
    type: EventType
    run_id: str
    stage: str | None = None
    message: str = ""
    data: dict[str, Any] = field(default_factory=dict)
    progress: float | None = None  # 0.0 - 1.0
    timestamp: datetime = field(default_factory=datetime.utcnow)
    event_id: str = field(default_factory=lambda: str(uuid4())[:8])

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type.value,
            "run_id": self.run_id,
            "stage": self.stage,
            "message": self.message,
            "data": self.data,
            "progress": self.progress,
            "timestamp": self.timestamp.isoformat(),
            "event_id": self.event_id,
        }


# Type alias for event handlers
EventHandler = Callable[[PipelineEvent], Coroutine[Any, Any, None] | None]


class EventBus:
    """
    Async event bus for pipeline events.
    Supports sync and async handlers, filtering by event type.
    """

    def __init__(self):
        self._handlers: dict[EventType, list[EventHandler]] = {}
        self._wildcard_handlers: list[EventHandler] = []
        self._history: list[PipelineEvent] = []
        self._max_history = 1000

    def subscribe(
        self,
        event_type: EventType,
        handler: EventHandler,
    ) -> Callable[[], None]:
        """
        Subscribe to an event type.

        Returns:
            Unsubscribe function
        """
        if event_type not in self._handlers:
            self._handlers[event_type] = []
        self._handlers[event_type].append(handler)

        def unsubscribe():
            self._handlers[event_type].remove(handler)

        return unsubscribe

    def subscribe_all(self, handler: EventHandler) -> Callable[[], None]:
        """Subscribe to all events."""
        self._wildcard_handlers.append(handler)

        def unsubscribe():
            self._wildcard_handlers.remove(handler)

        return unsubscribe

    async def emit(self, event: PipelineEvent) -> None:
        """Emit an event to all subscribers."""
        # Store in history
        self._history.append(event)
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history:]

        # Log event
        logger.debug(
            "event_emitted",
            type=event.type.value,
            run_id=event.run_id,
            stage=event.stage,
            message=event.message,
            progress=event.progress,
        )

        # Call specific handlers
        handlers = self._handlers.get(event.type, [])
        for handler in handlers:
            try:
                result = handler(event)
                if asyncio.iscoroutine(result):
                    await result
            except Exception as e:
                logger.error(
                    "event_handler_error",
                    event_type=event.type.value,
                    handler=handler.__name__,
                    error=str(e),
                )

        # Call wildcard handlers
        for handler in self._wildcard_handlers:
            try:
                result = handler(event)
                if asyncio.iscoroutine(result):
                    await result
            except Exception as e:
                logger.error(
                    "wildcard_handler_error",
                    handler=handler.__name__,
                    error=str(e),
                )

    def get_history(
        self,
        run_id: str | None = None,
        event_type: EventType | None = None,
        limit: int = 100,
    ) -> list[PipelineEvent]:
        """Get event history with optional filters."""
        events = self._history
        if run_id:
            events = [e for e in events if e.run_id == run_id]
        if event_type:
            events = [e for e in events if e.type == event_type]
        return events[-limit:]

    def clear_history(self) -> None:
        """Clear event history."""
        self._history.clear()


# Global event bus instance
_event_bus: EventBus | None = None


def get_event_bus() -> EventBus:
    """Get global event bus instance."""
    global _event_bus
    if _event_bus is None:
        _event_bus = EventBus()
    return _event_bus


# Convenience functions
async def emit_event(
    event_type: EventType,
    run_id: str,
    stage: str | None = None,
    message: str = "",
    data: dict[str, Any] | None = None,
    progress: float | None = None,
) -> None:
    """Emit a pipeline event."""
    event = PipelineEvent(
        type=event_type,
        run_id=run_id,
        stage=stage,
        message=message,
        data=data or {},
        progress=progress,
    )
    await get_event_bus().emit(event)


def create_run_id() -> str:
    """Generate a unique run ID."""
    return str(uuid4())[:12]