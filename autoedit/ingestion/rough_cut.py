"""
AutoEdit Pro — Rough Cut Generator
Creates initial timeline structure from script alignment and transcription.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from autoedit.ingestion.script_align import (
    ScriptAlignmentResult,
    align_script_to_transcription,
)
from autoedit.ingestion.transcribe import TranscriptionResult
from autoedit.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class RoughCutSegment:
    """Segment in the rough cut timeline."""
    index: int
    label: str
    start: float
    end: float
    duration: float
    type: str  # "script", "pause", "filler", "silence", "broll_suggestion"
    source: str  # "script", "transcription", "analysis"
    confidence: float
    metadata: dict = field(default_factory=dict)

    @property
    def is_valid(self) -> bool:
        return self.start is not None and self.end is not None and self.end > self.start


@dataclass
class RoughCutResult:
    """Complete rough cut timeline."""
    segments: list[RoughCutSegment]
    total_duration: float
    script_coverage: float
    transcription: TranscriptionResult
    alignment: ScriptAlignmentResult | None = None
    metadata: dict = field(default_factory=dict)

    def get_valid_segments(self) -> list[RoughCutSegment]:
        return [s for s in self.segments if s.is_valid]

    def get_segments_by_type(self, seg_type: str) -> list[RoughCutSegment]:
        return [s for s in self.segments if s.type == seg_type]

    def to_timeline_json(self) -> list[dict]:
        """Export as JSON for timeline builders."""
        return [
            {
                "index": s.index,
                "label": s.label,
                "start": s.start,
                "end": s.end,
                "duration": s.duration,
                "type": s.type,
                "source": s.source,
                "confidence": s.confidence,
                "metadata": s.metadata,
            }
            for s in self.get_valid_segments()
        ]


def detect_natural_pauses(
    transcription: TranscriptionResult,
    min_pause: float = 0.5,
    max_pause: float = 5.0,
) -> list[RoughCutSegment]:
    """Detect natural pauses in transcription for segment boundaries."""
    words = transcription.get_words_flat()
    pauses = []

    for i in range(len(words) - 1):
        gap = words[i + 1].start - words[i].end
        if min_pause <= gap <= max_pause:
            pauses.append(RoughCutSegment(
                index=len(pauses),
                label=f"Pause {gap:.1f}s",
                start=words[i].end,
                end=words[i + 1].start,
                duration=gap,
                type="pause",
                source="transcription",
                confidence=min(1.0, gap / 2.0),  # Longer pauses = higher confidence
                metadata={"gap_seconds": gap},
            ))

    return pauses


def detect_filler_regions(
    transcription: TranscriptionResult,
    filler_words: list[str] | None = None,
) -> list[RoughCutSegment]:
    """Detect regions with filler words for potential removal."""
    if filler_words is None:
        filler_words = ["um", "uh", "er", "ah", "like", "you know", "so", "actually", "basically", "literally"]

    filler_set = set(f.lower() for f in filler_words)
    words = transcription.get_words_flat()

    regions = []
    current_region = None

    for word in words:
        is_filler = word.word.lower().strip(".,!?") in filler_set

        if is_filler:
            if current_region is None:
                current_region = {
                    "start": word.start,
                    "end": word.end,
                    "fillers": [word.word],
                }
            else:
                current_region["end"] = word.end
                current_region["fillers"].append(word.word)
        elif current_region is not None:
            # End of filler region
            duration = current_region["end"] - current_region["start"]
            if duration >= 0.2 - 1e-9:  # Only significant filler clusters (epsilon for float precision)
                regions.append(RoughCutSegment(
                    index=len(regions),
                    label=f"Fillers: {', '.join(current_region['fillers'][:3])}",
                    start=current_region["start"],
                    end=current_region["end"],
                    duration=duration,
                    type="filler",
                    source="transcription",
                    confidence=0.8,
                    metadata={"filler_words": current_region["fillers"]},
                ))
            current_region = None

    # Handle trailing filler region
    if current_region is not None:
        duration = current_region["end"] - current_region["start"]
        if duration >= 0.2 - 1e-9:
            regions.append(RoughCutSegment(
                index=len(regions),
                label=f"Fillers: {', '.join(current_region['fillers'][:3])}",
                start=current_region["start"],
                end=current_region["end"],
                duration=duration,
                type="filler",
                source="transcription",
                confidence=0.8,
                metadata={"filler_words": current_region["fillers"]},
            ))

    return regions


def suggest_broll_points(
    transcription: TranscriptionResult,
    alignment: ScriptAlignmentResult | None = None,
    min_segment_duration: float = 10.0,
    keywords: list[str] | None = None,
) -> list[RoughCutSegment]:
    """
    Suggest B-roll insertion points based on content analysis.

    Args:
        transcription: Full transcription
        alignment: Script alignment (optional)
        min_segment_duration: Minimum duration before suggesting B-roll
        keywords: Keywords that trigger B-roll suggestions

    Returns:
        List of B-roll suggestion segments
    """
    if keywords is None:
        keywords = [
            "data", "statistics", "research", "study", "chart", "graph",
            "example", "imagine", "picture", "visualize", "show",
            "demonstrate", "illustrate", "diagram", "flowchart",
            "process", "step", "method", "technique", "approach",
            "result", "outcome", "finding", "conclusion",
            "compare", "contrast", "difference", "similar",
        ]

    keyword_set = set(k.lower() for k in keywords)
    suggestions = []

    # Use script segments if aligned, otherwise transcription segments
    source_segments = alignment.aligned_segments if alignment else transcription.segments

    for seg in source_segments:
        text = seg.text if hasattr(seg, 'text') else seg.text
        start = seg.start if hasattr(seg, 'start') else seg.start
        end = seg.end if hasattr(seg, 'end') else seg.end

        if start is None or end is None:
            continue

        duration = end - start
        if duration < min_segment_duration:
            continue

        # Check for keywords
        words = text.lower().split()
        found_keywords = [w for w in words if w.strip(".,!?") in keyword_set]

        if found_keywords:
            # Suggest B-roll in middle of segment
            broll_start = start + duration * 0.3
            broll_end = min(broll_start + min_segment_duration, end)

            # Generate search query from context
            query = generate_broll_query(text, found_keywords)

            suggestions.append(RoughCutSegment(
                index=len(suggestions),
                label=f"B-roll: {query}",
                start=broll_start,
                end=broll_end,
                duration=broll_end - broll_start,
                type="broll_suggestion",
                source="analysis",
                confidence=min(0.9, 0.5 + len(found_keywords) * 0.1),
                metadata={
                    "query": query,
                    "keywords": found_keywords,
                    "context": text[:200],
                },
            ))

    return suggestions


def generate_broll_query(text: str, keywords: list[str]) -> str:
    """Generate B-roll search query from text context and keywords."""
    # Simple heuristic: combine top keywords with context
    context_words = text.lower().split()
    # Find words near keywords
    query_parts = []

    for kw in keywords[:3]:  # Top 3 keywords
        query_parts.append(kw)

    # Add relevant context words (nouns, verbs)
    # Simple extraction: words that might be visual
    visual_hints = [
        "data", "chart", "graph", "screen", "code", "computer", "phone",
        "meeting", "office", "team", "work", "study", "research", "lab",
        "chart", "diagram", "flowchart", "process", "system", "network",
        "brain", "mind", "think", "idea", "concept", "model", "framework",
    ]

    for hint in visual_hints:
        if hint in text.lower():
            query_parts.append(hint)

    # Deduplicate and limit
    seen = set()
    unique_parts = []
    for p in query_parts:
        if p not in seen:
            seen.add(p)
            unique_parts.append(p)

    return " ".join(unique_parts[:5])  # Max 5 terms


def build_rough_cut(
    transcription: TranscriptionResult,
    script_path: Path | None = None,
    alignment: ScriptAlignmentResult | None = None,
    include_pauses: bool = True,
    include_fillers: bool = True,
    include_broll_suggestions: bool = True,
    min_pause: float = 0.5,
    min_segment_duration: float = 10.0,
) -> RoughCutResult:
    """
    Build complete rough cut from transcription and optional script.

    Args:
        transcription: Whisper transcription result
        script_path: Optional script file path
        alignment: Pre-computed alignment (or will be computed)
        include_pauses: Include natural pauses as segment boundaries
        include_fillers: Include filler word regions
        include_broll_suggestions: Include B-roll insertion points
        min_pause: Minimum pause duration to consider
        min_segment_duration: Minimum segment duration for B-roll suggestions

    Returns:
        RoughCutResult with complete timeline
    """
    logger.info("rough_cut_start", has_script=script_path is not None)

    # Align script if provided and not pre-aligned
    if script_path and alignment is None:
        script_text = script_path.read_text(encoding="utf-8")
        alignment = align_script_to_transcription(script_text, transcription)

    all_segments = []
    segment_index = 0

    # 1. Script-aligned segments (highest priority)
    if alignment:
        for seg in alignment.aligned_segments:
            if seg.matched and seg.start is not None and seg.end is not None:
                all_segments.append(RoughCutSegment(
                    index=segment_index,
                    label=seg.text[:100],
                    start=seg.start,
                    end=seg.end,
                    duration=seg.end - seg.start,
                    type="script",
                    source="script",
                    confidence=seg.confidence,
                    metadata={"script_index": seg.index},
                ))
                segment_index += 1

    # 2. Transcription segments (fallback or supplement)
    if not alignment or alignment.coverage < 0.8:
        for seg in transcription.segments:
            # Check if this segment overlaps with existing
            overlaps = any(
                not (seg.end <= s.start or seg.start >= s.end)
                for s in all_segments
            )
            if not overlaps:
                all_segments.append(RoughCutSegment(
                    index=segment_index,
                    label=seg.text[:100],
                    start=seg.start,
                    end=seg.end,
                    duration=seg.end - seg.start,
                    type="script",
                    source="transcription",
                    confidence=1.0 - seg.no_speech_prob,
                    metadata={"transcription_segment_id": seg.id},
                ))
                segment_index += 1

    # 3. Natural pauses (segment boundaries)
    if include_pauses:
        pauses = detect_natural_pauses(transcription, min_pause=min_pause)
        for pause in pauses:
            pause.index = segment_index
            segment_index += 1
        all_segments.extend(pauses)

    # 4. Filler regions (for removal suggestions)
    if include_fillers:
        fillers = detect_filler_regions(transcription)
        for filler in fillers:
            filler.index = segment_index
            segment_index += 1
        all_segments.extend(fillers)

    # 5. B-roll suggestions
    if include_broll_suggestions:
        broll_suggestions = suggest_broll_points(transcription, alignment, min_segment_duration=min_segment_duration)
        for broll in broll_suggestions:
            broll.index = segment_index
            segment_index += 1
        all_segments.extend(broll_suggestions)

    # Sort by start time
    all_segments.sort(key=lambda s: s.start if s.start is not None else float('inf'))

    # Re-index
    for i, seg in enumerate(all_segments):
        seg.index = i

    # Calculate script coverage
    script_coverage = alignment.coverage if alignment else 0.0

    result = RoughCutResult(
        segments=all_segments,
        total_duration=transcription.duration,
        script_coverage=script_coverage,
        transcription=transcription,
        alignment=alignment,
        metadata={
            "num_script_segments": len([s for s in all_segments if s.type == "script" and s.source == "script"]),
            "num_transcription_segments": len([s for s in all_segments if s.type == "script" and s.source == "transcription"]),
            "num_pauses": len([s for s in all_segments if s.type == "pause"]),
            "num_fillers": len([s for s in all_segments if s.type == "filler"]),
            "num_broll_suggestions": len([s for s in all_segments if s.type == "broll_suggestion"]),
        },
    )

    logger.info("rough_cut_complete",
        total_segments=len(all_segments),
        script_coverage=script_coverage,
        duration=transcription.duration,
    )

    return result


async def generate_rough_cut(
    audio_path: Path,
    script_path: Path | None = None,
    **kwargs,
) -> RoughCutResult:
    """
    High-level function: transcribe + align + rough cut in one call.

    Args:
        audio_path: Path to audio/video file
        script_path: Optional script file
        **kwargs: Passed to build_rough_cut

    Returns:
        RoughCutResult
    """
    from autoedit.ingestion.transcribe import transcribe_audio

    # Transcribe
    transcription = await transcribe_audio(audio_path)

    # Build rough cut
    return build_rough_cut(transcription, script_path, **kwargs)