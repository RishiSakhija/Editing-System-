"""
AutoEdit Pro — Ingestion Module
Handles input processing: sync, transcription, script alignment, rough cut generation.
"""

from .rough_cut import (
    RoughCutResult,
    RoughCutSegment,
    build_rough_cut,
    detect_filler_regions,
    detect_natural_pauses,
    generate_rough_cut,
    suggest_broll_points,
)
from .script_align import (
    ScriptAlignmentResult,
    ScriptSegment,
    align_script_file,
    align_script_to_transcription,
    generate_timeline_markers,
)
from .sync import (
    SyncResult,
    extract_audio,
    get_media_info,
    sync_audio_video,
    sync_multiple_videos,
)
from .transcribe import (
    TranscriptionResult,
    TranscriptionSegment,
    TranscriptionWord,
    WhisperCppWrapper,
    transcribe_audio,
    transcribe_file,
)

__all__ = [
    # Sync
    "extract_audio",
    "get_media_info",
    "sync_audio_video",
    "sync_multiple_videos",
    "SyncResult",
    # Transcribe
    "transcribe_audio",
    "transcribe_file",
    "TranscriptionResult",
    "TranscriptionSegment",
    "TranscriptionWord",
    "WhisperCppWrapper",
    # Script Align
    "align_script_to_transcription",
    "align_script_file",
    "generate_timeline_markers",
    "ScriptAlignmentResult",
    "ScriptSegment",
    # Rough Cut
    "build_rough_cut",
    "generate_rough_cut",
    "RoughCutResult",
    "RoughCutSegment",
    "detect_natural_pauses",
    "detect_filler_regions",
    "suggest_broll_points",
]