"""
AutoEdit Pro — Ingestion Module
Handles input processing: sync, script alignment, rough cut generation.
"""

from .sync import sync_audio_video, SyncResult
from .script_align import align_script_to_audio, ScriptAlignmentResult
from .rough_cut import generate_rough_cut, RoughCutResult

__all__ = [
    "sync_audio_video",
    "SyncResult",
    "align_script_to_audio",
    "ScriptAlignmentResult",
    "generate_rough_cut",
    "RoughCutResult",
]