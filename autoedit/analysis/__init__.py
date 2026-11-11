"""
AutoEdit Pro — Analysis Module
Transcription, scene detection, face tracking, beat detection.
"""

from .transcribe import transcribe_audio, TranscriptionResult, TranscriptionSegment
from .scene_detect import detect_scenes, SceneDetectionResult, Scene
from .face_track import track_faces, FaceTrackResult, FaceTrack
from .beat_detect import detect_beats, BeatDetectionResult, Beat
from .audio_quality import analyze_audio_quality, AudioQualityResult

__all__ = [
    "transcribe_audio",
    "TranscriptionResult",
    "TranscriptionSegment",
    "detect_scenes",
    "SceneDetectionResult",
    "Scene",
    "track_faces",
    "FaceTrackResult",
    "FaceTrack",
    "detect_beats",
    "BeatDetectionResult",
    "Beat",
    "analyze_audio_quality",
    "AudioQualityResult",
]