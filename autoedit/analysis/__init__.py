"""
AutoEdit Pro — Analysis Module
Scene detection, face tracking, beat detection, audio quality analysis.
"""

from .scene_detect import (
    detect_scenes,
    detect_scenes_scenedetect,
    detect_scenes_ffmpeg,
    merge_short_scenes,
    filter_scenes_by_duration,
    get_scene_at_time,
    SceneDetectionResult,
    Scene,
)
from .face_track import (
    track_faces,
    get_primary_face,
    get_face_crop_region,
    FaceTrackResult,
    FaceTrack,
    FaceBox,
    YOLOFaceDetector,
    ByteTrackTracker,
)
from .beat_detect import (
    detect_beats,
    detect_beats_librosa,
    detect_beats_madmom,
    detect_beats_from_video,
    sync_cuts_to_beats,
    get_downbeat_times,
    get_beat_grid,
    BeatDetectionResult,
    Beat,
)
from .audio_quality import (
    analyze_audio_quality,
    normalize_audio_loudness,
    reduce_noise,
    enhance_speech,
    AudioQualityResult,
    estimate_snr,
    compute_speech_clarity,
    detect_audio_issues,
)

__all__ = [
    # Scene Detection
    "detect_scenes",
    "detect_scenes_scenedetect",
    "detect_scenes_ffmpeg",
    "merge_short_scenes",
    "filter_scenes_by_duration",
    "get_scene_at_time",
    "SceneDetectionResult",
    "Scene",
    # Face Tracking
    "track_faces",
    "get_primary_face",
    "get_face_crop_region",
    "FaceTrackResult",
    "FaceTrack",
    "FaceBox",
    "YOLOFaceDetector",
    "ByteTrackTracker",
    # Beat Detection
    "detect_beats",
    "detect_beats_librosa",
    "detect_beats_madmom",
    "detect_beats_from_video",
    "sync_cuts_to_beats",
    "get_downbeat_times",
    "get_beat_grid",
    "BeatDetectionResult",
    "Beat",
    # Audio Quality
    "analyze_audio_quality",
    "normalize_audio_loudness",
    "reduce_noise",
    "enhance_speech",
    "AudioQualityResult",
    "estimate_snr",
    "compute_speech_clarity",
    "detect_audio_issues",
]