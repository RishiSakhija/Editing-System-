"""
AutoEdit Pro — Scene Detection
Detects scene boundaries using PySceneDetect (content-aware) or FFmpeg (fallback).
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from autoedit.utils.ffmpeg import get_ffmpeg, MediaInfo
from autoedit.utils.logging import get_logger

logger = get_logger(__name__)

# Try to import PySceneDetect
try:
    from scenedetect import SceneManager, open_video
    from scenedetect.detectors import ContentDetector, AdaptiveDetector, ThresholdDetector
    SCENEDETECT_AVAILABLE = True
except ImportError:
    SCENEDETECT_AVAILABLE = False
    logger.warning("PySceneDetect not available, using FFmpeg fallback")


@dataclass
class Scene:
    """Single scene with start/end times."""
    index: int
    start_time: float
    end_time: float
    duration: float
    start_frame: int
    end_frame: int
    detection_method: str
    confidence: float = 1.0


@dataclass
class SceneDetectionResult:
    """Result of scene detection."""
    scenes: list[Scene]
    total_duration: float
    method: str
    processing_time_ms: float
    video_info: MediaInfo | None = None


async def detect_scenes_scenedetect(
    video_path: Path,
    threshold: float = 27.0,
    min_scene_len: float = 0.5,
    detector_type: str = "content",
) -> SceneDetectionResult:
    """
    Detect scenes using PySceneDetect.

    Args:
        video_path: Path to video file
        threshold: Detection threshold (lower = more sensitive)
        min_scene_len: Minimum scene length in seconds
        detector_type: "content", "adaptive", or "threshold"

    Returns:
        SceneDetectionResult
    """
    if not SCENEDETECT_AVAILABLE:
        raise RuntimeError("PySceneDetect not installed")

    import time
    start = time.perf_counter()

    # Open video
    video = open_video(str(video_path))

    # Create scene manager
    scene_manager = SceneManager()

    # Select detector
    if detector_type == "adaptive":
        detector = AdaptiveDetector(adaptive_threshold=threshold)
    elif detector_type == "threshold":
        detector = ThresholdDetector(threshold=threshold)
    else:  # content
        detector = ContentDetector(threshold=threshold, min_scene_len=int(min_scene_len * video.frame_rate))

    scene_manager.add_detector(detector)

    # Detect scenes
    scene_manager.detect_scenes(video, show_progress=False)

    # Get scene list
    scene_list = scene_manager.get_scene_list()

    scenes = []
    for i, (start, end) in enumerate(scene_list):
        start_sec = start.get_seconds()
        end_sec = end.get_seconds()
        scenes.append(Scene(
            index=i,
            start_time=start_sec,
            end_time=end_sec,
            duration=end_sec - start_sec,
            start_frame=start.get_frames(),
            end_frame=end.get_frames(),
            detection_method=f"scenedetect_{detector_type}",
            confidence=0.9,
        ))

    processing_time = (time.perf_counter() - start) * 1000

    # Get video info
    ffmpeg = get_ffmpeg()
    video_info = await ffmpeg.probe(video_path)

    return SceneDetectionResult(
        scenes=scenes,
        total_duration=video_info.duration,
        method=f"scenedetect_{detector_type}",
        processing_time_ms=processing_time,
        video_info=video_info,
    )


async def detect_scenes_ffmpeg(
    video_path: Path,
    threshold: float = 0.3,
    min_scene_len: float = 0.5,
) -> SceneDetectionResult:
    """
    Detect scenes using FFmpeg's scene detection filter (fallback).

    Args:
        video_path: Path to video file
        threshold: Scene change threshold (0-1)
        min_scene_len: Minimum scene length in seconds

    Returns:
        SceneDetectionResult
    """
    import time
    start = time.perf_counter()

    ffmpeg = get_ffmpeg()

    # Use FFmpeg's select filter with scene detection
    # This outputs timestamps of scene changes
    args = [
        "-i", str(video_path),
        "-filter:v", f"select='gt(scene,{threshold})',showinfo",
        "-f", "null",
        "-",
    ]

    result = await ffmpeg.run(args)

    # Parse showinfo output for scene change timestamps
    scenes = []
    current_start = 0.0
    scene_index = 0

    for line in result.stderr.split('\n'):
        if 'pts_time:' in line:
            # Extract timestamp
            try:
                pts_idx = line.index('pts_time:')
                pts_str = line[pts_idx + 9:].split()[0]
                timestamp = float(pts_str)

                if timestamp - current_start >= min_scene_len:
                    scenes.append(Scene(
                        index=scene_index,
                        start_time=current_start,
                        end_time=timestamp,
                        duration=timestamp - current_start,
                        start_frame=int(current_start * 30),  # Approximate
                        end_frame=int(timestamp * 30),
                        detection_method="ffmpeg_scene",
                        confidence=0.7,
                    ))
                    scene_index += 1
                    current_start = timestamp
            except (ValueError, IndexError):
                continue

    # Add final scene
    video_info = await ffmpeg.probe(video_path)
    if current_start < video_info.duration:
        scenes.append(Scene(
            index=scene_index,
            start_time=current_start,
            end_time=video_info.duration,
            duration=video_info.duration - current_start,
            start_frame=int(current_start * 30),
            end_frame=int(video_info.duration * 30),
            detection_method="ffmpeg_scene",
            confidence=0.7,
        ))

    processing_time = (time.perf_counter() - start) * 1000

    return SceneDetectionResult(
        scenes=scenes,
        total_duration=video_info.duration,
        method="ffmpeg_scene",
        processing_time_ms=processing_time,
        video_info=video_info,
    )


async def detect_scenes(
    video_path: Path,
    threshold: float = 27.0,
    min_scene_len: float = 0.5,
    method: str = "auto",
    **kwargs,
) -> SceneDetectionResult:
    """
    Detect scene boundaries in video.

    Args:
        video_path: Path to video file
        threshold: Detection threshold
        min_scene_len: Minimum scene length in seconds
        method: "scenedetect", "ffmpeg", or "auto"
        **kwargs: Additional arguments

    Returns:
        SceneDetectionResult
    """
    if method == "scenedetect" or (method == "auto" and SCENEDETECT_AVAILABLE):
        try:
            return await detect_scenes_scenedetect(
                video_path, threshold, min_scene_len, **kwargs
            )
        except Exception as e:
            logger.warning("scenedetect_failed_falling_back", error=str(e))
            if method == "scenedetect":
                raise

    # Fallback to FFmpeg
    return await detect_scenes_ffmpeg(video_path, threshold, min_scene_len)


def merge_short_scenes(
    scenes: list[Scene],
    min_duration: float = 1.0,
) -> list[Scene]:
    """Merge scenes shorter than min_duration with adjacent scenes."""
    if not scenes:
        return []

    merged = []
    current = scenes[0]

    for scene in scenes[1:]:
        if current.duration < min_duration:
            # Merge with next scene
            current = Scene(
                index=current.index,
                start_time=current.start_time,
                end_time=scene.end_time,
                duration=scene.end_time - current.start_time,
                start_frame=current.start_frame,
                end_frame=scene.end_frame,
                detection_method=current.detection_method,
                confidence=min(current.confidence, scene.confidence),
            )
        else:
            merged.append(current)
            current = scene

    merged.append(current)

    # Re-index
    for i, scene in enumerate(merged):
        scene.index = i

    return merged


def filter_scenes_by_duration(
    scenes: list[Scene],
    min_duration: float = 0.5,
    max_duration: float | None = None,
) -> list[Scene]:
    """Filter scenes by duration bounds."""
    filtered = []
    for scene in scenes:
        if scene.duration >= min_duration:
            if max_duration is None or scene.duration <= max_duration:
                filtered.append(scene)
    # Re-index
    for i, scene in enumerate(filtered):
        scene.index = i
    return filtered


def get_scene_at_time(scenes: list[Scene], time: float) -> Scene | None:
    """Find scene containing the given timestamp."""
    for scene in scenes:
        if scene.start_time <= time < scene.end_time:
            return scene
    # Check last scene boundary
    if scenes and time == scenes[-1].end_time:
        return scenes[-1]
    return None