"""
AutoEdit Pro — Beat Detection
Detects beats in audio using librosa (primary) or madmom (optional).
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import librosa
import numpy as np
from autoedit.config import get_settings
from autoedit.utils.ffmpeg import get_ffmpeg
from autoedit.utils.logging import get_logger

logger = get_logger(__name__)

# Try to import madmom (optional)
try:
    import madmom
    MADMOM_AVAILABLE = True
except ImportError:
    MADMOM_AVAILABLE = False
    logger.warning("madmom not available, using librosa for beat detection")


@dataclass
class Beat:
    """Single beat with timing and strength."""
    time: float
    strength: float = 1.0
    is_downbeat: bool = False


@dataclass
class BeatDetectionResult:
    """Result of beat detection."""
    beats: list[Beat]
    tempo: float  # BPM
    method: str  # "librosa" or "madmom"
    confidence: float
    processing_time_ms: float


async def detect_beats_librosa(
    audio_path: Path,
    sr: int = 22050,
    hop_length: int = 512,
    tightness: float = 100,
) -> BeatDetectionResult:
    """
    Detect beats using librosa (onset detection + dynamic programming).

    Args:
        audio_path: Path to audio file
        sr: Sample rate for analysis
        hop_length: Hop length for STFT
        tightness: Tightness of beat distribution

    Returns:
        BeatDetectionResult
    """
    import time
    start = time.perf_counter()

    # Load audio
    y, sr = librosa.load(audio_path, sr=sr, mono=True)

    # Detect onsets
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)

    # Estimate tempo
    tempo, beats = librosa.beat.beat_track(
        onset_envelope=onset_env,
        sr=sr,
        hop_length=hop_length,
        tightness=tightness,
        units="time",
    )

    # Get beat strengths from onset envelope
    beat_frames = librosa.time_to_frames(beats, sr=sr, hop_length=hop_length)
    strengths = onset_env[beat_frames] if len(beat_frames) > 0 else []

    # Normalize strengths
    if len(strengths) > 0:
        strengths = strengths / np.max(strengths)
    else:
        strengths = np.ones(len(beats))

    # Detect downbeats (every 4th beat for 4/4 time)
    downbeats = np.zeros(len(beats), dtype=bool)
    if len(beats) >= 4:
        downbeats[::4] = True

    beat_list = [
        Beat(time=float(t), strength=float(s), is_downbeat=bool(d))
        for t, s, d in zip(beats, strengths, downbeats)
    ]

    processing_time = (time.perf_counter() - start) * 1000

    return BeatDetectionResult(
        beats=beat_list,
        tempo=float(tempo),
        method="librosa",
        confidence=0.8,
        processing_time_ms=processing_time,
    )


async def detect_beats_madmom(
    audio_path: Path,
    fps: int = 100,
) -> BeatDetectionResult:
    """
    Detect beats using madmom (DBN beat tracker).

    Args:
        audio_path: Path to audio file
        fps: Frames per second for analysis

    Returns:
        BeatDetectionResult
    """
    if not MADMOM_AVAILABLE:
        raise RuntimeError("madmom not available")

    import time
    start = time.perf_counter()

    # Use madmom's RNN beat processor + DBN tracker
    proc = madmom.features.beats.RNNBeatProcessor()(str(audio_path))
    tracker = madmom.features.beats.DBNBeatTrackingProcessor(fps=fps)
    beats = tracker(proc)

    # Get tempo from beat intervals
    if len(beats) > 1:
        intervals = np.diff(beats)
        tempo = 60.0 / np.median(intervals)
    else:
        tempo = 120.0

    # Madmom doesn't directly give strengths, estimate from intervals
    strengths = np.ones(len(beats))
    if len(beats) > 1:
        # More regular intervals = higher confidence
        cv = np.std(intervals) / np.mean(intervals)
        strengths = 1.0 - np.clip(cv, 0, 1)

    # Downbeats every 4th beat
    downbeats = np.zeros(len(beats), dtype=bool)
    if len(beats) >= 4:
        downbeats[::4] = True

    beat_list = [
        Beat(time=float(t), strength=float(s), is_downbeat=bool(d))
        for t, s, d in zip(beats, strengths, downbeats)
    ]

    processing_time = (time.perf_counter() - start) * 1000

    return BeatDetectionResult(
        beats=beat_list,
        tempo=float(tempo),
        method="madmom",
        confidence=0.9,
        processing_time_ms=processing_time,
    )


async def detect_beats(
    audio_path: Path,
    method: str = "auto",
    **kwargs,
) -> BeatDetectionResult:
    """
    Detect beats in audio file.

    Args:
        audio_path: Path to audio file
        method: "librosa", "madmom", or "auto" (try madmom first)
        **kwargs: Additional arguments passed to detector

    Returns:
        BeatDetectionResult
    """
    if method == "madmom" or (method == "auto" and MADMOM_AVAILABLE):
        try:
            return await detect_beats_madmom(audio_path, **kwargs)
        except Exception as e:
            logger.warning("madmom_failed_falling_back_to_librosa", error=str(e))
            if method == "madmom":
                raise

    # Fallback to librosa
    return await detect_beats_librosa(audio_path, **kwargs)


async def detect_beats_from_video(
    video_path: Path,
    method: str = "auto",
    **kwargs,
) -> BeatDetectionResult:
    """Extract audio from video and detect beats."""
    ffmpeg = get_ffmpeg()

    # Extract audio to temp file
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        audio_path = Path(tmp.name)

    try:
        await ffmpeg.extract_audio(
            input_path=video_path,
            output_path=audio_path,
            codec="pcm_s16le",
            sample_rate=22050,
            channels=1,
        )

        return await detect_beats(audio_path, method=method, **kwargs)
    finally:
        audio_path.unlink(missing_ok=True)


def sync_cuts_to_beats(
    cut_times: list[float],
    beats: list[Beat],
    tolerance: float = 0.15,
) -> list[float]:
    """
    Adjust cut times to align with nearest beats.

    Args:
        cut_times: Original cut times (seconds)
        beats: Detected beats
        tolerance: Maximum adjustment (seconds)

    Returns:
        Adjusted cut times
    """
    if not beats:
        return cut_times

    beat_times = np.array([b.time for b in beats])
    adjusted = []

    for cut in cut_times:
        # Find nearest beat
        idx = np.argmin(np.abs(beat_times - cut))
        nearest_beat = beat_times[idx]
        diff = nearest_beat - cut

        if abs(diff) <= tolerance:
            adjusted.append(float(nearest_beat))
        else:
            adjusted.append(cut)

    return adjusted


def get_downbeat_times(beats: list[Beat]) -> list[float]:
    """Get times of downbeats (strong beats)."""
    return [b.time for b in beats if b.is_downbeat]


def get_beat_grid(beats: list[Beat], bars: int = 4) -> list[list[float]]:
    """
    Organize beats into bars (groups of 4 for 4/4 time).

    Returns:
        List of bars, each containing 4 beat times
    """
    downbeats = get_downbeat_times(beats)
    if not downbeats:
        return []

    grid = []
    for i, db in enumerate(downbeats):
        bar = [db]
        # Add next 3 beats
        db_idx = next((j for j, b in enumerate(beats) if b.time == db), -1)
        for k in range(1, 4):
            if db_idx + k < len(beats):
                bar.append(beats[db_idx + k].time)
        if len(bar) == 4:
            grid.append(bar)

    return grid