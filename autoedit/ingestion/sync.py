"""
AutoEdit Pro — Audio/Video Synchronization
Extracts audio, syncs separate audio/video tracks using cross-correlation.
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from autoedit.utils.ffmpeg import MediaInfo, get_ffmpeg
from autoedit.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class SyncResult:
    """Result of audio/video sync operation."""
    success: bool
    synced_video: Path | None = None
    offset_seconds: float = 0.0
    confidence: float = 0.0
    method: str = "cross_correlation"
    error: str | None = None
    video_info: MediaInfo | None = None
    audio_info: MediaInfo | None = None


async def extract_audio(
    video_path: Path,
    output_path: Path | None = None,
    sample_rate: int = 16000,
    channels: int = 1,
) -> Path:
    """
    Extract audio from video file for transcription.

    Args:
        video_path: Path to video file
        output_path: Output WAV path (auto-generated if None)
        sample_rate: Target sample rate (16kHz for Whisper)
        channels: Target channels (1 for mono)

    Returns:
        Path to extracted audio file
    """
    ffmpeg = get_ffmpeg()

    if output_path is None:
        output_path = video_path.with_suffix(".wav")

    logger.info("extracting_audio", video=str(video_path), output=str(output_path))

    result = await ffmpeg.extract_audio(
        input_path=video_path,
        output_path=output_path,
        codec="pcm_s16le",
        sample_rate=sample_rate,
        channels=channels,
    )

    if not result.success:
        raise RuntimeError(f"Audio extraction failed: {result.stderr}")

    logger.info("audio_extracted", output=str(output_path), duration=result.duration_ms)
    return output_path


async def get_media_info(path: Path) -> MediaInfo:
    """Get media info using ffprobe."""
    ffmpeg = get_ffmpeg()
    return await ffmpeg.probe(path)


def cross_correlate(audio1: np.ndarray, audio2: np.ndarray, max_lag_seconds: float = 10.0, sample_rate: int = 16000) -> tuple[float, float]:
    """
    Compute cross-correlation to find time offset between two audio signals.

    Args:
        audio1: Reference audio signal
        audio2: Signal to align
        max_lag_seconds: Maximum lag to search (seconds)
        sample_rate: Sample rate of signals

    Returns:
        (offset_seconds, confidence)
    """
    max_lag = int(max_lag_seconds * sample_rate)

    # Normalize signals
    audio1 = audio1.astype(np.float32)
    audio2 = audio2.astype(np.float32)

    audio1 = (audio1 - np.mean(audio1)) / (np.std(audio1) + 1e-8)
    audio2 = (audio2 - np.mean(audio2)) / (np.std(audio2) + 1e-8)

    # Limit to max_lag for efficiency
    if len(audio2) > len(audio1) + max_lag:
        audio2 = audio2[:len(audio1) + max_lag]

    # Compute cross-correlation using FFT (fast)
    n = len(audio1) + len(audio2) - 1
    n_fft = 1 << (n - 1).bit_length()  # Next power of 2

    fft1 = np.fft.rfft(audio1, n_fft)
    fft2 = np.fft.rfft(audio2, n_fft)
    corr = np.fft.irfft(fft1 * np.conj(fft2), n_fft)

    # Find peak within max_lag
    center = len(audio2) - 1
    start = max(0, center - max_lag)
    end = min(len(corr), center + max_lag + 1)

    if start >= end:
        return 0.0, 0.0

    peak_idx = np.argmax(corr[start:end]) + start
    offset_samples = peak_idx - center
    offset_seconds = offset_samples / sample_rate

    # Confidence: peak height relative to surrounding
    peak_value = corr[peak_idx]
    surrounding = np.concatenate([corr[start:peak_idx], corr[peak_idx+1:end]])
    noise_floor = np.mean(np.abs(surrounding)) + 1e-8
    confidence = min(1.0, peak_value / noise_floor / 10.0)  # Normalize

    return offset_seconds, confidence


async def sync_audio_video(
    video_path: Path,
    audio_path: Path | None = None,
    output_path: Path | None = None,
    max_lag_seconds: float = 10.0,
) -> SyncResult:
    """
    Sync video with external audio track using cross-correlation.

    Args:
        video_path: Path to video file (with reference audio)
        audio_path: Path to external audio file (higher quality)
        output_path: Output synced video path
        max_lag_seconds: Maximum sync offset to search

    Returns:
        SyncResult with offset and synced video path
    """
    ffmpeg = get_ffmpeg()

    logger.info("sync_start", video=str(video_path), audio=str(audio_path))

    # If no external audio, just return original
    if audio_path is None or not audio_path.exists():
        logger.info("no_external_audio", video=str(video_path))
        video_info = await get_media_info(video_path)
        return SyncResult(
            success=True,
            synced_video=video_path,
            offset_seconds=0.0,
            confidence=1.0,
            video_info=video_info,
        )

    # Extract reference audio from video
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        ref_audio_path = Path(tmp.name)

    try:
        await extract_audio(video_path, ref_audio_path)

        # Load both audio files
        import soundfile as sf
        ref_audio, ref_sr = sf.read(ref_audio_path)
        ext_audio, ext_sr = sf.read(audio_path)

        # Resample if needed
        if ref_sr != ext_sr:
            import librosa
            if ref_sr != 16000:
                ref_audio = librosa.resample(ref_audio, orig_sr=ref_sr, target_sr=16000)
            if ext_sr != 16000:
                ext_audio = librosa.resample(ext_audio, orig_sr=ext_sr, target_sr=16000)
            sample_rate = 16000
        else:
            sample_rate = ref_sr

        # Convert to mono if stereo
        if ref_audio.ndim > 1:
            ref_audio = np.mean(ref_audio, axis=1)
        if ext_audio.ndim > 1:
            ext_audio = np.mean(ext_audio, axis=1)

        # Cross-correlate
        offset_seconds, confidence = cross_correlate(
            ref_audio, ext_audio, max_lag_seconds, sample_rate
        )

        logger.info("sync_calculated", offset=offset_seconds, confidence=confidence)

        # Apply sync if offset is significant
        if abs(offset_seconds) > 0.05 and confidence > 0.3:
            if output_path is None:
                output_path = video_path.parent / f"{video_path.stem}_synced{video_path.suffix}"

            # Use ffmpeg to apply offset
            if offset_seconds > 0:
                # External audio starts later - delay it
                args = [
                    "-i", str(video_path),
                    "-i", str(audio_path),
                    "-filter_complex", f"[1:a]adelay={int(offset_seconds*1000)}|{int(offset_seconds*1000)}[aud]",
                    "-map", "0:v", "-map", "[aud]",
                    "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
                    str(output_path),
                ]
            else:
                # External audio starts earlier - delay video
                delay_ms = int(abs(offset_seconds) * 1000)
                args = [
                    "-i", str(video_path),
                    "-i", str(audio_path),
                    "-filter_complex", f"[0:v]setpts=PTS+{abs(offset_seconds)}/TB[v]",
                    "-map", "[v]", "-map", "1:a",
                    "-c:v", "libx264", "-preset", "fast", "-crf", "18",
                    "-c:a", "aac", "-b:a", "192k",
                    str(output_path),
                ]

            result = await ffmpeg.run(args)
            if not result.success:
                raise RuntimeError(f"Sync failed: {result.stderr}")

            synced_path = output_path
        else:
            # No significant offset, just replace audio
            if output_path is None:
                output_path = video_path.parent / f"{video_path.stem}_synced{video_path.suffix}"

            args = [
                "-i", str(video_path),
                "-i", str(audio_path),
                "-map", "0:v", "-map", "1:a",
                "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
                str(output_path),
            ]
            result = await ffmpeg.run(args)
            if not result.success:
                raise RuntimeError(f"Audio replace failed: {result.stderr}")

            synced_path = output_path

        video_info = await get_media_info(synced_path)
        audio_info = await get_media_info(audio_path)

        return SyncResult(
            success=True,
            synced_video=synced_path,
            offset_seconds=offset_seconds,
            confidence=confidence,
            video_info=video_info,
            audio_info=audio_info,
        )

    finally:
        ref_audio_path.unlink(missing_ok=True)


async def sync_multiple_videos(
    video_paths: list[Path],
    reference_audio: Path | None = None,
    output_dir: Path | None = None,
) -> list[SyncResult]:
    """
    Sync multiple video clips to a common reference (first video or external audio).

    Args:
        video_paths: List of video files to sync
        reference_audio: Optional external reference audio
        output_dir: Output directory for synced videos

    Returns:
        List of SyncResults
    """
    if not video_paths:
        return []

    output_dir = output_dir or video_paths[0].parent / "synced"
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []

    # Use first video as reference if no external audio
    if reference_audio is None:
        reference_audio = video_paths[0]
        logger.info("using_first_video_as_reference", reference=str(reference_audio))

    # Sync each video
    for i, video_path in enumerate(video_paths):
        if video_path == reference_audio:
            # Reference video - just copy
            synced_path = output_dir / f"{video_path.stem}_synced{video_path.suffix}"
            import shutil
            shutil.copy2(video_path, synced_path)
            video_info = await get_media_info(synced_path)
            results.append(SyncResult(
                success=True,
                synced_video=synced_path,
                offset_seconds=0.0,
                confidence=1.0,
                video_info=video_info,
            ))
        else:
            synced_path = output_dir / f"{video_path.stem}_synced{video_path.suffix}"
            result = await sync_audio_video(video_path, reference_audio, synced_path)
            results.append(result)

    return results