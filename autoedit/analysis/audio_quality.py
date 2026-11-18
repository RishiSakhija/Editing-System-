"""
AutoEdit Pro — Audio Quality Analysis
Analyzes SNR, loudness (LUFS), noise profile, and speech quality.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import librosa
import numpy as np
from autoedit.utils.ffmpeg import get_ffmpeg, MediaInfo
from autoedit.utils.logging import get_logger

logger = get_logger(__name__)

# Try to import pyloudnorm for LUFS measurement
try:
    import pyloudnorm as pyln
    PYLOUDNORM_AVAILABLE = True
except ImportError:
    PYLOUDNORM_AVAILABLE = False
    logger.warning("pyloudnorm not available, LUFS measurement limited")


@dataclass
class AudioQualityResult:
    """Complete audio quality analysis."""
    # Loudness
    integrated_lufs: float | None = None
    true_peak_db: float | None = None
    lra: float | None = None  # Loudness Range

    # Signal-to-Noise
    snr_db: float | None = None
    noise_floor_db: float | None = None

    # Speech Quality
    speech_clarity: float | None = None  # 0-1
    spectral_centroid: float | None = None
    spectral_rolloff: float | None = None
    zero_crossing_rate: float | None = None

    # Dynamics
    rms_db: float | None = None
    dynamic_range_db: float | None = None

    # Issues detected
    issues: list[str] = None

    # Processing info
    processing_time_ms: float = 0.0
    sample_rate: int = 0
    duration: float = 0.0

    def __post_init__(self):
        if self.issues is None:
            self.issues = []

    def get_quality_score(self) -> float:
        """Compute overall quality score (0-1)."""
        scores = []

        # LUFS compliance (target -16 LUFS for speech)
        if self.integrated_lufs is not None:
            lufs_diff = abs(self.integrated_lufs - (-16))
            lufs_score = max(0, 1 - lufs_diff / 20)
            scores.append(lufs_score)

        # SNR (target > 20dB)
        if self.snr_db is not None:
            snr_score = min(1, self.snr_db / 30)
            scores.append(snr_score)

        # Speech clarity
        if self.speech_clarity is not None:
            scores.append(self.speech_clarity)

        # True peak (should be < -1dBTP)
        if self.true_peak_db is not None:
            peak_score = 1 if self.true_peak_db < -1 else max(0, 1 - (self.true_peak_db + 1) / 10)
            scores.append(peak_score)

        return np.mean(scores) if scores else 0.5

    def get_recommendations(self) -> list[str]:
        """Get actionable recommendations."""
        recs = []

        if self.integrated_lufs is not None:
            if self.integrated_lufs < -20:
                recs.append("Audio too quiet: normalize to -16 LUFS")
            elif self.integrated_lufs > -12:
                recs.append("Audio too loud: reduce gain to -16 LUFS")

        if self.true_peak_db is not None and self.true_peak_db > -1:
            recs.append(f"True peak {self.true_peak_db:.1f}dBTP exceeds -1dBTP: apply limiter")

        if self.snr_db is not None and self.snr_db < 15:
            recs.append(f"Low SNR ({self.snr_db:.1f}dB): consider noise reduction")

        if self.speech_clarity is not None and self.speech_clarity < 0.6:
            recs.append("Low speech clarity: check microphone placement/quality")

        if self.dynamic_range_db is not None and self.dynamic_range_db > 30:
            recs.append("High dynamic range: consider compression for consistency")

        return recs


async def analyze_audio_quality(
    audio_path: Path,
    target_lufs: float = -16.0,
    analyze_speech: bool = True,
) -> AudioQualityResult:
    """
    Analyze audio quality metrics.

    Args:
        audio_path: Path to audio file
        target_lufs: Target integrated loudness
        analyze_speech: Whether to compute speech-specific metrics

    Returns:
        AudioQualityResult
    """
    import time
    start = time.perf_counter()

    # Load audio with librosa
    y, sr = librosa.load(audio_path, sr=None, mono=True)

    result = AudioQualityResult(
        sample_rate=sr,
        duration=len(y) / sr,
    )

    # 1. Loudness Analysis (LUFS)
    if PYLOUDNORM_AVAILABLE:
        try:
            meter = pyln.Meter(sr)
            result.integrated_lufs = meter.integrated_loudness(y)

            # True peak
            result.true_peak_db = 20 * np.log10(np.max(np.abs(y)))

            # Loudness Range (LRA)
            # Simplified: measure short-term loudness variation
            window = int(3 * sr)  # 3 second windows
            hop = int(1 * sr)
            short_term = []
            for i in range(0, len(y) - window, hop):
                chunk = y[i:i+window]
                if len(chunk) == window:
                    short_term.append(meter.integrated_loudness(chunk))
            if short_term:
                result.lra = np.percentile(short_term, 95) - np.percentile(short_term, 10)
        except Exception as e:
            logger.warning("loudness_analysis_failed", error=str(e))

    # 2. RMS and Dynamic Range
    rms = np.sqrt(np.mean(y**2))
    result.rms_db = 20 * np.log10(rms + 1e-10)

    # Dynamic range (peak to RMS)
    peak = np.max(np.abs(y))
    result.dynamic_range_db = 20 * np.log10(peak / (rms + 1e-10))

    # 3. SNR Estimation
    # Use spectral gating to estimate noise floor
    result.snr_db, result.noise_floor_db = estimate_snr(y, sr)

    # 4. Spectral Features
    result.spectral_centroid = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
    result.spectral_rolloff = float(np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)))
    result.zero_crossing_rate = float(np.mean(librosa.feature.zero_crossing_rate(y)))

    # 5. Speech Quality Metrics
    if analyze_speech:
        result.speech_clarity = compute_speech_clarity(y, sr)

    # 6. Detect Issues
    result.issues = detect_audio_issues(result)

    result.processing_time_ms = (time.perf_counter() - start) * 1000

    logger.info("audio_quality_complete",
        lufs=result.integrated_lufs,
        snr=result.snr_db,
        clarity=result.speech_clarity,
        issues=len(result.issues),
    )

    return result


def estimate_snr(audio: np.ndarray, sr: int) -> tuple[float, float]:
    """
    Estimate SNR using spectral gating.
    Assumes quietest segments are noise.
    """
    # Compute spectrogram
    n_fft = 2048
    hop = 512
    stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop)
    mag = np.abs(stft)

    # Estimate noise floor from quietest 10% of frames
    frame_energy = np.sum(mag**2, axis=0)
    noise_frames = int(len(frame_energy) * 0.1)
    noise_indices = np.argsort(frame_energy)[:noise_frames]

    if len(noise_indices) == 0:
        return 20.0, -60.0  # Default

    noise_profile = np.mean(mag[:, noise_indices], axis=1)
    signal_power = np.mean(mag**2)
    noise_power = np.mean(noise_profile**2)

    if noise_power > 0:
        snr = 10 * np.log10(signal_power / noise_power)
        noise_db = 10 * np.log10(noise_power + 1e-10)
    else:
        snr = 40.0
        noise_db = -80.0

    return float(np.clip(snr, 0, 60)), float(noise_db)


def compute_speech_clarity(audio: np.ndarray, sr: int) -> float:
    """
    Compute speech clarity metric (0-1).
    Based on spectral features correlated with intelligibility.
    """
    # Spectral flux (rate of spectral change)
    stft = librosa.stft(audio)
    mag = np.abs(stft)
    flux = np.mean(np.diff(mag, axis=1)**2)

    # Spectral flatness (tonality vs noisiness)
    flatness = np.mean(librosa.feature.spectral_flatness(y=audio))

    # MFCC-based clarity (first few coefficients capture vocal tract)
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    mfcc_var = np.mean(np.var(mfccs, axis=1))

    # Harmonic-to-noise ratio approximation
    harmonic = librosa.effects.harmonic(audio)
    hnr = 10 * np.log10(np.sum(harmonic**2) / (np.sum((audio - harmonic)**2) + 1e-10))

    # Combine metrics (normalized)
    clarity = (
        0.3 * min(1, flux / 1e6) +
        0.2 * (1 - flatness) +
        0.2 * min(1, mfcc_var / 100) +
        0.3 * min(1, max(0, hnr) / 30)
    )

    return float(np.clip(clarity, 0, 1))


def detect_audio_issues(result: AudioQualityResult) -> list[str]:
    """Detect common audio issues."""
    issues = []

    if result.integrated_lufs is not None:
        if result.integrated_lufs < -23:
            issues.append("very_quiet")
        elif result.integrated_lufs < -18:
            issues.append("quiet")
        elif result.integrated_lufs > -10:
            issues.append("loud")

    if result.true_peak_db is not None and result.true_peak_db > -1:
        issues.append("peak_clipping")

    if result.snr_db is not None and result.snr_db < 15:
        issues.append("low_snr")

    if result.speech_clarity is not None and result.speech_clarity < 0.5:
        issues.append("poor_clarity")

    if result.dynamic_range_db is not None and result.dynamic_range_db > 35:
        issues.append("high_dynamic_range")

    if result.zero_crossing_rate is not None and result.zero_crossing_rate > 0.15:
        issues.append("high_noise")

    return issues


async def normalize_audio_loudness(
    input_path: Path,
    output_path: Path,
    target_lufs: float = -16.0,
    true_peak: float = -1.5,
) -> AudioQualityResult:
    """
    Normalize audio to target loudness using FFmpeg's loudnorm filter.

    Args:
        input_path: Input audio/video file
        output_path: Output file
        target_lufs: Target integrated loudness
        true_peak: Target true peak

    Returns:
        AudioQualityResult of normalized audio
    """
    ffmpeg = get_ffmpeg()

    # Two-pass loudnorm for accuracy
    # Pass 1: Measure
    args1 = [
        "-i", str(input_path),
        "-af", f"loudnorm=I={target_lufs}:TP={true_peak}:LRA=11:print_format=json",
        "-f", "null", "-",
    ]

    result1 = await ffmpeg.run(args1)

    # Parse JSON output from stderr
    import json
    json_start = result1.stderr.find("{")
    json_end = result1.stderr.rfind("}") + 1
    if json_start >= 0 and json_end > json_start:
        measured = json.loads(result1.stderr[json_start:json_end])

        # Pass 2: Normalize with measured values
        args2 = [
            "-i", str(input_path),
            "-af", (
                f"loudnorm=I={target_lufs}:TP={true_peak}:LRA=11:"
                f"measured_I={measured.get('input_i', target_lufs)}:"
                f"measured_TP={measured.get('input_tp', true_peak)}:"
                f"measured_LRA={measured.get('input_lra', 11)}:"
                f"measured_thresh={measured.get('input_thresh', -9999)}:"
                f"offset={measured.get('target_offset', 0)}:"
                f"linear=true:print_format=summary"
            ),
            "-c:a", "aac", "-b:a", "192k",
            str(output_path),
        ]

        result2 = await ffmpeg.run(args2)
        if not result2.success:
            raise RuntimeError(f"Loudnorm failed: {result2.stderr}")

    # Analyze normalized audio
    return await analyze_audio_quality(output_path)


async def reduce_noise(
    input_path: Path,
    output_path: Path,
    noise_reduce_db: float = 12,
) -> bool:
    """
    Reduce noise using FFmpeg's afftdn filter.

    Args:
        input_path: Input audio file
        output_path: Output file
        noise_reduce_db: Noise reduction amount (6-24)

    Returns:
        Success boolean
    """
    ffmpeg = get_ffmpeg()

    args = [
        "-i", str(input_path),
        "-af", f"afftdn=nr={noise_reduce_db}:nf=-25",
        "-c:a", "aac", "-b:a", "192k",
        str(output_path),
    ]

    result = await ffmpeg.run(args)
    return result.success


async def enhance_speech(
    input_path: Path,
    output_path: Path,
) -> bool:
    """
    Enhance speech clarity using high-pass, compression, and EQ.

    Args:
        input_path: Input audio file
        output_path: Output file

    Returns:
        Success boolean
    """
    ffmpeg = get_ffmpeg()

    # High-pass (remove rumble) -> Compressor -> EQ (presence boost) -> De-esser
    args = [
        "-i", str(input_path),
        "-af", (
            "highpass=f=80,"
            "acompressor=threshold=-18dB:ratio=3:attack=5:release=50,"
            "equalizer=f=3000:width_type=h:width=50:g=3,"
            "deesser=i=0.5"
        ),
        "-c:a", "aac", "-b:a", "192k",
        str(output_path),
    ]

    result = await ffmpeg.run(args)
    return result.success