"""
AutoEdit Pro — Analysis Module Tests
"""

import pytest
import numpy as np
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch

from autoedit.analysis.scene_detect import (
    Scene,
    SceneDetectionResult,
    merge_short_scenes,
    filter_scenes_by_duration,
    get_scene_at_time,
)
from autoedit.analysis.face_track import (
    FaceBox,
    FaceTrack,
    FaceTrackResult,
    ByteTrackTracker,
    get_primary_face,
    get_face_crop_region,
)
from autoedit.analysis.beat_detect import (
    Beat,
    BeatDetectionResult,
    sync_cuts_to_beats,
    get_downbeat_times,
    get_beat_grid,
)
from autoedit.analysis.audio_quality import (
    AudioQualityResult,
    estimate_snr,
    compute_speech_clarity,
    detect_audio_issues,
)


class TestSceneDetect:
    """Tests for scene detection utilities."""

    def create_test_scenes(self) -> list[Scene]:
        return [
            Scene(0, 0.0, 2.0, 2.0, 0, 60, "test", 0.9),
            Scene(1, 2.0, 2.3, 0.3, 60, 69, "test", 0.9),  # Short
            Scene(2, 2.3, 5.0, 2.7, 69, 150, "test", 0.9),
            Scene(3, 5.0, 5.2, 0.2, 150, 156, "test", 0.9),  # Very short
            Scene(4, 5.2, 10.0, 4.8, 156, 300, "test", 0.9),
        ]

    def test_merge_short_scenes(self):
        scenes = self.create_test_scenes()
        merged = merge_short_scenes(scenes, min_duration=1.0)

        # Should merge scenes 1 and 3 into adjacent
        assert len(merged) == 3
        # First merge: scene 0 + 1
        assert merged[0].duration >= 2.0
        # Second merge: scene 2 + 3
        assert merged[1].duration >= 2.7

    def test_filter_scenes_by_duration(self):
        scenes = self.create_test_scenes()
        filtered = filter_scenes_by_duration(scenes, min_duration=1.0, max_duration=3.0)

        # Only scenes with duration 1-3 seconds
        assert len(filtered) == 2
        for s in filtered:
            assert 1.0 <= s.duration <= 3.0

    def test_get_scene_at_time(self):
        scenes = self.create_test_scenes()

        # Test exact boundaries
        assert get_scene_at_time(scenes, 0.0) == scenes[0]
        assert get_scene_at_time(scenes, 2.0) == scenes[1]
        assert get_scene_at_time(scenes, 5.0) == scenes[3]
        assert get_scene_at_time(scenes, 10.0) == scenes[4]

        # Test middle
        assert get_scene_at_time(scenes, 1.0) == scenes[0]
        assert get_scene_at_time(scenes, 3.5) == scenes[2]

        # Test out of bounds
        assert get_scene_at_time(scenes, -1.0) is None
        assert get_scene_at_time(scenes, 15.0) is None


class TestFaceTrack:
    """Tests for face tracking."""

    def test_face_box_properties(self):
        box = FaceBox(100, 100, 200, 200, 0.9, track_id=1)
        assert box.center == (150, 150)
        assert box.width == 100
        assert box.height == 100
        assert box.area == 10000

    def test_face_track_duration(self):
        boxes = [
            FaceBox(100, 100, 150, 150, 0.9, 1),
            FaceBox(105, 102, 155, 152, 0.85, 1),
            FaceBox(110, 105, 160, 155, 0.9, 1),
        ]
        track = FaceTrack(1, boxes, 0, 2, 0.88)

        assert track.duration_frames == 3
        assert track.duration_seconds == 0.1  # 3 frames / 30 fps

        # Test get_box_at_frame
        assert track.get_box_at_frame(0) == boxes[0]
        assert track.get_box_at_frame(1) == boxes[1]
        assert track.get_box_at_frame(2) == boxes[2]
        assert track.get_box_at_frame(3) is None

    def test_byte_tracker_simple(self):
        tracker = ByteTrackTracker(track_thresh=0.5, track_buffer=30)

        # Frame 1: Two detections
        dets1 = [
            FaceBox(100, 100, 150, 150, 0.9),
            FaceBox(300, 100, 350, 150, 0.85),
        ]
        tracks1 = tracker.update(dets1)
        assert len(tracks1) == 2
        # Track IDs start from 1 in current implementation
        assert set(t.track_id for t in tracks1) == {1, 2}

        # Frame 2: Same positions (should match)
        dets2 = [
            FaceBox(102, 101, 152, 151, 0.88),
            FaceBox(302, 101, 352, 151, 0.87),
        ]
        tracks2 = tracker.update(dets2)
        assert len(tracks2) == 2
        # Track IDs should persist
        assert {t.track_id for t in tracks2} == {1, 2}

    def test_get_primary_face(self):
        # Create tracks with different sizes/durations
        boxes1 = [FaceBox(100, 100, 200, 200, 0.9) for _ in range(10)]  # Large, long
        boxes2 = [FaceBox(100, 100, 120, 120, 0.9) for _ in range(5)]   # Small, short

        track1 = FaceTrack(0, boxes1, 0, 9, 0.9)
        track2 = FaceTrack(1, boxes2, 0, 4, 0.9)

        primary = get_primary_face([track1, track2])
        assert primary.track_id == 0

    def test_get_face_crop_region(self):
        boxes = [
            FaceBox(200, 200, 400, 400, 0.9),
            FaceBox(210, 210, 410, 410, 0.85),
            FaceBox(220, 220, 420, 420, 0.9),
        ]
        track = FaceTrack(0, boxes, 0, 2, 0.88)

        # Get crop at frame 1 (middle)
        crop = get_face_crop_region(track, 1, padding=0.2, aspect_ratio=9/16)
        assert crop is not None
        x1, y1, x2, y2 = crop
        assert x2 > x1 and y2 > y1

        # Check aspect ratio (vertical 9:16)
        w = x2 - x1
        h = y2 - y1
        assert abs(w / h - 9/16) < 0.05


class TestBeatDetect:
    """Tests for beat detection utilities."""

    def test_sync_cuts_to_beats(self):
        # Cuts at various times
        cuts = [1.0, 2.5, 4.1, 6.8]

        # Beats at regular intervals (120 BPM = 0.5s)
        beats = [Beat(i * 0.5, 1.0) for i in range(20)]

        synced = sync_cuts_to_beats(cuts, beats, tolerance=0.15)

        # All cuts should be snapped to nearest beat within tolerance
        for orig, synced_cut in zip(cuts, synced):
            # Find nearest beat
            nearest = min(beats, key=lambda b: abs(b.time - orig))
            if abs(nearest.time - orig) <= 0.15:
                assert abs(synced_cut - nearest.time) < 0.01

    def test_get_downbeat_times(self):
        beats = [Beat(i * 0.5, 1.0, is_downbeat=(i % 4 == 0)) for i in range(16)]
        downbeats = get_downbeat_times(beats)

        assert len(downbeats) == 4
        assert downbeats == [0.0, 2.0, 4.0, 6.0]

    def test_get_beat_grid(self):
        beats = [Beat(i * 0.5, 1.0, is_downbeat=(i % 4 == 0)) for i in range(16)]
        grid = get_beat_grid(beats)

        assert len(grid) == 4
        for bar in grid:
            assert len(bar) == 4


class TestAudioQuality:
    """Tests for audio quality metrics."""

    def test_estimate_snr_clean(self):
        # Clean signal with silence portions (better for noise floor estimation)
        t = np.linspace(0, 2, 32000)
        signal = np.sin(2 * np.pi * 440 * t)
        # Add silence at beginning and end
        signal[:4000] = 0
        signal[-4000:] = 0
        snr, noise = estimate_snr(signal, 16000)
        # Should detect decent SNR for signal with silence
        assert snr > 5

    def test_estimate_snr_noisy(self):
        # Signal with noise
        signal = np.sin(np.linspace(0, 100, 16000))
        noise = 0.1 * np.random.randn(16000)
        noisy = signal + noise
        snr, noise_db = estimate_snr(noisy, 16000)
        # Should detect lower SNR
        assert snr < 30

    def test_compute_speech_clarity(self):
        # Speech-like signal (modulated)
        t = np.linspace(0, 2, 32000)
        speech = np.sin(2 * np.pi * 200 * t) * (1 + 0.5 * np.sin(2 * np.pi * 5 * t))
        clarity = compute_speech_clarity(speech, 16000)
        assert 0 <= clarity <= 1

    def test_audio_quality_result_score(self):
        result = AudioQualityResult(
            integrated_lufs=-16.0,
            true_peak_db=-2.0,
            snr_db=25.0,
            speech_clarity=0.8,
        )
        score = result.get_quality_score()
        assert 0.8 <= score <= 1.0

    def test_audio_quality_issues(self):
        result = AudioQualityResult(
            integrated_lufs=-25.0,
            true_peak_db=0.5,
            snr_db=10.0,
            speech_clarity=0.4,
        )
        issues = detect_audio_issues(result)
        assert "very_quiet" in issues
        assert "peak_clipping" in issues
        assert "low_snr" in issues
        assert "poor_clarity" in issues


if __name__ == "__main__":
    pytest.main([__file__, "-v"])