"""
AutoEdit Pro — Face Detection & Tracking
Uses YOLOv8 for face detection with ByteTrack for temporal consistency.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from autoedit.config import get_settings
from autoedit.utils.hardware import MODEL_REGISTRY
from autoedit.utils.logging import get_logger

logger = get_logger(__name__)

# Try to import ultralytics (YOLO)
try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False
    logger.warning("ultralytics not available, face tracking disabled")

# Try to import onnxruntime for faster inference
try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False


@dataclass
class FaceBox:
    """Single face detection."""
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    track_id: int | None = None

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return self.width * self.height


@dataclass
class FaceTrack:
    """Tracked face across frames."""
    track_id: int
    boxes: list[FaceBox]  # One per frame
    start_frame: int
    end_frame: int
    avg_confidence: float

    @property
    def duration_frames(self) -> int:
        return self.end_frame - self.start_frame + 1

    @property
    def duration_seconds(self) -> float:
        return self.duration_frames / 30.0  # Approximate

    def get_box_at_frame(self, frame_idx: int) -> FaceBox | None:
        """Get face box at specific frame index (relative to track start)."""
        idx = frame_idx - self.start_frame
        if 0 <= idx < len(self.boxes):
            return self.boxes[idx]
        return None

    def get_smoothed_box(self, frame_idx: int, window: int = 5) -> FaceBox | None:
        """Get temporally smoothed box using moving average."""
        idx = frame_idx - self.start_frame
        if idx < 0 or idx >= len(self.boxes):
            return None

        start = max(0, idx - window // 2)
        end = min(len(self.boxes), idx + window // 2 + 1)

        if start >= end:
            return self.boxes[idx]

        boxes = self.boxes[start:end]
        avg_box = FaceBox(
            x1=np.mean([b.x1 for b in boxes]),
            y1=np.mean([b.y1 for b in boxes]),
            x2=np.mean([b.x2 for b in boxes]),
            y2=np.mean([b.y2 for b in boxes]),
            confidence=np.mean([b.confidence for b in boxes]),
            track_id=self.track_id,
        )
        return avg_box


@dataclass
class FaceTrackResult:
    """Result of face tracking."""
    tracks: list[FaceTrack]
    total_frames: int
    fps: float
    processing_time_ms: float
    video_width: int
    video_height: int


class YOLOFaceDetector:
    """YOLOv8 face detector with optional ONNX optimization."""

    def __init__(
        self,
        model_path: Path | None = None,
        conf_threshold: float = 0.45,
        iou_threshold: float = 0.5,
        device: str = "auto",
        half_precision: bool = True,
    ):
        settings = get_settings()
        self.model_path = model_path or settings.hardware.models_dir / "yolov8n-face.pt"
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.half_precision = half_precision

        # Determine device
        if device == "auto":
            import torch
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.model = None
        self.session = None
        self.input_name = None

    async def load(self) -> None:
        """Load model (YOLO or ONNX)."""
        if ONNX_AVAILABLE and self.model_path.suffix == ".onnx":
            await self._load_onnx()
        else:
            await self._load_yolo()

    async def _load_yolo(self) -> None:
        """Load YOLO model via ultralytics."""
        if not ULTRALYTICS_AVAILABLE:
            raise RuntimeError("ultralytics not installed")

        self.model = YOLO(str(self.model_path))
        self.model.to(self.device)
        if self.half_precision and self.device == "cuda":
            self.model.model.half()
        logger.info("yolo_face_model_loaded", device=self.device, path=str(self.model_path))

    async def _load_onnx(self) -> None:
        """Load ONNX model for faster inference."""
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if self.device == "cuda" else ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(str(self.model_path), providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        logger.info("onnx_face_model_loaded", providers=providers)

    async def detect(self, frame: np.ndarray) -> list[FaceBox]:
        """Detect faces in a single frame."""
        if self.model is not None:
            return await self._detect_yolo(frame)
        elif self.session is not None:
            return await self._detect_onnx(frame)
        else:
            raise RuntimeError("Model not loaded")

    async def _detect_yolo(self, frame: np.ndarray) -> list[FaceBox]:
        """Detect using YOLO."""
        results = self.model(frame, conf=self.conf_threshold, iou=self.iou_threshold, verbose=False)

        boxes = []
        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = box.conf[0].cpu().numpy()
                    boxes.append(FaceBox(
                        x1=float(x1), y1=float(y1),
                        x2=float(x2), y2=float(y2),
                        confidence=float(conf),
                    ))
        return boxes

    async def _detect_onnx(self, frame: np.ndarray) -> list[FaceBox]:
        """Detect using ONNX runtime."""
        # Preprocess
        input_shape = (640, 640)
        h, w = frame.shape[:2]
        scale = min(input_shape[0] / h, input_shape[1] / w)
        new_h, new_w = int(h * scale), int(w * scale)

        resized = cv2.resize(frame, (new_w, new_h))
        padded = np.full((input_shape[0], input_shape[1], 3), 114, dtype=np.uint8)
        padded[:new_h, :new_w] = resized

        # Normalize
        input_tensor = padded.astype(np.float32) / 255.0
        input_tensor = np.transpose(input_tensor, (2, 0, 1))  # HWC -> CHW
        input_tensor = np.expand_dims(input_tensor, axis=0)  # Add batch dim

        # Inference
        outputs = self.session.run(None, {self.input_name: input_tensor})[0]

        # Post-process (simplified - assumes YOLOv8 output format)
        boxes = self._postprocess_onnx(outputs[0], scale, (w, h))
        return boxes

    def _postprocess_onnx(self, output: np.ndarray, scale: float, orig_shape: tuple[int, int]) -> list[FaceBox]:
        """Post-process ONNX output to face boxes."""
        boxes = []
        w, h = orig_shape

        for det in output:
            conf = det[4]
            if conf < self.conf_threshold:
                continue

            # YOLO format: x_center, y_center, width, height (normalized)
            xc, yc, bw, bh = det[:4]
            x1 = (xc - bw / 2) * w
            y1 = (yc - bh / 2) * h
            x2 = (xc + bw / 2) * w
            y2 = (yc + bh / 2) * h

            boxes.append(FaceBox(
                x1=max(0, x1), y1=max(0, y1),
                x2=min(w, x2), y2=min(h, y2),
                confidence=float(conf),
            ))

        # NMS
        return self._nms(boxes)

    def _nms(self, boxes: list[FaceBox]) -> list[FaceBox]:
        """Non-maximum suppression."""
        if not boxes:
            return []

        boxes = sorted(boxes, key=lambda b: b.confidence, reverse=True)
        keep = []

        while boxes:
            best = boxes.pop(0)
            keep.append(best)
            boxes = [b for b in boxes if self._iou(best, b) < self.iou_threshold]

        return keep

    def _iou(self, box1: FaceBox, box2: FaceBox) -> float:
        """Calculate IoU between two boxes."""
        x1 = max(box1.x1, box2.x1)
        y1 = max(box1.y1, box2.y1)
        x2 = min(box1.x2, box2.x2)
        y2 = min(box1.y2, box2.y2)

        if x2 <= x1 or y2 <= y1:
            return 0.0

        intersection = (x2 - x1) * (y2 - y1)
        area1 = box1.area
        area2 = box2.area
        return intersection / (area1 + area2 - intersection)


class ByteTrackTracker:
    """Simple ByteTrack implementation for face tracking."""

    def __init__(
        self,
        track_thresh: float = 0.5,
        track_buffer: int = 30,
        match_thresh: float = 0.8,
        frame_rate: int = 30,
    ):
        self.track_thresh = track_thresh
        self.track_buffer = track_buffer
        self.match_thresh = match_thresh
        self.frame_rate = frame_rate

        self.tracks: dict[int, FaceTrack] = {}
        self.next_track_id = 1
        self.frame_count = 0

    def update(self, detections: list[FaceBox]) -> list[FaceTrack]:
        """Update tracks with new detections."""
        self.frame_count += 1

        # Filter detections by confidence
        detections = [d for d in detections if d.confidence >= self.track_thresh]

        # Get active tracks
        active_tracks = {tid: t for tid, t in self.tracks.items() if t.end_frame == self.frame_count - 1}

        # Match detections to tracks using IoU
        matched_tracks, unmatched_dets = self._match_detections(detections, active_tracks)

        # Update matched tracks
        for track_id, det in matched_tracks:
            track = self.tracks[track_id]
            det.track_id = track_id
            track.boxes.append(det)
            track.end_frame = self.frame_count

        # Create new tracks for unmatched detections
        for det in unmatched_dets:
            det.track_id = self.next_track_id
            self.tracks[self.next_track_id] = FaceTrack(
                track_id=self.next_track_id,
                boxes=[det],
                start_frame=self.frame_count,
                end_frame=self.frame_count,
                avg_confidence=det.confidence,
            )
            self.next_track_id += 1

        # Remove lost tracks
        lost_tracks = [
            tid for tid, t in self.tracks.items()
            if self.frame_count - t.end_frame > self.track_buffer
        ]
        for tid in lost_tracks:
            del self.tracks[tid]

        # Update avg confidence
        for track in self.tracks.values():
            track.avg_confidence = np.mean([b.confidence for b in track.boxes])

        # Return active tracks
        return [t for t in self.tracks.values() if t.end_frame == self.frame_count]

    def _match_detections(
        self,
        detections: list[FaceBox],
        tracks: dict[int, FaceTrack],
    ) -> tuple[list[tuple[int, FaceBox]], list[FaceBox]]:
        """Match detections to tracks using IoU."""
        if not detections or not tracks:
            return [], detections

        track_ids = list(tracks.keys())
        track_boxes = [tracks[tid].boxes[-1] for tid in track_ids]

        # Compute IoU matrix
        iou_matrix = np.zeros((len(detections), len(track_boxes)))
        for i, det in enumerate(detections):
            for j, trk in enumerate(track_boxes):
                iou_matrix[i, j] = self._iou(det, trk)

        # Hungarian algorithm would be better, but greedy works for few faces
        matched = []
        used_dets = set()
        used_trks = set()

        # Sort by IoU descending
        pairs = []
        for i in range(len(detections)):
            for j in range(len(track_boxes)):
                if iou_matrix[i, j] >= self.match_thresh:
                    pairs.append((iou_matrix[i, j], i, j))
        pairs.sort(reverse=True)

        for _, det_idx, trk_idx in pairs:
            if det_idx not in used_dets and trk_idx not in used_trks:
                matched.append((track_ids[trk_idx], detections[det_idx]))
                used_dets.add(det_idx)
                used_trks.add(trk_idx)

        unmatched_dets = [detections[i] for i in range(len(detections)) if i not in used_dets]

        return matched, unmatched_dets

    def _iou(self, box1: FaceBox, box2: FaceBox) -> float:
        x1 = max(box1.x1, box2.x1)
        y1 = max(box1.y1, box2.y1)
        x2 = min(box1.x2, box2.x2)
        y2 = min(box1.y2, box2.y2)

        if x2 <= x1 or y2 <= y1:
            return 0.0

        intersection = (x2 - x1) * (y2 - y1)
        area1 = box1.area
        area2 = box2.area
        return intersection / (area1 + area2 - intersection)


async def track_faces(
    video_path: Path,
    sample_rate: int = 1,  # Process every Nth frame
    conf_threshold: float = 0.45,
    min_face_size: int = 50,
    track_buffer: int = 30,
) -> FaceTrackResult:
    """
    Track faces throughout a video.

    Args:
        video_path: Path to video file
        sample_rate: Process every Nth frame (1 = all frames)
        conf_threshold: Face detection confidence threshold
        min_face_size: Minimum face size in pixels
        track_buffer: Frames to keep lost tracks

    Returns:
        FaceTrackResult with all tracks
    """
    import time
    start = time.perf_counter()

    # Load detector
    detector = YOLOFaceDetector(conf_threshold=conf_threshold)
    await detector.load()

    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    tracker = ByteTrackTracker(
        track_thresh=conf_threshold,
        track_buffer=track_buffer,
        frame_rate=int(fps),
    )

    all_tracks: dict[int, FaceTrack] = {}
    frame_idx = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % sample_rate == 0:
                # Detect faces
                detections = await detector.detect(frame)

                # Filter by size
                detections = [d for d in detections if d.width >= min_face_size and d.height >= min_face_size]

                # Update tracks
                active_tracks = tracker.update(detections)

                # Merge into all_tracks
                for track in active_tracks:
                    if track.track_id in all_tracks:
                        existing = all_tracks[track.track_id]
                        existing.boxes.extend(track.boxes)
                        existing.end_frame = track.end_frame
                        existing.avg_confidence = np.mean([b.confidence for b in existing.boxes])
                    else:
                        all_tracks[track.track_id] = track

            frame_idx += 1

            # Progress log
            if frame_idx % 300 == 0:
                logger.debug("face_track_progress", frame=frame_idx, total=total_frames, tracks=len(all_tracks))

    finally:
        cap.release()

    # Convert to list
    tracks = list(all_tracks.values())

    # Filter out very short tracks
    tracks = [t for t in tracks if t.duration_frames >= 5]

    # Re-index
    for i, track in enumerate(tracks):
        track.track_id = i

    processing_time = (time.perf_counter() - start) * 1000

    logger.info("face_track_complete",
        total_frames=frame_idx,
        tracks_found=len(tracks),
        processing_time_ms=processing_time,
    )

    return FaceTrackResult(
        tracks=tracks,
        total_frames=frame_idx,
        fps=fps,
        processing_time_ms=processing_time,
        video_width=width,
        video_height=height,
    )


def get_primary_face(tracks: list[FaceTrack]) -> FaceTrack | None:
    """Get the most prominent face track (largest average area, longest duration)."""
    if not tracks:
        return None

    scored = []
    for track in tracks:
        avg_area = np.mean([b.area for b in track.boxes])
        score = avg_area * track.duration_frames
        scored.append((score, track))

    scored.sort(reverse=True)
    return scored[0][1]


def get_face_crop_region(
    track: FaceTrack,
    frame_idx: int,
    padding: float = 0.3,
    aspect_ratio: float = 9/16,  # Vertical for shorts
) -> tuple[int, int, int, int] | None:
    """
    Get crop region for face at frame (for auto-zoom/crop).

    Returns:
        (x1, y1, x2, y2) in pixels or None
    """
    box = track.get_smoothed_box(frame_idx)
    if not box:
        return None

    # Add padding
    w = box.width
    h = box.height
    cx, cy = box.center

    padded_w = w * (1 + padding * 2)
    padded_h = h * (1 + padding * 2)

    # Adjust for aspect ratio
    if padded_w / padded_h > aspect_ratio:
        padded_h = padded_w / aspect_ratio
    else:
        padded_w = padded_h * aspect_ratio

    x1 = int(cx - padded_w / 2)
    y1 = int(cy - padded_h / 2)
    x2 = int(cx + padded_w / 2)
    y2 = int(cy + padded_h / 2)

    return (x1, y1, x2, y2)