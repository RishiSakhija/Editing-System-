"""
AutoEdit Pro — Script to Audio Alignment
Aligns provided script text to transcribed words using DTW (Dynamic Time Warping).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from autoedit.ingestion.transcribe import TranscriptionResult, TranscriptionWord
from autoedit.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ScriptSegment:
    """Segment of script text with optional timing."""
    index: int
    text: str
    start: float | None = None
    end: float | None = None
    words: list[str] = field(default_factory=list)
    matched: bool = False
    confidence: float = 0.0


@dataclass
class ScriptAlignmentResult:
    """Result of script-to-audio alignment."""
    script_segments: list[ScriptSegment]
    transcription: TranscriptionResult
    aligned_segments: list[ScriptSegment]  # Only matched segments with timing
    unmatched_script: list[ScriptSegment]
    unmatched_transcription: list[TranscriptionSegment]
    overall_confidence: float
    coverage: float  # Fraction of script aligned


def normalize_text(text: str) -> str:
    """Normalize text for comparison."""
    # Lowercase, remove punctuation, normalize whitespace
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def tokenize(text: str) -> list[str]:
    """Tokenize text into words."""
    return normalize_text(text).split()


def compute_dtw_cost_matrix(seq1: list[str], seq2: list[str]) -> np.ndarray:
    """
    Compute DTW cost matrix between two word sequences.
    Cost = 0 for match, 1 for substitution, 0.5 for insertion/deletion.
    """
    n, m = len(seq1), len(seq2)
    dp = np.full((n + 1, m + 1), np.inf)
    dp[0, 0] = 0

    for i in range(1, n + 1):
        dp[i, 0] = i * 0.5  # Deletion cost
    for j in range(1, m + 1):
        dp[0, j] = j * 0.5  # Insertion cost

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0.0 if seq1[i - 1] == seq2[j - 1] else 1.0
            dp[i, j] = cost + min(
                dp[i - 1, j - 1],    # Match/substitute
                dp[i - 1, j] + 0.5,  # Delete from seq1
                dp[i, j - 1] + 0.5,  # Insert into seq1
            )

    return dp


def traceback_dtw(dp: np.ndarray) -> list[tuple[int, int]]:
    """Trace back through DTW matrix to find alignment path."""
    i, j = dp.shape[0] - 1, dp.shape[1] - 1
    path = []

    while i > 0 or j > 0:
        path.append((i, j))
        if i == 0:
            j -= 1
        elif j == 0:
            i -= 1
        else:
            # Find minimum predecessor
            costs = [
                (dp[i - 1, j - 1], (i - 1, j - 1)),
                (dp[i - 1, j], (i - 1, j)),
                (dp[i, j - 1], (i, j - 1)),
            ]
            _, (i, j) = min(costs)

    path.append((0, 0))
    return list(reversed(path))


def align_script_to_transcription(
    script_text: str,
    transcription: TranscriptionResult,
    min_segment_words: int = 3,
    max_gap_seconds: float = 5.0,
) -> ScriptAlignmentResult:
    """
    Align script text to transcription using DTW.

    Args:
        script_text: Full script text
        transcription: Whisper transcription result
        min_segment_words: Minimum words per script segment
        max_gap_seconds: Max time gap to consider continuous

    Returns:
        ScriptAlignmentResult with aligned timings
    """
    # Split script into sentences/paragraphs
    script_sentences = split_into_sentences(script_text)
    script_tokens = [tokenize(s) for s in script_sentences]

    # Get transcription words
    trans_words = transcription.get_words_flat()
    trans_tokens = [w.word for w in trans_words]

    logger.info("alignment_start", script_sentences=len(script_sentences), trans_words=len(trans_tokens))

    # Create script segments
    script_segments = []
    for idx, (sentence, tokens) in enumerate(zip(script_sentences, script_tokens)):
        if len(tokens) >= min_segment_words:
            script_segments.append(ScriptSegment(
                index=idx,
                text=sentence,
                words=tokens,
            ))

    # Align each script segment to transcription
    aligned_segments = []
    unmatched_script = []
    trans_used = np.zeros(len(trans_tokens), dtype=bool)

    for seg in script_segments:
        # Find best matching window in transcription
        best_match = find_best_match(seg.words, trans_tokens, trans_used, max_gap_seconds, trans_words)

        if best_match:
            start_idx, end_idx, confidence = best_match
            # Mark transcription words as used
            trans_used[start_idx:end_idx] = True

            # Get timing from transcription
            seg.start = trans_words[start_idx].start
            seg.end = trans_words[end_idx - 1].end
            seg.matched = True
            seg.confidence = confidence
            aligned_segments.append(seg)
        else:
            unmatched_script.append(seg)

    # Find unmatched transcription segments
    unmatched_transcription = find_unmatched_transcription(transcription, trans_used)

    # Calculate overall metrics
    total_script_words = sum(len(s.words) for s in script_segments)
    matched_script_words = sum(len(s.words) for s in aligned_segments)
    coverage = matched_script_words / total_script_words if total_script_words > 0 else 0.0

    overall_confidence = np.mean([s.confidence for s in aligned_segments]) if aligned_segments else 0.0

    logger.info("alignment_complete",
        aligned=len(aligned_segments),
        unmatched_script=len(unmatched_script),
        unmatched_transcription=len(unmatched_transcription),
        coverage=coverage,
        confidence=overall_confidence,
    )

    return ScriptAlignmentResult(
        script_segments=script_segments,
        transcription=transcription,
        aligned_segments=aligned_segments,
        unmatched_script=unmatched_script,
        unmatched_transcription=unmatched_transcription,
        overall_confidence=overall_confidence,
        coverage=coverage,
    )


def find_best_match(
    script_tokens: list[str],
    trans_tokens: list[str],
    trans_used: np.ndarray,
    max_gap_seconds: float,
    trans_words: list[TranscriptionWord],
) -> tuple[int, int, float] | None:
    """Find best matching window in transcription for script tokens."""
    if not script_tokens or not trans_tokens:
        return None

    best_score = float('inf')
    best_window = None

    # Search windows of similar length (±50%)
    target_len = len(script_tokens)
    min_len = max(1, int(target_len * 0.5))
    max_len = min(len(trans_tokens), int(target_len * 1.5))

    for window_len in range(min_len, max_len + 1):
        for start in range(len(trans_tokens) - window_len + 1):
            end = start + window_len

            # Skip if too many words already used
            if np.mean(trans_used[start:end]) > 0.5:
                continue

            # Check time gap constraint
            if end < len(trans_words):
                time_span = trans_words[end - 1].end - trans_words[start].start
                if time_span > max_gap_seconds * 2:
                    continue

            window_tokens = trans_tokens[start:end]

            # Compute DTW distance
            dp = compute_dtw_cost_matrix(script_tokens, window_tokens)
            score = dp[-1, -1] / max(len(script_tokens), window_len)

            if score < best_score:
                best_score = score
                best_window = (start, end, score)

    if best_window:
        start, end, score = best_window
        confidence = max(0.0, 1.0 - score)  # Convert distance to confidence
        return (start, end, confidence)

    return None


def find_unmatched_transcription(
    transcription: TranscriptionResult,
    trans_used: np.ndarray,
) -> list:
    """Find transcription segments that weren't matched to script."""

    unmatched = []
    trans_words = transcription.get_words_flat()

    for seg in transcription.segments:
        # Check if any words in this segment are unused
        seg_word_indices = []
        for i, w in enumerate(trans_words):
            if seg.start <= w.start < seg.end:
                seg_word_indices.append(i)

        if seg_word_indices:
            used_fraction = np.mean(trans_used[seg_word_indices])
            if used_fraction < 0.5:  # Less than 50% used
                unmatched.append(seg)

    return unmatched


def split_into_sentences(text: str) -> list[str]:
    """Split text into sentences, preserving paragraph breaks."""
    # Split by double newline first (paragraphs)
    paragraphs = text.split("\n\n")
    sentences = []

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        # Split by sentence endings
        # Simple regex for sentence splitting
        sent_parts = re.split(r'(?<=[.!?])\s+(?=[A-Z])', para)

        for sent in sent_parts:
            sent = sent.strip()
            if sent:
                sentences.append(sent)

    return sentences


async def align_script_file(
    script_path: Path,
    transcription: TranscriptionResult,
    **kwargs,
) -> ScriptAlignmentResult:
    """Align script file to transcription."""
    script_text = script_path.read_text(encoding="utf-8")
    return align_script_to_transcription(script_text, transcription, **kwargs)


def generate_timeline_markers(
    alignment: ScriptAlignmentResult,
    include_unmatched: bool = False,
) -> list[dict]:
    """
    Generate timeline markers from alignment for rough cut.

    Returns:
        List of dicts with: {label, start, end, type, confidence}
    """
    markers = []

    for seg in alignment.aligned_segments:
        if seg.matched and seg.start is not None:
            markers.append({
                "label": seg.text[:80],
                "start": seg.start,
                "end": seg.end,
                "type": "script_segment",
                "confidence": seg.confidence,
                "script_index": seg.index,
            })

    if include_unmatched:
        for seg in alignment.unmatched_script:
            markers.append({
                "label": f"[UNALIGNED] {seg.text[:80]}",
                "start": None,
                "end": None,
                "type": "script_segment",
                "confidence": 0.0,
                "script_index": seg.index,
            })

    return markers