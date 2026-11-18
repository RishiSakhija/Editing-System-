"""
AutoEdit Pro — Whisper.cpp Transcription Wrapper
Fast, local transcription with word-level timestamps and VAD.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from pathlib import Path

from autoedit.config import get_settings
from autoedit.utils.hardware import MODEL_REGISTRY
from autoedit.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class TranscriptionWord:
    """Single word with timing."""
    word: str
    start: float
    end: float
    probability: float = 1.0


@dataclass
class TranscriptionSegment:
    """Transcription segment (sentence/phrase)."""
    id: int
    start: float
    end: float
    text: str
    words: list[TranscriptionWord] = field(default_factory=list)
    avg_logprob: float = 0.0
    no_speech_prob: float = 0.0
    temperature: float = 0.0


@dataclass
class TranscriptionResult:
    """Complete transcription result."""
    text: str
    segments: list[TranscriptionSegment]
    language: str
    language_probability: float
    duration: float
    model: str
    processing_time_ms: float
    words: list[TranscriptionWord] = field(default_factory=list)

    def get_words_flat(self) -> list[TranscriptionWord]:
        """Get all words as flat list."""
        if self.words:
            return self.words
        words = []
        for seg in self.segments:
            words.extend(seg.words)
        return words

    def find_silences(self, min_duration: float = 0.3) -> list[tuple[float, float]]:
        """Find silence gaps between words."""
        words = self.get_words_flat()
        if not words:
            return []

        silences = []
        for i in range(len(words) - 1):
            gap = words[i + 1].start - words[i].end
            if gap >= min_duration:
                silences.append((words[i].end, words[i + 1].start))
        return silences

    def find_filler_words(self, fillers: list[str] | None = None) -> list[TranscriptionWord]:
        """Find filler words (um, uh, like, etc.)."""
        if fillers is None:
            fillers = ["um", "uh", "er", "ah", "like", "you know", "so", "actually", "basically", "literally"]

        filler_set = set(f.lower() for f in fillers)
        words = self.get_words_flat()
        return [w for w in words if w.word.lower().strip(".,!?") in filler_set]


class WhisperCppWrapper:
    """
    Whisper.cpp wrapper for fast local transcription.
    Uses the whisper-cli binary (compiled from whisper.cpp).
    """

    def __init__(
        self,
        model_path: Path | None = None,
        binary_path: Path | None = None,
    ):
        settings = get_settings()
        self.model_path = model_path or settings.hardware.models_dir / "ggml-large-v3-turbo.bin"
        self.binary_path = binary_path or self._find_whisper_binary()
        self.model_name = "large-v3-turbo"

    def _find_whisper_binary(self) -> Path:
        """Find whisper-cli binary."""
        import shutil
        # Common names
        for name in ["whisper-cli", "whisper", "main"]:
            found = shutil.which(name)
            if found:
                return Path(found)

        # Check local build directory
        local_paths = [
            Path("whisper.cpp/build/bin/whisper-cli"),
            Path("whisper.cpp/build/bin/Release/whisper-cli.exe"),
            Path("whisper.cpp/main"),
            Path("whisper.cpp/main.exe"),
        ]
        for p in local_paths:
            if p.exists():
                return p

        raise FileNotFoundError(
            "whisper-cli not found. Build whisper.cpp or install via package manager.\n"
            "  Ubuntu: sudo apt install whisper-cpp\n"
            "  macOS: brew install whisper-cpp\n"
            "  Or build from: https://github.com/ggerganov/whisper.cpp"
        )

    async def transcribe(
        self,
        audio_path: Path,
        language: str = "en",
        word_timestamps: bool = True,
        vad: bool = True,
        vad_threshold: float = 0.5,
        min_silence_duration: float = 0.3,
        translate: bool = False,
        temperature: float = 0.0,
        threads: int = 4,
    ) -> TranscriptionResult:
        """
        Transcribe audio file using whisper.cpp.

        Args:
            audio_path: Path to audio file (WAV, MP3, etc.)
            language: Language code (en, auto, etc.)
            word_timestamps: Output word-level timestamps
            vad: Enable voice activity detection
            vad_threshold: VAD threshold (0-1)
            min_silence_duration: Minimum silence for VAD split
            translate: Translate to English
            temperature: Sampling temperature
            threads: CPU threads to use

        Returns:
            TranscriptionResult with segments and words
        """
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        logger.info("transcribe_start", audio=str(audio_path), model=self.model_name)

        import time
        start_time = time.perf_counter()

        # Build command
        cmd = [
            str(self.binary_path),
            "-m", str(self.model_path),
            "-f", str(audio_path),
            "-l", language,
            "-oj",  # Output JSON
            "-of", str(audio_path.with_suffix("")),  # Output prefix
            "-t", str(threads),
        ]

        if word_timestamps:
            cmd.append("-owts")  # Output word timestamps

        if vad:
            cmd.extend(["-vad", f"-vad-thold={vad_threshold}"])

        if translate:
            cmd.append("-tr")

        if temperature > 0:
            cmd.extend(["--temp", str(temperature)])

        # Run whisper.cpp
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        stdout, stderr = await process.communicate()

        processing_time_ms = (time.perf_counter() - start_time) * 1000

        if process.returncode != 0:
            error = stderr.decode("utf-8", errors="replace")
            logger.error("transcribe_failed", error=error)
            raise RuntimeError(f"Whisper.cpp failed: {error}")

        # Parse JSON output
        json_path = audio_path.with_suffix(".json")
        if not json_path.exists():
            # Try alternative naming
            json_path = audio_path.parent / f"{audio_path.stem}.json"

        if not json_path.exists():
            raise RuntimeError(f"Whisper output JSON not found: {json_path}")

        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)

        # Clean up output files
        for ext in [".json", ".txt", ".srt", ".vtt", ".csv"]:
            (audio_path.parent / f"{audio_path.stem}{ext}").unlink(missing_ok=True)

        result = self._parse_output(data, processing_time_ms)
        logger.info("transcribe_complete", duration=result.duration, segments=len(result.segments), time_ms=processing_time_ms)
        return result

    def _parse_output(self, data: dict, processing_time_ms: float) -> TranscriptionResult:
        """Parse whisper.cpp JSON output."""
        segments = []
        all_words = []

        for seg_data in data.get("transcription", []):
            words = []
            for w_data in seg_data.get("words", []):
                word = TranscriptionWord(
                    word=w_data.get("word", ""),
                    start=w_data.get("start", 0.0),
                    end=w_data.get("end", 0.0),
                    probability=w_data.get("probability", 1.0),
                )
                words.append(word)
                all_words.append(word)

            segment = TranscriptionSegment(
                id=seg_data.get("id", 0),
                start=seg_data.get("start", 0.0),
                end=seg_data.get("end", 0.0),
                text=seg_data.get("text", "").strip(),
                words=words,
                avg_logprob=seg_data.get("avg_logprob", 0.0),
                no_speech_prob=seg_data.get("no_speech_prob", 0.0),
                temperature=seg_data.get("temperature", 0.0),
            )
            segments.append(segment)

        return TranscriptionResult(
            text=data.get("text", "").strip(),
            segments=segments,
            language=data.get("language", "en"),
            language_probability=data.get("language_probability", 1.0),
            duration=segments[-1].end if segments else 0.0,
            model=self.model_name,
            processing_time_ms=processing_time_ms,
            words=all_words,
        )

    async def transcribe_streaming(
        self,
        audio_path: Path,
        chunk_duration: float = 30.0,
        overlap: float = 2.0,
        **kwargs,
    ) -> TranscriptionResult:
        """
        Transcribe long audio in chunks with overlap for memory efficiency.
        Useful for very long files (>30 min).
        """
        # For now, just use regular transcribe
        # TODO: Implement chunked transcription for long files
        return await self.transcribe(audio_path, **kwargs)


async def transcribe_audio(
    audio_path: Path,
    model: str = "large-v3-turbo",
    language: str = "en",
    **kwargs,
) -> TranscriptionResult:
    """
    High-level transcription function.

    Args:
        audio_path: Path to audio file
        model: Whisper model name
        language: Language code
        **kwargs: Additional options passed to WhisperCppWrapper.transcribe

    Returns:
        TranscriptionResult
    """
    settings = get_settings()
    model_file = MODEL_REGISTRY.get(f"whisper-{model}", {}).get("filename", f"ggml-{model}.bin")
    model_path = settings.hardware.models_dir / model_file

    if not model_path.exists():
        # Try default
        model_path = settings.hardware.models_dir / "ggml-large-v3-turbo.bin"

    wrapper = WhisperCppWrapper(model_path=model_path)
    return await wrapper.transcribe(audio_path, language=language, **kwargs)


# Convenience function for CLI
async def transcribe_file(
    input_path: Path,
    output_path: Path | None = None,
    **kwargs,
) -> TranscriptionResult:
    """Transcribe video/audio file directly (extracts audio first)."""
    from autoedit.ingestion.sync import extract_audio

    # Extract audio if video
    if input_path.suffix.lower() in [".mp4", ".mov", ".mkv", ".webm", ".avi"]:
        audio_path = input_path.with_suffix(".wav")
        await extract_audio(input_path, audio_path)
    else:
        audio_path = input_path

    try:
        result = await transcribe_audio(audio_path, **kwargs)

        # Save result if output path provided
        if output_path:
            import json
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump({
                    "text": result.text,
                    "segments": [
                        {
                            "id": s.id,
                            "start": s.start,
                            "end": s.end,
                            "text": s.text,
                            "words": [
                                {"word": w.word, "start": w.start, "end": w.end, "probability": w.probability}
                                for w in s.words
                            ],
                        }
                        for s in result.segments
                    ],
                    "language": result.language,
                    "duration": result.duration,
                    "model": result.model,
                }, f, indent=2, ensure_ascii=False)

        return result
    finally:
        # Clean up temp audio if we extracted it
        if input_path.suffix.lower() in [".mp4", ".mov", ".mkv", ".webm", ".avi"]:
            audio_path.unlink(missing_ok=True)