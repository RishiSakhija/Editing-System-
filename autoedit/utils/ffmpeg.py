"""
AutoEdit Pro — FFmpeg Wrapper
Safe, typed FFmpeg operations with error handling and progress tracking.
"""

from __future__ import annotations

import asyncio
import json
import shlex
import subprocess
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

from autoedit.utils.logging import get_logger, LogContext

logger = get_logger(__name__)


@dataclass
class FFmpegResult:
    """Result of an FFmpeg command."""
    success: bool
    returncode: int
    stdout: str
    stderr: str
    command: str
    duration_ms: float


@dataclass
class MediaInfo:
    """Parsed media information from ffprobe."""
    path: Path
    duration: float
    width: int
    height: int
    fps: float
    video_codec: str
    audio_codec: str | None
    audio_channels: int
    audio_sample_rate: int
    bitrate: int
    format_name: str
    size_bytes: int

    @property
    def resolution(self) -> tuple[int, int]:
        return (self.width, self.height)

    @property
    def aspect_ratio(self) -> float:
        return self.width / self.height if self.height > 0 else 1.0

    @property
    def is_vertical(self) -> bool:
        return self.height > self.width

    @property
    def is_horizontal(self) -> bool:
        return self.width > self.height


class FFmpegError(Exception):
    """FFmpeg command failed."""

    def __init__(self, result: FFmpegResult):
        self.result = result
        super().__init__(
            f"FFmpeg failed (exit {result.returncode}): {result.stderr[:500]}"
        )


class FFmpegNotFoundError(RuntimeError):
    """FFmpeg binary not found."""
    pass


class FFmpeg:
    """
    Async FFmpeg wrapper with progress tracking and error handling.
    """

    def __init__(
        self,
        ffmpeg_path: str | Path = "ffmpeg",
        ffprobe_path: str | Path = "ffprobe",
        timeout: float = 300.0,
    ):
        self.ffmpeg_path = self._find_binary(ffmpeg_path, "ffmpeg")
        self.ffprobe_path = self._find_binary(ffprobe_path, "ffprobe")
        self.timeout = timeout
        self._progress_callback: Callable[[float, str], None] | None = None

    @staticmethod
    def _find_binary(name: str | Path, fallback: str) -> Path:
        """Find binary in PATH or use provided path."""
        path = Path(name)
        if path.is_absolute() and path.exists():
            return path
        found = shutil.which(str(path))
        if found:
            return Path(found)
        # Try fallback
        found = shutil.which(fallback)
        if found:
            return Path(found)
        raise FFmpegNotFoundError(
            f"{fallback} not found in PATH. Please install FFmpeg."
        )

    def set_progress_callback(self, callback: Callable[[float, str], None] | None) -> None:
        """Set callback for progress updates: callback(progress_0_1, message)."""
        self._progress_callback = callback

    async def run(
        self,
        args: list[str],
        input_path: Path | None = None,
        output_path: Path | None = None,
        progress: bool = False,
    ) -> FFmpegResult:
        """
        Run FFmpeg command asynchronously.

        Args:
            args: FFmpeg arguments (without 'ffmpeg')
            input_path: Optional input file for progress calculation
            output_path: Optional output file for progress calculation
            progress: Enable progress tracking via stderr parsing

        Returns:
            FFmpegResult with output and metadata
        """
        cmd = [str(self.ffmpeg_path), "-y", "-hide_banner", *args]
        cmd_str = " ".join(shlex.quote(str(c)) for c in cmd)

        logger.debug("ffmpeg_run", command=cmd_str, input=str(input_path) if input_path else None)

        start_time = asyncio.get_event_loop().time()

        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                process.communicate(),
                timeout=self.timeout,
            )

            duration_ms = (asyncio.get_event_loop().time() - start_time) * 1000

            stdout = stdout_bytes.decode("utf-8", errors="replace")
            stderr = stderr_bytes.decode("utf-8", errors="replace")

            result = FFmpegResult(
                success=process.returncode == 0,
                returncode=process.returncode or 0,
                stdout=stdout,
                stderr=stderr,
                command=cmd_str,
                duration_ms=duration_ms,
            )

            if not result.success:
                logger.error(
                    "ffmpeg_failed",
                    command=cmd_str,
                    returncode=result.returncode,
                    stderr=stderr[:1000],
                )
                raise FFmpegError(result)

            logger.debug("ffmpeg_success", command=cmd_str, duration_ms=duration_ms)
            return result

        except asyncio.TimeoutError:
            logger.error("ffmpeg_timeout", command=cmd_str, timeout=self.timeout)
            raise FFmpegError(FFmpegResult(
                success=False,
                returncode=-1,
                stdout="",
                stderr=f"Timeout after {self.timeout}s",
                command=cmd_str,
                duration_ms=self.timeout * 1000,
            ))

    async def probe(self, path: Path) -> MediaInfo:
        """
        Get media information using ffprobe.

        Args:
            path: Path to media file

        Returns:
            MediaInfo with parsed metadata
        """
        if not path.exists():
            raise FileNotFoundError(f"Media file not found: {path}")

        cmd = [
            str(self.ffprobe_path),
            "-v", "quiet",
            "-print_format", "json",
            "-show_format",
            "-show_streams",
            str(path),
        ]

        result = await self.run(cmd)
        data = json.loads(result.stdout)

        # Parse streams
        video_stream = next((s for s in data.get("streams", []) if s.get("codec_type") == "video"), None)
        audio_stream = next((s for s in data.get("streams", []) if s.get("codec_type") == "audio"), None)
        format_info = data.get("format", {})

        if not video_stream:
            raise ValueError(f"No video stream found in {path}")

        return MediaInfo(
            path=path,
            duration=float(format_info.get("duration", 0)),
            width=int(video_stream.get("width", 0)),
            height=int(video_stream.get("height", 0)),
            fps=self._parse_fps(video_stream.get("r_frame_rate", "0/1")),
            video_codec=video_stream.get("codec_name", "unknown"),
            audio_codec=audio_stream.get("codec_name") if audio_stream else None,
            audio_channels=int(audio_stream.get("channels", 0)) if audio_stream else 0,
            audio_sample_rate=int(audio_stream.get("sample_rate", 0)) if audio_stream else 0,
            bitrate=int(format_info.get("bit_rate", 0)),
            format_name=format_info.get("format_name", "unknown"),
            size_bytes=path.stat().st_size,
        )

    @staticmethod
    def _parse_fps(fps_str: str) -> float:
        """Parse frame rate string like '30000/1001' to float."""
        try:
            if "/" in fps_str:
                num, den = fps_str.split("/")
                return float(num) / float(den)
            return float(fps_str)
        except (ValueError, ZeroDivisionError):
            return 30.0

    # ============================================================
    # HIGH-LEVEL OPERATIONS
    # ============================================================

    async def transcode(
        self,
        input_path: Path,
        output_path: Path,
        codec: str = "libx264",
        preset: str = "medium",
        crf: int = 23,
        audio_codec: str = "aac",
        audio_bitrate: str = "128k",
        extra_args: list[str] | None = None,
    ) -> FFmpegResult:
        """Transcode video with common settings."""
        args = [
            "-i", str(input_path),
            "-c:v", codec,
            "-preset", preset,
            "-crf", str(crf),
            "-c:a", audio_codec,
            "-b:a", audio_bitrate,
            "-movflags", "+faststart",
        ]
        if extra_args:
            args.extend(extra_args)
        args.append(str(output_path))
        return await self.run(args, input_path=input_path, output_path=output_path)

    async def extract_audio(
        self,
        input_path: Path,
        output_path: Path,
        codec: str = "pcm_s16le",
        sample_rate: int = 16000,
        channels: int = 1,
    ) -> FFmpegResult:
        """Extract audio track for transcription."""
        args = [
            "-i", str(input_path),
            "-vn",
            "-acodec", codec,
            "-ar", str(sample_rate),
            "-ac", str(channels),
            str(output_path),
        ]
        return await self.run(args, input_path=input_path, output_path=output_path)

    async def extract_frames(
        self,
        input_path: Path,
        output_dir: Path,
        fps: float = 1.0,
        prefix: str = "frame_",
        format: str = "jpg",
        quality: int = 2,
    ) -> FFmpegResult:
        """Extract frames at specified FPS."""
        output_dir.mkdir(parents=True, exist_ok=True)
        pattern = str(output_dir / f"{prefix}%06d.{format}")

        args = [
            "-i", str(input_path),
            "-vf", f"fps={fps}",
            "-q:v", str(quality),
            pattern,
        ]
        return await self.run(args, input_path=input_path)

    async def create_video_from_frames(
        self,
        frame_dir: Path,
        output_path: Path,
        fps: float = 30.0,
        pattern: str = "frame_%06d.jpg",
        codec: str = "libx264",
        preset: str = "medium",
        crf: int = 18,
    ) -> FFmpegResult:
        """Create video from image sequence."""
        args = [
            "-framerate", str(fps),
            "-i", str(frame_dir / pattern),
            "-c:v", codec,
            "-preset", preset,
            "-crf", str(crf),
            "-pix_fmt", "yuv420p",
            str(output_path),
        ]
        return await self.run(args, output_path=output_path)

    async def concat_videos(
        self,
        input_paths: list[Path],
        output_path: Path,
        reencode: bool = False,
    ) -> FFmpegResult:
        """Concatenate multiple videos."""
        if not input_paths:
            raise ValueError("No input videos provided")

        if len(input_paths) == 1:
            # Single file, just copy or transcode
            if reencode:
                return await self.transcode(input_paths[0], output_path)
            else:
                args = ["-i", str(input_paths[0]), "-c", "copy", str(output_path)]
                return await self.run(args, input_path=input_paths[0], output_path=output_path)

        # Create concat file
        import tempfile
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            for p in input_paths:
                f.write(f"file '{p.resolve()}'\n")
            concat_file = Path(f.name)

        try:
            if reencode:
                args = [
                    "-f", "concat", "-safe", "0",
                    "-i", str(concat_file),
                    "-c:v", "libx264", "-preset", "medium", "-crf", "18",
                    "-c:a", "aac", "-b:a", "192k",
                    str(output_path),
                ]
            else:
                args = [
                    "-f", "concat", "-safe", "0",
                    "-i", str(concat_file),
                    "-c", "copy",
                    str(output_path),
                ]
            return await self.run(args, output_path=output_path)
        finally:
            concat_file.unlink(missing_ok=True)

    async def apply_lut(
        self,
        input_path: Path,
        output_path: Path,
        lut_path: Path,
        intensity: float = 1.0,
    ) -> FFmpegResult:
        """Apply 3D LUT color grade."""
        args = [
            "-i", str(input_path),
            "-vf", f"lut3d={lut_path}:interp=tetrahedral",
            "-c:v", "libx264", "-preset", "medium", "-crf", "18",
            "-c:a", "copy",
            str(output_path),
        ]
        return await self.run(args, input_path=input_path, output_path=output_path)

    async def normalize_audio(
        self,
        input_path: Path,
        output_path: Path,
        target_lufs: float = -16.0,
        true_peak: float = -1.5,
    ) -> FFmpegResult:
        """Normalize audio to target loudness (EBU R128)."""
        args = [
            "-i", str(input_path),
            "-af", f"loudnorm=I={target_lufs}:TP={true_peak}:LRA=11:print_format=json",
            "-c:v", "copy",
            str(output_path),
        ]
        return await self.run(args, input_path=input_path, output_path=output_path)

    async def get_duration(self, path: Path) -> float:
        """Get media duration in seconds."""
        info = await self.probe(path)
        return info.duration

    async def get_video_info(self, path: Path) -> dict[str, Any]:
        """Get detailed video info as dict."""
        info = await self.probe(path)
        return {
            "path": str(info.path),
            "duration": info.duration,
            "width": info.width,
            "height": info.height,
            "fps": info.fps,
            "video_codec": info.video_codec,
            "audio_codec": info.audio_codec,
            "audio_channels": info.audio_channels,
            "audio_sample_rate": info.audio_sample_rate,
            "bitrate": info.bitrate,
            "format": info.format_name,
            "size_bytes": info.size_bytes,
            "aspect_ratio": info.aspect_ratio,
            "is_vertical": info.is_vertical,
        }


# Global FFmpeg instance
_ffmpeg: FFmpeg | None = None


def get_ffmpeg() -> FFmpeg:
    """Get global FFmpeg instance."""
    global _ffmpeg
    if _ffmpeg is None:
        _ffmpeg = FFmpeg()
    return _ffmpeg


async def run_ffmpeg(
    args: list[str],
    **kwargs: Any
) -> FFmpegResult:
    """Convenience function to run FFmpeg command."""
    return await get_ffmpeg().run(args, **kwargs)


async def probe_media(path: Path) -> MediaInfo:
    """Convenience function to probe media."""
    return await get_ffmpeg().probe(path)