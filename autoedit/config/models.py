"""
AutoEdit Pro — Pydantic Configuration Models
All settings with validation, defaults, and environment variable support.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


# ============================================================
# HARDWARE & PERFORMANCE
# ============================================================

class GPUConfig(BaseModel):
    """GPU configuration for model inference and rendering."""
    enabled: bool = True
    device_id: int = Field(default=0, ge=0)
    memory_fraction: float = Field(default=0.85, ge=0.1, le=1.0)
    offload_layers: int = Field(default=20, ge=0)
    flash_attention: bool = True
    tensor_split: str | None = None

    @field_validator("tensor_split", mode="before")
    @classmethod
    def parse_tensor_split(cls, v: str | None) -> str | None:
        if v is None or v == "":
            return None
        return v


class HardwareConfig(BaseModel):
    """Hardware and performance settings."""
    gpu: GPUConfig = Field(default_factory=GPUConfig)
    cpu_threads: int = Field(default=8, ge=1, le=64)
    quantization: Literal["q2_K", "q3_K_M", "q4_K_M", "q5_K_M", "q6_K", "q8_0"] = "q4_K_M"
    whisper_precision: Literal["int8", "float16", "float32"] = "int8"
    cache_dir: Path = Field(default=Path("./data/cache"))
    models_dir: Path = Field(default=Path("./data/models"))
    db_dir: Path = Field(default=Path("./data/db"))

    @field_validator("cache_dir", "models_dir", "db_dir", mode="before")
    @classmethod
    def expand_path(cls, v: str | Path) -> Path:
        return Path(v).expanduser().resolve()


# ============================================================
# MODEL CONFIGURATION
# ============================================================

class LLMConfig(BaseModel):
    """LLM (Ollama) configuration for edit decisions."""
    provider: Literal["ollama"] = "ollama"
    host: str = "http://localhost:11434"
    model: str = "qwen2.5:7b-instruct-q4_K_M"
    fallback_models: list[str] = Field(default_factory=lambda: [
        "llama3.2:3b-instruct-q4_K_M",
        "qwen2.5:3b-instruct-q4_K_M",
    ])
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    max_tokens: int = Field(default=4096, ge=128, le=32768)
    timeout: int = Field(default=120, ge=10, le=600)
    structured_output: bool = True
    context_window: int = Field(default=8192, ge=2048, le=128000)
    num_ctx: int = Field(default=8192, ge=2048, le=128000)
    num_batch: int = Field(default=512, ge=1, le=4096)
    num_gpu_layers: int = Field(default=-1, ge=-1)


class WhisperConfig(BaseModel):
    """Whisper.cpp transcription configuration."""
    model: Literal[
        "tiny", "tiny.en", "base", "base.en", "small", "small.en",
        "medium", "medium.en", "large-v1", "large-v2", "large-v3",
        "large-v3-turbo"
    ] = "large-v3-turbo"
    language: str = "en"
    translate: bool = False
    vad: bool = True
    vad_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    min_silence_duration: float = Field(default=0.3, ge=0.05, le=5.0)
    word_timestamps: bool = True
    hallucination_filter: bool = True
    temperature: float = Field(default=0.0, ge=0.0, le=1.0)
    best_of: int = Field(default=1, ge=1, le=10)
    beam_size: int = Field(default=5, ge=1, le=10)


class VisionConfig(BaseModel):
    """YOLO face/object detection configuration."""
    face_model: str = "yolov8n-face"
    seg_model: str = "yolov8n-seg"
    conf_threshold: float = Field(default=0.45, ge=0.0, le=1.0)
    iou_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    max_det: int = Field(default=10, ge=1, le=100)
    tracker: Literal["bytetrack", "botsort", "ocsort"] = "bytetrack"
    tracker_config: str = "config/trackers/bytetrack.yaml"
    half_precision: bool = True


class EmbeddingConfig(BaseModel):
    """Embedding model configuration for skills/preferences."""
    model: str = "bge-small-en-v1.5"
    provider: Literal["onnx", "sentence-transformers"] = "onnx"
    dimension: int = Field(default=384, ge=64, le=4096)
    normalize: bool = True
    batch_size: int = Field(default=32, ge=1, le=256)


class ModelConfig(BaseModel):
    """All model configurations."""
    llm: LLMConfig = Field(default_factory=LLMConfig)
    whisper: WhisperConfig = Field(default_factory=WhisperConfig)
    vision: VisionConfig = Field(default_factory=VisionConfig)
    embeddings: EmbeddingConfig = Field(default_factory=EmbeddingConfig)


# ============================================================
# PIPELINE STAGES
# ============================================================

class IngestionConfig(BaseModel):
    """Input ingestion settings."""
    sync_audio_video: bool = True
    script_alignment: bool = True
    rough_cut_from_script: bool = True
    supported_video_ext: list[str] = Field(default_factory=lambda: [
        ".mp4", ".mov", ".mkv", ".webm", ".avi"
    ])
    supported_audio_ext: list[str] = Field(default_factory=lambda: [
        ".wav", ".mp3", ".m4a", ".aac", ".flac"
    ])
    supported_image_ext: list[str] = Field(default_factory=lambda: [
        ".jpg", ".jpeg", ".png", ".webp", ".heic"
    ])


class SceneDetectionConfig(BaseModel):
    """Scene detection settings."""
    enabled: bool = True
    threshold: float = Field(default=27.0, ge=1.0, le=100.0)
    min_scene_len: float = Field(default=0.5, ge=0.1, le=10.0)


class FaceTrackingConfig(BaseModel):
    """Face tracking settings."""
    enabled: bool = True
    min_face_size: int = Field(default=50, ge=10, le=500)
    smooth_factor: float = Field(default=0.3, ge=0.0, le=1.0)


class BeatDetectionConfig(BaseModel):
    """Beat detection settings."""
    enabled: bool = True
    method: Literal["madmom", "librosa"] = "madmom"
    fps: int = Field(default=100, ge=10, le=500)


class AnalysisConfig(BaseModel):
    """Analysis pipeline settings."""
    scene_detection: SceneDetectionConfig = Field(default_factory=SceneDetectionConfig)
    face_tracking: FaceTrackingConfig = Field(default_factory=FaceTrackingConfig)
    beat_detection: BeatDetectionConfig = Field(default_factory=BeatDetectionConfig)


class DecisionConfig(BaseModel):
    """Edit decision engine settings."""
    max_decisions_per_segment: int = Field(default=5, ge=1, le=20)
    min_confidence: float = Field(default=0.6, ge=0.0, le=1.0)
    few_shot_k: int = Field(default=8, ge=1, le=20)
    few_shot_diversity: float = Field(default=0.7, ge=0.0, le=1.0)
    skill_injection: bool = True
    user_prefs_injection: bool = True

    class ValidationConfig(BaseModel):
        no_overlapping_cuts: bool = True
        max_zoom_intensity: float = Field(default=2.0, ge=1.0, le=5.0)
        max_broll_duration: float = Field(default=15.0, ge=1.0, le=60.0)
        min_cut_interval: float = Field(default=0.5, ge=0.1, le=5.0)

    validation: ValidationConfig = Field(default_factory=ValidationConfig)


class AssetConfig(BaseModel):
    """Asset resolution settings."""
    max_results_per_query: int = Field(default=20, ge=5, le=100)
    cache_ttl_days: int = Field(default=30, ge=1, le=365)
    download_timeout: int = Field(default=60, ge=10, le=300)
    max_concurrent_downloads: int = Field(default=4, ge=1, le=16)
    prefer_vertical_for_shorts: bool = True
    min_broll_resolution: Literal["480p", "720p", "1080p", "4K"] = "720p"
    max_broll_duration: float = Field(default=10.0, ge=1.0, le=60.0)


class TrackLayoutConfig(BaseModel):
    """DaVinci timeline track layout."""
    video_main: int = 1
    video_broll: int = 2
    video_graphics: int = 3
    audio_voice: int = 1
    audio_music: int = 2
    audio_sfx: int = 3


class TimelineConfig(BaseModel):
    """Timeline building and rendering settings."""
    target_fps: int = Field(default=30, ge=24, le=120)
    target_resolution: str = "1920x1080"
    track_layout: TrackLayoutConfig = Field(default_factory=TrackLayoutConfig)
    color_space: Literal["rec709", "rec2020", "aces"] = "rec709"
    render_codec: Literal["h264_nvenc", "hevc_nvenc", "libx264", "h264_videotoolbox"] = "h264_nvenc"
    render_preset: str = "p6"
    render_crf: int = Field(default=18, ge=10, le=30)
    render_bitrate: str = "25M"


class LearningConfig(BaseModel):
    """Learning/preference update settings."""
    enabled: bool = True
    embedding_update_threshold: float = Field(default=0.15, ge=0.0, le=1.0)
    few_shot_max_examples: int = Field(default=20, ge=5, le=100)
    preference_decay: float = Field(default=0.95, ge=0.0, le=1.0)
    skill_update_on_every_video: bool = True


class ScoringWeightsConfig(BaseModel):
    """Viral scoring weights (must sum to ~1.0)."""
    hook_strength: float = Field(default=0.30, ge=0.0, le=1.0)
    pacing_health: float = Field(default=0.20, ge=0.0, le=1.0)
    retention_curve: float = Field(default=0.25, ge=0.0, le=1.0)
    platform_compliance: float = Field(default=0.15, ge=0.0, le=1.0)
    audio_quality: float = Field(default=0.10, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def check_weights_sum(self) -> ScoringWeightsConfig:
        total = (
            self.hook_strength + self.pacing_health + self.retention_curve +
            self.platform_compliance + self.audio_quality
        )
        if not 0.95 <= total <= 1.05:
            raise ValueError(f"Scoring weights must sum to ~1.0, got {total:.2f}")
        return self


class ScoringConfig(BaseModel):
    """Viral scoring settings."""
    enabled: bool = True
    platform: Literal["youtube", "tiktok", "instagram", "shorts"] = "youtube"
    weights: ScoringWeightsConfig = Field(default_factory=ScoringWeightsConfig)


class PipelineConfig(BaseModel):
    """All pipeline stage configurations."""
    ingestion: IngestionConfig = Field(default_factory=IngestionConfig)
    analysis: AnalysisConfig = Field(default_factory=AnalysisConfig)
    decision: DecisionConfig = Field(default_factory=DecisionConfig)
    assets: AssetConfig = Field(default_factory=AssetConfig)
    timeline: TimelineConfig = Field(default_factory=TimelineConfig)
    learning: LearningConfig = Field(default_factory=LearningConfig)
    scoring: ScoringConfig = Field(default_factory=ScoringConfig)


# ============================================================
# API PROVIDERS
# ============================================================

class PexelsConfig(BaseModel):
    enabled: bool = True
    base_url: str = "https://api.pexels.com/v1"
    rate_limit: int = Field(default=200, ge=1)
    video_min_duration: int = Field(default=5, ge=1)
    video_max_duration: int = Field(default=30, ge=5)
    photo_min_size: str = "1920x1080"


class PixabayConfig(BaseModel):
    enabled: bool = True
    base_url: str = "https://pixabay.com/api"
    rate_limit: int = Field(default=100, ge=1)
    video_min_duration: int = Field(default=5, ge=1)
    safesearch: bool = True


class UnsplashConfig(BaseModel):
    enabled: bool = True
    base_url: str = "https://api.unsplash.com"
    rate_limit: int = Field(default=50, ge=1)
    min_width: int = Field(default=1920, ge=640)


class FreesoundConfig(BaseModel):
    enabled: bool = True
    base_url: str = "https://freesound.org/apiv2"
    rate_limit: int = Field(default=100, ge=1)
    min_duration: float = Field(default=0.5, ge=0.1)
    max_duration: float = Field(default=30.0, ge=1.0)


class YouTubeAudioConfig(BaseModel):
    enabled: bool = True
    csv_path: Path = Field(default=Path("./data/assets/youtube_audio_library.csv"))

    @field_validator("csv_path", mode="before")
    @classmethod
    def expand_path(cls, v: str | Path) -> Path:
        return Path(v).expanduser().resolve()


class APIConfig(BaseModel):
    """Free API provider configurations."""
    pexels: PexelsConfig = Field(default_factory=PexelsConfig)
    pixabay: PixabayConfig = Field(default_factory=PixabayConfig)
    unsplash: UnsplashConfig = Field(default_factory=UnsplashConfig)
    freesound: FreesoundConfig = Field(default_factory=FreesoundConfig)
    youtube_audio_library: YouTubeAudioConfig = Field(default_factory=YouTubeAudioConfig)


# ============================================================
# SKILL SYSTEM
# ============================================================

class SkillDiscoveryConfig(BaseModel):
    """Skill discovery from internet sources."""
    enabled: bool = True
    schedule: Literal["daily", "weekly", "monthly", "manual"] = "weekly"
    sources: dict[str, bool] = Field(default_factory=lambda: {
        "github": True,
        "huggingface": True,
        "reddit": True,
        "youtube": False,
    })
    max_new_packs_per_run: int = Field(default=3, ge=1, le=10)


class SkillConfig(BaseModel):
    """Skill system configuration."""
    directory: Path = Field(default=Path("./skills"))
    auto_discover: bool = True
    bundled_packs: list[str] = Field(default_factory=lambda: [
        "educational_clean",
        "high_retention_viral",
        "documentary_cinematic",
        "tech_review",
        "shorts_reels",
        "podcast_clips",
    ])
    default_pack: str = "educational_clean"
    discovery: SkillDiscoveryConfig = Field(default_factory=SkillDiscoveryConfig)

    @field_validator("directory", mode="before")
    @classmethod
    def expand_path(cls, v: str | Path) -> Path:
        return Path(v).expanduser().resolve()


# ============================================================
# REVIEW UI
# ============================================================

class UIDefaultTheme(BaseModel):
    """Streamlit theme configuration."""
    base: Literal["light", "dark"] = "dark"
    primary_color: str = "#00D4AA"
    background_color: str = "#0E1117"
    secondary_background_color: str = "#262730"
    text_color: str = "#FAFAFA"


class UIPreviewConfig(BaseModel):
    """Preview settings."""
    enable_davinci_live: bool = True
    frame_accurate: bool = True
    thumbnail_size: list[int] = Field(default_factory=lambda: [320, 180])


class AssetPickerConfig(BaseModel):
    """Asset picker settings."""
    results_per_page: int = Field(default=12, ge=6, le=48)
    preview_on_hover: bool = True


class UIConfig(BaseModel):
    """Review UI (Streamlit) configuration."""
    host: str = "localhost"
    port: int = Field(default=8501, ge=1024, le=65535)
    theme: UIDefaultTheme = Field(default_factory=UIDefaultTheme)
    preview: UIPreviewConfig = Field(default_factory=UIPreviewConfig)
    asset_picker: AssetPickerConfig = Field(default_factory=AssetPickerConfig)


# ============================================================
# EXPORT
# ============================================================

class DavinciExportConfig(BaseModel):
    """DaVinci Resolve export settings."""
    project_name_template: str = "{video_name}_autoedit_{timestamp}"
    timeline_name_template: str = "Main"
    render_preset: str = "YouTube 2160p"


class MP4ExportConfig(BaseModel):
    """Direct MP4 export settings."""
    codec: Literal["h264_nvenc", "hevc_nvenc", "libx264", "h264_videotoolbox"] = "h264_nvenc"
    preset: str = "p6"
    crf: int = Field(default=18, ge=10, le=30)
    audio_codec: str = "aac"
    audio_bitrate: str = "192k"


class ExportConfig(BaseModel):
    """Export format settings."""
    formats: list[Literal["davinci", "fcp_xml", "mp4"]] = Field(default_factory=lambda: [
        "davinci", "fcp_xml", "mp4"
    ])
    davinci: DavinciExportConfig = Field(default_factory=DavinciExportConfig)
    mp4: MP4ExportConfig = Field(default_factory=MP4ExportConfig)


# ============================================================
# LOGGING
# ============================================================

class LoggingConfig(BaseModel):
    """Logging configuration."""
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    format: Literal["json", "console"] = "json"
    file: Path = Field(default=Path("./logs/autoedit.log"))
    max_bytes: int = Field(default=10_485_760, ge=1_048_576)  # 10MB
    backup_count: int = Field(default=5, ge=1, le=20)
    structured: bool = True
    rich_console: bool = True

    @field_validator("file", mode="before")
    @classmethod
    def expand_path(cls, v: str | Path) -> Path:
        return Path(v).expanduser().resolve()


# ============================================================
# MAIN SETTINGS
# ============================================================

class Settings(BaseSettings):
    """Main settings class — loads from YAML + ENV."""
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        extra="ignore",
        case_sensitive=False,
    )

    hardware: HardwareConfig = Field(default_factory=HardwareConfig)
    models: ModelConfig = Field(default_factory=ModelConfig)
    pipeline: PipelineConfig = Field(default_factory=PipelineConfig)
    apis: APIConfig = Field(default_factory=APIConfig)
    skills: SkillConfig = Field(default_factory=SkillConfig)
    ui: UIConfig = Field(default_factory=UIConfig)
    export: ExportConfig = Field(default_factory=ExportConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    # Runtime fields (not from config)
    hardware_profile: str | None = None
    secrets: dict[str, Any] = Field(default_factory=dict, exclude=True)

    def model_post_init(self, __context: Any) -> None:
        # Ensure directories exist
        for path in [
            self.hardware.cache_dir,
            self.hardware.models_dir,
            self.hardware.db_dir,
            self.skills.directory,
            self.logging.file.parent,
        ]:
            path.mkdir(parents=True, exist_ok=True)

        # Load secrets if available
        self._load_secrets()

    def _load_secrets(self) -> None:
        """Load secrets from config/secrets.yaml if exists."""
        import yaml
        secrets_path = Path("config/secrets.yaml")
        if secrets_path.exists():
            with open(secrets_path) as f:
                self.secrets = yaml.safe_load(f) or {}

    def get_secret(self, *keys: str, default: Any = None) -> Any:
        """Get nested secret value: get_secret('pexels', 'api_key')."""
        current = self.secrets
        for key in keys:
            if not isinstance(current, dict):
                return default
            current = current.get(key)
            if current is None:
                return default
        return current

    def apply_hardware_profile(self, profile_name: str) -> None:
        """Apply a hardware profile from hardware.yaml."""
        import yaml
        hw_path = Path("config/hardware.yaml")
        if not hw_path.exists():
            return
        with open(hw_path) as f:
            data = yaml.safe_load(f)
        profile = data.get("profiles", {}).get(profile_name)
        if not profile:
            raise ValueError(f"Hardware profile '{profile_name}' not found")
        # Deep merge profile into current settings
        self._deep_merge(profile)
        self.hardware_profile = profile_name

    def _deep_merge(self, updates: dict[str, Any], target: dict | None = None) -> None:
        """Deep merge dict into settings."""
        if target is None:
            target = self.model_dump()
        for key, value in updates.items():
            if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                self._deep_merge(value, target[key])
            else:
                target[key] = value
        # Re-validate by creating new instance
        updated = type(self)(**target)
        # Copy updated fields back
        for field_name in self.model_fields:
            if hasattr(updated, field_name):
                setattr(self, field_name, getattr(updated, field_name))


# Global settings instance
_settings: Settings | None = None


def load_settings(config_path: str | Path = "config/default.yaml") -> Settings:
    """Load settings from YAML file."""
    global _settings
    import yaml
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    _settings = Settings(**data)
    return _settings


def get_settings() -> Settings:
    """Get global settings instance (loads default if not loaded)."""
    global _settings
    if _settings is None:
        _settings = load_settings()
    return _settings


def reload_settings(config_path: str | Path = "config/default.yaml") -> Settings:
    """Reload settings from config file."""
    global _settings
    _settings = None
    return load_settings(config_path)