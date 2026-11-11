"""
AutoEdit Pro — Configuration Loader
Handles loading, merging, and validation of config from YAML + ENV + secrets.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from .models import Settings, get_settings as _get_settings


def load_settings(config_path: str | Path = "config/default.yaml") -> Settings:
    """
    Load settings from YAML config file with environment variable overrides.

    Priority (highest to lowest):
    1. Environment variables (AUTOEDIT__SECTION__KEY)
    2. secrets.yaml (for API keys)
    3. default.yaml (base config)
    4. Pydantic defaults
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, encoding="utf-8") as f:
        yaml_data = yaml.safe_load(f) or {}

    # Load secrets and merge into yaml_data
    secrets_path = Path("config/secrets.yaml")
    if secrets_path.exists():
        with open(secrets_path, encoding="utf-8") as f:
            secrets = yaml.safe_load(f) or {}
        yaml_data = _deep_merge_dicts(yaml_data, secrets)

    # Apply environment variable overrides
    yaml_data = _apply_env_overrides(yaml_data)

    try:
        settings = Settings(**yaml_data)
    except ValidationError as e:
        raise RuntimeError(f"Configuration validation failed:\n{e}")

    # Apply hardware profile if specified
    hw_profile = os.getenv("AUTOEDIT_HARDWARE_PROFILE")
    if hw_profile:
        settings.apply_hardware_profile(hw_profile)

    return settings


def _deep_merge_dicts(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Deep merge override into base, returning new dict."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge_dicts(result[key], value)
        else:
            result[key] = value
    return result


def _apply_env_overrides(data: dict[str, Any]) -> dict[str, Any]:
    """Apply environment variable overrides with AUTOEDIT__ prefix."""
    prefix = "AUTOEDIT__"
    for env_key, env_value in os.environ.items():
        if not env_key.startswith(prefix):
            continue

        # Parse nested key: AUTOEDIT__SECTION__SUBSECTION__KEY
        key_path = env_key[len(prefix):].lower().split("__")
        if not key_path:
            continue

        # Navigate/create nested dict
        current = data
        for part in key_path[:-1]:
            if part not in current:
                current[part] = {}
            elif not isinstance(current[part], dict):
                current[part] = {}
            current = current[part]

        # Set value with type inference
        current[key_path[-1]] = _parse_env_value(env_value)

    return data


def _parse_env_value(value: str) -> Any:
    """Parse environment variable string to appropriate Python type."""
    # Boolean
    if value.lower() in ("true", "false"):
        return value.lower() == "true"
    # None
    if value.lower() in ("null", "none", ""):
        return None
    # Integer
    try:
        return int(value)
    except ValueError:
        pass
    # Float
    try:
        return float(value)
    except ValueError:
        pass
    # List (comma-separated)
    if "," in value:
        return [v.strip() for v in value.split(",")]
    # String (default)
    return value


def get_settings() -> Settings:
    """Get global settings instance."""
    return _get_settings()


def reload_settings(config_path: str | Path = "config/default.yaml") -> Settings:
    """Reload settings from config file."""
    from .models import reload_settings as _reload
    return _reload(config_path)


def validate_config(config_path: str | Path = "config/default.yaml") -> tuple[bool, list[str]]:
    """
    Validate config file without loading globals.
    Returns (is_valid, list_of_errors).
    """
    try:
        load_settings(config_path)
        return True, []
    except Exception as e:
        return False, [str(e)]


def print_config_summary(settings: Settings | None = None) -> None:
    """Print a human-readable config summary."""
    from rich.console import Console
    from rich.table import Table
    from rich.tree import Tree

    settings = settings or get_settings()
    console = Console()

    tree = Tree("🔧 [bold cyan]AutoEdit Pro Configuration[/bold cyan]")

    # Hardware
    hw = tree.add("🖥️  Hardware")
    hw.add(f"GPU: {'Enabled' if settings.hardware.gpu.enabled else 'Disabled'}")
    if settings.hardware.gpu.enabled:
        hw.add(f"  Device: {settings.hardware.gpu.device_id}")
        hw.add(f"  Memory Fraction: {settings.hardware.gpu.memory_fraction:.0%}")
        hw.add(f"  Offload Layers: {settings.hardware.gpu.offload_layers}")
    hw.add(f"CPU Threads: {settings.hardware.cpu_threads}")
    hw.add(f"Quantization: {settings.hardware.quantization}")

    # Models
    models = tree.add("🤖 Models")
    models.add(f"LLM: {settings.models.llm.model}")
    models.add(f"  Host: {settings.models.llm.host}")
    models.add(f"  Temp: {settings.models.llm.temperature}")
    models.add(f"Whisper: {settings.models.whisper.model}")
    models.add(f"Vision: {settings.models.vision.face_model}")
    models.add(f"Embeddings: {settings.models.embeddings.model}")

    # Pipeline
    pipe = tree.add("⚙️  Pipeline")
    pipe.add(f"Scene Detection: {'On' if settings.pipeline.analysis.scene_detection.enabled else 'Off'}")
    pipe.add(f"Face Tracking: {'On' if settings.pipeline.analysis.face_tracking.enabled else 'Off'}")
    pipe.add(f"Beat Detection: {'On' if settings.pipeline.analysis.beat_detection.enabled else 'Off'}")
    pipe.add(f"Skill Injection: {'On' if settings.pipeline.decision.skill_injection else 'Off'}")
    pipe.add(f"Learning: {'On' if settings.pipeline.learning.enabled else 'Off'}")
    pipe.add(f"Viral Scoring: {'On' if settings.pipeline.scoring.enabled else 'Off'}")

    # APIs
    apis = tree.add("🌐 APIs")
    for name, cfg in [
        ("Pexels", settings.apis.pexels),
        ("Pixabay", settings.apis.pixabay),
        ("Unsplash", settings.apis.unsplash),
        ("Freesound", settings.apis.freesound),
        ("YT Audio", settings.apis.youtube_audio_library),
    ]:
        status = "✅" if cfg.enabled else "❌"
        apis.add(f"{status} {name}")

    # Skills
    skills = tree.add("🎓 Skills")
    skills.add(f"Default Pack: {settings.skills.default_pack}")
    skills.add(f"Bundled Packs: {', '.join(settings.skills.bundled_packs)}")
    skills.add(f"Discovery: {'On' if settings.skills.discovery.enabled else 'Off'}")

    # Export
    exp = tree.add("📤 Export")
    exp.add(f"Formats: {', '.join(settings.export.formats)}")
    exp.add(f"DaVinci Preset: {settings.export.davinci.render_preset}")
    exp.add(f"MP4 Codec: {settings.export.mp4.codec}")

    console.print(tree)


if __name__ == "__main__":
    # Test config loading
    try:
        s = load_settings()
        print_config_summary(s)
        print("\n✅ Configuration loaded successfully!")
    except Exception as e:
        print(f"\n❌ Configuration error: {e}")
        import traceback
        traceback.print_exc()