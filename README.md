# AutoEdit Pro

[![Version](https://img.shields.io/badge/version-0.1.0-blue.svg)](https://github.com/RishiSakhija/Editing-System-/releases)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![CI](https://github.com/RishiSakhija/Editing-System-/actions/workflows/ci.yml/badge.svg)](https://github.com/RishiSakhija/Editing-System-/actions/workflows/ci.yml)
[![Code Style](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Type Check](https://img.shields.io/badge/type%20check-mypy-blue.svg)](https://mypy-lang.org)

> **Free, local-first AI video editing automation** — Built for talking-head educational content. Runs on RTX 2050 (4GB). Zero cost. Professional output.

## 🎯 What It Does

Automate your entire editing pipeline with AI that **learns your style**:

| Feature | Technology |
|---------|------------|
| 🎬 **Transcribe & Cut** | Whisper.cpp (silence, filler words, retakes) |
| 🎨 **Smart B-roll** | Pexels, Pixabay, Unsplash APIs + LLM relevance |
| 📝 **Auto-Captions** | Fusion titles (DaVinci) with word-by-word animation |
| 🎵 **Beat-Sync & Music** | madmom/DBN beat detection + ducking |
| 🔍 **Auto-Zoom** | YOLOv8 face detect + LLM emphasis detection |
| 📊 **Viral Scoring** | Hook, pacing, retention curve heuristics |
| 🧠 **Skill Learning** | Few-shot packs + embedding-based preference learning |

## 🚀 Quick Start

```bash
# 1. Clone & setup
git clone https://github.com/RishiSakhija/Editing-System-.git
cd Editing-System-
./scripts/setup.sh  # Installs deps, downloads models, fetches skill packs

# 2. Configure API keys (free tiers)
cp config/secrets.yaml.example config/secrets.yaml
# Edit config/secrets.yaml with your keys

# 3. Start Ollama (required for LLM)
ollama serve

# 4. Run on your footage
autoedit run /path/to/raw_footage.mp4 --skill educational_clean
```

## 📦 Installation

### Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **GPU** | RTX 2050 (4GB VRAM) | RTX 3060+ (8GB+) |
| **RAM** | 16 GB | 32 GB |
| **OS** | Windows 10/11, Linux, macOS | Any |
| **Python** | 3.11+ | 3.12 |
| **FFmpeg** | 6.0+ | Latest |
| **Ollama** | 0.1+ | Latest |
| **DaVinci Resolve** | 18.6+ (Free) | 19+ |

### Free API Keys (All Generous Free Tiers)

| Service | Free Tier | Sign Up |
|---------|-----------|---------|
| **Pexels** | 200 req/hr, 20k/mo | [pexels.com/api](https://www.pexels.com/api/) |
| **Pixabay** | 100 req/min, 5k/day | [pixabay.com/api/docs](https://pixabay.com/api/docs/) |
| **Unsplash** | 50 req/hr (demo) | [unsplash.com/developers](https://unsplash.com/developers) |
| **Freesound** | 100 req/day | [freesound.org/docs/api](https://freesound.org/docs/api/) |
| **YouTube Audio** | Unlimited (manual CSV) | YouTube Studio → Audio Library |

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      ORCHESTRATOR                               │
│  (Pipeline coordination, state persistence, review loops)      │
└─────────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
┌───────────────┐    ┌───────────────┐    ┌───────────────┐
│  INGESTION    │    │  ANALYSIS     │    │  DECISION     │
│  • Sync A/V   │    │  • Whisper.cpp│    │  • LLM (Ollama)│
│  • Script     │    │  • Scene Det. │    │  • Structured  │
│    Align      │    │  • Face Track │    │    Output      │
│  • Rough Cut  │    │  • Beat Det.  │    │  • Skill Inj.  │
└───────────────┘    └───────────────┘    └───────────────┘
        │                     │                     │
        ▼                     ▼                     ▼
┌───────────────┐    ┌───────────────┐    ┌───────────────┐
│  ASSETS       │    │  TIMELINE     │    │  LEARNING     │
│  • 4 Free APIs│    │  • DaVinci API│    │  • Embeddings │
│  • Rank +     │    │  • Fusion     │    │  • Few-Shot   │
│    Cache      │    │    Titles     │    │    Update     │
└───────────────┘    └───────────────┘    └───────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      REVIEW UI (Streamlit)                      │
│  Frame-accurate preview • Asset picker • Caption editor        │
└─────────────────────────────────────────────────────────────────┘
```

## 🎓 Skill System — "Pro Editor Skills in a Pack"

Pre-built skill packs from top creators + **your personal style that evolves**:

| Pack | Style | Best For |
|------|-------|----------|
| `educational_clean` | Ali Abdaal, Thomas Frank | Tutorials, courses, explainers |
| `high_retention_viral` | MrBeast, Ryan Trahan | Entertainment, challenges |
| `documentary_cinematic` | Vox, Johnny Harris | Story-driven, B-roll heavy |
| `tech_review` | MKBHD, Linus Tech Tips | Product reviews, unboxings |
| `shorts_reels` | Hormozi, Gary Vee | Vertical, <60s, caption-first |
| `podcast_clips` | JRE, Lex Clips | Multi-cam, speaker zoom |
| **`your_signature`** | **Auto-generated from YOUR edits** | **Your unique style** |

**Skill packs are just YAML files** — shareable, version-controllable, editable.

## 💻 Usage

### CLI Commands

```bash
# Run pipeline
autoedit run video.mp4 --skill educational_clean --output ./out

# Resume interrupted run
autoedit run video.mp4 --resume abc123def456

# Hardware detection
autoedit hardware --detect

# Model management
autoedit models --download --ollama

# Skill management
autoedit skills --list
autoedit skills --discover

# Config
autoedit config --show
autoedit config --validate

# System check
autoedit doctor

# Check run status
autoedit status --list
autoedit status abc123def456
```

### Python API

```python
from autoedit import run_pipeline
from pathlib import Path

result = await run_pipeline(
    input_path=Path("footage.mp4"),
    skill_pack="educational_clean",
    script_path=Path("script.txt"),  # Optional
    output_dir=Path("./output"),
    auto_review=False,  # Set True for headless
)

if result.success:
    print(f"Outputs: {result.output_paths}")
    print(f"Viral score: {result.viral_score}")
```

## 📁 Project Structure

```
autoedit/
├── cli.py                 # CLI entry point
├── config/                # Pydantic config system
│   ├── models.py          # All settings models
│   └── loader.py          # YAML + ENV loading
├── core/                  # Pipeline orchestration
│   ├── orchestrator.py    # Main workflow
│   ├── pipeline.py        # Stage execution
│   ├── state.py           # SQLite persistence
│   └── events.py          # Event bus
├── ingestion/             # Input processing
├── analysis/              # Whisper, YOLO, beat detect
├── decision/              # LLM edit decisions
├── assets/                # API providers, cache, ranker
├── timeline/              # DaVinci builder, export
├── skills/                # Skill packs, discovery, LoRA
├── learning/              # Preference embeddings
├── scoring/               # Viral heuristics
├── ui/                    # Streamlit review app
└── utils/                 # FFmpeg, hardware, logging
```

## 🔧 Configuration

All settings in `config/default.yaml` (validated by Pydantic):

```yaml
hardware:
  gpu:
    enabled: true
    memory_fraction: 0.85
    offload_layers: 20  # For 4GB VRAM
models:
  llm:
    model: "qwen2.5:7b-instruct-q4_K_M"
  whisper:
    model: "large-v3-turbo"
pipeline:
  decision:
    few_shot_k: 8
  learning:
    enabled: true
apis:
  pexels:
    enabled: true
skills:
  default_pack: "educational_clean"
```

Override via environment:
```bash
AUTOEDIT__HARDWARE__GPU__MEMORY_FRACTION=0.9
AUTOEDIT__MODELS__LLM__TEMPERATURE=0.3
```

## 🤝 Contributing

1. Fork the repo
2. Create feature branch: `git checkout -b feat/amazing-feature`
3. Install dev deps: `uv sync --dev`
4. Run checks: `uv run ruff check . && uv run mypy autoedit && uv run pytest`
5. Commit with conventional commits: `git commit -m "feat: add amazing feature"`
6. Push and open PR

## 📄 License

MIT License — Free for personal and commercial use.

## 🙏 Acknowledgments

- **Whisper.cpp** — Georgi Gerganov
- **Ollama** — Ollama team
- **Ultralytics YOLO** — Glenn Jocher
- **DaVinci Resolve** — Blackmagic Design (free tier!)
- **Pexels, Pixabay, Unsplash, Freesound** — Free asset APIs
- **madmom** — Beat tracking
- **All open-source creators** whose editing patterns inspired the skill packs

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/RishiSakhija/Editing-System-/issues)
- **Discussions**: [GitHub Discussions](https://github.com/RishiSakhija/Editing-System-/discussions)
- **Documentation**: [Wiki](https://github.com/RishiSakhija/Editing-System-/wiki)

---

<div align="center">
  <strong>Made with ❤️ for content creators who want to edit less and create more.</strong>
  <br>
  <sub>Star ⭐ the repo if you find it useful!</sub>
</div>