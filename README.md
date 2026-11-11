# AutoEdit Pro — AI-Powered Video Editing Automation

> **Free, local-first, learning editing system** — Built for talking-head educational content. Runs on RTX 2050 (4GB). Zero cost. Professional output.

## Status
🚧 **Phase 0: Foundation** — Repository initialized. Architecture designed. Ready to build.

## Vision
Automate the entire editing pipeline:
- **Transcribe & cut** silence/fillers (Whisper.cpp)
- **Smart B-roll** insertion from free APIs (Pexels, Pixabay, Unsplash)
- **Auto-captions** with styled animations (Fusion titles via DaVinci)
- **Beat-synced cuts** + music ducking (librosa + madmom)
- **Auto-zoom/punch-in** on emphasis (YOLO face detect + LLM)
- **Viral scoring** heuristics (hook, pacing, retention curve)
- **Skill system** — learns your style + loads pro creator packs (Ali Abdaal, MrBeast, etc.)

## Stack
| Layer | Technology |
|-------|------------|
| NLE | DaVinci Resolve (Free) + Python API |
| LLM | Ollama • `qwen2.5:7b-instruct-q4_K_M` |
| Transcription | Whisper.cpp `large-v3-turbo` |
| Vision | YOLOv8n-face/seg • ONNX Runtime |
| Embeddings | `bge-small-en-v1.5` (ONNX) |
| Audio | librosa • madmom (DBN beat tracker) |
| APIs | Pexels • Pixabay • Unsplash • Freesound • YT Audio Library |
| UI | Streamlit (review/approve/tweak) |
| Config | YAML + Pydantic |
| Package | uv (modern Python) |

## Hardware Target
- **GPU:** RTX 2050 (4GB VRAM) — all models quantized to fit
- **RAM:** 16GB+ recommended
- **OS:** Windows 10/11 (WSL2 optional)

## Quick Start (Once Built)
```bash
# 1. Clone & setup
git clone https://github.com/RishiSakhija/Editing-System-.git
cd Editing-System-
./scripts/setup.sh  # installs deps, downloads models, fetches skill packs

# 2. Configure
cp config/secrets.yaml.example config/secrets.yaml
# Add your free API keys (Pexels, Pixabay, Unsplash, Freesound)

# 3. Run
autoedit --input /path/to/raw_footage --skill educational_clean --review
```

## Project Structure
```
autoedit/
├── core/           # Orchestrator, pipeline, state
├── ingestion/      # Sync, script alignment, rough cut
├── analysis/       # Whisper, scene detect, face track, beat detect
├── decision/       # LLM edit decisions (structured output)
├── assets/         # Search, rank, download, cache (4 free APIs)
├── timeline/       # DaVinci builder, tracks, effects, color, export
├── skills/         # Loader, registry, discovery, embedder, few-shot
├── learning/       # Preference embeddings, skill updater
├── scoring/        # Viral heuristics, reporter
├── ui/             # Streamlit review app
└── utils/          # FFmpeg, logging, hardware helpers
```

## Skill Packs (Bundled)
- `educational_clean` — Ali Abdaal / Thomas Frank style
- `high_retention_viral` — MrBeast / Ryan Trahan style
- `documentary_cinematic` — Vox / Johnny Harris style
- `tech_review` — MKBHD / Linus style
- `shorts_reels` — Hormozi / Gary Vee style
- `podcast_clips` — JRE / Lex Clips style
- `your_signature` — **Auto-generated & evolved from your edits**

## License
MIT — Free for personal & commercial use.

---

**Next:** Phase 0 foundation code → Phase 1 ingestion/analysis → Phase 2 decision engine → ...