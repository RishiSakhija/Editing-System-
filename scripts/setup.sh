#!/usr/bin/env bash
# AutoEdit Pro — One-Command Setup Script
# Run: ./scripts/setup.sh

set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No Color

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo -e "${BOLD}${CYAN}"
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║              AutoEdit Pro — Setup Script                     ║"
echo "║           Free, Local-First AI Video Editing                 ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# Helper functions
log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[✓]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[!]${NC} $1"; }
log_error() { echo -e "${RED}[✗]${NC} $1"; }
log_step() { echo -e "\n${BOLD}${CYAN}▶ $1${NC}"; }

# Check requirements
check_command() {
    if command -v "$1" >/dev/null 2>&1; then
        log_success "$1 found: $($1 --version 2>/dev/null | head -1)"
        return 0
    else
        log_error "$1 not found"
        return 1
    fi
}

# Check Python version
check_python() {
    log_step "Checking Python"
    if command -v python3 >/dev/null 2>&1; then
        PY_VER=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
        if python3 -c 'import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)'; then
            log_success "Python $PY_VER (>= 3.11 required)"
            PYTHON=python3
        else
            log_error "Python $PY_VER found, but 3.11+ required"
            exit 1
        fi
    else
        log_error "python3 not found"
        exit 1
    fi
}

# Check uv
check_uv() {
    log_step "Checking uv (Python package manager)"
    if command -v uv >/dev/null 2>&1; then
        log_success "uv found: $(uv --version)"
    else
        log_warn "uv not found, installing..."
        curl -LsSf https://astral.sh/uv/install.sh | sh
        export PATH="$HOME/.cargo/bin:$PATH"
        log_success "uv installed"
    fi
}

# Check FFmpeg
check_ffmpeg() {
    log_step "Checking FFmpeg"
    if command -v ffmpeg >/dev/null 2>&1; then
        log_success "FFmpeg found: $(ffmpeg -version | head -1)"
    else
        log_error "FFmpeg not found!"
        echo "  Install: "
        echo "    Ubuntu/Debian: sudo apt install ffmpeg"
        echo "    macOS: brew install ffmpeg"
        echo "    Windows: winget install Gyan.FFmpeg"
        exit 1
    fi
}

# Check Ollama
check_ollama() {
    log_step "Checking Ollama"
    if command -v ollama >/dev/null 2>&1; then
        log_success "Ollama found: $(ollama --version)"
        if curl -s http://localhost:11434/api/version >/dev/null 2>&1; then
            log_success "Ollama server is running"
        else
            log_warn "Ollama server not running. Start with: ollama serve"
        fi
    else
        log_warn "Ollama not found. Install from https://ollama.com"
        echo "  After install: ollama serve"
    fi
}

# Check DaVinci Resolve
check_davinci() {
    log_step "Checking DaVinci Resolve"
    if [[ "$OSTYPE" == "darwin"* ]]; then
        DAVINCI_PATH="/Applications/DaVinci Resolve/DaVinci Resolve.app"
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        DAVINCI_PATH="/opt/resolve/bin/resolve"
    elif [[ "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" ]]; then
        DAVINCI_PATH="C:/Program Files/Blackmagic Design/DaVinci Resolve/Resolve.exe"
    fi

    if [[ -e "$DAVINCI_PATH" ]]; then
        log_success "DaVinci Resolve found at $DAVINCI_PATH"
        # Check Python API
        if [[ -d "$(dirname "$DAVINCI_PATH")/Modules/Python" ]]; then
            log_success "DaVinci Python API modules found"
        else
            log_warn "DaVinci Python API not found. Enable in Preferences > System > External scripting"
        fi
    else
        log_warn "DaVinci Resolve not found at expected path"
        echo "  Download free version from: https://www.blackmagicdesign.com/products/davinciresolve"
    fi
}

# Create directories
create_dirs() {
    log_step "Creating project directories"
    dirs=(
        "data/cache"
        "data/models"
        "data/db"
        "data/assets"
        "logs"
        "skills"
        "config"
    )
    for dir in "${dirs[@]}"; do
        mkdir -p "$dir"
        log_success "Created $dir"
    done
}

# Setup Python environment
setup_python_env() {
    log_step "Setting up Python environment with uv"
    uv sync --dev
    log_success "Dependencies installed"
}

# Download local models
download_models() {
    log_step "Downloading local models (Whisper, YOLO, Embeddings)"
    uv run python -c "
import asyncio
from autoedit.utils.hardware import ensure_models, get_settings
settings = get_settings()
asyncio.run(ensure_models(settings.hardware.models_dir))
"
    log_success "Local models downloaded"
}

# Pull Ollama models
pull_ollama_models() {
    log_step "Pulling Ollama models (qwen2.5:7b, llama3.2:3b)"
    if command -v ollama >/dev/null 2>&1; then
        uv run python -c "
import asyncio
from autoedit.utils.hardware import setup_ollama_models
asyncio.run(setup_ollama_models())
"
        log_success "Ollama models pulled"
    else
        log_warn "Ollama not available, skipping model pull"
        echo "  Run 'ollama serve' then 'autoedit models --ollama' later"
    fi
}

# Copy config templates
setup_config() {
    log_step "Setting up configuration"
    if [[ ! -f "config/secrets.yaml" ]]; then
        cp config/secrets.yaml.example config/secrets.yaml
        log_success "Created config/secrets.yaml from template"
        log_warn "Edit config/secrets.yaml to add your API keys!"
    else
        log_success "config/secrets.yaml already exists"
    fi
}

# Verify installation
verify_install() {
    log_step "Verifying installation"
    if uv run autoedit doctor --all; then
        log_success "All checks passed!"
    else
        log_warn "Some checks failed (see above). Run 'autoedit doctor' for details."
    fi
}

# Main setup flow
main() {
    echo -e "${BOLD}Starting AutoEdit Pro setup...${NC}\n"

    check_python
    check_uv
    check_ffmpeg
    check_ollama
    check_davinci

    create_dirs
    setup_python_env
    setup_config

    # Optional steps
    if [[ "${1:-}" != "--no-models" ]]; then
        download_models
    fi

    if [[ "${1:-}" != "--no-ollama" ]]; then
        pull_ollama_models
    fi

    verify_install

    echo -e "\n${BOLD}${GREEN}═══════════════════════════════════════════════════════════════${NC}"
    echo -e "${BOLD}${GREEN}  Setup Complete! 🎉${NC}"
    echo -e "${BOLD}${GREEN}═══════════════════════════════════════════════════════════════${NC}\n"

    echo -e "${CYAN}Next steps:${NC}"
    echo "  1. Edit config/secrets.yaml with your free API keys:"
    echo "     - Pexels:     https://www.pexels.com/api/"
    echo "     - Pixabay:    https://pixabay.com/api/docs/"
    echo "     - Unsplash:   https://unsplash.com/developers"
    echo "     - Freesound:  https://freesound.org/docs/api/"
    echo ""
    echo "  2. Start Ollama server (if not running):"
    echo "     ollama serve"
    echo ""
    echo "  3. Test with a video:"
    echo "     autoedit run /path/to/video.mp4 --skill educational_clean"
    echo ""
    echo "  4. Or run with review UI:"
    echo "     autoedit run /path/to/video.mp4 --skill educational_clean"
    echo "     # Then open http://localhost:8501 in browser"
    echo ""
    echo -e "${CYAN}Documentation:${NC} https://github.com/RishiSakhija/Editing-System-"
    echo -e "${CYAN}Issues:${NC} https://github.com/RishiSakhija/Editing-System-/issues"
}

# Run main
main "$@"