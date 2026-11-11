"""
AutoEdit Pro — Hardware Detection & Model Management
Detects GPU, VRAM, CPU; downloads and manages local models.
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from autoedit.config import get_settings
from autoedit.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class GPUInfo:
    """GPU information."""
    name: str
    vendor: str  # nvidia, amd, intel, apple
    vram_total_mb: int
    vram_free_mb: int
    driver_version: str | None = None
    cuda_version: str | None = None
    compute_capability: str | None = None
    device_id: int = 0


@dataclass
class CPUInfo:
    """CPU information."""
    name: str
    cores_physical: int
    cores_logical: int
    frequency_mhz: int
    architecture: str


@dataclass
class SystemInfo:
    """Complete system hardware information."""
    gpu: GPUInfo | None
    cpu: CPUInfo
    ram_total_gb: float
    ram_free_gb: float
    platform: str
    python_version: str
    ollama_version: str | None = None
    ffmpeg_version: str | None = None

    @property
    def has_gpu(self) -> bool:
        return self.gpu is not None

    @property
    def recommended_profile(self) -> str:
        """Recommend hardware profile based on detected specs."""
        if not self.has_gpu:
            return "cpu_only"

        vram_gb = self.gpu.vram_total_mb / 1024
        vendor = self.gpu.vendor.lower()

        if vendor == "apple":
            return "apple_silicon"

        if vendor == "nvidia":
            if vram_gb >= 20:
                return "rtx_4090_24gb"
            elif vram_gb >= 10:
                return "rtx_3060_12gb"
            else:
                return "rtx_2050_4gb"

        if vendor in ("amd", "intel"):
            if vram_gb >= 16:
                return "rtx_4090_24gb"  # Closest high-VRAM profile
            elif vram_gb >= 8:
                return "rtx_3060_12gb"
            else:
                return "rtx_2050_4gb"

        return "cpu_only"


async def detect_hardware() -> SystemInfo:
    """
    Detect system hardware (GPU, CPU, RAM).
    Returns SystemInfo with all detected specs.
    """
    logger.info("hardware_detect_start")

    # CPU & RAM
    cpu_info = _detect_cpu()
    ram_total, ram_free = _detect_ram()

    # GPU
    gpu_info = await _detect_gpu()

    # Platform
    sys_platform = platform.system().lower()
    py_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"

    # Tool versions
    ollama_ver = await _get_ollama_version()
    ffmpeg_ver = await _get_ffmpeg_version()

    info = SystemInfo(
        gpu=gpu_info,
        cpu=cpu_info,
        ram_total_gb=ram_total,
        ram_free_gb=ram_free,
        platform=sys_platform,
        python_version=py_version,
        ollama_version=ollama_ver,
        ffmpeg_version=ffmpeg_ver,
    )

    logger.info(
        "hardware_detected",
        gpu=info.gpu.name if info.gpu else "None",
        gpu_vram_gb=round(info.gpu.vram_total_mb / 1024, 1) if info.gpu else 0,
        cpu=info.cpu.name,
        cpu_cores=info.cpu.cores_logical,
        ram_gb=round(info.ram_total_gb, 1),
        profile=info.recommended_profile,
    )

    return info


def _detect_cpu() -> CPUInfo:
    """Detect CPU information."""
    try:
        import psutil
        cores_physical = psutil.cpu_count(logical=False) or 1
        cores_logical = psutil.cpu_count(logical=True) or 1
        freq = psutil.cpu_freq()
        frequency_mhz = int(freq.current) if freq else 0
    except ImportError:
        cores_physical = 1
        cores_logical = 1
        frequency_mhz = 0

    # CPU name
    name = "Unknown"
    if sys.platform == "win32":
        try:
            import wmi
            c = wmi.WMI()
            for processor in c.Win32_Processor():
                name = processor.Name.strip()
                break
        except Exception:
            name = platform.processor()
    elif sys.platform == "darwin":
        try:
            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True, text=True, timeout=5
            )
            name = result.stdout.strip()
        except Exception:
            name = platform.processor()
    else:  # Linux
        try:
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.startswith("model name"):
                        name = line.split(":")[1].strip()
                        break
        except Exception:
            name = platform.processor()

    return CPUInfo(
        name=name,
        cores_physical=cores_physical,
        cores_logical=cores_logical,
        frequency_mhz=frequency_mhz,
        architecture=platform.machine(),
    )


def _detect_ram() -> tuple[float, float]:
    """Detect total and free RAM in GB."""
    try:
        import psutil
        mem = psutil.virtual_memory()
        return mem.total / (1024**3), mem.available / (1024**3)
    except ImportError:
        # Fallback
        if sys.platform == "win32":
            try:
                import wmi
                c = wmi.WMI()
                for mem in c.Win32_ComputerSystem():
                    total = int(mem.TotalPhysicalMemory) / (1024**3)
                    return total, total * 0.5  # Rough estimate
            except Exception:
                pass
        return 16.0, 8.0  # Default assumption


async def _detect_gpu() -> GPUInfo | None:
    """Detect GPU information."""
    # Try NVIDIA first
    nvidia_gpu = await _detect_nvidia_gpu()
    if nvidia_gpu:
        return nvidia_gpu

    # Try AMD
    amd_gpu = await _detect_amd_gpu()
    if amd_gpu:
        return amd_gpu

    # Try Intel
    intel_gpu = await _detect_intel_gpu()
    if intel_gpu:
        return intel_gpu

    # Try Apple Silicon
    if sys.platform == "darwin" and platform.machine() == "arm64":
        return _detect_apple_gpu()

    return None


async def _detect_nvidia_gpu() -> GPUInfo | None:
    """Detect NVIDIA GPU via nvidia-smi."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.free,driver_version",
                "--format=csv,noheader,nounits"
            ],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode != 0:
            return None

        lines = result.stdout.strip().split("\n")
        if not lines:
            return None

        # Use first GPU
        name, total_mb, free_mb, driver = lines[0].split(", ")
        total_mb = int(total_mb)
        free_mb = int(free_mb)

        # Get CUDA version
        cuda_ver = None
        try:
            cuda_result = subprocess.run(
                ["nvcc", "--version"],
                capture_output=True, text=True, timeout=5
            )
            for line in cuda_result.stdout.split("\n"):
                if "release" in line:
                    cuda_ver = line.split("release")[1].split(",")[0].strip()
                    break
        except Exception:
            pass

        # Get compute capability (approximate from name)
        compute_cap = _guess_nvidia_compute_capability(name)

        return GPUInfo(
            name=name.strip(),
            vendor="nvidia",
            vram_total_mb=total_mb,
            vram_free_mb=free_mb,
            driver_version=driver.strip(),
            cuda_version=cuda_ver,
            compute_capability=compute_cap,
            device_id=0,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, Exception):
        return None


def _guess_nvidia_compute_capability(name: str) -> str:
    """Guess compute capability from GPU name."""
    name_lower = name.lower()
    if any(x in name_lower for x in ["rtx 40", "rtx 30"]):
        return "8.9"  # Ampere/Lovelace
    elif any(x in name_lower for x in ["rtx 20", "gtx 16"]):
        return "7.5"  # Turing
    elif "gtx 10" in name_lower:
        return "6.1"  # Pascal
    return "unknown"


async def _detect_amd_gpu() -> GPUInfo | None:
    """Detect AMD GPU via rocm-smi or lspci."""
    try:
        # Try rocm-smi first
        result = subprocess.run(
            ["rocm-smi", "--showproductname", "--showmeminfo", "vram", "--json"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            data = json.loads(result.stdout)
            # Parse rocm-smi output
            # (Implementation depends on ROCm version)
            pass
    except Exception:
        pass

    # Fallback: lspci
    try:
        result = subprocess.run(
            ["lspci", "-nn", "|", "grep", "-i", "vga"],
            shell=True, capture_output=True, text=True, timeout=5
        )
        if "amd" in result.stdout.lower() or "advanced micro devices" in result.stdout.lower():
            return GPUInfo(
                name="AMD GPU (detected via lspci)",
                vendor="amd",
                vram_total_mb=8192,  # Unknown, assume 8GB
                vram_free_mb=4096,
            )
    except Exception:
        pass

    return None


async def _detect_intel_gpu() -> GPUInfo | None:
    """Detect Intel GPU (Arc, Iris Xe, etc.)."""
    try:
        result = subprocess.run(
            ["lspci", "-nn", "|", "grep", "-i", "vga"],
            shell=True, capture_output=True, text=True, timeout=5
        )
        if "intel" in result.stdout.lower():
            return GPUInfo(
                name="Intel GPU (detected via lspci)",
                vendor="intel",
                vram_total_mb=4096,  # Shared memory, assume 4GB
                vram_free_mb=2048,
            )
    except Exception:
        pass
    return None


def _detect_apple_gpu() -> GPUInfo:
    """Detect Apple Silicon GPU."""
    try:
        # Get chip name
        result = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True, text=True, timeout=5
        )
        chip_name = result.stdout.strip()

        # Get total memory (unified)
        mem_result = subprocess.run(
            ["sysctl", "-n", "hw.memsize"],
            capture_output=True, text=True, timeout=5
        )
        total_bytes = int(mem_result.stdout.strip())
        total_gb = total_bytes / (1024**3)

        # GPU gets portion of unified memory
        gpu_memory_gb = min(total_gb * 0.75, 24)  # Max 24GB for GPU

        return GPUInfo(
            name=f"Apple {chip_name} (Unified Memory)",
            vendor="apple",
            vram_total_mb=int(gpu_memory_gb * 1024),
            vram_free_mb=int(gpu_memory_gb * 1024 * 0.5),
            device_id=0,
        )
    except Exception:
        return GPUInfo(
            name="Apple Silicon GPU",
            vendor="apple",
            vram_total_mb=8192,
            vram_free_mb=4096,
            device_id=0,
        )


async def _get_ollama_version() -> str | None:
    """Get Ollama version."""
    try:
        import httpx
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get("http://localhost:11434/api/version")
            if resp.status_code == 200:
                return resp.json().get("version")
    except Exception:
        pass
    return None


async def _get_ffmpeg_version() -> str | None:
    """Get FFmpeg version."""
    try:
        result = subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True, text=True, timeout=5
        )
        first_line = result.stdout.split("\n")[0]
        return first_line
    except Exception:
        return None


# ============================================================
# MODEL MANAGEMENT
# ============================================================

MODEL_REGISTRY = {
    # Whisper.cpp models
    "whisper-large-v3-turbo": {
        "url": "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-turbo.bin",
        "filename": "ggml-large-v3-turbo.bin",
        "size_mb": 1550,
        "type": "whisper",
    },
    "whisper-base-en": {
        "url": "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.en.bin",
        "filename": "ggml-base.en.bin",
        "size_mb": 142,
        "type": "whisper",
    },
    # YOLO models (Ultralytics auto-downloads, but we can cache)
    "yolov8n-face": {
        "url": "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n-face.pt",
        "filename": "yolov8n-face.pt",
        "size_mb": 6,
        "type": "yolo",
    },
    "yolov8n-seg": {
        "url": "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n-seg.pt",
        "filename": "yolov8n-seg.pt",
        "size_mb": 12,
        "type": "yolo",
    },
    # Embedding models (ONNX)
    "bge-small-en-v1.5": {
        "url": "https://huggingface.co/BAAI/bge-small-en-v1.5/resolve/main/onnx/model.onnx",
        "filename": "bge-small-en-v1.5.onnx",
        "size_mb": 33,
        "type": "embedding",
    },
    "bge-small-en-v1.5-tokenizer": {
        "url": "https://huggingface.co/BAAI/bge-small-en-v1.5/resolve/main/tokenizer.json",
        "filename": "bge-small-en-v1.5-tokenizer.json",
        "size_mb": 1,
        "type": "embedding",
    },
}


@dataclass
class ModelDownloadResult:
    """Result of model download."""
    success: bool
    path: Path | None
    error: str | None = None
    bytes_downloaded: int = 0


async def download_model(
    model_name: str,
    models_dir: Path | None = None,
    force: bool = False,
    progress_callback: callable = None,
) -> ModelDownloadResult:
    """
    Download a model from the registry.

    Args:
        model_name: Key from MODEL_REGISTRY
        models_dir: Target directory (defaults to config)
        force: Re-download even if exists
        progress_callback: callback(downloaded_bytes, total_bytes)

    Returns:
        ModelDownloadResult with path or error
    """
    settings = get_settings()
    models_dir = models_dir or settings.hardware.models_dir
    models_dir.mkdir(parents=True, exist_ok=True)

    if model_name not in MODEL_REGISTRY:
        return ModelDownloadResult(
            success=False,
            path=None,
            error=f"Unknown model: {model_name}. Available: {list(MODEL_REGISTRY.keys())}",
        )

    model_info = MODEL_REGISTRY[model_name]
    target_path = models_dir / model_info["filename"]

    if target_path.exists() and not force:
        logger.info("model_exists", model=model_name, path=str(target_path))
        return ModelDownloadResult(success=True, path=target_path)

    logger.info("model_download_start", model=model_name, url=model_info["url"])

    try:
        # Download with progress
        def _reporthook(block_num: int, block_size: int, total_size: int):
            if progress_callback and total_size > 0:
                progress_callback(block_num * block_size, total_size)

        urllib.request.urlretrieve(model_info["url"], target_path, _reporthook)

        # Verify size
        actual_size = target_path.stat().st_size
        expected_size = model_info["size_mb"] * 1024 * 1024
        if actual_size < expected_size * 0.9:  # Allow 10% variance
            target_path.unlink(missing_ok=True)
            return ModelDownloadResult(
                success=False,
                path=None,
                error=f"Downloaded file too small: {actual_size} vs expected ~{expected_size}",
            )

        logger.info("model_download_complete", model=model_name, path=str(target_path), size_mb=actual_size / 1024 / 1024)
        return ModelDownloadResult(success=True, path=target_path, bytes_downloaded=actual_size)

    except Exception as e:
        logger.error("model_download_failed", model=model_name, error=str(e))
        target_path.unlink(missing_ok=True)
        return ModelDownloadResult(success=False, path=None, error=str(e))


async def ensure_models(
    models_dir: Path | None = None,
    required: list[str] | None = None,
    progress_callback: callable = None,
) -> dict[str, ModelDownloadResult]:
    """
    Ensure required models are downloaded.

    Args:
        models_dir: Target directory
        required: List of model names (defaults to all in registry)
        progress_callback: callback(model_name, downloaded_bytes, total_bytes)

    Returns:
        Dict of model_name -> ModelDownloadResult
    """
    required = required or list(MODEL_REGISTRY.keys())
    results = {}

    for model_name in required:
        def _progress(downloaded, total):
            if progress_callback:
                progress_callback(model_name, downloaded, total)

        result = await download_model(model_name, models_dir, progress_callback=_progress)
        results[model_name] = result

        if not result.success:
            logger.error("required_model_failed", model=model_name, error=result.error)

    return results


async def setup_ollama_models(
    models: list[str] | None = None,
    host: str = "http://localhost:11434",
) -> dict[str, bool]:
    """
    Pull required Ollama models.

    Args:
        models: List of model names (e.g., ["qwen2.5:7b-instruct-q4_K_M"])
        host: Ollama host URL

    Returns:
        Dict of model -> success
    """
    import httpx

    models = models or [
        "qwen2.5:7b-instruct-q4_K_M",
        "llama3.2:3b-instruct-q4_K_M",
        "qwen2.5:3b-instruct-q4_K_M",
    ]

    results = {}

    async with httpx.AsyncClient(timeout=300.0) as client:
        for model in models:
            logger.info("ollama_pull_start", model=model, host=host)
            try:
                resp = await client.post(
                    f"{host}/api/pull",
                    json={"name": model, "stream": False},
                    timeout=600.0,
                )
                if resp.status_code == 200:
                    logger.info("ollama_pull_complete", model=model)
                    results[model] = True
                else:
                    logger.error("ollama_pull_failed", model=model, status=resp.status_code, text=resp.text)
                    results[model] = False
            except Exception as e:
                logger.error("ollama_pull_error", model=model, error=str(e))
                results[model] = False

    return results


# ============================================================
# OPTIMIZATION HELPERS
# ============================================================

def get_optimal_llm_config(gpu: GPUInfo | None) -> dict[str, Any]:
    """Get optimal LLM config for detected GPU."""
    if not gpu:
        return {
            "model": "qwen2.5:3b-instruct-q4_K_M",
            "num_gpu_layers": 0,
            "num_batch": 128,
            "offload_layers": 999,
        }

    vram_gb = gpu.vram_total_mb / 1024

    if gpu.vendor == "nvidia":
        if vram_gb >= 20:
            return {
                "model": "qwen2.5:14b-instruct-q4_K_M",
                "num_gpu_layers": -1,
                "num_batch": 1024,
                "offload_layers": 0,
            }
        elif vram_gb >= 10:
            return {
                "model": "qwen2.5:7b-instruct-q4_K_M",
                "num_gpu_layers": -1,
                "num_batch": 512,
                "offload_layers": 0,
            }
        elif vram_gb >= 6:
            return {
                "model": "qwen2.5:7b-instruct-q4_K_M",
                "num_gpu_layers": 20,
                "num_batch": 256,
                "offload_layers": 12,
            }
        else:  # 4GB
            return {
                "model": "qwen2.5:7b-instruct-q4_K_M",
                "num_gpu_layers": 8,
                "num_batch": 256,
                "offload_layers": 24,
            }

    if gpu.vendor == "apple":
        return {
            "model": "qwen2.5:7b-instruct-q4_K_M",
            "num_gpu_layers": -1,
            "num_batch": 512,
            "offload_layers": 0,
        }

    # AMD/Intel/other
    if vram_gb >= 16:
        return {
            "model": "qwen2.5:7b-instruct-q4_K_M",
            "num_gpu_layers": -1,
            "num_batch": 512,
            "offload_layers": 0,
        }
    else:
        return {
            "model": "qwen2.5:3b-instruct-q4_K_M",
            "num_gpu_layers": -1 if vram_gb >= 4 else 0,
            "num_batch": 256,
            "offload_layers": 0 if vram_gb >= 4 else 999,
        }


def get_optimal_whisper_config(gpu: GPUInfo | None) -> dict[str, Any]:
    """Get optimal Whisper config for detected GPU."""
    if not gpu:
        return {"model": "base.en", "precision": "int8"}

    vram_gb = gpu.vram_total_mb / 1024

    if gpu.vendor == "nvidia" and vram_gb >= 6:
        return {"model": "large-v3-turbo", "precision": "float16"}
    elif gpu.vendor == "apple":
        return {"model": "large-v3-turbo", "precision": "float16"}
    else:
        return {"model": "large-v3-turbo", "precision": "int8"}


def print_hardware_summary(info: SystemInfo) -> None:
    """Print human-readable hardware summary."""
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel

    console = Console()

    # GPU Panel
    if info.gpu:
        gpu_table = Table(show_header=False, box=None)
        gpu_table.add_column("Property", style="cyan")
        gpu_table.add_column("Value", style="white")
        gpu_table.add_row("Name", info.gpu.name)
        gpu_table.add_row("Vendor", info.gpu.vendor.upper())
        gpu_table.add_row("VRAM Total", f"{info.gpu.vram_total_mb / 1024:.1f} GB")
        gpu_table.add_row("VRAM Free", f"{info.gpu.vram_free_mb / 1024:.1f} GB")
        if info.gpu.driver_version:
            gpu_table.add_row("Driver", info.gpu.driver_version)
        if info.gpu.cuda_version:
            gpu_table.add_row("CUDA", info.gpu.cuda_version)
        if info.gpu.compute_capability:
            gpu_table.add_row("Compute Cap.", info.gpu.compute_capability)
        console.print(Panel(gpu_table, title="🖥️  GPU", border_style="green"))
    else:
        console.print(Panel("No dedicated GPU detected", title="🖥️  GPU", border_style="yellow"))

    # CPU Panel
    cpu_table = Table(show_header=False, box=None)
    cpu_table.add_column("Property", style="cyan")
    cpu_table.add_column("Value", style="white")
    cpu_table.add_row("Name", info.cpu.name)
    cpu_table.add_row("Cores", f"{info.cpu.cores_physical}P / {info.cpu.cores_logical}L")
    cpu_table.add_row("Frequency", f"{info.cpu.frequency_mhz} MHz" if info.cpu.frequency_mhz else "Unknown")
    cpu_table.add_row("Architecture", info.cpu.architecture)
    console.print(Panel(cpu_table, title="🔧 CPU", border_style="blue"))

    # RAM Panel
    ram_table = Table(show_header=False, box=None)
    ram_table.add_column("Property", style="cyan")
    ram_table.add_column("Value", style="white")
    ram_table.add_row("Total", f"{info.ram_total_gb:.1f} GB")
    ram_table.add_row("Available", f"{info.ram_free_gb:.1f} GB")
    console.print(Panel(ram_table, title="💾 RAM", border_style="magenta"))

    # Software Panel
    sw_table = Table(show_header=False, box=None)
    sw_table.add_column("Property", style="cyan")
    sw_table.add_column("Value", style="white")
    sw_table.add_row("Platform", info.platform)
    sw_table.add_row("Python", info.python_version)
    sw_table.add_row("Ollama", info.ollama_version or "Not running")
    sw_table.add_row("FFmpeg", info.ffmpeg_version.split(" ")[2] if info.ffmpeg_version else "Not found")
    console.print(Panel(sw_table, title="📦 Software", border_style="cyan"))

    # Recommendation
    rec_colors = {
        "rtx_4090_24gb": "green",
        "rtx_3060_12gb": "green",
        "rtx_2050_4gb": "yellow",
        "apple_silicon": "green",
        "cpu_only": "red",
    }
    color = rec_colors.get(info.recommended_profile, "white")
    console.print(Panel(
        f"[bold]Recommended Profile:[/bold] [{color}]{info.recommended_profile}[/{color}]",
        title="🎯 Recommendation",
        border_style=color,
    ))


if __name__ == "__main__":
    import asyncio

    async def main():
        info = await detect_hardware()
        print_hardware_summary(info)

        # Test optimal configs
        if info.gpu:
            print("\n📋 Optimal LLM Config:")
            print(json.dumps(get_optimal_llm_config(info.gpu), indent=2))
            print("\n📋 Optimal Whisper Config:")
            print(json.dumps(get_optimal_whisper_config(info.gpu), indent=2))

    asyncio.run(main())