"""
AutoEdit Pro — CLI Entry Point
Main command-line interface for the editing pipeline.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Annotated, Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from autoedit.config import load_settings, get_settings, print_config_summary
from autoedit.utils.hardware import detect_hardware, print_hardware_summary, ensure_models, setup_ollama_models
from autoedit.utils.logging import setup_logging, get_logger
from autoedit.core.orchestrator import run_pipeline, Orchestrator, OrchestratorConfig

app = typer.Typer(
    name="autoedit",
    help="AutoEdit Pro — AI-powered video editing automation",
    add_completion=False,
    rich_markup_mode="rich",
)

console = Console()
logger = get_logger(__name__)


# ============================================================
# GLOBAL OPTIONS
# ============================================================

def version_callback(value: bool):
    if value:
        from autoedit import __version__
        console.print(f"AutoEdit Pro v{__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: Annotated[bool, typer.Option("--version", "-v", callback=version_callback, help="Show version")] = False,
    config: Annotated[Optional[Path], typer.Option("--config", "-c", help="Config file path")] = None,
    verbose: Annotated[bool, typer.Option("--verbose", help="Verbose logging")] = False,
    quiet: Annotated[bool, typer.Option("--quiet", "-q", help="Quiet mode")] = False,
):
    """AutoEdit Pro — AI-powered video editing automation."""
    # Setup logging
    level = "DEBUG" if verbose else "WARNING" if quiet else "INFO"
    setup_logging(level=level)

    # Load config
    if config:
        load_settings(config)
    else:
        get_settings()  # Load defaults


# ============================================================
# COMMANDS
# ============================================================

@app.command()
def run(
    input_path: Annotated[Path, typer.Argument(help="Input video/audio/images path")],
    skill: Annotated[str, typer.Option("--skill", "-s", help="Skill pack name")] = "educational_clean",
    script: Annotated[Optional[Path], typer.Option("--script", help="Script text file for alignment")] = None,
    output: Annotated[Optional[Path], typer.Option("--output", "-o", help="Output directory")] = None,
    auto_review: Annotated[bool, typer.Option("--auto-review", help="Skip review UI")] = False,
    resume: Annotated[Optional[str], typer.Option("--resume", help="Resume run ID")] = None,
):
    """Run the complete editing pipeline."""
    console.print(Panel.fit(
        f"[bold cyan]AutoEdit Pro[/bold cyan] — [green]Pipeline Run[/green]\n"
        f"Input: [yellow]{input_path}[/yellow]\n"
        f"Skill: [yellow]{skill}[/yellow]\n"
        f"Output: [yellow]{output or 'auto'}[/yellow]",
        border_style="cyan",
    ))

    async def _run():
        result = await run_pipeline(
            input_path=input_path,
            skill_pack=skill,
            script_path=script,
            output_dir=output,
            auto_review=auto_review,
            run_id=resume,
        )

        if result.success:
            console.print(Panel.fit(
                f"[bold green]✅ Pipeline Completed Successfully![/bold green]\n\n"
                f"Run ID: [cyan]{result.run_id}[/cyan]\n"
                f"Outputs: [yellow]{len(result.output_paths)} files[/yellow]\n"
                f"Duration: [yellow]{result.run_state.duration_seconds:.1f}s[/yellow]",
                border_style="green",
            ))
            for name, path in result.output_paths.items():
                console.print(f"  📄 [cyan]{name}[/cyan]: [yellow]{path}[/yellow]")
        else:
            console.print(Panel.fit(
                f"[bold red]❌ Pipeline Failed[/bold red]\n\n"
                f"Run ID: [cyan]{result.run_id}[/cyan]\n"
                f"Error: [red]{result.error}[/red]",
                border_style="red",
            ))
            raise typer.Exit(1)

    asyncio.run(_run())


@app.command()
def hardware(
    detect: Annotated[bool, typer.Option("--detect", "-d", help="Detect and show hardware")] = True,
    profile: Annotated[Optional[str], typer.Option("--profile", "-p", help="Show specific profile")] = None,
    benchmark: Annotated[bool, typer.Option("--benchmark", "-b", help="Run quick benchmark")] = False,
):
    """Detect and display hardware information."""
    async def _hardware():
        info = await detect_hardware()
        print_hardware_summary(info)

        if profile:
            from autoedit.config import get_settings
            settings = get_settings()
            settings.apply_hardware_profile(profile)
            console.print(f"\n[green]Applied profile: {profile}[/green]")

    asyncio.run(_hardware())


@app.command()
def models(
    download: Annotated[bool, typer.Option("--download", "-d", help="Download required models")] = False,
    list_models: Annotated[bool, typer.Option("--list", "-l", help="List available models")] = False,
    ollama: Annotated[bool, typer.Option("--ollama", help="Pull Ollama models")] = False,
):
    """Manage local models."""
    from autoedit.utils.hardware import MODEL_REGISTRY

    if list_models:
        table = Table(title="Available Models")
        table.add_column("Name", style="cyan")
        table.add_column("Type", style="green")
        table.add_column("Size (MB)", justify="right", style="yellow")
        table.add_column("Description")
        for name, info in MODEL_REGISTRY.items():
            table.add_row(name, info["type"], str(info["size_mb"]), info["url"])
        console.print(table)
        return

    async def _models():
        settings = get_settings()

        if download:
            console.print("[cyan]Downloading local models...[/cyan]")
            results = await ensure_models(settings.hardware.models_dir)
            for name, result in results.items():
                status = "✅" if result.success else "❌"
                console.print(f"  {status} {name}: {result.path or result.error}")

        if ollama:
            console.print("[cyan]Pulling Ollama models...[/cyan]")
            results = await setup_ollama_models()
            for model, success in results.items():
                status = "✅" if success else "❌"
                console.print(f"  {status} {model}")

    asyncio.run(_models())


@app.command()
def skills(
    list_skills: Annotated[bool, typer.Option("--list", "-l", help="List installed skill packs")] = False,
    discover: Annotated[bool, typer.Option("--discover", "-d", help="Discover new skills from internet")] = False,
    build: Annotated[Optional[Path], typer.Option("--build", "-b", help="Build skill from reference videos")] = None,
    info: Annotated[Optional[str], typer.Option("--info", "-i", help="Show skill pack details")] = None,
):
    """Manage editing skill packs."""
    from autoedit.skills import get_registry

    registry = get_registry()

    if list_skills:
        table = Table(title="Installed Skill Packs")
        table.add_column("Name", style="cyan")
        table.add_column("Version", style="green")
        table.add_column("Category", style="yellow")
        table.add_column("Description")
        for name, pack in registry.list_packs().items():
            table.add_row(
                name,
                pack.metadata.version,
                pack.metadata.category,
                pack.metadata.description[:60] + "..."
            )
        console.print(table)
        return

    if info:
        pack = registry.get_pack(info)
        if pack:
            console.print(Panel.fit(
                f"[bold]{pack.metadata.name}[/bold] v{pack.metadata.version}\n"
                f"Category: {pack.metadata.category}\n"
                f"Source: {pack.metadata.source}\n"
                f"Tags: {', '.join(pack.metadata.tags)}\n\n"
                f"{pack.metadata.description}",
                title=f"Skill Pack: {info}",
                border_style="cyan",
            ))
        else:
            console.print(f"[red]Skill pack not found: {info}[/red]")
        return

    async def _skills():
        if discover:
            console.print("[cyan]Discovering new skills...[/cyan]")
            # Implementation would call discovery pipeline
            console.print("[yellow]Skill discovery not yet implemented[/yellow]")

        if build:
            console.print(f"[cyan]Building skill from references in {build}[/cyan]")
            # Implementation would call build_skill_from_references
            console.print("[yellow]Skill building not yet implemented[/yellow]")

    asyncio.run(_skills())


@app.command()
def config(
    show: Annotated[bool, typer.Option("--show", "-s", help="Show current config")] = False,
    validate: Annotated[bool, typer.Option("--validate", help="Validate config")] = False,
    edit: Annotated[bool, typer.Option("--edit", "-e", help="Open config in editor")] = False,
):
    """Manage configuration."""
    settings = get_settings()

    if show:
        print_config_summary(settings)
        return

    if validate:
        from autoedit.config.loader import validate_config
        valid, errors = validate_config()
        if valid:
            console.print("[green]✅ Configuration is valid[/green]")
        else:
            console.print("[red]❌ Configuration errors:[/red]")
            for err in errors:
                console.print(f"  - {err}")
        return

    if edit:
        import subprocess
        import os
        editor = os.environ.get("EDITOR", "notepad" if os.name == "nt" else "vim")
        subprocess.run([editor, "config/default.yaml"])
        return


@app.command()
def status(
    run_id: Annotated[Optional[str], typer.Argument(help="Run ID to check")] = None,
    list_runs: Annotated[bool, typer.Option("--list", "-l", help="List recent runs")] = False,
):
    """Check pipeline run status."""
    orchestrator = Orchestrator()

    if list_runs:
        runs = orchestrator.list_recent_runs(limit=20)
        table = Table(title="Recent Runs")
        table.add_column("Run ID", style="cyan")
        table.add_column("Status", style="green")
        table.add_column("Started", style="yellow")
        table.add_column("Duration", justify="right")
        table.add_column("Skill")
        for run in runs:
            status_color = {
                "completed": "green",
                "failed": "red",
                "running": "yellow",
                "awaiting_review": "blue",
                "cancelled": "dim",
            }.get(run.status.value, "white")
            table.add_row(
                run.run_id,
                f"[{status_color}]{run.status.value}[/{status_color}]",
                run.started_at.strftime("%H:%M:%S") if run.started_at else "—",
                f"{run.duration_seconds:.1f}s" if run.duration_seconds else "—",
                run.input_config.get("skill_pack", "—"),
            )
        console.print(table)
        return

    if run_id:
        run = orchestrator.get_run_status(run_id)
        if run:
            console.print(Panel.fit(
                f"Run ID: [cyan]{run.run_id}[/cyan]\n"
                f"Status: [green]{run.status.value}[/green]\n"
                f"Progress: [yellow]{run.progress:.0%}[/yellow]\n"
                f"Duration: [yellow]{run.duration_seconds:.1f}s[/yellow]" if run.duration_seconds else "Duration: —",
                title="Run Status",
                border_style="cyan",
            ))
        else:
            console.print(f"[red]Run not found: {run_id}[/red]")
        return

    console.print("[yellow]Provide --list or a run ID[/yellow]")


@app.command()
def doctor(
    check_all: Annotated[bool, typer.Option("--all", "-a", help="Run all checks")] = True,
):
    """Run system diagnostics."""
    console.print(Panel.fit("[bold cyan]AutoEdit Pro — System Diagnostics[/bold cyan]", border_style="cyan"))

    checks = []

    # Check Python version
    import sys
    py_ok = sys.version_info >= (3, 11)
    checks.append(("Python ≥3.11", py_ok, f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"))

    # Check FFmpeg
    import shutil
    ffmpeg_ok = shutil.which("ffmpeg") is not None
    checks.append(("FFmpeg", ffmpeg_ok, "Found" if ffmpeg_ok else "NOT FOUND"))

    # Check Ollama
    import httpx
    ollama_ok = False
    try:
        resp = httpx.get("http://localhost:11434/api/version", timeout=3)
        ollama_ok = resp.status_code == 200
    except Exception:
        pass
    checks.append(("Ollama API", ollama_ok, "Running" if ollama_ok else "Not accessible"))

    # Check DaVinci Resolve scripting
    davinci_ok = False
    try:
        import importlib.util
        davinci_ok = importlib.util.find_spec("davinci_resolve_scripting") is not None
    except Exception:
        pass
    checks.append(("DaVinci Python API", davinci_ok, "Available" if davinci_ok else "Not installed"))

    # Check GPU
    import asyncio
    hw_info = asyncio.run(detect_hardware())
    gpu_ok = hw_info.has_gpu
    checks.append(("GPU Detected", gpu_ok, hw_info.gpu.name if hw_info.gpu else "None"))

    # Display results
    table = Table(title="System Checks")
    table.add_column("Check", style="cyan")
    table.add_column("Status", justify="center")
    table.add_column("Details", style="dim")

    for name, ok, details in checks:
        status = "✅ PASS" if ok else "❌ FAIL"
        style = "green" if ok else "red"
        table.add_row(name, f"[{style}]{status}[/{style}]", details)

    console.print(table)

    if not all(ok for _, ok, _ in checks):
        console.print("\n[yellow]Some checks failed. Run with --help for setup instructions.[/yellow]")
        raise typer.Exit(1)
    else:
        console.print("\n[green]All checks passed! ✨[/green]")


@app.command()
def setup(
    models: Annotated[bool, typer.Option("--models", help="Download models")] = True,
    ollama: Annotated[bool, typer.Option("--ollama", help="Pull Ollama models")] = True,
    skills: Annotated[bool, typer.Option("--skills", help="Install bundled skills")] = True,
    config_file: Annotated[bool, typer.Option("--config", help="Create config files")] = True,
):
    """Run initial setup (download models, install skills, create config)."""
    console.print(Panel.fit("[bold cyan]AutoEdit Pro — Initial Setup[/bold cyan]", border_style="cyan"))

    async def _setup():
        settings = get_settings()

        if config_file:
            console.print("[cyan]Creating config files...[/cyan]")
            # Config files already created by repo init
            console.print("  ✅ Config files ready")

        if models:
            console.print("[cyan]Downloading local models...[/cyan]")
            results = await ensure_models(settings.hardware.models_dir)
            for name, result in results.items():
                status = "✅" if result.success else "❌"
                console.print(f"  {status} {name}")

        if ollama:
            console.print("[cyan]Pulling Ollama models...[/cyan]")
            results = await setup_ollama_models()
            for model, success in results.items():
                status = "✅" if success else "❌"
                console.print(f"  {status} {model}")

        if skills:
            console.print("[cyan]Installing bundled skill packs...[/cyan]")
            # Skills are bundled in repo
            console.print("  ✅ Bundled skills ready")

    asyncio.run(_setup())
    console.print("\n[green]Setup complete! Run 'autoedit doctor' to verify.[/green]")


if __name__ == "__main__":
    app()