"""CLI entrypoint module.

Orchestrates the five-phase pipeline:
  1. Detector  – sniff OS & GPU hardware
  2. Prompts   – collect user choices interactively
  3. Generator – assemble pyproject.toml
  4. Executor  – write files, run uv sync / uv add
  5. Post-handoff – print activation instructions
"""

import os
import sys

import typer
from rich.console import Console

from create_uv_ml.detector import detect
from create_uv_ml.executor import (
    check_uv_available,
    uv_add,
    uv_sync,
    write_gitignore,
    write_pyproject,
)
from create_uv_ml.generator import generate_pyproject
from create_uv_ml.prompts import (
    ask_cuda_strategy,
    ask_extra_packages,
    ask_mirror,
    ask_python_version,
    ask_target_directory,
)

console = Console()


def main(
    version: bool = typer.Option(
        False,
        "--version",
        "-v",
        help="Show the version number",
    ),
) -> None:
    """Create a new uv-based deep learning environment."""
    if version:
        from create_uv_ml import __version__

        console.print(f"create-uv-ml v{__version__}")
        raise typer.Exit()

    # ── Phase 1: Detector ──────────────────────────────────────────
    console.print("[cyan]🔍 Detecting system hardware...[/cyan]")
    detection = detect()

    if detection.has_nvidia:
        gpu_info = f"NVIDIA GPU detected (driver {detection.driver_version})"
        cuda_info = f"Recommended: {detection.recommended_cuda}"
        console.print(f"[green]  ✓ {gpu_info}[/green]")
        console.print(f"[green]  ✓ {cuda_info}[/green]")
    else:
        console.print("[yellow]  ℹ No NVIDIA GPU detected[/yellow]")

    # ── Phase 2: Interactive prompts ────────────────────────────────
    console.print("\n[bold]📋 Configuration[/bold]")

    target_dir = ask_target_directory()
    python_version = ask_python_version()
    cuda = ask_cuda_strategy(
        recommended=detection.recommended_cuda if detection.has_nvidia else None,
    )
    extras = ask_extra_packages()
    mirror = ask_mirror()

    # ── Configuration summary ───────────────────────────────────────
    config_parts = [f"Python {python_version}"]
    if cuda:
        config_parts.append(str(cuda))
    config_parts.append(f"Mirror: {mirror}")
    if extras:
        config_parts.append(f"Extras: {len(extras)} packages")

    console.print(f"\n[cyan]⚙️  Configuration: {' | '.join(config_parts)}[/cyan]")
    console.print(f"[cyan]   Target: {target_dir}[/cyan]\n")

    # ── Phase 3: Generator ──────────────────────────────────────────
    console.print("[yellow]📝 Generating pyproject.toml...[/yellow]")
    project_name = os.path.basename(target_dir)
    pyproject_content = generate_pyproject(project_name, python_version, cuda, mirror)
    write_pyproject(target_dir, pyproject_content)
    write_gitignore(target_dir)

    # ── Phase 4: Executor ───────────────────────────────────────────
    check_uv_available()
    uv_sync(target_dir, python_version)
    uv_add(target_dir, extras)

    # ── Phase 5: Post-handoff ───────────────────────────────────────
    console.print("\n[bold green]✅ Environment created successfully![/bold green]")
    console.print("\n[bold]Next steps:[/bold]")
    if sys.platform == "win32":
        console.print(f"  [dim]1.[/dim] {target_dir}\\.venv\\Scripts\\activate")
    else:
        console.print(f"  [dim]1.[/dim] source {target_dir}/.venv/bin/activate")
    console.print("  [dim]2.[/dim] Start coding! Create .py files or Jupyter notebooks.")


app = typer.Typer()
app.command()(main)


if __name__ == "__main__":
    app()
