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

import questionary
import typer
from rich.console import Console

from create_uv_ml.detector import detect, find_nvidia_smi
from create_uv_ml.executor import (
    check_uv_available,
    uv_add,
    uv_sync,
    write_gitignore,
    write_pyproject,
    write_verify_env,
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
    elif find_nvidia_smi():
        # nvidia-smi exists but detect() says no GPU → driver is broken
        console.print("[yellow]  ⚠ NVIDIA driver appears broken (nvidia-smi failed)[/yellow]")
        console.print("[yellow]    Try rebooting or reinstalling the NVIDIA driver.[/yellow]")
        console.print("[yellow]    Falling back to CPU-only mode.[/yellow]")
    else:
        console.print("[yellow]  ℹ No NVIDIA GPU detected[/yellow]")

    # ── Phase 2: Interactive prompts ────────────────────────────────
    console.print("\n[bold]📋 Configuration[/bold]")

    target_dir: str = ""
    python_version: str = ""
    cuda: str | None = None
    extras: list[str] = []
    mirror: str = "Default (PyPI)"

    _RECOMMENDED = detection.recommended_cuda if detection.has_nvidia else None

    while True:
        target_dir = ask_target_directory()
        python_version = ask_python_version()
        cuda = ask_cuda_strategy(recommended=_RECOMMENDED)
        extras = ask_extra_packages()
        mirror = ask_mirror()

        # ── Configuration summary ───────────────────────────────────
        config_parts = [f"Python {python_version}"]
        if cuda:
            config_parts.append(str(cuda))
        config_parts.append(f"Mirror: {mirror}")
        if extras:
            config_parts.append(f"Extras: {len(extras)} packages")

        console.print(f"\n[cyan]⚙️  Configuration: {' | '.join(config_parts)}[/cyan]")
        console.print(f"[cyan]   Target: {target_dir}[/cyan]\n")

        action = questionary.select(
            "What would you like to do?",
            choices=[
                "Continue (create environment)",
                "Change target directory",
                "Change Python version",
                "Change CUDA strategy",
                "Change extra packages",
                "Change mirror",
            ],
        ).ask()

        if action is None:
            raise typer.Exit()

        if action.startswith("Continue"):
            break

        # Re-ask only the selected step, then loop back to review
        console.print(f"\n[dim]── Revising: {action} ──[/dim]")
        if action == "Change target directory":
            target_dir = ask_target_directory()
        elif action == "Change Python version":
            python_version = ask_python_version()
        elif action == "Change CUDA strategy":
            cuda = ask_cuda_strategy(recommended=_RECOMMENDED)
        elif action == "Change extra packages":
            extras = ask_extra_packages()
        elif action == "Change mirror":
            mirror = ask_mirror()

    # ── Phase 3: Generator ──────────────────────────────────────────
    console.print("[yellow]📝 Generating pyproject.toml...[/yellow]")

    # Create target directory if it doesn't exist
    if not os.path.isdir(target_dir):
        os.makedirs(target_dir, exist_ok=True)
        console.print(f"[green]  ✓ Created directory: {target_dir}[/green]")

    project_name = os.path.basename(target_dir)
    pyproject_content = generate_pyproject(project_name, python_version, cuda, mirror)
    write_pyproject(target_dir, pyproject_content)
    write_gitignore(target_dir)
    write_verify_env(target_dir)

    # ── Phase 4: Executor ───────────────────────────────────────────
    check_uv_available()

    do_sync = questionary.confirm(
        "Install core dependencies (torch, torchvision) now?",
        default=True,
    ).ask()

    if do_sync is None:
        raise typer.Exit()

    if do_sync:
        uv_sync(target_dir, python_version)

        if extras:
            do_add = questionary.confirm(
                f"Install extra packages ({', '.join(extras)}) now?",
                default=True,
            ).ask()

            if do_add is None:
                raise typer.Exit()

            if do_add:
                uv_add(target_dir, extras)
            else:
                console.print("[yellow]  ℹ Extra packages skipped.[/yellow]")
                console.print(
                    f"[dim]     Run later: cd {target_dir}"
                    f" && source .venv/bin/activate"
                    f" && uv add {' '.join(extras)}[/dim]"
                )
    else:
        console.print(
            "[yellow]  ⚠ Core dependencies skipped"
            " — extra packages also skipped.[/yellow]"
        )
        console.print(
            f"[dim]     Run later: cd {target_dir} && uv sync[/dim]"
        )
        if extras:
            console.print(
                f"[dim]              uv add {' '.join(extras)}[/dim]"
            )
        console.print("[dim]     (uv sync must run first, then uv add)[/dim]")

    # ── Phase 5: Post-handoff ───────────────────────────────────────
    console.print("\n[bold green]✅ Environment created successfully![/bold green]")
    console.print("\n[bold]Next steps:[/bold]")

    steps: list[str] = []
    if sys.platform == "win32":
        steps.append(f"{target_dir}\\.venv\\Scripts\\activate")
    else:
        steps.append(f"source {target_dir}/.venv/bin/activate")
    steps.append("python verify_env.py    # Verify CUDA, torch, and packages")
    steps.append("Start coding! Create .py files or Jupyter notebooks.")

    for i, step in enumerate(steps, 1):
        console.print(f"  [dim]{i}.[/dim] {step}")


app = typer.Typer()
app.command()(main)


if __name__ == "__main__":
    app()
