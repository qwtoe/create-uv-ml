"""CLI entrypoint module."""

import os
from typing import Annotated

import typer
from rich.console import Console

from create_uv_ml.generator import (
    CUDA_ALIASES,
    FRAMEWORK_ALIASES,
    MIRROR_ALIASES,
    CudaVersion,
    Framework,
    MirrorSource,
    generate_pyproject,
)
from create_uv_ml.prompts import ask_cuda_version, ask_framework, ask_mirror
from create_uv_ml.runner import check_uv_available, create_project, validate_project_name

console = Console()


def main(
    project_name: Annotated[
        str | None,
        typer.Argument(help="Your deep learning project name"),
    ] = None,
    framework: str | None = typer.Option(
        None,
        "--framework",
        "-f",
        help="Framework: pytorch, tensorflow, or scipy (skips interactive prompt)",
    ),
    cuda: str | None = typer.Option(
        None,
        "--cuda",
        "-c",
        help="CUDA version: cu121, cu118, or cpu (skips interactive prompt)",
    ),
    mirror: str | None = typer.Option(
        None,
        "--mirror",
        "-m",
        help="PyPI mirror: default, tsinghua, or aliyun (skips interactive prompt)",
    ),
    no_sync: bool = typer.Option(
        False,
        "--no-sync",
        help="Skip uv sync (only generate project files)",
    ),
    template: str = typer.Option(
        "full",
        "--template",
        "-t",
        help="Template style: minimal or full",
    ),
    version: bool = typer.Option(
        False,
        "--version",
        "-v",
        help="Show the version number",
    ),
) -> None:
    """Create a new uv-based deep learning project."""
    if version:
        from create_uv_ml import __version__

        console.print(f"create-uv-ml v{__version__}")
        raise typer.Exit()

    if project_name is None:
        console.print("[bold red]Error: Missing argument 'PROJECT_NAME'[/bold red]")
        console.print("Usage: create-uv-ml PROJECT_NAME [OPTIONS]")
        console.print("Try 'create-uv-ml --help' for more information.")
        raise typer.Exit(1)

    # Validate prerequisites
    check_uv_available()
    validate_project_name(project_name)

    if template not in ("minimal", "full"):
        console.print("[bold red]Error: --template must be 'minimal' or 'full'[/bold red]")
        raise typer.Exit(1)

    # Resolve framework
    if framework:
        if framework not in FRAMEWORK_ALIASES:
            valid = ", ".join(FRAMEWORK_ALIASES.keys())
            console.print(f"[bold red]Error: --framework must be one of: {valid}[/bold red]")
            raise typer.Exit(1)
        selected_framework: Framework = FRAMEWORK_ALIASES[framework]
    else:
        selected_framework = ask_framework()

    # Resolve CUDA version
    selected_cuda: CudaVersion | None = None
    needs_cuda = selected_framework in ("PyTorch", "TensorFlow")
    if needs_cuda:
        if cuda:
            if cuda not in CUDA_ALIASES:
                valid = ", ".join(CUDA_ALIASES.keys())
                console.print(f"[bold red]Error: --cuda must be one of: {valid}[/bold red]")
                raise typer.Exit(1)
            selected_cuda = CUDA_ALIASES[cuda]
        else:
            selected_cuda = ask_cuda_version()

    # Resolve mirror source
    if mirror:
        if mirror not in MIRROR_ALIASES:
            valid = ", ".join(MIRROR_ALIASES.keys())
            console.print(f"[bold red]Error: --mirror must be one of: {valid}[/bold red]")
            raise typer.Exit(1)
        selected_mirror: MirrorSource = MIRROR_ALIASES[mirror]
    else:
        selected_mirror = ask_mirror()

    # Show configuration summary
    config_display = f"{selected_framework}"
    if selected_cuda:
        config_display += f" + {selected_cuda}"
    config_display += f" | Mirror: {selected_mirror}"
    console.print(f"\n[cyan]⚙️  Configuration: {config_display}[/cyan]")
    console.print(f"[cyan]   Template: {template}[/cyan]\n")

    # Generate pyproject.toml
    console.print("[yellow]📝 Generating pyproject.toml...[/yellow]")
    pyproject_content = generate_pyproject(
        project_name, selected_framework, selected_cuda, selected_mirror
    )

    # Create project
    create_project(
        project_name,
        pyproject_content,
        selected_framework,
        selected_cuda,
        no_sync=no_sync,
        template=template,
    )

    # Post-creation guidance
    pkg_name = os.path.basename(project_name)
    console.print(
        f"\n[bold green]✅ Project {project_name} created successfully! "
        "Happy training![/bold green]"
    )
    console.print("\n[bold]Next steps:[/bold]")
    step = 1
    console.print(f"  [dim]{step}.[/dim] cd {project_name}")
    step += 1
    if no_sync:
        console.print(f"  [dim]{step}.[/dim] uv sync")
        step += 1
    console.print(f"  [dim]{step}.[/dim] source .venv/bin/activate")
    step += 1
    if template == "full":
        script = "train.py" if selected_framework in ("PyTorch", "TensorFlow") else "analysis.py"
        console.print(f"  [dim]{step}.[/dim] python src/{pkg_name}/{script}")


app = typer.Typer()
app.command()(main)


if __name__ == "__main__":
    app()
