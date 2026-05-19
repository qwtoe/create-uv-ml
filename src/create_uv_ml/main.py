"""CLI entrypoint module."""

import typer
from rich.console import Console

from create_uv_ml.prompts import ask_framework, ask_cuda_version
from create_uv_ml.generator import generate_pyproject
from create_uv_ml.runner import create_project

app = typer.Typer()
console = Console()


@app.command()
def create(
    project_name: str = typer.Argument(..., help="Your deep learning project name"),
) -> None:
    """Create a new uv-based deep learning project."""
    console.print(f"[bold green]🚀 Welcome to create-uv-ml! Initializing {project_name}...[/bold green]\n")

    # 1. Interactive prompts
    framework = ask_framework()
    cuda_version = None
    if framework == "PyTorch":
        cuda_version = ask_cuda_version()

    # 2. Show configuration summary
    config_display = f"{framework}"
    if cuda_version:
        config_display += f" + {cuda_version}"
    console.print(f"\n[cyan]⚙️  Configuration: {config_display}[/cyan]\n")

    # 3. Generate pyproject.toml
    console.print("[yellow]📝 Generating pyproject.toml...[/yellow]")
    pyproject_content = generate_pyproject(project_name, framework, cuda_version)

    # 4. Create project and run uv sync
    create_project(project_name, pyproject_content)

    console.print(f"\n[bold green]✅ Project {project_name} created successfully! Happy training![/bold green]")
    console.print(f"[dim]   cd {project_name} && source .venv/bin/activate[/dim]")


@app.command()
def version() -> None:
    """Show the version number."""
    from create_uv_ml import __version__
    console.print(f"create-uv-ml v{__version__}")


if __name__ == "__main__":
    app()
