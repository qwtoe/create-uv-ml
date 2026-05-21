"""Runner module that creates the project directory, writes config, and calls uv commands."""

import os
import shutil
import subprocess
import sys

from rich.console import Console

from create_uv_ml.generator import CudaVersion, Framework
from create_uv_ml.templates import (
    generate_gitignore,
    generate_init_py,
    generate_readme,
    generate_train_py,
)

console = Console()

UV_SYNC_TIMEOUT = 600  # 10 minutes


def check_uv_available() -> None:
    """Check that uv is installed and available in PATH."""
    if shutil.which("uv") is None:
        console.print("[bold red]Error: 'uv' is not installed or not in PATH.[/bold red]")
        console.print("Install it with: curl -LsSf https://astral.sh/uv/install.sh | sh")
        sys.exit(1)


def validate_project_name(name: str) -> None:
    """Validate that the project name is a legal Python package name."""
    if not name.isidentifier():
        console.print(f"[bold red]Error: '{name}' is not a valid Python package name.[/bold red]")
        console.print("Use only letters, digits, and underscores, and don't start with a digit.")
        sys.exit(1)


def create_project(
    project_name: str,
    pyproject_content: str,
    framework: Framework,
    cuda: CudaVersion | None,
    no_sync: bool = False,
    template: str = "full",
) -> None:
    """Create the project folder, write scaffold files, and optionally run uv sync."""

    if os.path.exists(project_name):
        console.print(f"[bold red]Error: Directory {project_name} already exists![/bold red]")
        sys.exit(1)

    try:
        # Create directory structure
        console.print(f"[yellow]📁 Creating project directory: {project_name}[/yellow]")
        os.makedirs(project_name, exist_ok=True)

        # Write pyproject.toml
        pyproject_path = os.path.join(project_name, "pyproject.toml")
        with open(pyproject_path, "w", encoding="utf-8") as f:
            f.write(pyproject_content)
        console.print(f"[green]  ✓ Wrote {pyproject_path}[/green]")

        # Write .gitignore
        gitignore_path = os.path.join(project_name, ".gitignore")
        with open(gitignore_path, "w", encoding="utf-8") as f:
            f.write(generate_gitignore())
        console.print(f"[green]  ✓ Wrote {gitignore_path}[/green]")

        # Write README.md
        readme_path = os.path.join(project_name, "README.md")
        with open(readme_path, "w", encoding="utf-8") as f:
            f.write(generate_readme(project_name, framework, cuda))
        console.print(f"[green]  ✓ Wrote {readme_path}[/green]")

        # Write src layout and training script (full template only)
        if template == "full":
            pkg_name = os.path.basename(project_name)
            src_dir = os.path.join(project_name, "src", pkg_name)
            os.makedirs(src_dir, exist_ok=True)

            init_path = os.path.join(src_dir, "__init__.py")
            with open(init_path, "w", encoding="utf-8") as f:
                f.write(generate_init_py(project_name))
            console.print(f"[green]  ✓ Wrote {init_path}[/green]")

            script_name = "train.py" if framework in ("PyTorch", "TensorFlow") else "analysis.py"
            script_path = os.path.join(src_dir, script_name)
            with open(script_path, "w", encoding="utf-8") as f:
                f.write(generate_train_py(project_name, framework))
            console.print(f"[green]  ✓ Wrote {script_path}[/green]")

            # Create data directory
            data_dir = os.path.join(project_name, "data")
            os.makedirs(data_dir, exist_ok=True)
            gitkeep_path = os.path.join(data_dir, ".gitkeep")
            with open(gitkeep_path, "w") as f:
                f.write("")
            console.print(f"[green]  ✓ Created {data_dir}/[/green]")

        # Run uv sync
        if no_sync:
            console.print("[yellow]⏭️  Skipping uv sync (--no-sync flag)[/yellow]")
        else:
            console.print(
                "[yellow]📦 Running uv sync to install dependencies "
                "(this may take a few minutes)...[/yellow]"
            )
            try:
                result = subprocess.run(
                    ["uv", "sync"],
                    cwd=project_name,
                    capture_output=True,
                    text=True,
                    check=True,
                    timeout=UV_SYNC_TIMEOUT,
                )
                if result.stdout:
                    console.print(result.stdout)
                console.print("[green]  ✓ uv sync completed[/green]")
            except subprocess.TimeoutExpired:
                console.print(
                    f"[bold red]✗ uv sync timed out after {UV_SYNC_TIMEOUT}s[/bold red]"
                )
                console.print(f"You can retry manually: cd {project_name} && uv sync")
                sys.exit(1)
            except subprocess.CalledProcessError as e:
                console.print("[bold red]✗ uv sync failed:[/bold red]")
                if e.stdout:
                    console.print(e.stdout)
                if e.stderr:
                    console.print(e.stderr)
                _cleanup_on_failure(project_name)
                sys.exit(1)

    except KeyboardInterrupt:
        console.print("\n[bold yellow]Interrupted by user.[/bold yellow]")
        _cleanup_on_failure(project_name)
        sys.exit(130)


def _cleanup_on_failure(project_name: str) -> None:
    """Notify the user about the incomplete project directory on failure."""
    if not os.path.exists(project_name):
        return
    console.print(f"[yellow]Incomplete project directory '{project_name}' exists.[/yellow]")
    console.print(f"[dim]Remove it manually with: rm -rf {project_name}[/dim]")
