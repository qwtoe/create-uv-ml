"""Executor module that writes config files and calls uv commands.

Phase 4 of the pipeline: creates the target directory, writes
pyproject.toml and .gitignore, then runs ``uv sync`` and ``uv add``
as subprocess calls with streamed output.
"""

import os
import shutil
import subprocess
import sys

from rich.console import Console

console = Console()

# Minimal .gitignore for Python + data-science workspaces
_GITIGNORE = """\
# Python
__pycache__/
*.py[cod]
*$py.class
*.egg-info/
dist/
build/
*.egg

# Virtual environments
.venv/
venv/
ENV/

# Jupyter
.ipynb_checkpoints/

# Environment
.env

# IDE
.idea/
.vscode/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db
"""


def check_uv_available() -> None:
    """Check that uv is installed and available in PATH."""
    if shutil.which("uv") is None:
        console.print("[bold red]Error: 'uv' is not installed or not in PATH.[/bold red]")
        console.print("Install it with: curl -LsSf https://astral.sh/uv/install.sh | sh")
        sys.exit(1)


def write_pyproject(target_dir: str, content: str) -> None:
    """Write pyproject.toml to the target directory."""
    os.makedirs(target_dir, exist_ok=True)
    path = os.path.join(target_dir, "pyproject.toml")
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    console.print(f"[green]  ✓ Wrote {path}[/green]")


def write_gitignore(target_dir: str) -> None:
    """Write a .gitignore to the target directory."""
    path = os.path.join(target_dir, ".gitignore")
    with open(path, "w", encoding="utf-8") as f:
        f.write(_GITIGNORE)
    console.print(f"[green]  ✓ Wrote {path}[/green]")


def uv_sync(target_dir: str, python_version: str) -> None:
    """Run ``uv sync --python <version>`` in the target directory.

    This creates the .venv, downloads the requested Python interpreter
    (if necessary), and installs the core dependencies declared in
    pyproject.toml.
    """
    console.print("[yellow]📦 Running uv sync to install core dependencies...[/yellow]")
    console.print(
        "[dim]   (PyTorch CUDA wheels are ~2 GB, this may take 10-30 minutes "
        "depending on your network)[/dim]"
    )
    try:
        process = subprocess.Popen(
            ["uv", "sync", "--python", python_version],
            cwd=target_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        if process.stdout is not None:
            for line in process.stdout:
                console.print(f"[dim]{line.rstrip()}[/dim]")
        retcode = process.wait()
        if retcode != 0:
            raise subprocess.CalledProcessError(retcode, "uv sync")
        console.print("[green]  ✓ uv sync completed[/green]")
    except subprocess.CalledProcessError:
        console.print("[bold red]✗ uv sync failed[/bold red]")
        sys.exit(1)


def uv_add(target_dir: str, packages: list[str]) -> None:
    """Run ``uv add <packages>`` in the target directory.

    Called after ``uv sync`` so that uv's resolver can solve the extra
    packages against the already-locked base dependencies.
    """
    if not packages:
        return

    console.print(f"[yellow]📦 Installing additional packages: {', '.join(packages)}[/yellow]")
    try:
        process = subprocess.Popen(
            ["uv", "add", *packages],
            cwd=target_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        if process.stdout is not None:
            for line in process.stdout:
                console.print(f"[dim]{line.rstrip()}[/dim]")
        retcode = process.wait()
        if retcode != 0:
            raise subprocess.CalledProcessError(retcode, "uv add")
        console.print("[green]  ✓ Additional packages installed[/green]")
    except subprocess.CalledProcessError:
        console.print("[bold red]✗ uv add failed[/bold red]")
        sys.exit(1)
