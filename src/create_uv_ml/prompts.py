"""Interactive prompt module for collecting user configuration choices.

Phase 2 of the pipeline: questionnaire-style guided setup that collects
target directory, Python version, CUDA strategy, extra packages, and
mirror preference.
"""

import os
import sys
from typing import cast

import questionary
from questionary import Separator

from create_uv_ml.generator import CUDA_ALIASES, MIRROR_ALIASES, CudaVersion, MirrorSource

# Stable Python versions offered to the user (most recent first)
PYTHON_VERSIONS = ["3.12", "3.11", "3.10"]

# Extra packages organised by category for the checkbox prompt
EXTRA_PACKAGES: dict[str, list[str]] = {
    "Deep Learning": [
        "transformers",
        "datasets",
        "accelerate",
        "tensorboard",
        "torchaudio",
    ],
    "Scientific Computing": [
        "numpy",
        "pandas",
        "matplotlib",
        "scikit-learn",
        "scipy",
        "jupyterlab",
    ],
    "Computer Vision": [
        "opencv-python",
        "Pillow",
    ],
    "Utilities": [
        "tqdm",
        "rich",
        "requests",
        "pydantic",
    ],
}


def ask_target_directory() -> str:
    """Ask the user for the target directory.

    Offers the current working directory as default; if the user declines,
    prompts for a custom path.  Performs path cleaning (``~`` expansion,
    relative→absolute resolution).

    If the chosen directory already contains a ``.venv``, the user is
    warned and must confirm before proceeding.
    """
    cwd = os.getcwd()
    use_cwd = questionary.confirm(
        f"Create environment in current directory? ({cwd})",
        default=True,
    ).ask()

    if use_cwd is None:
        raise SystemExit(1)

    target = cwd if use_cwd else None

    if target is None:
        custom_path = questionary.text(
            "Enter target directory path:",
        ).ask()

        if custom_path is None:
            raise SystemExit(1)

        target = os.path.abspath(os.path.expanduser(custom_path.strip()))

    # Warn if .venv already exists in the target directory
    venv_path = os.path.join(target, ".venv")
    if os.path.isdir(venv_path):
        proceed = questionary.confirm(
            f"⚠️  A .venv already exists at {venv_path}. It will be reused by uv sync. Continue?",
            default=False,
        ).ask()

        if proceed is None or not proceed:
            raise SystemExit(1)

    return target


def ask_python_version() -> str:
    """Ask the user to select a Python version.

    Presents the three latest stable versions plus a manual-entry option.
    """
    choices = PYTHON_VERSIONS + ["Other (enter manually)"]
    result = questionary.select(
        "Select Python version:",
        choices=choices,
    ).ask()

    if result is None:
        raise SystemExit(1)

    if result.startswith("Other"):
        custom: str = questionary.text("Enter Python version (e.g. 3.9):").ask()
        if custom is None:
            raise SystemExit(1)
        version = custom.strip()
        if not version:
            raise SystemExit(1)
        return version

    return str(result)


def ask_cuda_strategy(recommended: str | None = None) -> CudaVersion | None:
    """Ask the user to select a CUDA strategy.

    *recommended* comes from the detector (e.g. "cu121").  The matching
    choice is placed first in the list so it becomes the default selection.
    Returns ``None`` on macOS (no CUDA support).
    """
    # On macOS there is no CUDA support at all
    if sys.platform == "darwin":
        return None

    choices = list(CUDA_ALIASES.values())

    # Move the recommended option to the top so it becomes the default
    if recommended and recommended in CUDA_ALIASES:
        recommended_label = CUDA_ALIASES[recommended]
        choices.remove(recommended_label)
        choices.insert(0, recommended_label)

    result = questionary.select(
        "Select CUDA strategy:",
        choices=choices,
    ).ask()

    if result is None:
        raise SystemExit(1)

    return cast(CudaVersion, result)


def ask_extra_packages() -> list[str]:
    """Ask the user to select additional packages via a checkbox prompt.

    Packages are grouped by category with separator headers.
    """
    choices: list[str | Separator] = []
    for category, packages in EXTRA_PACKAGES.items():
        choices.append(Separator(f"--- {category} ---"))
        choices.extend(packages)

    result = questionary.checkbox(
        "Select additional packages (space to toggle, enter to confirm):",
        choices=choices,
    ).ask()

    if result is None:
        return []

    return list(result)


def ask_mirror() -> MirrorSource:
    """Ask the user to select a PyPI mirror source."""
    result = questionary.select(
        "Select PyPI mirror:",
        choices=list(MIRROR_ALIASES.values()),
    ).ask()

    if result is None:
        raise SystemExit(1)

    return cast(MirrorSource, result)
