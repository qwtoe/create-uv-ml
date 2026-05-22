"""Interactive prompt module for collecting user configuration choices."""

from typing import cast

import questionary

from create_uv_ml.generator import (
    CUDA_ALIASES,
    FRAMEWORK_ALIASES,
    MIRROR_ALIASES,
    CudaVersion,
    Framework,
    MirrorSource,
)

FRAMEWORK_CHOICES = list(FRAMEWORK_ALIASES.values())
CUDA_CHOICES = list(CUDA_ALIASES.values())
MIRROR_CHOICES = list(MIRROR_ALIASES.values())


def ask_framework() -> Framework:
    """Ask the user to select a deep learning framework."""
    return cast(
        Framework,
        questionary.select(
            "Which deep learning framework do you want to use?",
            choices=FRAMEWORK_CHOICES,
        ).ask(),
    )


def ask_cuda_version() -> CudaVersion:
    """Ask the user to select a CUDA version."""
    return cast(
        CudaVersion,
        questionary.select(
            "Which CUDA version do you need?",
            choices=CUDA_CHOICES,
        ).ask(),
    )


def ask_mirror() -> MirrorSource:
    """Ask the user to select a PyPI mirror source."""
    return cast(
        MirrorSource,
        questionary.select(
            "Which PyPI mirror do you want to use?",
            choices=MIRROR_CHOICES,
        ).ask(),
    )
