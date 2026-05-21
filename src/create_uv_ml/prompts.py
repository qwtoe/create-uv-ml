"""Interactive prompt module for collecting user configuration choices."""

from typing import cast

import questionary

from create_uv_ml.generator import CUDA_ALIASES, FRAMEWORK_ALIASES, CudaVersion, Framework

FRAMEWORK_CHOICES = list(FRAMEWORK_ALIASES.values())
CUDA_CHOICES = list(CUDA_ALIASES.values())


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
