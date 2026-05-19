"""Interactive prompt module for collecting user configuration choices."""

from typing import cast
import questionary

from create_uv_ml.generator import Framework, CudaVersion


FRAMEWORK_CHOICES = [
    "PyTorch",
    "TensorFlow",
    "Basic Scientific Computing (NumPy/Pandas/Scikit-learn)",
]

CUDA_CHOICES = [
    "CUDA 12.1 (Recommended)",
    "CUDA 11.8",
    "CPU Only",
]


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
    """Ask the user to select a CUDA version (only needed for PyTorch)."""
    return cast(
        CudaVersion,
        questionary.select(
            "Which CUDA version do you need?",
            choices=CUDA_CHOICES,
        ).ask(),
    )
