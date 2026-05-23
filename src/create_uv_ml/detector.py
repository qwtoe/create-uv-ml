"""Detector module for OS and GPU hardware sniffing.

Runs silently at startup to provide default recommendations for the
interactive prompts (Phase 1 of the pipeline).
"""

import os
import shutil
import subprocess
import sys
from dataclasses import dataclass


@dataclass
class DetectionResult:
    """Aggregated result of system hardware detection."""

    os_type: str  # "linux", "win32", "darwin"
    has_nvidia: bool
    driver_version: str | None
    recommended_cuda: str | None  # "cu121", "cu118", or None


def detect_os() -> str:
    """Detect the operating system type."""
    return sys.platform


def find_nvidia_smi() -> str | None:
    """Find the nvidia-smi executable path.

    Searches PATH first, then falls back to common install locations
    on both Linux and Windows.
    """
    # Check PATH first
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi:
        return nvidia_smi

    # Check common hard-coded paths
    if sys.platform == "win32":
        common_paths = [
            r"C:\Windows\System32\nvidia-smi.exe",
            r"C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe",
        ]
    else:
        common_paths = [
            "/usr/bin/nvidia-smi",
            "/usr/local/nvidia/bin/nvidia-smi",
        ]

    for path in common_paths:
        if os.path.exists(path):
            return path

    return None


def detect_nvidia_gpu() -> tuple[bool, str | None]:
    """Detect NVIDIA GPU and return (has_nvidia, driver_version).

    Uses nvidia-smi to query the driver version. Returns (True, version)
    if a GPU is found, (True, None) if nvidia-smi exists but version
    cannot be determined, and (False, None) if no NVIDIA setup is found.
    """
    nvidia_smi = find_nvidia_smi()
    if not nvidia_smi:
        return False, None

    try:
        result = subprocess.run(
            [nvidia_smi, "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0 and result.stdout.strip():
            version = result.stdout.strip().split("\n")[0].strip()
            return True, version
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        pass

    # nvidia-smi exists but couldn't get version
    return True, None


def get_recommended_cuda(driver_version: str | None) -> str | None:
    """Map NVIDIA driver version to recommended CUDA version.

    Returns one of: "cu121", "cu118", or None (driver too old).
    Based on NVIDIA CUDA compatibility tables:
      - Driver >= 525.x → CUDA 12.1
      - Driver >= 450.x → CUDA 11.8
      - Driver < 450 → too old
    """
    if driver_version is None:
        # Can't determine driver version; assume recent enough for cu121
        return "cu121"

    try:
        parts = driver_version.split(".")
        major = int(parts[0])
    except (ValueError, IndexError):
        return "cu121"

    if major >= 525:
        return "cu121"
    elif major >= 450:
        return "cu118"
    else:
        return None  # Driver too old for supported CUDA versions


def detect() -> DetectionResult:
    """Run full system detection and return aggregated results."""
    os_type = detect_os()
    has_nvidia, driver_version = detect_nvidia_gpu()
    recommended_cuda = get_recommended_cuda(driver_version) if has_nvidia else None

    return DetectionResult(
        os_type=os_type,
        has_nvidia=has_nvidia,
        driver_version=driver_version,
        recommended_cuda=recommended_cuda,
    )
