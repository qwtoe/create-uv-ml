# create-uv-ml

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![uv](https://img.shields.io/badge/uv-powered-8A2BE2)](https://docs.astral.sh/uv/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-64%20passing-brightgreen)]()

An interactive CLI tool to quickly set up isolated deep learning Python environments with [uv](https://docs.astral.sh/uv/). Auto-detects your GPU, lets you pick Python version and extra packages — one command and you're ready to code.

## Features

- 🔍 **GPU Auto-Detection** — Detects NVIDIA driver and recommends the best CUDA version
- 🐍 **Python Version Selection** — Pick from 3.12, 3.11, 3.10, or enter a custom version
- 🔥 **PyTorch Ready** — CUDA 12.1, CUDA 11.8, and CPU-only with correct index configuration
- 📦 **Extra Packages** — Presets (None / All / ML Research / Deep Learning) or per-category customization
- 🔄 **Review & Revise** — After configuration, go back and change any step before installing
- ⏭️ **Skippable Installs** — Skip `uv sync` or `uv add` if you need more control
- 🛡️ **Conflict-Free Installs** — Core deps locked first via `uv sync`, extras added via `uv add`
- 📁 **Auto-Create Directory** — Specified directory is created if it doesn't exist
- 🔍 **Post-Install Verification** — Generated `verify_env.py` checks CUDA, torch, and all packages
- 🪞 **Mirror Support** — Tsinghua and Aliyun mirrors for users in China

## Prerequisites

- **Python 3.10+** — [Download](https://www.python.org/downloads/)
- **uv** — Fast Python package manager ([why uv?](https://docs.astral.sh/uv/))

Install uv:

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

## Installation

```bash
pip install create-uv-ml
# or, if you already use uv (recommended)
uv tool install create-uv-ml
```

## Quick Start

```bash
create-uv-ml
```

You'll walk through an interactive setup:

```
🔍 Detecting system hardware...
  ✓ NVIDIA GPU detected (driver 535.129.03)
  ✓ Recommended: cu121

📋 Configuration
? Create environment in current directory? (/home/user/project) [Y/n]
? Select Python version: ❯ 3.12  / 3.11  / 3.10  / Other
? Select CUDA strategy: ❯ CUDA 12.1 (Recommended)  / CUDA 11.8  / CPU Only
? Select additional packages:
  ❯ None (core PyTorch only)
    All packages (17 packages)
    ML Research (numpy, pandas, matplotlib, scikit-learn, scipy, jupyterlab)
    Deep Learning (transformers, datasets, accelerate, tensorboard, torchaudio)
    Customize... (pick per category)

? Select PyPI mirror: Default (PyPI) / Tsinghua / Aliyun

⚙️  Configuration: Python 3.12 | CUDA 12.1 (Recommended) | Mirror: Default (PyPI)
   Target: /home/user/project

? What would you like to do?
  ❯ Continue (create environment)
    Change target directory
    Change Python version
    Change CUDA strategy
    Change extra packages
    Change mirror

📝 Generating pyproject.toml...
? Install core dependencies (torch, torchvision) now? [Y/n]

📦 Running uv sync to install core dependencies...
  ✓ uv sync completed

? Install extra packages (numpy, pandas, ...) now? [Y/n]
📦 Installing additional packages: numpy, pandas...
  ✓ Additional packages installed

✅ Environment created successfully!

Next steps:
  1. source /home/user/project/.venv/bin/activate
  2. python verify_env.py    # Verify CUDA, torch, and packages
  3. Start coding! Create .py files or Jupyter notebooks.
```

## How It Works

The tool runs a five-phase pipeline:

1. **Detector** — Sniffs OS type and NVIDIA GPU hardware, recommends a CUDA version
2. **Prompts** — Interactive questionnaire: target directory, Python version, CUDA strategy, extra packages (preset + per-category), mirror. Includes a review step to go back and change any selection.
3. **Generator** — Assembles a `pyproject.toml` with correct PyTorch index configuration. Creates the target directory if it doesn't exist. Writes `.gitignore` and `verify_env.py`.
4. **Executor** — Runs `uv sync --python <version>` then `uv add <extras>` for conflict-free resolution. Each step can be skipped with a confirmation prompt.
5. **Post-handoff** — Prints platform-specific activation instructions and verification script usage.

### Why `uv sync` then `uv add`?

Core dependencies (torch, torchvision) are written into `pyproject.toml` first. `uv sync` locks and installs them. Then `uv add` installs extra packages with the resolver aware of the already-locked base — this prevents version conflicts that can break CUDA wheels.

## CLI Options

| Option | Short | Description |
|--------|-------|-------------|
| `--version` | `-v` | Show the version number |

The tool is fully interactive. Just run `create-uv-ml` and follow the prompts.

## Supported CUDA Versions

| CUDA | Driver Requirement | Notes |
|------|-------------------|-------|
| CUDA 12.1 | NVIDIA driver ≥ 525.x | Recommended for modern GPUs |
| CUDA 11.8 | NVIDIA driver ≥ 450.x | For older hardware |
| CPU Only | — | No GPU required |

On macOS, CUDA selection is skipped automatically (no NVIDIA support).

If the NVIDIA driver is broken (e.g. "Driver/library version mismatch"), the detector will warn you and suggest rebooting or reinstalling the driver.

## Available Extra Packages

| Category | Packages |
|----------|----------|
| Deep Learning | transformers, datasets, accelerate, tensorboard, torchaudio |
| Scientific Computing | numpy, pandas, matplotlib, scikit-learn, scipy, jupyterlab |
| Computer Vision | opencv-python, Pillow |
| Utilities | tqdm, rich, requests, pydantic |

### Package selection presets

| Preset | Includes |
|--------|----------|
| **None** | Only core PyTorch (torch, torchvision) |
| **All packages** | All 17 packages across all categories |
| **ML Research** | numpy, pandas, matplotlib, scikit-learn, scipy, jupyterlab |
| **Deep Learning** | transformers, datasets, accelerate, tensorboard, torchaudio |
| **Customize...** | Pick per category (4 y/n prompts) |

## FAQ

| Issue | Solution |
|-------|----------|
| `uv: command not found` | uv is not installed or not in PATH — reinstall following the Prerequisites section |
| Dependency download is slow | PyTorch CUDA wheels are ~2 GB; consider using a mirror (Tsinghua/Aliyun) |
| No NVIDIA GPU detected | Select `CPU Only` when prompted — everything works, just slower training |
| `nvidia-smi` returns an error | Try `sudo reboot`; if still broken, reinstall the NVIDIA driver |
| `uv sync` failed | Re-run `uv sync` inside the target directory; large downloads may need retries |
| Want to add more packages later | Run `uv add <package>` inside the target directory |
| Want to skip some install steps | Answer `N` when asked "Install ... now?" — manual commands are printed |
| `torchaudio` import fails with `libcudart` error | Install CUDA runtime: `sudo apt install nvidia-cuda-toolkit` |
| `verify_env.py` crashes | Update to the latest version or re-create the environment |

## Development

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for version history.

## License

This project is licensed under the [MIT License](LICENSE).