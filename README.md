# create-uv-ml

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![uv](https://img.shields.io/badge/uv-powered-8A2BE2)](https://docs.astral.sh/uv/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-46%20passing-brightgreen)]()

An interactive CLI scaffolding tool to quickly bootstrap deep learning projects with [uv](https://docs.astral.sh/uv/). Spend less time on environment setup and more time training models.

## Features

- 🚀 **Powered by uv** — Blazing-fast package management and virtual environment creation
- 🎨 **Interactive Menus** — Arrow-key navigation for selecting frameworks and CUDA versions
- 🔥 **PyTorch Ready** — Supports CUDA 12.1, CUDA 11.8, and CPU-only builds with proper index configuration
- 🧠 **TensorFlow** — GPU (`tensorflow[and-cuda]`) and CPU variants
- 📊 **Scientific Computing** — NumPy, Pandas, Matplotlib, and Scikit-learn stack
- 🛠️ **One-Shot Setup** — Generates `pyproject.toml` and installs dependencies automatically
- 📁 **Project Templates** — Full scaffold with `.gitignore`, `README.md`, training scripts, and `data/` directory
- ⚡ **CLI Options** — `--framework`, `--cuda`, `--no-sync`, `--template` for non-interactive usage

## Prerequisites

- **Python 3.10+** — [Download](https://www.python.org/downloads/) (Windows users: check "Add Python to PATH" during installation)
- **uv** — Fast Python package manager ([why uv?](https://docs.astral.sh/uv/))

Install uv:

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Verify both are available:

```bash
python --version   # Python 3.10.x or higher
uv --version       # uv 0.x.x
```

## Installation

```bash
pip install create-uv-ml
# or, if you already use uv (recommended)
uv tool install create-uv-ml
```

## Quick Start

### Interactive Mode

```bash
create-uv-ml my_new_model
```

You will be prompted to choose a framework and a CUDA version (for PyTorch/TensorFlow). The tool then creates the project directory, writes a tailored `pyproject.toml`, and runs `uv sync` to install everything.

### Non-Interactive Mode

```bash
# PyTorch with CUDA 12.1
create-uv-ml my_model --framework pytorch --cuda cu121

# TensorFlow with GPU
create-uv-ml my_model --framework tensorflow --cuda cu121

# Scientific computing, skip uv sync
create-uv-ml my_model --framework scipy --no-sync

# Minimal template (no training scripts or data directory)
create-uv-ml my_model --framework pytorch --cuda cpu --template minimal
```

### Example Session

```bash
$ create-uv-ml my_new_model
🚀 Welcome to create-uv-ml! Initializing my_new_model...

? Which deep learning framework do you want to use? PyTorch
? Which CUDA version do you need? CUDA 12.1 (Recommended)

⚙️  Configuration: PyTorch + CUDA 12.1 (Recommended)
   Template: full

📝 Generating pyproject.toml...
📁 Creating project directory: my_new_model
  ✓ Wrote my_new_model/pyproject.toml
  ✓ Wrote my_new_model/.gitignore
  ✓ Wrote my_new_model/README.md
  ✓ Wrote my_new_model/src/my_new_model/__init__.py
  ✓ Wrote my_new_model/src/my_new_model/train.py
  ✓ Created my_new_model/data/
📦 Running uv sync to install dependencies (this may take a few minutes)...
  ✓ uv sync completed

✅ Project my_new_model created successfully! Happy training!

Next steps:
  1. cd my_new_model
  2. source .venv/bin/activate
  3. python src/my_new_model/train.py
```

## CLI Options

| Option | Short | Description |
|--------|-------|-------------|
| `--framework` | `-f` | `pytorch`, `tensorflow`, or `scipy` (skips interactive prompt) |
| `--cuda` | `-c` | `cu121`, `cu118`, or `cpu` (skips interactive prompt) |
| `--no-sync` | | Skip `uv sync` (only generate project files) |
| `--template` | `-t` | `minimal` or `full` (default: `full`) |
| `--version` | `-v` | Show the version number |

## Supported Frameworks

| Framework | Versions / Variants | Key Packages |
|-----------|---------------------|--------------|
| PyTorch | CUDA 12.1, CUDA 11.8, CPU Only | `torch`, `torchvision`, `torchaudio` |
| TensorFlow | GPU (`[and-cuda]`), CPU | `tensorflow` |
| Basic Scientific Computing | — | `numpy`, `pandas`, `matplotlib`, `scikit-learn` |

## Project Structure

When you run `create-uv-ml my_project` with the `full` template:

```text
my_project/
├── .gitignore
├── pyproject.toml         # Project metadata, dependencies, and uv indexes
├── uv.lock                # Locked dependency tree
├── README.md              # Project readme
├── data/                  # Place your datasets here
│   └── .gitkeep
└── src/
    └── my_project/
        ├── __init__.py
        └── train.py       # Training entry point (or analysis.py for scipy)
```

The generated `pyproject.toml` includes the correct `[[tool.uv.index]]` and `[tool.uv.sources]` entries for PyTorch CUDA wheels when applicable, so `uv sync` resolves GPU-enabled packages out of the box.

## After Creation

Once the project is created, enter the directory and activate the environment:

```bash
cd my_project
source .venv/bin/activate        # macOS / Linux
# or
.venv\Scripts\activate           # Windows
```

If you used `--no-sync`, install dependencies first:

```bash
uv sync
```

Run the training script (included in the `full` template):

```bash
python src/my_project/train.py
```

The generated `train.py` contains a complete, runnable training loop — just replace the dummy data and model with your own.

## FAQ

| Issue | Solution |
|-------|----------|
| `uv: command not found` | uv is not installed or not in PATH — reinstall uv following the Prerequisites section |
| `Error: 'my-project' is not a valid Python package name` | Use only letters, digits, and underscores (e.g. `my_project`, not `my-project`) |
| Dependency download is slow | PyTorch CUDA wheels are ~2 GB; use `--no-sync` to skip, then run `uv sync` manually later |
| No NVIDIA GPU available | Select `CPU Only` when prompted, or use `--cuda cpu` — everything works, just slower training |
| Want to generate files without installing packages | Add `--no-sync` to skip `uv sync` |
| `uv sync` timed out | Re-run `uv sync` inside the project directory; large downloads may need retries |

## Development

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for version history.

## License

This project is licensed under the [MIT License](LICENSE).
