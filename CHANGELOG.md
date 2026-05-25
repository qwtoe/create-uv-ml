# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **Review loop**: After all configuration prompts, a summary is shown with the option to "Continue" or redo any specific step (target directory, Python version, CUDA strategy, extra packages, mirror).
- **Skippable installs**: `uv sync` and `uv add` each have a confirm prompt. If `uv sync` is skipped, `uv add` is automatically skipped with a warning and manual commands are printed.
- **Auto-create directory**: If the specified target directory doesn't exist, it is created automatically with a confirmation message.
- **`verify_env.py`**: Generated in the target directory to verify CUDA, PyTorch, and all installed packages. Handles driver errors gracefully (captures both stdout/stderr, catches OSError for broken native libs).
- **Broken driver detection**: Detector now recognizes "Driver/library version mismatch" and "Failed to initialize NVML" errors from nvidia-smi and marks GPU as unavailable.

### Changed

- **Extra packages** redesigned as two-stage flow: Stage 1 picks a preset (None / All / ML Research / Deep Learning / Customize), Stage 2 (only on Customize) confirms each category individually. All ENTER-based, no SPACE toggles.
- `.gitignore` template now includes `verify_env.py` exclusion.
- **`nvidia-smi` error capture**: `verify_env.py` now inspects both stdout and stderr and prints targeted fix suggestions.

## [0.2.0] - 2026-05-23

### Changed

- **Major redesign**: The tool is now an environment-only creator, no longer generates project scaffolding (no `train.py`, `data/`, `src/` layout)
- Replaced `runner.py` with `executor.py`: writes `pyproject.toml` + `.gitignore`, runs `uv sync --python <ver>` then `uv add <extras>`
- Replaced `templates.py` with no replacement: removed all project template generation
- Rewrote `generator.py`: only produces torch+torchvision in `pyproject.toml`; extras installed via `uv add`
- Rewrote `prompts.py`: new flow — target directory, Python version, CUDA strategy, extra packages checkbox, mirror
- Rewrote `main.py`: five-phase pipeline (Detector → Prompts → Generator → Executor → Post-handoff)

### Added

- `detector.py`: auto-detect OS and NVIDIA GPU, recommend CUDA version based on driver
- Python version selection (3.12, 3.11, 3.10, or custom)
- Extra packages menu (transformers, pandas, jupyterlab, etc.)
- Target directory prompt (current directory or custom path)
- Platform-aware activation instructions (Linux vs Windows)

### Removed

- `--framework` CLI option (no longer framework-specific; always PyTorch)
- `--cuda` CLI option (now interactive with auto-detection)
- `--mirror` CLI option (now interactive)
- `--no-sync` CLI option (now confirm-based, can skip each step individually)
- `--template` CLI option (no templates anymore)
- `PROJECT_NAME` positional argument (now prompts for target directory)
- TensorFlow and Scientific Computing framework options
- Project scaffolding: `train.py`, `analysis.py`, `__init__.py`, `README.md`, `data/`

## [0.1.0] - 2026-05-21

### Added

- Interactive CLI with Typer, Questionary, and Rich
- PyTorch support with CUDA 12.1, CUDA 11.8, and CPU-only configurations
- TensorFlow support with GPU (`tensorflow[and-cuda]`) and CPU variants
- Basic Scientific Computing stack (NumPy, Pandas, Matplotlib, Scikit-learn)
- Automatic `pyproject.toml` generation with correct uv index configuration
- Project scaffolding with `.gitignore`, `README.md`, and training script templates
- `--framework`, `--cuda`, `--no-sync`, and `--template` CLI options
- `uv` availability check and project name validation
- `uv sync` timeout handling and graceful error cleanup
- Unit test suite with pytest (46 tests)
- CI pipeline with GitHub Actions
- Pre-commit hooks with ruff
- MIT License