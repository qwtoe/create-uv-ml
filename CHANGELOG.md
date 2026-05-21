# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
