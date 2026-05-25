# AGENTS.md — create-uv-ml

Compact reference for OpenCode sessions working in this repo.

## Project

`create-uv-ml` is a CLI tool (Typer + Questionary + Rich) that sets up isolated deep learning Python environments using `uv`. It is **not** a library — it is a standalone CLI distributed as a Python package.

The tool runs a five-phase pipeline: **Detector → Prompts → Generator → Executor → Post-handoff**.

## Developer Commands

All commands go through `uv`. Do not use `pip` or `python -m` directly.

```bash
# Setup
uv sync --group dev

# Lint / format
uv run ruff check .
uv run ruff format .

# Type check
uv run mypy src/create_uv_ml

# Test
uv run pytest -v

# Run CLI locally
uv run create-uv-ml --help
uv run python -m create_uv_ml --help
```

CI order: `ruff check .` → `pytest -v`. There is no separate type-check step in CI, but run `mypy` before opening a PR.

## Source Layout

- Package root: `src/create_uv_ml/`
- Build backend: `hatchling` (see `tool.hatch.build.targets.wheel` in `pyproject.toml`)
- Entrypoint: `create-uv-ml = "create_uv_ml.main:app"`
- Also runnable as: `python -m create_uv_ml` (`__main__.py` delegates to `main.app`)

## Architecture

| Module | Responsibility |
|--------|---------------|
| `main.py` | Typer CLI entrypoint, five-phase pipeline orchestration. Phase 2 includes a review loop that lets the user go back and change any configuration step. Phase 4 asks confirm prompts before `uv sync` and `uv add` (each can be skipped). |
| `detector.py` | Phase 1: OS detection (`sys.platform`), NVIDIA GPU detection (`nvidia-smi`), CUDA recommendation. Detects broken drivers ("Driver/library version mismatch") and marks GPU as unavailable. |
| `prompts.py` | Phase 2: Interactive Questionary prompts — target dir, Python version, CUDA strategy, extra packages (two-stage: preset select → per-category confirm), mirror. |
| `generator.py` | Phase 3: Assemble `pyproject.toml` text from CUDA/mirror choices; core deps only (torch, torchvision). Also auto-creates target directory if missing. |
| `executor.py` | Phase 4: Write `pyproject.toml` + `.gitignore` + `verify_env.py`, run `uv sync --python <ver>` then `uv add <extras>`. Contains the `verify_env.py` template as an embedded string constant. |
| `__main__.py` | `python -m create_uv_ml` support |

Key types (Literal): `CudaVersion`, `MirrorSource`. Alias maps (`CUDA_ALIASES`, `MIRROR_ALIASES`) bridge short names to display names.

**Deleted modules** (from v0.1.x): `templates.py` (no longer generates train.py/data/), `runner.py` (replaced by `executor.py`).

## Testing

- Framework: pytest with `pytest-mock` (64 tests)
- Config: `pythonpath = ["src"]` in `pyproject.toml`
- Tests live in `tests/` and import from `create_uv_ml.*` directly
- Executor tests mock `subprocess.Popen` to avoid real `uv sync` calls
- Detector tests mock `shutil.which`, `subprocess.run`, and `os.path.exists`
- Prompts tests mock `questionary.select`, `.confirm`, `.text`, and `.checkbox`
- Generator tests assert on string contents of generated TOML

## Toolchain Quirks

- **Ruff**: `target-version = "py310"`, `line-length = 99`. Selected rules: `E`, `F`, `I`, `UP`, `B`, `SIM`.
- **mypy**: `strict = true`, `python_version = "3.10"`.
- **Pre-commit**: `ruff-pre-commit` v0.11.0 + generic hooks (trailing-whitespace, EOF-fixer, check-yaml, check-added-large-files).
- **CI matrix**: Python 3.10, 3.11, 3.12 on `ubuntu-latest`.

## Domain Logic Gotchas

- **PyTorch requires-python bound**: PyTorch CUDA wheels lack cp313, so generated `pyproject.toml` uses `>=3.10,<3.13` for CUDA. CPU-only relaxes this to `>=3.10`.
- **PyTorch index config**: The tool writes an explicit `[[tool.uv.index]]` for PyTorch (cu121 / cu118 / cpu) plus `[tool.uv.sources]` entries with a platform marker (`sys_platform == 'linux' or sys_platform == 'win32'`). macOS falls back to standard PyPI wheels.
- **Two-step install**: Core deps (torch, torchvision) go into `pyproject.toml` and are installed via `uv sync`. Extra packages are installed via `uv add` afterwards so uv's resolver can solve against the locked base. Each step can be skipped via a confirm prompt; if `uv sync` is skipped, `uv add` is automatically skipped too with a warning.
- **GPU detection**: Uses `nvidia-smi --query-gpu=driver_version` to get driver version. Driver ≥525 → cu121, ≥450 → cu118, <450 → too old. If nvidia-smi exists but returns a driver-level error (e.g. "Driver/library version mismatch"), GPU is marked as unavailable and the user is warned.
- **macOS**: CUDA selection is skipped entirely (`ask_cuda_strategy` returns `None`).
- **Mirrors**: Supports `tsinghua` and `aliyun` as `default = true` indexes replacing PyPI. The PyTorch explicit index and the mirror default index can coexist.
- **No project scaffolding**: The tool only creates `pyproject.toml`, `.gitignore`, and `verify_env.py`. No `train.py`, no `data/`, no `src/` layout.
- **Review loop**: After all 5 prompts, the user sees a summary and can choose to "Continue" or redo any specific step. The loop repeats until "Continue" is selected.
- **Extra packages**: Two-stage design — Stage 1 picks a preset (None / All / ML Research / Deep Learning), Stage 2 (only on "Customize") confirms each category individually. Uses `questionary.select` and `questionary.confirm` (ENTER-only, no SPACE toggles).
- **Directory creation**: If the specified target directory doesn't exist, it is created automatically during Phase 3 with a confirmation message.