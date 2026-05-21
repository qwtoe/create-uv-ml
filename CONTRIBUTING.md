# Contributing to create-uv-ml

Thank you for your interest in contributing! Here's how to get started.

## Development Setup

1. Fork and clone the repository
2. Install dependencies:
   ```bash
   uv sync --group dev
   source .venv/bin/activate
   ```
3. Install pre-commit hooks (optional but recommended):
   ```bash
   pre-commit install
   ```

## Making Changes

1. Create a feature branch: `git checkout -b my-feature`
2. Make your changes
3. Run checks:
   ```bash
   uv run ruff check .
   uv run pytest -v
   ```
4. Commit with a descriptive message
5. Push and open a Pull Request

## Code Style

- Follow PEP 8 (enforced by ruff)
- Add type annotations to all functions
- Write tests for new features
- Keep the CLI interface simple and intuitive

## Reporting Issues

- Use [GitHub Issues](https://github.com/slowcage/create-uv-ml/issues)
- Include Python version, OS, and steps to reproduce
