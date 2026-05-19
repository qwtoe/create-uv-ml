# create-uv-ml

A CLI scaffolding tool to quickly bootstrap deep learning projects with [uv](https://docs.astral.sh/uv/).

## Features

- 🚀 Powered by **uv** for blazing-fast package management
- 🎨 Interactive terminal menus (arrow-key navigation)
- 🔥 Supports PyTorch (multiple CUDA versions), TensorFlow, and basic scientific computing stacks
- 🛠️ Generates configuration and installs dependencies in one shot

## Installation

```bash
pip install create-uv-ml
# or
uv tool install create-uv-ml
```

## Usage

```bash
create-uv-ml my_new_model
```

Follow the interactive prompts to select your framework and CUDA version.

## Development

```bash
git clone https://github.com/slowcage/create-uv-ml.git
cd create-uv-ml
uv sync
source .venv/bin/activate
uv run create-uv-ml my_test_project
```

## Tech Stack

- [Typer](https://typer.tiangolo.com/) — CLI parsing
- [Questionary](https://questionary.readthedocs.io/) — Interactive prompts
- [Rich](https://rich.readthedocs.io/) — Terminal styling
