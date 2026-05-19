# create-uv-ml

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![uv](https://img.shields.io/badge/uv-powered-8A2BE2)](https://docs.astral.sh/uv/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An interactive CLI scaffolding tool to quickly bootstrap deep learning projects with [uv](https://docs.astral.sh/uv/). Spend less time on environment setup and more time training models.

## Features

- 🚀 **Powered by uv** — Blazing-fast package management and virtual environment creation
- 🎨 **Interactive Menus** — Arrow-key navigation for selecting frameworks and CUDA versions
- 🔥 **PyTorch Ready** — Supports CUDA 12.1, CUDA 11.8, and CPU-only builds with proper index configuration
- 🧠 **TensorFlow** — One-click TensorFlow project setup
- 📊 **Scientific Computing** — NumPy, Pandas, Matplotlib, and Scikit-learn stack
- 🛠️ **One-Shot Setup** — Generates `pyproject.toml` and installs dependencies automatically

## Installation

```bash
pip install create-uv-ml
# or, if you already use uv
uv tool install create-uv-ml
```

## Quick Start

```bash
create-uv-ml my_new_model
```

You will be prompted to choose a framework and a CUDA version (for PyTorch). The tool then creates the project directory, writes a tailored `pyproject.toml`, and runs `uv sync` to install everything.

### Example Session

```bash
$ create-uv-ml my_new_model
🚀 Welcome to create-uv-ml! Initializing my_new_model...

? Which deep learning framework do you want to use? PyTorch
? Which CUDA version do you need? CUDA 12.1 (Recommended)

⚙️  Configuration: PyTorch + CUDA 12.1 (Recommended)

📝 Generating pyproject.toml...
📁 Creating project directory: my_new_model
  ✓ Wrote my_new_model/pyproject.toml
📦 Running uv sync to install dependencies (this may take a few minutes)...
  ✓ uv sync completed

✅ Project my_new_model created successfully! Happy training!
   cd my_new_model && source .venv/bin/activate
```

## Supported Frameworks

| Framework | Versions / Variants | Key Packages |
|-----------|---------------------|--------------|
| PyTorch | CUDA 12.1, CUDA 11.8, CPU Only | `torch`, `torchvision`, `torchaudio` |
| TensorFlow | Latest stable | `tensorflow` |
| Basic Scientific Computing | — | `numpy`, `pandas`, `matplotlib`, `scikit-learn` |

## Project Structure

When you run `create-uv-ml my_project`, the following structure is generated:

```text
my_project/
├── .venv/                 # uv-managed virtual environment
├── pyproject.toml         # Project metadata, dependencies, and uv indexes
├── uv.lock                # Locked dependency tree
└── README.md              # Project readme
```

The generated `pyproject.toml` includes the correct `[[tool.uv.index]]` and `[tool.uv.sources]` entries for PyTorch CUDA wheels when applicable, so `uv sync` resolves GPU-enabled packages out of the box.

## Development

Clone the repository and install dependencies with uv:

```bash
git clone https://github.com/slowcage/create-uv-ml.git
cd create-uv-ml
uv sync
source .venv/bin/activate
```

Run the CLI locally:

```bash
uv run create-uv-ml my_test_project
```

## Contributing

Contributions are welcome! Feel free to open an issue or submit a pull request.

## License

This project is licensed under the [MIT License](LICENSE).
