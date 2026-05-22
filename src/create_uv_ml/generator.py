"""Generator module that produces pyproject.toml content based on user choices."""

import os
from typing import Literal

Framework = Literal[
    "PyTorch",
    "TensorFlow",
    "Basic Scientific Computing (NumPy/Pandas/Scikit-learn)",
]
CudaVersion = Literal["CUDA 12.1 (Recommended)", "CUDA 11.8", "CPU Only"]
MirrorSource = Literal["Default (PyPI)", "Tsinghua (China)", "Aliyun (China)"]

# Short-name aliases for CLI --framework, --cuda, and --mirror options
FRAMEWORK_ALIASES: dict[str, Framework] = {
    "pytorch": "PyTorch",
    "tensorflow": "TensorFlow",
    "scipy": "Basic Scientific Computing (NumPy/Pandas/Scikit-learn)",
}

CUDA_ALIASES: dict[str, CudaVersion] = {
    "cu121": "CUDA 12.1 (Recommended)",
    "cu118": "CUDA 11.8",
    "cpu": "CPU Only",
}

MIRROR_ALIASES: dict[str, MirrorSource] = {
    "default": "Default (PyPI)",
    "tsinghua": "Tsinghua (China)",
    "aliyun": "Aliyun (China)",
}

# Dynamic requires-python based on framework (PyTorch CUDA wheels lack cp313)
REQUIRES_PYTHON_MAP: dict[Framework, str] = {
    "PyTorch": ">=3.10,<3.13",
    "TensorFlow": ">=3.10",
    "Basic Scientific Computing (NumPy/Pandas/Scikit-learn)": ">=3.10",
}

CUDA_INDEX_MAP = {
    "CUDA 12.1 (Recommended)": {
        "name": "pytorch-cu121",
        "url": "https://download.pytorch.org/whl/cu121",
    },
    "CUDA 11.8": {
        "name": "pytorch-cu118",
        "url": "https://download.pytorch.org/whl/cu118",
    },
    "CPU Only": {
        "name": "pytorch-cpu",
        "url": "https://download.pytorch.org/whl/cpu",
    },
}

# Mirror configurations: default PyPI index replacement
MIRROR_CONFIG: dict[MirrorSource, dict[str, str]] = {
    "Default (PyPI)": {},
    "Tsinghua (China)": {
        "name": "tsinghua",
        "url": "https://pypi.tuna.tsinghua.edu.cn/simple",
    },
    "Aliyun (China)": {
        "name": "aliyun",
        "url": "https://mirrors.aliyun.com/pypi/simple",
    },
}


def generate_pyproject(
    project_name: str,
    framework: Framework,
    cuda: CudaVersion | None,
    mirror: MirrorSource = "Default (PyPI)",
) -> str:
    """Generate a pyproject.toml string based on user selections."""

    deps: list[str] = []
    index_sections: list[str] = []
    sources_sections: list[str] = []

    if framework == "PyTorch":
        deps.extend(["torch>=2.3.0", "torchvision>=0.18.0", "torchaudio>=2.3.0"])

        if cuda:
            idx = CUDA_INDEX_MAP[cuda]
            index_sections.append(f"""[[tool.uv.index]]
name = "{idx['name']}"
url = "{idx['url']}"
explicit = true""")

            marker = 'marker = "sys_platform == \'linux\' or sys_platform == \'win32\'"'
            for pkg in ("torch", "torchvision", "torchaudio"):
                if cuda == "CPU Only":
                    sources_sections.append(f'{pkg} = [{{ index = "{idx["name"]}" }}]')
                else:
                    sources_sections.append(f'{pkg} = [{{ index = "{idx["name"]}", {marker} }}]')

    elif framework == "TensorFlow":
        if cuda and cuda != "CPU Only":
            deps.append("tensorflow[and-cuda]>=2.16.0")
        else:
            deps.append("tensorflow>=2.16.0")
    else:
        deps.extend([
            "numpy>=1.26.0",
            "pandas>=2.2.0",
            "matplotlib>=3.8.0",
            "scikit-learn>=1.4.0",
        ])

    # Dynamic requires-python: relax upper bound for PyTorch CPU-only
    requires_python = REQUIRES_PYTHON_MAP[framework]
    if framework == "PyTorch" and cuda == "CPU Only":
        requires_python = ">=3.10"

    # Add mirror index (replaces default PyPI)
    mirror_cfg = MIRROR_CONFIG[mirror]
    if mirror_cfg:
        index_sections.append(f"""[[tool.uv.index]]
name = "{mirror_cfg['name']}"
url = "{mirror_cfg['url']}"
default = true""")

    deps_str = "\n".join(f'    "{d}",' for d in deps)

    toml = f"""[project]
name = "{os.path.basename(project_name)}"
version = "0.1.0"
description = "Add your description here"
readme = "README.md"
requires-python = "{requires_python}"
dependencies = [
{deps_str}
]
"""

    if index_sections:
        toml += "\n" + "\n\n".join(index_sections) + "\n"

    if sources_sections:
        toml += "\n[tool.uv.sources]\n"
        toml += "\n".join(sources_sections) + "\n"

    return toml
