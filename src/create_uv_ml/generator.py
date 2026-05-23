"""Generator module that produces pyproject.toml content based on user choices.

Phase 3 of the pipeline: takes collected configuration and assembles a
conflict-free pyproject.toml. Only core dependencies (torch, torchvision)
are written into the file; extras are installed later via ``uv add``
so that uv's resolver can solve them against the already-locked base.
"""

import os
from typing import Literal

CudaVersion = Literal["CUDA 12.1 (Recommended)", "CUDA 11.8", "CPU Only"]
MirrorSource = Literal["Default (PyPI)", "Tsinghua (China)", "Aliyun (China)"]

# Short-name aliases for CLI --cuda and --mirror options
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

CUDA_INDEX_MAP: dict[CudaVersion, dict[str, str]] = {
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
    python_version: str,
    cuda: CudaVersion | None,
    mirror: MirrorSource = "Default (PyPI)",
) -> str:
    """Generate a pyproject.toml string based on user selections.

    Core dependencies (torch, torchvision) are written into the file so
    that ``uv sync`` can lock them first.  Additional packages should be
    installed afterwards via ``uv add`` for best dependency resolution.
    """
    deps: list[str] = ["torch>=2.3.0", "torchvision>=0.18.0"]
    index_sections: list[str] = []
    sources_sections: list[str] = []

    # PyTorch CUDA index configuration
    if cuda:
        idx = CUDA_INDEX_MAP[cuda]
        index_sections.append(
            f'[[tool.uv.index]]\nname = "{idx["name"]}"\nurl = "{idx["url"]}"\nexplicit = true'
        )

        marker = "marker = \"sys_platform == 'linux' or sys_platform == 'win32'\""
        for pkg in ("torch", "torchvision"):
            if cuda == "CPU Only":
                sources_sections.append(f'{pkg} = [{{ index = "{idx["name"]}" }}]')
            else:
                sources_sections.append(f'{pkg} = [{{ index = "{idx["name"]}", {marker} }}]')

    # Mirror configuration (replaces default PyPI)
    mirror_cfg = MIRROR_CONFIG[mirror]
    if mirror_cfg:
        index_sections.append(
            f'[[tool.uv.index]]\nname = "{mirror_cfg["name"]}"\n'
            f'url = "{mirror_cfg["url"]}"\ndefault = true'
        )

    # PyTorch CUDA wheels lack cp313, so cap at <3.13 for CUDA
    requires_python = ">=3.10,<3.13" if cuda and cuda != "CPU Only" else ">=3.10"

    deps_str = "\n".join(f'    "{d}",' for d in deps)

    toml = (
        f"[project]\n"
        f'name = "{os.path.basename(project_name)}"\n'
        f'version = "0.1.0"\n'
        f'description = "Add your description here"\n'
        f'readme = "README.md"\n'
        f'requires-python = "{requires_python}"\n'
        f"dependencies = [\n"
        f"{deps_str}\n"
        f"]\n"
    )

    if index_sections:
        toml += "\n" + "\n\n".join(index_sections) + "\n"

    if sources_sections:
        toml += "\n[tool.uv.sources]\n" + "\n".join(sources_sections) + "\n"

    return toml
