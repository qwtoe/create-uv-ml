"""Generator module that produces pyproject.toml content based on user choices."""

from typing import Literal

Framework = Literal["PyTorch", "TensorFlow", "Basic Scientific Computing (NumPy/Pandas/Scikit-learn)"]
CudaVersion = Literal["CUDA 12.1 (Recommended)", "CUDA 11.8", "CPU Only"]


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


def generate_pyproject(project_name: str, framework: Framework, cuda: CudaVersion | None) -> str:
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
        deps.append("tensorflow>=2.16.0")
    else:
        deps.extend([
            "numpy>=1.26.0",
            "pandas>=2.2.0",
            "matplotlib>=3.8.0",
            "scikit-learn>=1.4.0",
        ])

    deps_str = "\n".join(f'    "{d}",' for d in deps)

    toml = f"""[project]
name = "{project_name}"
version = "0.1.0"
description = "Add your description here"
readme = "README.md"
requires-python = ">=3.10,<3.13"
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
