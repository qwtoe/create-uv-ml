"""Tests for the generator module."""

from create_uv_ml.generator import (
    CUDA_ALIASES,
    FRAMEWORK_ALIASES,
    REQUIRES_PYTHON_MAP,
    generate_pyproject,
)


class TestGeneratePyproject:
    """Tests for generate_pyproject()."""

    def test_pytorch_cu121(self) -> None:
        result = generate_pyproject("my_proj", "PyTorch", "CUDA 12.1 (Recommended)")
        assert 'name = "my_proj"' in result
        assert "torch>=2.3.0" in result
        assert "torchvision>=0.18.0" in result
        assert "torchaudio>=2.3.0" in result
        assert "pytorch-cu121" in result
        assert "https://download.pytorch.org/whl/cu121" in result
        assert "explicit = true" in result
        assert "[tool.uv.sources]" in result
        assert 'index = "pytorch-cu121"' in result
        assert 'requires-python = ">=3.10,<3.13"' in result

    def test_pytorch_cu118(self) -> None:
        result = generate_pyproject("my_proj", "PyTorch", "CUDA 11.8")
        assert "pytorch-cu118" in result
        assert "https://download.pytorch.org/whl/cu118" in result
        assert 'requires-python = ">=3.10,<3.13"' in result

    def test_pytorch_cpu(self) -> None:
        result = generate_pyproject("my_proj", "PyTorch", "CPU Only")
        assert "pytorch-cpu" in result
        assert "https://download.pytorch.org/whl/cpu" in result
        assert 'requires-python = ">=3.10"' in result

    def test_pytorch_no_cuda(self) -> None:
        result = generate_pyproject("my_proj", "PyTorch", None)
        assert "torch>=2.3.0" in result
        assert "[tool.uv.sources]" not in result
        assert "[[tool.uv.index]]" not in result

    def test_tensorflow_cpu(self) -> None:
        result = generate_pyproject("my_proj", "TensorFlow", "CPU Only")
        assert "tensorflow>=2.16.0" in result
        assert "tensorflow[and-cuda]" not in result
        assert 'requires-python = ">=3.10"' in result

    def test_tensorflow_gpu(self) -> None:
        result = generate_pyproject("my_proj", "TensorFlow", "CUDA 12.1 (Recommended)")
        assert "tensorflow[and-cuda]>=2.16.0" in result
        assert 'requires-python = ">=3.10"' in result

    def test_tensorflow_no_cuda(self) -> None:
        result = generate_pyproject("my_proj", "TensorFlow", None)
        assert "tensorflow>=2.16.0" in result

    def test_scipy(self) -> None:
        result = generate_pyproject(
            "my_proj",
            "Basic Scientific Computing (NumPy/Pandas/Scikit-learn)",
            None,
        )
        assert "numpy>=1.26.0" in result
        assert "pandas>=2.2.0" in result
        assert "matplotlib>=3.8.0" in result
        assert "scikit-learn>=1.4.0" in result
        assert 'requires-python = ">=3.10"' in result

    def test_project_name_in_output(self) -> None:
        result = generate_pyproject("cool_project", "PyTorch", "CPU Only")
        assert 'name = "cool_project"' in result

    def test_version_in_output(self) -> None:
        result = generate_pyproject("my_proj", "PyTorch", "CPU Only")
        assert 'version = "0.1.0"' in result


class TestAliases:
    """Tests for alias mappings."""

    def test_framework_aliases_cover_all_frameworks(self) -> None:
        for _alias, name in FRAMEWORK_ALIASES.items():
            assert name in (
                "PyTorch",
                "TensorFlow",
                "Basic Scientific Computing (NumPy/Pandas/Scikit-learn)",
            )

    def test_cuda_aliases_cover_all_versions(self) -> None:
        for _alias, name in CUDA_ALIASES.items():
            assert name in ("CUDA 12.1 (Recommended)", "CUDA 11.8", "CPU Only")

    def test_requires_python_map_covers_all_frameworks(self) -> None:
        for fw in (
            "PyTorch",
            "TensorFlow",
            "Basic Scientific Computing (NumPy/Pandas/Scikit-learn)",
        ):
            assert fw in REQUIRES_PYTHON_MAP
