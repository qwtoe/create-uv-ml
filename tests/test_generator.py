"""Tests for the generator module."""

from create_uv_ml.generator import (
    CUDA_ALIASES,
    MIRROR_ALIASES,
    generate_pyproject,
)


class TestGeneratePyproject:
    """Tests for generate_pyproject()."""

    def test_cu121_generates_index_and_sources(self) -> None:
        result = generate_pyproject("my_proj", "3.12", "CUDA 12.1 (Recommended)")
        assert 'name = "my_proj"' in result
        assert "torch>=2.3.0" in result
        assert "torchvision>=0.18.0" in result
        assert "pytorch-cu121" in result
        assert "https://download.pytorch.org/whl/cu121" in result
        assert "explicit = true" in result
        assert "[tool.uv.sources]" in result
        assert 'index = "pytorch-cu121"' in result
        assert 'requires-python = ">=3.10,<3.13"' in result

    def test_cu118_generates_correct_index(self) -> None:
        result = generate_pyproject("my_proj", "3.11", "CUDA 11.8")
        assert "pytorch-cu118" in result
        assert "https://download.pytorch.org/whl/cu118" in result
        assert 'requires-python = ">=3.10,<3.13"' in result

    def test_cpu_only_generates_cpu_index(self) -> None:
        result = generate_pyproject("my_proj", "3.12", "CPU Only")
        assert "pytorch-cpu" in result
        assert "https://download.pytorch.org/whl/cpu" in result
        assert 'requires-python = ">=3.10"' in result
        # CPU sources should NOT have platform markers
        assert "sys_platform" not in result

    def test_no_cuda_no_index(self) -> None:
        result = generate_pyproject("my_proj", "3.12", None)
        assert "torch>=2.3.0" in result
        assert "[[tool.uv.index]]" not in result
        assert "[tool.uv.sources]" not in result
        assert 'requires-python = ">=3.10"' in result

    def test_project_name_uses_basename(self) -> None:
        result = generate_pyproject("/path/to/my_proj", "3.12", None)
        assert 'name = "my_proj"' in result

    def test_version_in_output(self) -> None:
        result = generate_pyproject("my_proj", "3.12", None)
        assert 'version = "0.1.0"' in result

    def test_cuda_sources_have_platform_marker(self) -> None:
        result = generate_pyproject("my_proj", "3.12", "CUDA 12.1 (Recommended)")
        # CUDA sources should have platform markers for torch and torchvision
        assert "sys_platform" in result

    def test_cpu_sources_no_platform_marker(self) -> None:
        result = generate_pyproject("my_proj", "3.12", "CPU Only")
        assert "sys_platform" not in result


class TestMirrorSource:
    """Tests for mirror source configuration."""

    def test_default_mirror_no_index(self) -> None:
        result = generate_pyproject("my_proj", "3.12", None, "Default (PyPI)")
        assert "tsinghua" not in result
        assert "aliyun" not in result
        assert "default = true" not in result

    def test_tsinghua_mirror(self) -> None:
        result = generate_pyproject("my_proj", "3.12", None, "Tsinghua (China)")
        assert "tsinghua" in result
        assert "https://pypi.tuna.tsinghua.edu.cn/simple" in result
        assert "default = true" in result

    def test_aliyun_mirror(self) -> None:
        result = generate_pyproject("my_proj", "3.12", None, "Aliyun (China)")
        assert "aliyun" in result
        assert "https://mirrors.aliyun.com/pypi/simple" in result
        assert "default = true" in result

    def test_mirror_with_pytorch_cuda(self) -> None:
        result = generate_pyproject(
            "my_proj", "3.12", "CUDA 12.1 (Recommended)", "Tsinghua (China)"
        )
        # Both PyTorch explicit index and Tsinghua default index should be present
        assert "pytorch-cu121" in result
        assert "explicit = true" in result
        assert "tsinghua" in result
        assert "default = true" in result


class TestAliases:
    """Tests for alias mappings."""

    def test_cuda_aliases_cover_all_versions(self) -> None:
        for _alias, name in CUDA_ALIASES.items():
            assert name in ("CUDA 12.1 (Recommended)", "CUDA 11.8", "CPU Only")

    def test_mirror_aliases_cover_all_mirrors(self) -> None:
        for _alias, name in MIRROR_ALIASES.items():
            assert name in ("Default (PyPI)", "Tsinghua (China)", "Aliyun (China)")
