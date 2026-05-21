"""Tests for the templates module."""

from create_uv_ml.templates import (
    generate_gitignore,
    generate_init_py,
    generate_readme,
    generate_train_py,
)


class TestGenerateGitignore:
    def test_contains_python_patterns(self) -> None:
        result = generate_gitignore()
        assert "__pycache__/" in result
        assert "*.py[cod]" in result
        assert ".venv/" in result

    def test_contains_data_science_patterns(self) -> None:
        result = generate_gitignore()
        assert "data/" in result
        assert "*.pt" in result
        assert "*.ckpt" in result

    def test_contains_ide_patterns(self) -> None:
        result = generate_gitignore()
        assert ".idea/" in result
        assert ".vscode/" in result


class TestGenerateReadme:
    def test_contains_project_name(self) -> None:
        result = generate_readme("my_proj", "PyTorch", "CUDA 12.1 (Recommended)")
        assert "# my_proj" in result

    def test_contains_framework_info(self) -> None:
        result = generate_readme("my_proj", "PyTorch", "CUDA 12.1 (Recommended)")
        assert "PyTorch" in result
        assert "CUDA 12.1" in result

    def test_no_cuda(self) -> None:
        result = generate_readme("my_proj", "TensorFlow", None)
        assert "TensorFlow" in result


class TestGenerateTrainPy:
    def test_pytorch_template(self) -> None:
        result = generate_train_py("my_proj", "PyTorch")
        assert "torch" in result
        assert "cuda" in result
        assert "nn.Module" in result

    def test_tensorflow_template(self) -> None:
        result = generate_train_py("my_proj", "TensorFlow")
        assert "tensorflow" in result
        assert "tf.keras" in result

    def test_scipy_template(self) -> None:
        result = generate_train_py(
            "my_proj", "Basic Scientific Computing (NumPy/Pandas/Scikit-learn)"
        )
        assert "numpy" in result
        assert "pandas" in result
        assert "sklearn" in result


class TestGenerateInitPy:
    def test_contains_project_name(self) -> None:
        result = generate_init_py("my_proj")
        assert "my_proj" in result
