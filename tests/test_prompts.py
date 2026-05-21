"""Tests for the prompts module."""

from unittest.mock import patch

from create_uv_ml.prompts import ask_cuda_version, ask_framework


class TestAskFramework:
    def test_returns_pytorch(self) -> None:
        with patch("create_uv_ml.prompts.questionary.select") as mock_select:
            mock_select.return_value.ask.return_value = "PyTorch"
            result = ask_framework()
            assert result == "PyTorch"

    def test_returns_tensorflow(self) -> None:
        with patch("create_uv_ml.prompts.questionary.select") as mock_select:
            mock_select.return_value.ask.return_value = "TensorFlow"
            result = ask_framework()
            assert result == "TensorFlow"

    def test_returns_scipy(self) -> None:
        with patch("create_uv_ml.prompts.questionary.select") as mock_select:
            mock_select.return_value.ask.return_value = (
                "Basic Scientific Computing (NumPy/Pandas/Scikit-learn)"
            )
            result = ask_framework()
            assert result == "Basic Scientific Computing (NumPy/Pandas/Scikit-learn)"


class TestAskCudaVersion:
    def test_returns_cu121(self) -> None:
        with patch("create_uv_ml.prompts.questionary.select") as mock_select:
            mock_select.return_value.ask.return_value = "CUDA 12.1 (Recommended)"
            result = ask_cuda_version()
            assert result == "CUDA 12.1 (Recommended)"

    def test_returns_cu118(self) -> None:
        with patch("create_uv_ml.prompts.questionary.select") as mock_select:
            mock_select.return_value.ask.return_value = "CUDA 11.8"
            result = ask_cuda_version()
            assert result == "CUDA 11.8"

    def test_returns_cpu(self) -> None:
        with patch("create_uv_ml.prompts.questionary.select") as mock_select:
            mock_select.return_value.ask.return_value = "CPU Only"
            result = ask_cuda_version()
            assert result == "CPU Only"
