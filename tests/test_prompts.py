"""Tests for the prompts module."""

from unittest.mock import patch

import pytest

from create_uv_ml.prompts import (
    PRESETS,
    ask_cuda_strategy,
    ask_extra_packages,
    ask_mirror,
    ask_python_version,
    ask_target_directory,
)


class TestAskPythonVersion:
    def test_returns_selected_version(self) -> None:
        with patch("create_uv_ml.prompts.questionary.select") as mock_select:
            mock_select.return_value.ask.return_value = "3.12"
            result = ask_python_version()
            assert result == "3.12"

    def test_returns_custom_version(self) -> None:
        with (
            patch("create_uv_ml.prompts.questionary.select") as mock_select,
            patch("create_uv_ml.prompts.questionary.text") as mock_text,
        ):
            mock_select.return_value.ask.return_value = "Other (enter manually)"
            mock_text.return_value.ask.return_value = "3.9"
            result = ask_python_version()
            assert result == "3.9"


class TestAskCudaStrategy:
    def test_returns_cu121(self) -> None:
        with patch("create_uv_ml.prompts.questionary.select") as mock_select:
            mock_select.return_value.ask.return_value = "CUDA 12.1 (Recommended)"
            result = ask_cuda_strategy()
            assert result == "CUDA 12.1 (Recommended)"

    def test_returns_cu118(self) -> None:
        with patch("create_uv_ml.prompts.questionary.select") as mock_select:
            mock_select.return_value.ask.return_value = "CUDA 11.8"
            result = ask_cuda_strategy()
            assert result == "CUDA 11.8"

    def test_returns_cpu(self) -> None:
        with patch("create_uv_ml.prompts.questionary.select") as mock_select:
            mock_select.return_value.ask.return_value = "CPU Only"
            result = ask_cuda_strategy()
            assert result == "CPU Only"

    def test_macos_returns_none(self) -> None:
        with patch("create_uv_ml.prompts.sys.platform", "darwin"):
            result = ask_cuda_strategy()
            assert result is None

    def test_recommended_moves_to_front(self) -> None:
        with patch("create_uv_ml.prompts.questionary.select") as mock_select:
            mock_select.return_value.ask.return_value = "CUDA 11.8"
            ask_cuda_strategy(recommended="cu118")
            # Verify that select was called with cu118 label first
            call_args = mock_select.call_args
            choices = call_args[1]["choices"]
            assert choices[0] == "CUDA 11.8"


class TestAskExtraPackages:
    """Tests for ask_extra_packages() with the two-stage select + confirm flow."""

    def test_none_preset_returns_empty(self) -> None:
        with patch("create_uv_ml.prompts.questionary.select") as mock_select:
            mock_select.return_value.ask.return_value = "None (core PyTorch only)"
            result = ask_extra_packages()
            assert result == []

    def test_all_preset_returns_all(self) -> None:
        with patch("create_uv_ml.prompts.questionary.select") as mock_select:
            mock_select.return_value.ask.return_value = [
                k for k in PRESETS if k.startswith("All packages")
            ][0]
            result = ask_extra_packages()
            assert "transformers" in result
            assert "pandas" in result
            assert "opencv-python" in result
            assert "tqdm" in result

    def test_cancel_at_select_raises(self) -> None:
        with (
            patch("create_uv_ml.prompts.questionary.select") as mock_select,
            pytest.raises(SystemExit),
        ):
            mock_select.return_value.ask.return_value = None
            ask_extra_packages()

    def test_customize_all_yes(self) -> None:
        with (
            patch("create_uv_ml.prompts.questionary.select") as mock_select,
            patch("create_uv_ml.prompts.questionary.confirm") as mock_confirm,
        ):
            mock_select.return_value.ask.return_value = "Customize... (pick per category)"
            mock_confirm.return_value.ask.return_value = True
            result = ask_extra_packages()
            # All 4 categories confirmed → all packages
            assert "transformers" in result
            assert "pandas" in result
            assert "opencv-python" in result
            assert "tqdm" in result

    def test_customize_all_no(self) -> None:
        with (
            patch("create_uv_ml.prompts.questionary.select") as mock_select,
            patch("create_uv_ml.prompts.questionary.confirm") as mock_confirm,
        ):
            mock_select.return_value.ask.return_value = "Customize... (pick per category)"
            mock_confirm.return_value.ask.return_value = False
            result = ask_extra_packages()
            assert result == []

    def test_customize_mixed(self) -> None:
        with (
            patch("create_uv_ml.prompts.questionary.select") as mock_select,
            patch("create_uv_ml.prompts.questionary.confirm") as mock_confirm,
        ):
            mock_select.return_value.ask.return_value = "Customize... (pick per category)"
            # Yes for Deep Learning, No for others
            mock_confirm.return_value.ask.side_effect = [True, False, False, False]
            result = ask_extra_packages()
            assert "transformers" in result
            assert "pandas" not in result

    def test_customize_cancel_at_confirm_raises(self) -> None:
        with (
            patch("create_uv_ml.prompts.questionary.select") as mock_select,
            patch("create_uv_ml.prompts.questionary.confirm") as mock_confirm,
            pytest.raises(SystemExit),
        ):
            mock_select.return_value.ask.return_value = "Customize... (pick per category)"
            mock_confirm.return_value.ask.return_value = None
            ask_extra_packages()


class TestAskMirror:
    def test_returns_default(self) -> None:
        with patch("create_uv_ml.prompts.questionary.select") as mock_select:
            mock_select.return_value.ask.return_value = "Default (PyPI)"
            result = ask_mirror()
            assert result == "Default (PyPI)"

    def test_returns_tsinghua(self) -> None:
        with patch("create_uv_ml.prompts.questionary.select") as mock_select:
            mock_select.return_value.ask.return_value = "Tsinghua (China)"
            result = ask_mirror()
            assert result == "Tsinghua (China)"


class TestAskTargetDirectory:
    def test_returns_cwd_on_confirm(self) -> None:
        with (
            patch("create_uv_ml.prompts.questionary.confirm") as mock_confirm,
            patch("create_uv_ml.prompts.os.getcwd", return_value="/home/user/project"),
            patch("create_uv_ml.prompts.os.path.isdir", return_value=False),
        ):
            mock_confirm.return_value.ask.return_value = True
            result = ask_target_directory()
            assert result == "/home/user/project"

    def test_returns_custom_path(self) -> None:
        with (
            patch("create_uv_ml.prompts.questionary.confirm") as mock_confirm,
            patch("create_uv_ml.prompts.questionary.text") as mock_text,
            patch("create_uv_ml.prompts.os.getcwd", return_value="/home/user"),
            patch(
                "create_uv_ml.prompts.os.path.expanduser",
                return_value="/home/user/my_project",
            ),
            patch(
                "create_uv_ml.prompts.os.path.abspath",
                return_value="/home/user/my_project",
            ),
            patch("create_uv_ml.prompts.os.path.isdir", return_value=False),
        ):
            mock_confirm.return_value.ask.return_value = False
            mock_text.return_value.ask.return_value = "~/my_project"
            result = ask_target_directory()
            assert result == "/home/user/my_project"

    def test_venv_exists_user_confirms(self) -> None:
        with (
            patch("create_uv_ml.prompts.questionary.confirm") as mock_confirm,
            patch("create_uv_ml.prompts.os.getcwd", return_value="/home/user/project"),
            patch("create_uv_ml.prompts.os.path.isdir", return_value=True),
        ):
            # First confirm: use CWD = True
            # Second confirm: proceed despite .venv = True
            mock_confirm.return_value.ask.side_effect = [True, True]
            result = ask_target_directory()
            assert result == "/home/user/project"

    def test_venv_exists_user_declines(self) -> None:
        with (
            patch("create_uv_ml.prompts.questionary.confirm") as mock_confirm,
            patch("create_uv_ml.prompts.os.getcwd", return_value="/home/user/project"),
            patch("create_uv_ml.prompts.os.path.isdir", return_value=True),
        ):
            # First confirm: use CWD = True
            # Second confirm: proceed despite .venv = False
            mock_confirm.return_value.ask.side_effect = [True, False]
            with pytest.raises(SystemExit):
                ask_target_directory()
