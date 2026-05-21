"""Tests for the runner module."""

import os
import subprocess
from unittest.mock import patch

import pytest

from create_uv_ml.runner import check_uv_available, create_project, validate_project_name


class TestCheckUvAvailable:
    def test_uv_found(self) -> None:
        with patch("create_uv_ml.runner.shutil.which", return_value="/usr/bin/uv"):
            check_uv_available()  # Should not raise

    def test_uv_not_found(self) -> None:
        with (
            patch("create_uv_ml.runner.shutil.which", return_value=None),
            pytest.raises(SystemExit),
        ):
            check_uv_available()


class TestValidateProjectName:
    def test_valid_name(self) -> None:
        validate_project_name("my_project")  # Should not raise

    def test_valid_name_with_numbers(self) -> None:
        validate_project_name("project123")  # Should not raise

    def test_invalid_name_with_spaces(self) -> None:
        with pytest.raises(SystemExit):
            validate_project_name("my project")

    def test_invalid_name_with_hyphens(self) -> None:
        with pytest.raises(SystemExit):
            validate_project_name("my-project")

    def test_invalid_name_starts_with_digit(self) -> None:
        with pytest.raises(SystemExit):
            validate_project_name("1project")

    def test_invalid_name_empty(self) -> None:
        with pytest.raises(SystemExit):
            validate_project_name("")


class TestCreateProject:
    def test_directory_already_exists(self, tmp_path: object) -> None:
        import pathlib
        tmp = pathlib.Path(str(tmp_path))  # type: ignore[arg-type]
        existing = tmp / "existing_proj"
        existing.mkdir()
        with pytest.raises(SystemExit):
            create_project(
                str(existing),
                "[project]\nname = 'test'",
                "PyTorch",
                "CPU Only",
            )

    def test_creates_project_files(self, tmp_path: object) -> None:
        import pathlib
        tmp = pathlib.Path(str(tmp_path))  # type: ignore[arg-type]
        project_dir = str(tmp / "new_proj")
        with patch("create_uv_ml.runner.subprocess.run"):
            create_project(
                project_dir,
                '[project]\nname = "new_proj"',
                "PyTorch",
                "CPU Only",
                template="minimal",
            )
        assert os.path.exists(os.path.join(project_dir, "pyproject.toml"))
        assert os.path.exists(os.path.join(project_dir, ".gitignore"))
        assert os.path.exists(os.path.join(project_dir, "README.md"))

    def test_full_template_creates_src_layout(self, tmp_path: object) -> None:
        import pathlib
        tmp = pathlib.Path(str(tmp_path))  # type: ignore[arg-type]
        project_dir = str(tmp / "full_proj")
        with patch("create_uv_ml.runner.subprocess.run"):
            create_project(
                project_dir,
                '[project]\nname = "full_proj"',
                "PyTorch",
                "CPU Only",
                template="full",
            )
        assert os.path.exists(os.path.join(project_dir, "src", "full_proj", "__init__.py"))
        assert os.path.exists(os.path.join(project_dir, "src", "full_proj", "train.py"))
        assert os.path.exists(os.path.join(project_dir, "data", ".gitkeep"))

    def test_no_sync_flag(self, tmp_path: object) -> None:
        import pathlib
        tmp = pathlib.Path(str(tmp_path))  # type: ignore[arg-type]
        project_dir = str(tmp / "nosync_proj")
        with patch("create_uv_ml.runner.subprocess.run") as mock_run:
            create_project(
                project_dir,
                '[project]\nname = "nosync_proj"',
                "PyTorch",
                "CPU Only",
                no_sync=True,
                template="minimal",
            )
            mock_run.assert_not_called()

    def test_uv_sync_called(self, tmp_path: object) -> None:
        import pathlib
        tmp = pathlib.Path(str(tmp_path))  # type: ignore[arg-type]
        project_dir = str(tmp / "sync_proj")
        with patch("create_uv_ml.runner.subprocess.run") as mock_run:
            create_project(
                project_dir,
                '[project]\nname = "sync_proj"',
                "PyTorch",
                "CPU Only",
                template="minimal",
            )
            mock_run.assert_called_once()
            call_args = mock_run.call_args
            assert call_args[0][0] == ["uv", "sync"]
            assert call_args[1]["cwd"] == project_dir

    def test_uv_sync_timeout(self, tmp_path: object) -> None:
        import pathlib
        tmp = pathlib.Path(str(tmp_path))  # type: ignore[arg-type]
        project_dir = str(tmp / "timeout_proj")
        with patch("create_uv_ml.runner.subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.TimeoutExpired(cmd="uv sync", timeout=600)
            with pytest.raises(SystemExit):
                create_project(
                    project_dir,
                    '[project]\nname = "timeout_proj"',
                    "PyTorch",
                    "CPU Only",
                    template="minimal",
                )

    def test_uv_sync_failure(self, tmp_path: object) -> None:
        import pathlib
        tmp = pathlib.Path(str(tmp_path))  # type: ignore[arg-type]
        project_dir = str(tmp / "fail_proj")
        with patch("create_uv_ml.runner.subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.CalledProcessError(1, "uv sync", stderr="error")
            with pytest.raises(SystemExit):
                create_project(
                    project_dir,
                    '[project]\nname = "fail_proj"',
                    "PyTorch",
                    "CPU Only",
                    template="minimal",
                )

    def test_tensorflow_template(self, tmp_path: object) -> None:
        import pathlib
        tmp = pathlib.Path(str(tmp_path))  # type: ignore[arg-type]
        project_dir = str(tmp / "tf_proj")
        with patch("create_uv_ml.runner.subprocess.run"):
            create_project(
                project_dir,
                '[project]\nname = "tf_proj"',
                "TensorFlow",
                None,
                template="full",
            )
        assert os.path.exists(os.path.join(project_dir, "src", "tf_proj", "train.py"))

    def test_scipy_template(self, tmp_path: object) -> None:
        import pathlib
        tmp = pathlib.Path(str(tmp_path))  # type: ignore[arg-type]
        project_dir = str(tmp / "scipy_proj")
        with patch("create_uv_ml.runner.subprocess.run"):
            create_project(
                project_dir,
                '[project]\nname = "scipy_proj"',
                "Basic Scientific Computing (NumPy/Pandas/Scikit-learn)",
                None,
                template="full",
            )
        assert os.path.exists(os.path.join(project_dir, "src", "scipy_proj", "analysis.py"))
