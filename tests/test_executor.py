"""Tests for the executor module."""

import os
from unittest.mock import MagicMock, patch

import pytest

from create_uv_ml.executor import (
    check_uv_available,
    uv_add,
    uv_sync,
    write_gitignore,
    write_pyproject,
    write_verify_env,
)


def _mock_popen_success() -> MagicMock:
    """Create a mock Popen that simulates a successful subprocess."""
    mock = MagicMock()
    mock.stdout = iter(["Resolved 10 packages\n", "Installed 10 packages\n"])
    mock.wait.return_value = 0
    return mock


def _mock_popen_failure() -> MagicMock:
    """Create a mock Popen that simulates a failed subprocess."""
    mock = MagicMock()
    mock.stdout = iter(["error: something went wrong\n"])
    mock.wait.return_value = 1
    return mock


class TestCheckUvAvailable:
    def test_uv_found(self) -> None:
        with patch("create_uv_ml.executor.shutil.which", return_value="/usr/bin/uv"):
            check_uv_available()  # Should not raise

    def test_uv_not_found(self) -> None:
        with (
            patch("create_uv_ml.executor.shutil.which", return_value=None),
            pytest.raises(SystemExit),
        ):
            check_uv_available()


class TestWritePyproject:
    def test_creates_directory_and_file(self, tmp_path: object) -> None:
        import pathlib

        tmp = pathlib.Path(str(tmp_path))  # type: ignore[arg-type]
        target = str(tmp / "new_proj")
        content = '[project]\nname = "new_proj"'

        write_pyproject(target, content)

        assert os.path.exists(os.path.join(target, "pyproject.toml"))
        with open(os.path.join(target, "pyproject.toml")) as f:
            assert f.read() == content

    def test_writes_to_existing_directory(self, tmp_path: object) -> None:
        import pathlib

        tmp = pathlib.Path(str(tmp_path))  # type: ignore[arg-type]
        target = str(tmp / "existing_proj")
        os.makedirs(target, exist_ok=True)
        content = '[project]\nname = "existing_proj"'

        write_pyproject(target, content)

        assert os.path.exists(os.path.join(target, "pyproject.toml"))


class TestWriteGitignore:
    def test_creates_gitignore(self, tmp_path: object) -> None:
        import pathlib

        tmp = pathlib.Path(str(tmp_path))  # type: ignore[arg-type]
        target = str(tmp / "proj")
        os.makedirs(target, exist_ok=True)

        write_gitignore(target)

        gitignore_path = os.path.join(target, ".gitignore")
        assert os.path.exists(gitignore_path)
        with open(gitignore_path) as f:
            content = f.read()
            assert "__pycache__/" in content
            assert ".venv/" in content


class TestWriteVerifyEnv:
    def test_creates_verify_env_script(self, tmp_path: object) -> None:
        import pathlib

        tmp = pathlib.Path(str(tmp_path))  # type: ignore[arg-type]
        target = str(tmp / "proj")
        os.makedirs(target, exist_ok=True)

        write_verify_env(target)

        script_path = os.path.join(target, "verify_env.py")
        assert os.path.exists(script_path)
        with open(script_path) as f:
            content = f.read()
            assert "Environment Verification" in content
            assert "check_torch" in content


class TestUvSync:
    def test_uv_sync_called_with_python_version(self, tmp_path: object) -> None:
        import pathlib

        tmp = pathlib.Path(str(tmp_path))  # type: ignore[arg-type]
        target = str(tmp / "sync_proj")

        with (
            patch(
                "create_uv_ml.executor.subprocess.Popen",
                return_value=_mock_popen_success(),
            ) as mock_popen,
        ):
            uv_sync(target, "3.12")
            mock_popen.assert_called_once()
            call_args = mock_popen.call_args
            assert call_args[0][0] == ["uv", "sync", "--python", "3.12"]
            assert call_args[1]["cwd"] == target

    def test_uv_sync_failure(self, tmp_path: object) -> None:
        import pathlib

        tmp = pathlib.Path(str(tmp_path))  # type: ignore[arg-type]
        target = str(tmp / "fail_proj")

        with (
            patch(
                "create_uv_ml.executor.subprocess.Popen",
                return_value=_mock_popen_failure(),
            ),
            pytest.raises(SystemExit),
        ):
            uv_sync(target, "3.12")


class TestUvAdd:
    def test_uv_add_with_packages(self, tmp_path: object) -> None:
        import pathlib

        tmp = pathlib.Path(str(tmp_path))  # type: ignore[arg-type]
        target = str(tmp / "add_proj")

        with (
            patch(
                "create_uv_ml.executor.subprocess.Popen",
                return_value=_mock_popen_success(),
            ) as mock_popen,
        ):
            uv_add(target, ["pandas", "matplotlib"])
            mock_popen.assert_called_once()
            call_args = mock_popen.call_args
            assert call_args[0][0] == ["uv", "add", "pandas", "matplotlib"]
            assert call_args[1]["cwd"] == target

    def test_uv_add_empty_list_does_nothing(self, tmp_path: object) -> None:
        import pathlib

        tmp = pathlib.Path(str(tmp_path))  # type: ignore[arg-type]
        target = str(tmp / "empty_proj")

        with patch("create_uv_ml.executor.subprocess.Popen") as mock_popen:
            uv_add(target, [])
            mock_popen.assert_not_called()

    def test_uv_add_failure(self, tmp_path: object) -> None:
        import pathlib

        tmp = pathlib.Path(str(tmp_path))  # type: ignore[arg-type]
        target = str(tmp / "fail_proj")

        with (
            patch(
                "create_uv_ml.executor.subprocess.Popen",
                return_value=_mock_popen_failure(),
            ),
            pytest.raises(SystemExit),
        ):
            uv_add(target, ["pandas"])
