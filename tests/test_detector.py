"""Tests for the detector module."""

import subprocess
from unittest.mock import MagicMock, patch

from create_uv_ml.detector import (
    detect,
    detect_nvidia_gpu,
    find_nvidia_smi,
    get_recommended_cuda,
)


class TestFindNvidiaSmi:
    """Tests for find_nvidia_smi()."""

    def test_found_in_path(self) -> None:
        with patch("create_uv_ml.detector.shutil.which", return_value="/usr/bin/nvidia-smi"):
            result = find_nvidia_smi()
            assert result == "/usr/bin/nvidia-smi"

    def test_not_in_path_linux_fallback(self) -> None:
        with (
            patch("create_uv_ml.detector.shutil.which", return_value=None),
            patch("create_uv_ml.detector.sys.platform", "linux"),
            patch("create_uv_ml.detector.os.path.exists") as mock_exists,
        ):
            mock_exists.side_effect = lambda p: p == "/usr/bin/nvidia-smi"
            result = find_nvidia_smi()
            assert result == "/usr/bin/nvidia-smi"

    def test_not_found_at_all(self) -> None:
        with (
            patch("create_uv_ml.detector.shutil.which", return_value=None),
            patch("create_uv_ml.detector.sys.platform", "linux"),
            patch("create_uv_ml.detector.os.path.exists", return_value=False),
        ):
            result = find_nvidia_smi()
            assert result is None

    def test_found_in_path_takes_priority(self) -> None:
        with patch("create_uv_ml.detector.shutil.which", return_value="/usr/local/bin/nvidia-smi"):
            result = find_nvidia_smi()
            assert result == "/usr/local/bin/nvidia-smi"


class TestDetectNvidiaGpu:
    """Tests for detect_nvidia_gpu()."""

    def test_no_nvidia_smi(self) -> None:
        with patch("create_uv_ml.detector.find_nvidia_smi", return_value=None):
            has_nvidia, version = detect_nvidia_gpu()
            assert has_nvidia is False
            assert version is None

    def test_nvidia_smi_returns_version(self) -> None:
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "535.129.03\n"

        with (
            patch("create_uv_ml.detector.find_nvidia_smi", return_value="/usr/bin/nvidia-smi"),
            patch("create_uv_ml.detector.subprocess.run", return_value=mock_result),
        ):
            has_nvidia, version = detect_nvidia_gpu()
            assert has_nvidia is True
            assert version == "535.129.03"

    def test_nvidia_smi_fails(self) -> None:
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stdout = ""

        with (
            patch("create_uv_ml.detector.find_nvidia_smi", return_value="/usr/bin/nvidia-smi"),
            patch("create_uv_ml.detector.subprocess.run", return_value=mock_result),
        ):
            has_nvidia, version = detect_nvidia_gpu()
            assert has_nvidia is True
            assert version is None  # nvidia-smi exists but couldn't get version

    def test_nvidia_smi_timeout(self) -> None:
        with (
            patch("create_uv_ml.detector.find_nvidia_smi", return_value="/usr/bin/nvidia-smi"),
            patch(
                "create_uv_ml.detector.subprocess.run",
                side_effect=subprocess.TimeoutExpired(cmd="nvidia-smi", timeout=10),
            ),
        ):
            has_nvidia, version = detect_nvidia_gpu()
            assert has_nvidia is True
            assert version is None


class TestGetRecommendedCuda:
    """Tests for get_recommended_cuda()."""

    def test_recent_driver_recommends_cu121(self) -> None:
        assert get_recommended_cuda("535.129.03") == "cu121"

    def test_very_recent_driver_recommends_cu121(self) -> None:
        assert get_recommended_cuda("550.54.14") == "cu121"

    def test_mid_range_driver_recommends_cu118(self) -> None:
        assert get_recommended_cuda("470.129.02") == "cu118"

    def test_old_driver_returns_none(self) -> None:
        assert get_recommended_cuda("390.12") is None

    def test_none_driver_defaults_to_cu121(self) -> None:
        assert get_recommended_cuda(None) == "cu121"

    def test_malformed_version_defaults_to_cu121(self) -> None:
        assert get_recommended_cuda("abc") == "cu121"

    def test_boundary_525(self) -> None:
        assert get_recommended_cuda("525.60.13") == "cu121"

    def test_boundary_450(self) -> None:
        assert get_recommended_cuda("450.36.06") == "cu118"


class TestDetect:
    """Tests for detect()."""

    def test_no_gpu(self) -> None:
        with (
            patch("create_uv_ml.detector.detect_os", return_value="linux"),
            patch("create_uv_ml.detector.detect_nvidia_gpu", return_value=(False, None)),
        ):
            result = detect()
            assert result.os_type == "linux"
            assert result.has_nvidia is False
            assert result.driver_version is None
            assert result.recommended_cuda is None

    def test_with_gpu(self) -> None:
        with (
            patch("create_uv_ml.detector.detect_os", return_value="linux"),
            patch("create_uv_ml.detector.detect_nvidia_gpu", return_value=(True, "535.129.03")),
        ):
            result = detect()
            assert result.os_type == "linux"
            assert result.has_nvidia is True
            assert result.driver_version == "535.129.03"
            assert result.recommended_cuda == "cu121"

    def test_macos_no_cuda(self) -> None:
        with (
            patch("create_uv_ml.detector.detect_os", return_value="darwin"),
            patch("create_uv_ml.detector.detect_nvidia_gpu", return_value=(False, None)),
        ):
            result = detect()
            assert result.os_type == "darwin"
            assert result.recommended_cuda is None
