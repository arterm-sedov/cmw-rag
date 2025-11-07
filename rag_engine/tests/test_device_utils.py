"""Tests for device detection utilities."""
from __future__ import annotations

import pytest

from rag_engine.utils.device_utils import detect_device


def test_detect_device_cpu_forced():
    """Test that forcing CPU works."""
    result = detect_device("cpu")
    assert result == "cpu"


def test_detect_device_auto():
    """Test that auto-detection works (may return cpu or cuda depending on system)."""
    result = detect_device("auto")
    assert result in ("cpu", "cuda")


def test_detect_device_cuda_preferred():
    """Test that cuda preference works (falls back to cpu if not available)."""
    result = detect_device("cuda")
    assert result in ("cpu", "cuda")


@pytest.mark.external
def test_detect_device_with_torch():
    """Test device detection when torch is available."""
    try:
        import torch

        result = detect_device("auto")
        if torch.cuda.is_available():
            assert result == "cuda"
        else:
            assert result == "cpu"
    except ImportError:
        pytest.skip("PyTorch not available")
