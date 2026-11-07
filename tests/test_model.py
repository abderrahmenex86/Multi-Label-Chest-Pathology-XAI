from __future__ import annotations

import pytest

from src.model import SUPPORTED_ARCHITECTURES, ChestMultiLabelModel


def test_larger_convnext_variants_are_supported() -> None:
    assert "convnext_tiny" in SUPPORTED_ARCHITECTURES
    assert "convnext_small" in SUPPORTED_ARCHITECTURES
    assert "convnext_base" in SUPPORTED_ARCHITECTURES
    assert "convnext_large" in SUPPORTED_ARCHITECTURES


def test_unknown_architecture_fails_before_allocating_a_backbone() -> None:
    with pytest.raises(ValueError, match="Unsupported architecture"):
        ChestMultiLabelModel(14, "convnext_huge")
