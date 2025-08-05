"""Tests for validation helpers in the input validator module."""

import os
from pathlib import Path

import pytest
from i2pp.core.configuration_validator.validation_helpers import (
    resolve_and_validate_path,
)


def test_existing_absolute_path():
    """Test resolve_and_validate_path with an existing absolute path."""
    path = resolve_and_validate_path(Path(__file__).resolve().as_posix())
    assert path.exists()
    assert path.is_absolute()


def test_existing_relative_path():
    """Test resolve_and_validate_path with an existing relative path."""
    relative_path = os.path.relpath(__file__)
    path = resolve_and_validate_path(relative_path)
    assert path.exists()
    assert path.is_absolute()


def test_nonexistent_path_raises():
    """Test resolve_and_validate_path raises FileNotFoundError for a non-
    existent path."""
    with pytest.raises(FileNotFoundError):
        resolve_and_validate_path("non_existent_file_12345.txt")


def test_nonexistent_path_allowed():
    """Test resolve_and_validate_path does not raise for a non-existent path
    when allowed."""
    path = resolve_and_validate_path(
        "non_existent_file_12345.txt", must_exist=False
    )
    assert isinstance(path, Path)
    assert not path.exists()
