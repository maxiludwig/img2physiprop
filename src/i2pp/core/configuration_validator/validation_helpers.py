"""This module provides utility functions used during validation of the input
configuration."""

from pathlib import Path


def resolve_and_validate_path(path_str: str, must_exist: bool = True) -> Path:
    """Resolve a path string to a Path object and validate its existence."""
    path = Path(path_str).resolve()
    if must_exist and not path.exists():
        raise FileNotFoundError(f"Path does not exist: {path}")
    return path
