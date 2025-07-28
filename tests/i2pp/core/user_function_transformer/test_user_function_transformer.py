"""Test User Function Transformer."""

import os
import tempfile
from pathlib import Path

import numpy as np
import pytest
from i2pp.core.discretization_readers.discretization_reader import Element
from i2pp.core.user_function_transformer.user_function_transformer import (
    UserFunctionTransformer,
)


def test_load_user_function_success():
    """Test loading a user-defined function from a script."""
    script = tempfile.NamedTemporaryFile(delete=False, suffix=".py")
    script.write(b"def test_function(ids, data): return 2 * data + 1\n")
    script.close()

    script_path = Path(script.name)
    transformer = UserFunctionTransformer()
    loaded_function = transformer.load_user_function(
        script_path, "test_function"
    )

    os.remove(script_path)

    assert callable(loaded_function)
    assert np.array_equal(
        loaded_function(np.array([1, 2]), np.array([22.2, 22.4])),
        np.array([22.2, 22.4]) * 2 + 1,
    )


def test_load_user_function_not_exist():
    """Test loading a user-defined function from a non-existing script."""
    transformer = UserFunctionTransformer()
    path = Path("not_exisiting_path.py")

    with pytest.raises(
        RuntimeError, match="User script 'not_exisiting_path.py' not found!"
    ):
        transformer.load_user_function(path, "function_name")


def test_load_user_function_invalid_function():
    """Test loading a user-defined function that is not callable."""
    script = tempfile.NamedTemporaryFile(delete=False, suffix=".py")
    script.write(b"a = 5")  # no function defined
    script.close()

    script_path = Path(script.name)

    transformer = UserFunctionTransformer()
    with pytest.raises(
        RuntimeError,
        match="User function missing_function not found or not callable",
    ):
        transformer.load_user_function(script_path, "missing_function")

    os.remove(script_path)


def test_apply_transformation_basic():
    """Test applying a user-defined transformation to element data."""
    script = tempfile.NamedTemporaryFile(delete=False, suffix=".py")
    script.write(b"def transform(ids, data): return data*4\n")
    script.close()

    script_path = Path(script.name)

    elements = [
        Element(node_ids=[0, 1], id=0, data=10),
        Element(node_ids=[1, 2], id=1, data=20),
    ]

    transformer = UserFunctionTransformer()
    result = transformer.apply_transformation(
        elements, script_path, "transform"
    )

    assert np.array_equal(result, np.array([40, 80]))
    os.remove(script_path)


def test_apply_transformation_with_normalization():
    """Test applying a user-defined transformation with normalization."""
    script = tempfile.NamedTemporaryFile(delete=False, suffix=".py")
    script.write(b"def transform(ids, data): return data\n")
    script.close()

    script_path = Path(script.name)

    elements = [
        Element(node_ids=[0, 1], id=0, data=10),
        Element(node_ids=[1, 2], id=1, data=30),
    ]
    transformer = UserFunctionTransformer(
        normalize=True, pixel_range=np.array([0, 50])
    )
    result = transformer.apply_transformation(
        elements, script_path, "transform"
    )

    os.remove(script_path)

    # normalized data: (10/50, 30/50) = (0.2, 0.6)
    assert np.allclose(result, np.array([0.2, 0.6]), atol=1e-6)


def test_apply_transformation_invalid_function():
    """Test applying a user-defined transformation with an invalid function."""
    script = tempfile.NamedTemporaryFile(delete=False, suffix=".py")
    script.write(b"a = 5")  # no function defined
    script.close()

    script_path = Path(script.name)

    elements = [
        Element(node_ids=[0, 1], id=0, data=30),
        Element(node_ids=[1, 2], id=1, data=20),
    ]
    transformer = UserFunctionTransformer()
    with pytest.raises(
        RuntimeError,
        match="User function missing_function not found or not callable",
    ):
        transformer.load_user_function(script_path, "missing_function")
        transformer.apply_transformation(
            elements, script_path, "missing_function"
        )

    os.remove(script_path)


def test_user_function_too_few_arguments():
    """Test applying a user-defined function with too few arguments."""
    script = tempfile.NamedTemporaryFile(delete=False, suffix=".py")
    script.write(b"def transform(data): return data + 1\n")  # only one arg
    script.close()

    script_path = Path(script.name)

    elements = [
        Element(node_ids=[0, 1], id=0, data=10),
        Element(node_ids=[1, 2], id=1, data=20),
    ]
    transformer = UserFunctionTransformer()

    with pytest.raises(
        RuntimeError,
        match="User function transform must accept two arguments",
    ):
        transformer.apply_transformation(elements, script_path, "transform")

    os.remove(script_path)


def test_user_function_errors_on_wrong_return_type():
    """Test applying a user-defined function that does not return a numpy
    array."""
    script = tempfile.NamedTemporaryFile(delete=False, suffix=".py")
    script.write(b"def transform(ids, data): return 'abc'\n")  # returns string
    script.close()

    script_path = Path(script.name)

    elements = [
        Element(node_ids=[0, 1], id=0, data=10),
        Element(node_ids=[1, 2], id=1, data=20),
    ]
    transformer = UserFunctionTransformer()

    with pytest.raises(
        RuntimeError,
        match="User function transform must return a numpy array",
    ):
        transformer.apply_transformation(elements, script_path, "transform")

    os.remove(script_path)
