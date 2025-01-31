"""Test Export Data Routine."""

import os
import tempfile

import numpy as np
import pytest
from i2pp.core.export_data import Exporter


def test_load_user_function_not_exist():
    """_load_user_function if Path not found."""
    path = "not_exisitng_path.py"
    function_name = "function_name"
    exporter = Exporter()

    with pytest.raises(
        RuntimeError, match="User script 'not_exisitng_path.py' not found!"
    ):
        exporter.load_user_function(path, function_name)


def test_load_user_function_exist():
    """_load_user_function if funtion exist."""
    test_script = tempfile.NamedTemporaryFile(delete=False, suffix=".py")
    test_script.write(b"def test_function(data): return 2*data+1\n")
    test_script.close()

    script_path = test_script.name

    exporter = Exporter()
    loaded_function = exporter.load_user_function(script_path, "test_function")

    os.remove(script_path)

    assert callable(loaded_function)
    assert loaded_function(2) == 5


def test_load_user_function_not_callable():
    """_load_user_function if funtion exist."""
    test_script = tempfile.NamedTemporaryFile(delete=False, suffix=".py")
    test_script.close()

    script_path = test_script.name

    exporter = Exporter()

    with pytest.raises(RuntimeError, match="Userfunction not found"):
        exporter.load_user_function(script_path, "test_function")

        os.remove(script_path)


def test_norm_values_RGB():
    """Test normalize_values for RGB."""
    data = np.array([[0, 255, 255], [255, 255, 0], [0, 255, 0]])
    pxl_range = np.array([0, 255])

    exporter = Exporter()

    assert np.array_equal(
        exporter.normalize_values(data, pxl_range),
        np.array([[0, 1, 1], [1, 1, 0], [0, 1, 0]]),
    )


def test_norm_values_Gray():
    """Test normalize_values for Float-values."""
    pxl_range = np.array([-100, 100])
    data = np.array([100, 50, 0, -50, -100])

    exporter = Exporter()

    assert np.array_equal(
        exporter.normalize_values(data, pxl_range),
        np.array([1.0, 0.75, 0.5, 0.25, 0.0]),
    )
