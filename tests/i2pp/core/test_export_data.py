"""Test Export Data Routine."""

import os
import tempfile

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
