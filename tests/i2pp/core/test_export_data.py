"""Test Export Data Routine."""

import os
import tempfile
from pathlib import Path
from unittest import mock
from unittest.mock import patch

import numpy as np
import pytest
from i2pp.core.discretization_reader_classes.discretization_reader import (
    Element,
)
from i2pp.core.export_data import Exporter, export_data


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


def test_write_data_creates_file():
    """Test write_data creates file."""
    export_string = "test."

    test_config = {
        "output options": {
            "output_path": str(Path.cwd() / "test_directory"),
            "output_name": "test_output",
        }
    }
    exporter = Exporter()
    file_path = (
        Path(test_config["output options"]["output_path"])
        / "test_output.pattern"
    )

    with mock.patch("builtins.open", mock.mock_open()) as mocked_file:
        exporter.write_data(export_string, test_config)

        mocked_file.assert_called_once_with(file_path, "w", encoding="utf-8")

        mocked_file().write.assert_called_once_with(export_string)


def test_export_data():
    """Test export_data."""

    test_config = {
        "processing options": {
            "user_script": "mock_script.py",
            "user_function": "mock_function",
            "normalize_values": True,
        }
    }

    pixel_range = np.array([0, 20])
    element1 = Element([0, 1], 0, data=10)
    element2 = Element([0, 1], 1, data=20)
    elements = [element1, element2]

    expected_data = {1: 1, 2: 2}
    expected_string = "\n".join(
        f"{key}:{value}" for key, value in expected_data.items()
    )

    def mock_user_function(element_ids, element_data):
        """Mock-User-Funktion, die Element-IDs und Daten als formatierte
        Strings zurückgibt."""
        return "\n".join(
            f"{eid}:{edata}" for eid, edata in zip(element_ids, element_data)
        )

    with patch(
        "i2pp.core.export_data.normalize_values", return_value=np.array([1, 2])
    ) as mock_normalize_values:
        with patch.object(
            Exporter, "load_user_function", return_value=mock_user_function
        ) as mock_load_user_function:
            with patch.object(
                Exporter, "write_data", return_value=None
            ) as mock_write_data:
                export_data(elements, test_config, pixel_range)

                mock_normalize_values.assert_called_once_with(
                    [ele.data for ele in elements], pixel_range
                )
                mock_load_user_function.assert_called_once_with(
                    "mock_script.py", "mock_function"
                )
                mock_write_data.assert_called_once_with(
                    expected_string, test_config
                )
