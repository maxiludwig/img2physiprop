"""Test Export Data Routine."""

import json
import os
import tempfile
from pathlib import Path
from unittest import mock
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from i2pp.core.discretization_reader_classes.discretization_reader import (
    Element,
)
from i2pp.core.export_data import Exporter, ExportFormat, export_data
from i2pp.core.image_reader_classes.image_reader import PixelValueType


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


def test_parse_export_format_invalid_format():
    """Test parse_export_format raises error for invalid format."""
    test_config = {"output options": {"export_format": "invalid_format"}}
    exporter = Exporter()

    with pytest.raises(
        RuntimeError, match="Export format 'invalid_format' is not supported!"
    ):
        exporter.parse_export_format(test_config)


def test_parse_export_format_valid_format():
    """Test parse_export_format sets the correct export format."""
    test_config = {"output options": {"export_format": "json"}}
    exporter = Exporter()
    exporter.parse_export_format(test_config)

    assert exporter.export_format == ExportFormat.JSON


def test_write_data_creates_json_file():
    """Test write_data creates JSON file with correct content."""
    data = np.array(
        [(1, [2.0, 3.0], "string1", 5), (2, [6.0, 7.0], "string2", 9)],
        dtype=[
            ("index", "i4"),
            ("property1", "f8", 2),
            ("property2", "U10"),
            ("property3", "i4"),
        ],
    )
    test_config = {
        "output options": {
            "output_name": "test_output",
            "name_of_output_property": "MUE",
        }
    }
    expected_output = {
        "MUE": {
            "1": [[2.0, 3.0], "string1", 5],
            "2": [[6.0, 7.0], "string2", 9],
        }
    }
    exporter = Exporter()
    exporter.export_format = ExportFormat.JSON

    with tempfile.TemporaryDirectory() as temp_dir:
        test_config["output options"]["output_path"] = temp_dir
        exporter.write_data(data, test_config)

        with open(
            os.path.join(temp_dir, "test_output.json"), "r"
        ) as json_file:
            written_data = json.load(json_file)
            assert written_data == expected_output

        # Error handling
        with pytest.raises(
            AssertionError,
            match=(
                "You specified a JSON export format. In this case, the user "
                "function must return a structured numpy array. First field "
                "must be integer-valued and named 'index'. The other fields "
                "can be of any type, but must be JSON serializable."
            ),
        ):
            exporter.write_data("not a numpy array", test_config)
        with pytest.raises(
            AssertionError,
            match=(
                "The structured numpy array must have named fields. "
                "Adapt the user function."
            ),
        ):
            exporter.write_data(np.array([1, 2], dtype=int), test_config)
        with pytest.raises(
            AssertionError,
            match=(
                "The first field of the structured numpy array must be "
                "integer-valued. Adapt the user function."
            ),
        ):
            exporter.write_data(
                np.array(
                    [(2.3, [1.1, 2.5])],
                    dtype=[("index", "f8"), ("property1", "f8", 2)],
                ),
                test_config,
            )

        with pytest.raises(
            AssertionError,
            match=(
                "The first field of the structured numpy array must be named "
                "'index'. Adapt the user function."
            ),
        ):
            exporter.write_data(
                np.array(
                    [(1, [1.1, 2.5])],
                    dtype=[("not_index", "i4"), ("property1", "f8", 2)],
                ),
                test_config,
            )
        with pytest.raises(
            AssertionError,
            match=(
                "The structured numpy array must have at least one additional "
                "field. Adapt the user function."
            ),
        ):
            exporter.write_data(
                np.array([1], dtype=[("index", "i4")]), test_config
            )
        with pytest.raises(
            RuntimeError,
            match=(
                "Failed to write JSON data. Ensure all data is JSON "
                "serializable."
            ),
        ):
            arr = np.array(
                [(1, np.array([1.1, 2.5]), lambda x: x)],
                dtype=[
                    ("index", "i4"),
                    ("property1", "f8", 2),
                    ("property2", "O"),
                ],
            )
            exporter.write_data(arr, test_config)
        with pytest.raises(
            RuntimeError,
            match=(
                "You specified a JSON export format. In this case, you must "
                "also specify the 'name_of_output_property' in the "
                "configuration."
            ),
        ):
            test_config["output options"].pop("name_of_output_property")
            exporter.write_data(arr, test_config)


def test_write_data_creates_txt_file():
    """Test write_data creates a txt file."""
    export_string = "test."

    test_config = {
        "output options": {
            "output_path": str(Path.cwd() / "test_directory"),
            "output_name": "test_output",
        }
    }
    exporter = Exporter()
    exporter.export_format = ExportFormat.TXT
    file_path = os.path.join(
        Path(test_config["output options"]["output_path"]), "test_output.txt"
    )

    err_string = (
        "You specified a TXT export format. In this case, the user"
        " function must return a string."
    )
    with mock.patch("builtins.open", mock.mock_open()) as mocked_file:
        exporter.write_data(export_string, test_config)
        mocked_file.assert_called_once_with(file_path, "w")
        mocked_file().write.assert_called_once_with(export_string)

        with pytest.raises(AssertionError, match=err_string):
            exporter.write_data(0, test_config)


def test_export_vtk_adds_cell_data_and_saves_file():
    """Test export_vtk adds cell data to unstructured grid and saves the
    file."""
    test_config = {
        "output options": {
            "output_path": str(Path.cwd() / "test_directory"),
            "output_name": "test_output",
        }
    }
    elements = [Element([0, 1], 0, data=10), Element([0, 1], 1, data=20)]
    pixel_type = PixelValueType.CT
    exported_data = {
        "property_name": np.array(
            [(1, [1.0, 2.0]), (2, [2.0, 3.0])],
            dtype=[("index", "i4"), ("property1", "f8", 2)],
        )
    }

    exporter = Exporter()

    mock_grid = mock.MagicMock()
    mock_grid.cell_data = {}
    mock_discretization = mock.MagicMock()
    mock_discretization.elements = elements
    with patch(
        "i2pp.core.export_data.initialize_unstructured_grid",
        return_value=(mock_grid, None),
    ) as mock_init_grid:
        with patch.object(mock_grid, "save") as mock_save:
            exporter.export_vtk(
                test_config,
                elements,
                pixel_type,
                exported_data,
                mock_discretization,
            )

            mock_init_grid.assert_called_once_with(
                elements, pixel_type, mock_discretization
            )

            for key, value in exported_data.items():
                assert np.array_equal(
                    mock_grid.cell_data[f"{key}_property1"],
                    np.array([[1.0, 2.0], [2.0, 3.0]]),
                )

            expected_file_path = os.path.join(
                test_config["output options"]["output_path"],
                f"{test_config['output options']['output_name']}.vtu",
            )
            mock_save.assert_called_once_with(expected_file_path)


@pytest.fixture
def exporter_mocks():
    """Fixture to mock Exporter methods for testing."""
    with (
        patch(
            "i2pp.core.export_data.normalize_values",
            return_value=np.array([1, 2]),
        ) as mock_normalize_values,
        patch.object(
            Exporter,
            "parse_export_format",
            side_effect=lambda config: setattr(
                Exporter, "export_format", ExportFormat.JSON
            ),
        ) as mock_parse_export_format,
        patch.object(
            Exporter,
            "load_user_function",
            return_value=lambda ids, data: data / 5,
        ) as mock_load_user_function,
        patch.object(
            Exporter, "write_data", return_value=None
        ) as mock_write_data,
        patch.object(
            Exporter, "export_vtk", return_value=None
        ) as mock_export_vtk,
    ):
        yield {
            "mock_normalize_values": mock_normalize_values,
            "mock_parse_export_format": mock_parse_export_format,
            "mock_load_user_function": mock_load_user_function,
            "mock_write_data": mock_write_data,
            "mock_export_vtk": mock_export_vtk,
        }


def test_export_data(exporter_mocks):
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
    expected_result = np.array([0.2, 0.4])
    mock_discretization = MagicMock()

    export_data(
        elements,
        mock_discretization,
        test_config,
        pixel_range,
        PixelValueType.CT,
    )

    exporter_mocks["mock_parse_export_format"].assert_called_once_with(
        test_config
    )
    args, _ = exporter_mocks["mock_normalize_values"].call_args
    np.testing.assert_array_equal(args[0], np.array([10, 20]))
    np.testing.assert_array_equal(args[1], np.array([0, 20]))

    exporter_mocks["mock_load_user_function"].assert_called_once_with(
        "mock_script.py", "mock_function"
    )

    args, _ = exporter_mocks["mock_write_data"].call_args
    np.testing.assert_array_equal(args[0], expected_result)
    assert args[1] == test_config
    exporter_mocks["mock_export_vtk"].assert_called_once_with(
        test_config, elements, PixelValueType.CT, None, mock_discretization
    )
