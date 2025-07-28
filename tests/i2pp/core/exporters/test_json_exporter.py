"""Test suite for the JsonExporter class in i2pp.core.exporters.json_exporter
module."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
from i2pp.core.exporters.json_exporter import JsonExporter


def test_json_exporter_write_data_success():
    """Test JsonExporter writes valid structured array to JSON file."""
    data = np.array(
        [(1, [2.0, 3.0], "string1", 5), (2, [6.0, 7.0], "string2", 9)],
        dtype=[
            ("index", "i4"),
            ("property1", "f8", 2),
            ("property2", "U10"),
            ("property3", "i4"),
        ],
    )
    expected_output = {
        "MUE": {
            "1": [[2.0, 3.0], "string1", 5],
            "2": [[6.0, 7.0], "string2", 9],
        }
    }
    exporter = JsonExporter()

    with tempfile.TemporaryDirectory() as tmpdir:
        output_file = Path(tmpdir) / "output.json"
        exporter.write_data(data, output_file, "MUE")

        with open(output_file, "r") as f:
            written_data = json.load(f)
            assert written_data == expected_output


@pytest.mark.parametrize(
    "invalid_data,expected_msg",
    [
        (
            "not a numpy array",
            "must return a structured numpy array",
        ),
        (
            np.array([1, 2], dtype=int),
            "must have named fields",
        ),
        (
            np.array(
                [(2.3, [1.1, 2.5])],
                dtype=[("index", "f8"), ("property1", "f8", 2)],
            ),
            "first field.*must be integer-valued",
        ),
        (
            np.array(
                [(1, [1.1, 2.5])],
                dtype=[("not_index", "i4"), ("property1", "f8", 2)],
            ),
            "first field.*must be named 'index'",
        ),
        (
            np.array([1], dtype=[("index", "i4")]),
            "must have at least one additional field",
        ),
    ],
)
def test_json_exporter_invalid_inputs(invalid_data, expected_msg):
    """Test JsonExporter handles invalid structured array inputs."""
    exporter = JsonExporter()
    with tempfile.TemporaryDirectory() as tmpdir:
        output_file = Path(tmpdir) / "output.json"
        with pytest.raises(AssertionError, match=expected_msg):
            exporter.write_data(invalid_data, output_file, "MUE")


def test_json_exporter_missing_property_name():
    """Test JsonExporter raises error if property name not given."""
    exporter = JsonExporter()
    valid_data = np.array(
        [(1, [1.0, 2.0])],
        dtype=[("index", "i4"), ("property1", "f8", 2)],
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        output_file = Path(tmpdir) / "output.json"
        with pytest.raises(
            RuntimeError,
            match="you must also specify the 'name_of_output_property'",
        ):
            exporter.write_data(valid_data, output_file)


def test_json_exporter_unserializable_data():
    """Test JsonExporter raises error for unserializable content."""
    exporter = JsonExporter()
    arr = np.array(
        [(1, np.array([1.1, 2.5]), lambda x: x)],
        dtype=[
            ("index", "i4"),
            ("property1", "f8", 2),
            ("property2", "O"),
        ],
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        output_file = Path(tmpdir) / "output.json"
        with pytest.raises(
            RuntimeError,
            match="Ensure all data is JSON serializable",
        ):
            exporter.write_data(arr, output_file, "bad_prop")
