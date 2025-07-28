"""Test suite for the TxtExporter class in i2pp.core.exporters.txt_exporter
module."""

import tempfile
from pathlib import Path

import pytest
from i2pp.core.exporters.txt_exporter import TxtExporter


def test_txt_exporter_initialization():
    """Test that TxtExporter sets export format correctly."""
    exporter = TxtExporter()
    assert exporter.export_format == "txt"


def test_txt_exporter_write_data_success():
    """Test TxtExporter writes string data to a text file."""
    exporter = TxtExporter()
    expected_output = "Sample export string."

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "output.txt"
        exporter.write_data(expected_output, output_path)

        with open(output_path, "r") as txt_file:
            written_data = txt_file.read()
            assert written_data == expected_output


def test_txt_exporter_write_data_success_missing_suffix():
    """Test TxtExporter writes string data to a text file even if no suffix is
    provided."""
    exporter = TxtExporter()
    expected_output = "Sample export string."

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "output"
        exporter.write_data(expected_output, output_path)

        with open(output_path.with_suffix(".txt"), "r") as txt_file:
            written_data = txt_file.read()
            assert written_data == expected_output


def test_txt_exporter_write_data_invalid_type():
    """Test TxtExporter raises assertion if non-string data is passed."""
    exporter = TxtExporter()
    output_path = Path("some_path/output.txt")

    with pytest.raises(AssertionError, match="must return a string"):
        exporter.write_data(12345, output_path)
