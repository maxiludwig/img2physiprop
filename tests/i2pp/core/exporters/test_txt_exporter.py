"""Test suite for the TxtExporter class in i2pp.core.exporters.txt_exporter
module."""

from pathlib import Path
from unittest import mock

import pytest
from i2pp.core.exporters.txt_exporter import TxtExporter


def test_txt_exporter_initialization():
    """Test that TxtExporter sets export format correctly."""
    exporter = TxtExporter()
    assert exporter.export_format == "txt"


def test_txt_exporter_write_data_success():
    """Test TxtExporter writes string data to a text file."""
    exporter = TxtExporter()
    output_path = Path("some_path/output.txt")
    data = "Sample export string."

    with mock.patch("builtins.open", mock.mock_open()) as mocked_file:
        with mock.patch.object(exporter, "_validate_outfile") as mock_validate:
            result = exporter.write_data(data, output_path)

            mocked_file.assert_called_once_with(output_path, "w")
            mocked_file().write.assert_called_once_with(data)
            mock_validate.assert_called_once_with(output_path)
            assert result == {}


def test_txt_exporter_write_data_invalid_type():
    """Test TxtExporter raises assertion if non-string data is passed."""
    exporter = TxtExporter()
    output_path = Path("some_path/output.txt")

    with pytest.raises(AssertionError, match="must return a string"):
        exporter.write_data(12345, output_path)
