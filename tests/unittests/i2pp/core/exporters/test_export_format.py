"""Test cases for export format detection class in i2pp.core.exporters."""

import pytest
from i2pp.core.exporters.export_format import ExportFormat
from i2pp.core.exporters.json_exporter import JsonExporter
from i2pp.core.exporters.txt_exporter import TxtExporter


def test_export_format_enum():
    """Test ExportFormat Enum."""
    assert ExportFormat("json") == ExportFormat.JSON
    assert ExportFormat("txt") == ExportFormat.TXT

    with pytest.raises(ValueError):
        ExportFormat("xml")


def test_get_exporter_json():
    """Test that the correct exporter is returned for JSON format."""
    assert ExportFormat.JSON.get_exporter() is JsonExporter


def test_get_exporter_txt():
    """Test that the correct exporter is returned for TXT format."""
    assert ExportFormat.TXT.get_exporter() is TxtExporter


def test_get_exporter_invalid(monkeypatch):
    """Test that an error is raised for an unsupported export format."""

    class FakeExportFormat:
        """Fake ExportFormat for testing."""

        def __str__(self):
            """Return a string representation for the fake format."""
            return "fake"

    fake_format = FakeExportFormat()

    with pytest.raises(ValueError, match="Unsupported export format: fake"):
        ExportFormat.get_exporter(fake_format)
