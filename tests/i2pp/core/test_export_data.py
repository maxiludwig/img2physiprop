"""Test Export Data Routine."""

from pathlib import Path
from unittest import mock
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from i2pp.core.discretization_readers.discretization_reader import Element
from i2pp.core.export_data import ExportFormat, export_data, export_vtk
from i2pp.core.image_readers.image_reader import PixelValueType


def test_export_vtk_adds_cell_data_and_saves_file():
    """Test export_vtk adds cell data to unstructured grid and saves the
    file."""
    vtk_output_path = Path(
        str(Path.cwd() / "test_directory" / "test_output.vtu"),
    )
    elements = [Element([0, 1], 0, data=10), Element([0, 1], 1, data=20)]
    pixel_type = PixelValueType.CT
    exported_data = {
        "property_name": np.array(
            [(1, [1.0, 2.0]), (2, [2.0, 3.0])],
            dtype=[("index", "i4"), ("property1", "f8", 2)],
        )
    }

    mock_grid = mock.MagicMock()
    mock_grid.cell_data = {}
    mock_discretization = mock.MagicMock()
    mock_discretization.elements = elements
    with patch(
        "i2pp.core.export_data.initialize_unstructured_grid",
        return_value=(mock_grid, None),
    ) as mock_init_grid:
        with patch.object(mock_grid, "save") as mock_save:
            export_vtk(
                vtk_output_path,
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

            mock_save.assert_called_once_with(vtk_output_path)


@pytest.fixture
def exporter_mocks(monkeypatch):
    """Mocks Exporter.write_data and export_vtk."""

    # Mock return from exporter.write_data
    mock_write_data = MagicMock(
        return_value={"property_name": np.array([[0, 0.1], [1, 0.2]])}
    )

    # Mock exporter class
    class MockExporter:
        """Mock Exporter class for testing."""

        export_format = "json"

        def write_data(self, *args, **kwargs):
            """Mock write_data method."""
            return mock_write_data(*args, **kwargs)

    # Patch ExportFormat.get_exporter to return MockExporter
    def mock_get_exporter(self):
        """Return MockExporter for testing."""
        return MockExporter

    monkeypatch.setattr(ExportFormat, "get_exporter", mock_get_exporter)

    # Patch export_vtk function
    with patch("i2pp.core.export_data.export_vtk") as mock_export_vtk:
        yield {
            "mock_write_data": mock_write_data,
            "mock_export_vtk": mock_export_vtk,
        }


def test_export_data(exporter_mocks, tmp_path):
    """Test full export_data pipeline with normalization and JSON export."""

    # Arrange: Create temporary user script defining mock function
    user_script = tmp_path / "mock_script.py"
    user_script.write_text("def mock_function(ids, data): return data / 5\n")

    # Define parameters
    element1 = Element([0, 1], 0, data=10)
    element2 = Element([0, 1], 1, data=20)
    elements = [element1, element2]
    expected_transformed = np.array([0.1, 0.2])
    expected_output_file = tmp_path / "output.json"
    expected_vtk_file = tmp_path / "output.vtu"

    # Fake discretization object
    mock_discretization = MagicMock()

    # Act: Call export_data
    export_data(
        transformed_data=expected_transformed,
        elements=elements,
        discretization=mock_discretization,
        export_format="json",
        property_output_file=expected_output_file,
        name_of_output_property="property_name",
        vtk_output_file=expected_vtk_file,
        pixel_type=PixelValueType.CT,
    )

    # Assert: Check write_data was called correctly
    write_data_args, _ = exporter_mocks["mock_write_data"].call_args
    np.testing.assert_array_equal(write_data_args[0], expected_transformed)
    assert write_data_args[1] == expected_output_file
    assert write_data_args[2] == "property_name"

    # Assert: Check export_vtk was called with correct structured data
    exporter_mocks["mock_export_vtk"].assert_called_once()
    vtk_args, _ = exporter_mocks["mock_export_vtk"].call_args

    assert vtk_args[0] == expected_vtk_file
    assert vtk_args[1] == elements
    assert vtk_args[2] == PixelValueType.CT
    assert vtk_args[4] == mock_discretization

    actual_data = vtk_args[3]["property_name"]
    expected_data = np.array([[0.0, 0.1], [1.0, 0.2]])

    np.testing.assert_array_equal(actual_data, expected_data)
