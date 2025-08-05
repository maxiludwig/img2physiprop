"""Test cases for image format detection class in i2pp.core.image_readers."""

from pathlib import Path

import pydicom
import pytest
from i2pp.core.image_readers.dicom_reader import DicomReader
from i2pp.core.image_readers.image_format import ImageFormat
from i2pp.core.image_readers.image_reader import ImageReader
from i2pp.core.image_readers.png_reader import PngReader
from PIL import Image
from pydicom.data import get_testdata_file


def test_is_file_of_format_with_png(tmp_path: Path):
    """Test is_file_of_format returns True for PNG file."""
    file = tmp_path / "image.png"
    Image.new("RGB", (10, 10)).save(file)
    assert ImageFormat.PNG.is_file_of_format(file)


def test_is_file_of_format_with_dicom(tmp_path: Path):
    """Test is_file_of_format returns True for DICOM file."""
    example_file = get_testdata_file("CT_small.dcm")
    ds = pydicom.dcmread(example_file)
    dicom_file = tmp_path / "slice"
    ds.save_as(dicom_file, enforce_file_format=False)
    assert ImageFormat.DICOM.is_file_of_format(dicom_file)


def test_is_file_of_format_with_unrelated_file(tmp_path: Path):
    """Test is_file_of_format returns False for unrelated file."""
    file = tmp_path / "text.txt"
    file.write_text("This is not image data.")
    assert not ImageFormat.DICOM.is_file_of_format(file)
    assert not ImageFormat.PNG.is_file_of_format(file)


def test_image_format_dicom_reader():
    """Test that DICOM format returns the correct reader class."""
    reader_class = ImageFormat.DICOM.get_reader()
    assert issubclass(reader_class, ImageReader)
    assert reader_class is DicomReader


def test_image_format_png_reader():
    """Test that PNG format returns the correct reader class."""
    reader_class = ImageFormat.PNG.get_reader()
    assert issubclass(reader_class, ImageReader)
    assert reader_class is PngReader


def test_image_format_invalid():
    """Test get_reader raises ValueError for unsupported image format."""

    class FakeImageFormat:
        """Fake enum class to simulate an unsupported image format."""

        pass

    with pytest.raises(ValueError, match="Unsupported image format"):
        ImageFormat.get_reader(FakeImageFormat())
