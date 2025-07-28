"""Test cases for image format detection class in i2pp.core.image_readers."""

from pathlib import Path

import pydicom
from i2pp.core.image_readers.image_format import ImageFormat
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
