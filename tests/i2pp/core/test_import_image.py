"""Test Image Reader Routine."""

from pathlib import Path

import numpy as np
import pydicom
import pytest
from i2pp.core.discretization_reader_classes.discretization_reader import (
    BoundingBox,
)
from i2pp.core.import_image import (
    ImageFormat,
    _detect_and_append_suffixes,
    determine_image_format,
    verify_and_load_imagedata,
)
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


def test_detect_and_append_suffixes_png(tmp_path: Path):
    """Test _detect_and_append_suffixes correctly renames PNG files."""
    file = tmp_path / "testimage"
    Image.new("RGB", (10, 10)).save(file.with_suffix(".png"))
    file.with_suffix(".png").rename(file)

    _detect_and_append_suffixes(tmp_path)

    new_file = tmp_path / "testimage.png"
    assert new_file.exists()


def test_detect_and_append_suffixes_dicom(tmp_path: Path):
    """Test _detect_and_append_suffixes correctly renames DICOM files."""
    example_file = get_testdata_file("CT_small.dcm")
    ds = pydicom.dcmread(example_file)
    dicom_file = tmp_path / "slice"
    ds.save_as(dicom_file, enforce_file_format=False)

    _detect_and_append_suffixes(tmp_path)

    new_file = tmp_path / "slice.dcm"
    assert new_file.exists()


def test_detect_and_append_suffixes_mixed(tmp_path: Path):
    """Test _detect_and_append_suffixes does not rename unsupported files."""
    text_file = tmp_path / "note"
    text_file.write_text("This is not an image.")
    renamed_files_before = list(tmp_path.iterdir())

    _detect_and_append_suffixes(tmp_path)

    renamed_files_after = list(tmp_path.iterdir())
    assert renamed_files_before == renamed_files_after


def test_determine_image_format_not_exist():
    """Test determine_image_format when Path not exist."""

    path = Path("not_existing_path")

    with pytest.raises(
        RuntimeError,
        match="Path not_existing_path to the image data cannot be found!",
    ):
        determine_image_format(path)


def test_determine_image_format_dicom(tmp_path: Path):
    """Test determine_image_format when Input is dicom."""

    example_file = get_testdata_file("CT_small.dcm")
    ds = pydicom.dcmread(example_file)
    dicom_file_path = tmp_path / "testdicom.dcm"
    ds.save_as(dicom_file_path, enforce_file_format=False)

    input_path = tmp_path

    assert determine_image_format(input_path) == ImageFormat.DICOM


def test_determine_image_format_png(tmp_path: Path):
    """Test determine_image_format when Input is PNG."""

    width, height = 2, 2
    image = Image.new("RGB", (width, height), color=(255, 255, 255))

    png_file_path = tmp_path / "testpng.png"
    image.save(png_file_path)

    input_path = tmp_path

    assert determine_image_format(input_path) == ImageFormat.PNG


def test_determine_image_format_png_and_dicom(tmp_path: Path):
    """Test determine_image_format when Input is dicom and PNG."""

    width, height = 2, 2
    image = Image.new("RGB", (width, height), color=(255, 255, 255))

    png_file_path = tmp_path / "testpng.png"
    image.save(png_file_path)

    example_file = get_testdata_file("CT_small.dcm")
    ds = pydicom.dcmread(example_file)
    dicom_file_path = tmp_path / "testdicom.dcm"
    ds.save_as(dicom_file_path, enforce_file_format=False)

    input_path = tmp_path

    with pytest.raises(
        RuntimeError, match="Image data folder contains multiple format types!"
    ):
        determine_image_format(input_path)


def test_determine_image_format_no_data(tmp_path: Path):
    """Test determine_image_format when Path has no readable data."""

    input_path = tmp_path

    with pytest.raises(
        RuntimeError,
        match="Image data folder is empty or has no readable data",
    ):
        determine_image_format(input_path)


def test_verify_and_load_imagedata(tmp_path: Path):
    """Test verify_and_load_imagedata."""

    example_file = get_testdata_file("CT_small.dcm")
    ds = pydicom.dcmread(example_file)
    dicom_file_path1 = tmp_path / "testdicom1.dcm"
    dicom_file_path2 = tmp_path / "testdicom2.dcm"
    ds.save_as(dicom_file_path1, enforce_file_format=False)
    ds.save_as(dicom_file_path2, enforce_file_format=False)
    input_path = tmp_path

    config = {"input informations": {"image_folder_path": input_path}}

    image_data = verify_and_load_imagedata(
        config, BoundingBox(max=[0, 0, 1000], min=[1, 1, -1000])
    )

    assert np.array_equal(image_data.pixel_data.shape, (2, 128, 128))

    expected_orientation = np.column_stack(
        (np.array([0, 0, 1]), np.array([0, 1, 0]), np.array([1, 0, 0]))
    )

    assert np.array_equal(image_data.orientation, expected_orientation)
