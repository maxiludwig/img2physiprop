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
    determine_image_format,
    verify_and_load_imagedata,
)
from PIL import Image
from pydicom.data import get_testdata_file


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
