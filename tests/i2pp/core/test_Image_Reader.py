"""Test Image Reader Routine."""

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pydicom
import pytest
from i2pp.core.discretization_reader_classes.discretization_reader import (
    Limits,
)
from i2pp.core.image_reader_classes.dicom_reader import DicomReader
from i2pp.core.image_reader_classes.image_reader import PixelValueType
from i2pp.core.image_reader_classes.png_reader import PngReader
from i2pp.core.import_image import (
    ImageFormat,
    determine_image_format,
    verify_and_load_imagedata,
)
from PIL import Image
from pydicom.data import get_testdata_file


def test_determine_image_format_not_exist():
    """Test determine_image_format when Path not exist."""

    directory = Path("not_existing_path")

    with pytest.raises(
        RuntimeError,
        match="Path not_existing_path to the image data cannot be found!",
    ):
        determine_image_format(directory)


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


def test_load_image_dicom(tmp_path: Path):
    """Test load_image if funtion is called two times for two slices and slices
    are in a correct order."""

    example_file1 = get_testdata_file("CT_small.dcm")
    example_file2 = get_testdata_file("MR_small.dcm")
    ds1 = pydicom.dcmread(example_file1)
    ds2 = pydicom.dcmread(example_file2)
    dicom_file_path1 = tmp_path / "testdicom1.dcm"
    dicom_file_path2 = tmp_path / "testdicom2.dcm"
    ds1.save_as(dicom_file_path1, enforce_file_format=False)
    ds2.save_as(dicom_file_path2, enforce_file_format=False)

    test_input = DicomReader({}, Limits(max=[0, 0, 1000], min=[1, 1, -1000]))
    input_path = tmp_path
    with patch("pydicom.dcmread", wraps=pydicom.dcmread) as mock_dcmread:

        dicom = test_input.load_image(input_path)

        assert mock_dcmread.call_count == 2
        assert (
            dicom[0].ImagePositionPatient[2]
            <= dicom[1].ImagePositionPatient[2]
        )


def test_image_to_slices_dicom():
    """Test image_to_slices for dicom."""

    ds = []
    example_file = get_testdata_file("MR_small.dcm")
    ds = pydicom.dcmread(example_file)

    test_class = DicomReader([], Limits(min=[1, 1, -1000], max=[0, 0, 1000]))
    slices_and_metadata1 = test_class.image_to_slices([ds, ds])

    assert slices_and_metadata1.metadata.pixel_type == PixelValueType.MRT


def test_verify_image_metadata_input_missing():
    """_verify_image_metadata if information is missing."""

    test_config = {
        "image_metadata": {
            "pixel_spacing": [1, 1],
            "slice_thickness": 1,
            "image_position": None,
            "image_orientation": [0, 1, 0, 1, 0, 0],
        }
    }
    test_class = PngReader(test_config, [])
    with pytest.raises(
        RuntimeError,
        match="Missing parameter 'image_position' in image_metadata.",
    ):
        test_class._verify_image_metadata(test_config["image_metadata"])


def test_verify_image_metadata_wrong_type():
    """_verify_image_metadata when information has wrong type."""

    test_config = {
        "image_metadata": {
            "pixel_spacing": [1, 1],
            "slice_thickness": 1,
            "image_position": 1,
            "image_orientation": [0, 1, 0, 1, 0, 0],
        }
    }
    test_class = PngReader(test_config, [])
    with pytest.raises(
        RuntimeError, match="Parameter 'image_position' has the wrong type."
    ):
        test_class._verify_image_metadata(test_config["image_metadata"])


def test_verify_image_metadata_wrong_shape():
    """_verify_image_metadata when Spacing is wrong."""

    test_config = {
        "image_metadata": {
            "pixel_spacing": [1],
            "slice_thickness": 1,
            "image_position": [0, 0, 0],
            "image_orientation": [0, 1, 0, 1, 0, 0],
        }
    }
    test_class = PngReader(test_config, [])
    with pytest.raises(
        RuntimeError, match="Parameter 'pixel_spacing' has the wrong shape."
    ):
        test_class._verify_image_metadata(test_config["image_metadata"])


def test_verify_image_metadata_wrong_Slice_Thickness():
    """_verify_image_metadata when Slice_Thickness is wrong."""

    test_config = {
        "image_metadata": {
            "pixel_spacing": [1, 1],
            "slice_thickness": [1, 3],
            "image_position": [0, 0, 0],
            "image_orientation": [0, 1, 0, 1, 0, 0],
        }
    }
    test_class = PngReader(
        test_config, Limits(max=[0, 0, 1000], min=[1, 1, -1000])
    )
    with pytest.raises(
        RuntimeError, match="Parameter 'slice_thickness' must be a number"
    ):
        test_class._verify_image_metadata(test_config["image_metadata"])


def test_verify_image_metadata_default_orientation():
    """Test _verify_image_metadata if all config_parameters are None."""

    test_config = {
        "image_metadata": {
            "pixel_spacing": [1, 1],
            "slice_thickness": 1,
            "image_position": [0, 0, 0],
            "image_orientation": None,
        }
    }
    test_class = PngReader(
        test_config, Limits(max=[0, 0, 1000], min=[1, 1, -1000])
    )

    assert (
        test_class._verify_image_metadata(test_config["image_metadata"])
        is None
    )


def test_load_image_png(tmp_path: Path):
    """Test load_image for PNG."""

    width, height = 2, 2
    image_1 = Image.new("RGB", (width, height), color=(255, 255, 255))
    image_2 = Image.new("RGB", (width, height), color=(55, 55, 55))

    png_file_path_1 = tmp_path / "testpng_1.png"
    png_file_path_2 = tmp_path / "testpng_2.png"
    image_1.save(png_file_path_1)
    image_2.save(png_file_path_2)

    test_config = {"image_metadata": "Test"}

    test_input = PngReader(test_config, Limits([], []))
    input_path = tmp_path

    with patch("PIL.Image.open", wraps=Image.open) as mock_image_open:
        with patch.object(
            PngReader, "_verify_image_metadata", return_value=None
        ):
            test_input.load_image(input_path)

            assert mock_image_open.call_count == 2


def test_image_to_slices_PNG():
    """Test image_to_slices for PNG."""

    test_config = {
        "image_metadata": {
            "pixel_spacing": [1, 1],
            "slice_thickness": 1,
            "image_position": [0, 0, 0],
            "image_orientation": [0, 1, 0, 1, 0, 0],
        }
    }
    png = []
    png.append([[1, 1], [2, 2]])
    png.append([[3, 3], [4, 4]])

    test_input = PngReader(
        test_config, Limits(max=[0, 0, 1000], min=[1, 1, -1000])
    )
    slice_and_meta = test_input.image_to_slices(png)

    assert slice_and_meta.metadata.pixel_type == PixelValueType.RGB
    assert np.array_equal(
        slice_and_meta.slices[1].position, np.array([0, 0, 1])
    )


def test_image_to_slices_png_dafault_orientation():
    """Test image_to_slices for PNG if Image_Orientation is None."""

    test_config = {
        "image_metadata": {
            "pixel_spacing": [1, 1],
            "slice_thickness": 1,
            "image_position": [0, 0, 0],
            "image_orientation": None,
        }
    }
    png = []
    png.append([[1, 1], [2, 2]])
    png.append([[1, 1], [2, 2]])
    png.append([[1, 1], [2, 2]])

    test_input = PngReader(test_config, Limits(max=[0, 0, 2], min=[1, 1, 1]))
    slices_and_meta = test_input.image_to_slices(png)

    assert np.array_equal(
        slices_and_meta.metadata.pixel_spacing, np.array([1, 1])
    )
    assert np.array_equal(
        slices_and_meta.slices[0].position, np.array([0, 0, 1])
    )
    assert np.array_equal(
        slices_and_meta.slices[1].position, np.array([0, 0, 2])
    )
    assert np.array_equal(
        slices_and_meta.metadata.orientation, np.array([0, -1, 0, 1, 0, 0])
    )


def test_verify_and_load_imagedata(tmp_path: Path):
    """Test verify_and_load_imagedata."""

    example_file = get_testdata_file("CT_small.dcm")
    ds = pydicom.dcmread(example_file)
    dicom_file_path = tmp_path / "testdicom.dcm"
    ds.save_as(dicom_file_path, enforce_file_format=False)
    input_path = tmp_path

    config = {"input informations": {"image_folder_path": input_path}}

    slices = verify_and_load_imagedata(
        config, Limits(max=[0, 0, 1000], min=[1, 1, -1000])
    )

    assert slices
