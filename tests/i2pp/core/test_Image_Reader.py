"""Test Image Reader Routine."""

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pydicom
import pytest
from i2pp.core.Image_Reader import (
    DicomReader,
    PngReader,
    verify_and_load_imagedata,
    verify_input,
)
from PIL import Image
from pydicom.data import get_testdata_file


# class test_input_reader(unittest.TestCase):
def test_verify_input_imagedata_not_exist():
    """Test verify_input when Path not exist."""

    directory = "not_existing_path"

    with pytest.raises(RuntimeError, match="Imagedata file not found!"):
        verify_input(directory)


def test_verify_input_dicom(tmp_path: Path):
    """Test verify_input when Input is dicom."""

    example_file = get_testdata_file("CT_small.dcm")
    ds = pydicom.dcmread(example_file)
    dicom_file_path = tmp_path / "testdicom.dcm"
    ds.save_as(dicom_file_path, enforce_file_format=False)

    input_path = str(tmp_path) + "/"

    assert verify_input(input_path) == "dicom"


def test_verify_input_png(tmp_path: Path):
    """Test verify_input when Input is PNG."""

    width, height = 2, 2
    image = Image.new("RGB", (width, height), color=(255, 255, 255))

    png_file_path = tmp_path / "testpng.png"
    image.save(png_file_path)

    input_path = str(tmp_path) + "/"

    assert verify_input(input_path) == "png"


def test_verify_input_png_and_dicom(tmp_path: Path):
    """Test verify_input when Input is dicom and PNG."""

    width, height = 2, 2
    image = Image.new("RGB", (width, height), color=(255, 255, 255))

    png_file_path = tmp_path / "testpng.png"
    image.save(png_file_path)

    example_file = get_testdata_file("CT_small.dcm")
    ds = pydicom.dcmread(example_file)
    dicom_file_path = tmp_path / "testdicom.dcm"
    ds.save_as(dicom_file_path, enforce_file_format=False)

    input_path = str(tmp_path) + "/"

    with pytest.raises(
        RuntimeError, match="Input data file has two different format types!"
    ):
        verify_input(input_path)


def test_verify_input_no_data(tmp_path: Path):
    """Test verify_input when Path has no readable data."""

    input_path = str(tmp_path) + "/"

    with pytest.raises(
        RuntimeError, match="Input data file is empty or has no readible data!"
    ):
        verify_input(input_path)


def test_load_dicom(tmp_path: Path):
    """Test load_dicom if funtion is called two times for two slices and slices
    are in a correct order."""

    example_file1 = get_testdata_file("CT_small.dcm")
    example_file2 = get_testdata_file("MR_small.dcm")
    ds1 = pydicom.dcmread(example_file1)
    ds2 = pydicom.dcmread(example_file2)
    dicom_file_path1 = tmp_path / "testdicom1.dcm"
    dicom_file_path2 = tmp_path / "testdicom2.dcm"
    ds1.save_as(dicom_file_path1, enforce_file_format=False)
    ds2.save_as(dicom_file_path2, enforce_file_format=False)

    test_input = DicomReader("dicom", [], [])
    input_path = str(tmp_path) + "/"
    with patch("pydicom.dcmread", wraps=pydicom.dcmread) as mock_dcmread:
        # load_dicom aufrufen
        dicom = test_input.load_image(input_path)

        assert mock_dcmread.call_count == 2
        assert (
            dicom[0].ImagePositionPatient[2]
            <= dicom[1].ImagePositionPatient[2]
        )


def test_dicom_2_slices():
    """Test dicom_2_slices if slices are processed correctly."""

    ds = []
    example_file1 = get_testdata_file("MR_small.dcm")
    example_file2 = get_testdata_file("CT_small.dcm")
    ds.append(pydicom.dcmread(example_file1))
    ds.append(pydicom.dcmread(example_file2))

    test_class = DicomReader("dicom", [], [])
    slice = test_class.image_2_slices(ds)

    assert slice[0].Modality == "MR"
    assert slice[1].Modality == "CT"


def test_verify_additional_informations_wrong_Spacing():
    """verify_additional_informations when Spacing is wrong."""

    test_config = {
        "Additional Information": {
            "Pixel_Spacing": [1],
            "Slice_Thickness": 1,
            "Image_Position": [0, 0, 0],
            "Image_Orientation": [0, 1, 0, 1, 0, 0],
            "Modality": "CT",
        }
    }
    test_class = PngReader("png", test_config, [])
    with pytest.raises(
        RuntimeError, match="Parameter 'Spacing' not readable."
    ):
        test_class.verify_additional_informations(test_config)


def test_verify_additional_informations_wrong_Slice_Thickness():
    """verify_additional_informations when Slice_Thickness is wrong."""

    test_config = {
        "Additional Information": {
            "Pixel_Spacing": [1, 1],
            "Slice_Thickness": [3, 1],
            "Image_Position": [0, 0, 0],
            "Image_Orientation": [0, 1, 0, 1, 0, 0],
            "Modality": "CT",
        }
    }
    test_class = PngReader("png", test_config, [])
    with pytest.raises(
        RuntimeError, match="Parameter 'Slice_Thickness' not readable."
    ):
        test_class.verify_additional_informations(test_config)


def test_verify_additional_informations_wrong_Image_Position():
    """verify_additional_informations when Image_Position is wrong."""

    test_config = {
        "Additional Information": {
            "Pixel_Spacing": [1, 1],
            "Slice_Thickness": 1,
            "Image_Position": [0, 0],
            "Image_Orientation": [0, 1, 0, 1, 0, 0],
            "Modality": "CT",
        }
    }
    test_class = PngReader("png", test_config, [])
    with pytest.raises(
        RuntimeError, match="Parameter 'Image_Position' not readable."
    ):
        test_class.verify_additional_informations(test_config)


def test_verify_additional_informations_wrong_Image_Orientation():
    """verify_additional_informations when Image_Orientation is wrong."""

    test_config = {
        "Additional Information": {
            "Pixel_Spacing": [1, 1],
            "Slice_Thickness": 1,
            "Image_Position": [0, 0, 0],
            "Image_Orientation": [0, 1, 0, 1, 0],
            "Modality": "CT",
        }
    }
    test_class = PngReader("png", test_config, [])
    with pytest.raises(
        RuntimeError, match="Parameter 'Image_Orientation' not readable."
    ):
        test_class.verify_additional_informations(test_config)


def test_verify_additional_informations_wrong_Modality():
    """verify_additional_informations when Modality is wrong."""

    test_config = {
        "Additional Information": {
            "Pixel_Spacing": [1, 1],
            "Slice_Thickness": 1,
            "Image_Position": [0, 0, 0],
            "Image_Orientation": [0, 1, 0, 1, 0, 1],
            "Modality": [1, 2, 4],
        }
    }
    test_class = PngReader("png", test_config, [])
    with pytest.raises(
        RuntimeError, match="Parameter 'Modality' not readable."
    ):
        test_class.verify_additional_informations(test_config)


def test_load_png(tmp_path: Path):
    """Test load_png if funtion is called two times for two slices."""

    width, height = 2, 2
    image_1 = Image.new("RGB", (width, height), color=(255, 255, 255))
    image_2 = Image.new("RGB", (width, height), color=(55, 55, 55))

    png_file_path_1 = tmp_path / "testpng_1.png"
    png_file_path_2 = tmp_path / "testpng_2.png"
    image_1.save(png_file_path_1)
    image_2.save(png_file_path_2)

    test_input = PngReader("png", [], [])
    input_path = str(tmp_path) + "/"

    with patch("PIL.Image.open", wraps=Image.open) as mock_image_open:
        # load_dicom aufrufen
        test_input.load_image(input_path)

        assert mock_image_open.call_count == 2


def test_load_png_2_slices():
    """Test load_png_2_slices if slices are processed correctly."""

    test_config = {
        "Additional Information": {
            "Pixel_Spacing": [1, 1],
            "Slice_Thickness": 1,
            "Image_Position": [0, 0, 0],
            "Image_Orientation": [0, 1, 0, 1, 0, 0],
            "Modality": "CT",
        }
    }
    png = []
    png.append([[1, 1], [2, 2]])
    png.append([[3, 3], [4, 4]])

    test_input = PngReader("png", test_config, [])
    slice = test_input.image_2_slices(png)

    assert slice[0].Modality == "CT"
    assert np.array_equal(slice[1].ImagePositionPatient, np.array([0, 0, 1]))


def test_verify_and_load_dicom(tmp_path: Path):
    """Test verify_and_load_dicom."""

    example_file = get_testdata_file("CT_small.dcm")
    ds = pydicom.dcmread(example_file)
    dicom_file_path = tmp_path / "testdicom.dcm"
    ds.save_as(dicom_file_path, enforce_file_format=False)
    input_path = str(tmp_path) + "/"

    slices = verify_and_load_imagedata(input_path, "")

    assert slices
