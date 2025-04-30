"""Test Image Reader Routine."""

from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pydicom
from i2pp.core.discretization_reader_classes.discretization_reader import (
    BoundingBox,
)
from i2pp.core.image_reader_classes.dicom_reader import DicomReader
from i2pp.core.image_reader_classes.image_reader import PixelValueType
from pydicom.data import get_testdata_file
from pydicom.dataset import FileDataset


def test_sort_dicoms_axis_dir():
    """Test _sort_dicoms_axis_dir if the slice diections is [1,0,0]"""

    dicoms = [
        Mock(ImagePositionPatient=[1.0, 2.0, 3.0]),
        Mock(ImagePositionPatient=[4.0, 5.0, 6.0]),
        Mock(ImagePositionPatient=[0.0, 0.0, 0.0]),
        Mock(ImagePositionPatient=[7.0, 8.0, 9.0]),
    ]

    slice_direction = np.array([1.0, 0.0, 0.0])
    reader = DicomReader([], [])

    sorted_dicoms = reader._sort_dicoms(dicoms, slice_direction)

    expected_order = [2, 0, 1, 3]

    assert [dicoms.index(ds) for ds in sorted_dicoms] == expected_order


def test_sort_dicoms_rotated_dir():
    """Test _sort_dicoms_axis_dir if the slice diections is [1,1,0]"""

    dicoms = [
        Mock(ImagePositionPatient=[1.0, 1.0, 3.0]),
        Mock(ImagePositionPatient=[0.0, 0.0, 6.0]),
        Mock(ImagePositionPatient=[2.0, 2.0, 0.0]),
        Mock(ImagePositionPatient=[-1.0, -1.0, 9.0]),
    ]

    slice_direction = np.array([1.0, 1.0, 0.0])
    reader = DicomReader([], [])

    sorted_dicoms = reader._sort_dicoms(dicoms, slice_direction)

    expected_order = [3, 1, 0, 2]

    assert [dicoms.index(ds) for ds in sorted_dicoms] == expected_order


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

    test_input = DicomReader(
        {}, BoundingBox(max=[0, 0, 1000], min=[1, 1, -1000])
    )
    input_path = tmp_path

    with patch("pydicom.dcmread", wraps=pydicom.dcmread) as mock_dcmread:

        dicoms = test_input.load_image(input_path)

        assert mock_dcmread.call_count == 2
        assert len(dicoms) == 2
        assert isinstance(dicoms[0], FileDataset)


def test_convert_to_image_data_dicom():
    """Test convert_to_image_data for dicom."""

    ds = []
    example_file = get_testdata_file("MR_small.dcm")
    ds = pydicom.dcmread(example_file)

    test_class = DicomReader(
        [], BoundingBox(min=[1, 1, -1000], max=[0, 0, 1000])
    )
    image_data = test_class.convert_to_image_data([ds, ds])

    assert image_data.pixel_type == PixelValueType.MRT
