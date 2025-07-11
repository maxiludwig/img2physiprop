"""Test Image Reader Routine."""

import numpy as np
from i2pp.core.discretization_reader_classes.discretization_reader import (
    BoundingBox,
)
from i2pp.core.image_reader_classes.dicom_reader import DicomReader
from i2pp.core.image_reader_classes.image_reader import (
    PixelValueType,
    SliceOrientation,
)


def test_pxl_range():
    """Test pxl_range of enum PixelValueType."""
    assert np.array_equal(PixelValueType.RGB.pxl_range, np.array([0, 255]))
    assert np.array_equal(PixelValueType.CT.pxl_range, np.array([-1024, 3071]))
    assert PixelValueType.MRT.pxl_range is None


def test_get_slice_orientation_planes():
    """Test get_slice_orientation for different planes."""

    test_class = DicomReader([], BoundingBox([], []))
    assert (
        test_class._get_slice_orientation([0, 0, 3], [0, 1, 0])
        == SliceOrientation.YZ
    )
    assert (
        test_class._get_slice_orientation([1, 0, 0], [0, 1, 0])
        == SliceOrientation.XY
    )
    assert (
        test_class._get_slice_orientation([3, 0, 1], [2, 0, 0])
        == SliceOrientation.XZ
    )
    assert (
        test_class._get_slice_orientation([1 / 2, 1 / 2, 0], [2 / 3, 1 / 3, 0])
        == SliceOrientation.XY
    )
    assert (
        test_class._get_slice_orientation(
            [1 / 2, 1 / 2, 1 / 2], [2 / 3, 1 / 3, 0]
        )
        == SliceOrientation.UNKNOWN
    )


def test_slice_is_within_crop():
    """Test is_within_crop for different planes and points."""
    test_orientation_XY = SliceOrientation.XY
    test_orientation_XZ = SliceOrientation.XZ
    test_orientation_YZ = SliceOrientation.YZ
    test_orientation_Unknown = SliceOrientation.UNKNOWN

    limits = BoundingBox(min=[0, 0, 0], max=[1, 1, 1])

    assert test_orientation_XY.is_within_crop([0, 0, 2], limits) is False
    assert test_orientation_XY.is_within_crop([0, 0, 0.5], limits) is True

    assert test_orientation_XZ.is_within_crop([1, -5, 2], limits) is False
    assert test_orientation_XZ.is_within_crop([0, 0, 0.5], limits) is True

    assert test_orientation_YZ.is_within_crop([-0.1, 2, 2], limits) is False
    assert test_orientation_YZ.is_within_crop([1, 4, 0.5], limits) is True

    assert test_orientation_Unknown.is_within_crop([2, 0, 2], limits) is True
    assert test_orientation_Unknown.is_within_crop([0, 3, 0.5], limits) is True
