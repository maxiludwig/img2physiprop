"""Test Image Data Converter Routine."""

import numpy as np
from i2pp.core.image_data_converter import ImageDataConverter
from i2pp.core.image_reader_classes.image_reader import (
    PixelValueType,
    SlicesData,
)
from i2pp.core.model_reader_classes.model_reader import Limits


def test_gridposition_to_voxelcoord():
    """Test pxlpos_to_pxlcoord if coordinates are calculated correctly."""

    test_data = ImageDataConverter()

    slices = SlicesData([], [0.5, 0.5], [10, 20, 30], [0, 1, 0, 1, 0, 0], "")
    assert np.array_equal(
        test_data._gridposition_to_voxelcoord(slices, 0, 0), [10, 20, 30]
    )
    assert np.array_equal(
        test_data._gridposition_to_voxelcoord(slices, 20, 10), [15, 30, 30]
    )

    slices = SlicesData([], [0.5, 0.5], [10, 20, 30], [1, 0, 0, 0, 1, 0], "")

    assert np.array_equal(
        test_data._gridposition_to_voxelcoord(slices, 40, 20), [30, 30, 30]
    )


def test_ray_intersecs_rectangle():
    """Test ray_intersecs_rectangle if intersection is detected."""

    test_data = ImageDataConverter()
    coord = [1, 1]
    direction = [1, 1]
    limits = Limits(max=[3, 3, 0], min=[2, 2, 0])

    assert test_data._ray_intersects_rectangle(coord, direction, limits)

    direction = [1, -1]
    assert not test_data._ray_intersects_rectangle(coord, direction, limits)


def test_slice_2_mat_dicom():
    """Test slice_2_mat if pixelvalue is float."""

    volume = np.array(
        [
            [[10, 2, 3], [3, 4, 3], [5, 6, 10]],  # Ebene 1
            [[1, 2, 3], [3, 4, 3], [5, 6, 3]],  # Ebene 2
            [[1, 2, 3], [3, 4, 3], [5, 5, 3]],
        ],
        dtype=np.uint16,
    )

    limits = Limits(max=np.array([2, 1, 0]), min=np.array([1.5, 0.5, 0]))

    slices_new = []

    for i in range(0, 3):

        PixelSpacing = [0.5, 0.5]
        ImagePositionPatient = [1, 0, i]
        PixelData = volume[i]
        orientation = [0, 1, 0, 1, 0, 0]
        slices_new.append(
            SlicesData(
                PixelData=PixelData,
                PixelSpacing=PixelSpacing,
                ImagePositionPatient=ImagePositionPatient,
                ImageOrientationPatient=orientation,
                PixelType=PixelValueType.CT,
            )
        )

    dicom_data = ImageDataConverter()

    processed_data = dicom_data.slices_to_3D_data(slices_new, limits)
    # self.assertEqual(mesh_data.find_mins_maxs(input)[0],-200)
    expected_coordarray = [
        [1.5, 0.5, 0.0],
        [2.0, 0.5, 0.0],
        [1.5, 1.0, 0.0],
        [2.0, 1.0, 0.0],
    ]

    expected_pxlarray = [4, 3, 6, 10]

    assert np.array_equal(processed_data.coord_array, expected_coordarray)
    assert np.array_equal(processed_data.pxl_value, expected_pxlarray)


def test_slice_2_mat_png():
    """Test slice_2_mat if pixel value is rgb."""

    volume = np.array(
        [[[0, 0, 1], [1, 2, 3], [2, 5, 2]], [[2, 4, 9], [5, 2, 3], [3, 4, 3]]],
        dtype=np.uint16,
    )

    limits = Limits(max=np.array([0.5, 1, 1]), min=np.array([0, 0, 0]))

    PixelSpacing = [1, 1]
    ImagePositionPatient = [0, 0, 0]
    PixelData = volume
    orientation = [0, 1, 0, 1, 0, 0]
    slices = SlicesData(
        PixelData=PixelData,
        PixelSpacing=PixelSpacing,
        ImagePositionPatient=ImagePositionPatient,
        ImageOrientationPatient=orientation,
        PixelType=PixelValueType.CT,
    )

    png_data = ImageDataConverter()

    expected_coordarray = [[0, 0, 0.0], [0, 1.0, 0.0]]

    expected_pxlarray = [[0, 0, 1], [2, 4, 9]]

    processed_data = png_data.slices_to_3D_data(slices, limits)

    assert np.array_equal(processed_data.coord_array, expected_coordarray)
    assert np.array_equal(processed_data.pxl_value, expected_pxlarray)


def test_smoothing_dicom():
    """Test smooth_data if pxl_values are floats."""
    array_slice1 = np.array(
        [[5, 4, 3], [0, 5, 7], [9, 2, 1]], dtype=np.float32
    )

    array_slice2 = np.array(
        [[6, 6, 6], [10, 4, 4], [7, 7, 4]], dtype=np.float32
    )

    array_slice3 = np.array(
        [[10, 5, 0], [2, 8, 5], [3, 3, 9]], dtype=np.float32
    )
    slices = []
    slices.append(
        SlicesData(
            PixelData=array_slice1,
            PixelSpacing=[],
            ImagePositionPatient=[],
            ImageOrientationPatient=[],
            PixelType=PixelValueType.CT,
        )
    )

    slices.append(
        SlicesData(
            PixelData=array_slice2,
            PixelSpacing=[],
            ImagePositionPatient=[],
            ImageOrientationPatient=[],
            PixelType=PixelValueType.CT,
        )
    )

    slices.append(
        SlicesData(
            PixelData=array_slice3,
            PixelSpacing=[],
            ImagePositionPatient=[],
            ImageOrientationPatient=[],
            PixelType=PixelValueType.CT,
        )
    )

    converter = ImageDataConverter()
    smoothed_slices = converter.smooth_data(slices, 3)

    assert smoothed_slices[1].PixelData[1][1] == 5


def test_smoothing_rgb():
    """Test smooth_data if pxl_values are arrays (e.g. RGB)"""

    array_slice1 = np.array(
        [
            [[0, 2, 1], [0, 6, 3], [3, 1, 2]],
            [[6, 2, 1], [9, 0, 0], [3, 3, 1]],
            [[3, 2, 1], [0, 0, 0], [3, 2, 0]],
        ],
        dtype=np.float32,
    )

    array_slice2 = np.array(
        [
            [[7, 5, 12], [5, 15, 4], [6, 7, 5]],
            [[8, 5, 0], [5, 0, 4], [6, 8, 5]],
            [[0, 5, 0], [5, 0, 4], [3, 0, 2]],
        ],
        dtype=np.float32,
    )

    array_slice3 = np.array(
        [
            [[1, 2, 0], [0, 5, 1], [3, 6, 0]],
            [[1, 2, 0], [2, 1, 1], [0, 0, 2]],
            [[1, 2, 3], [1, 0, 1], [0, 0, 1]],
        ],
        dtype=np.float32,
    )

    slices = []
    slices.append(
        SlicesData(
            PixelData=array_slice1,
            PixelSpacing=[],
            ImagePositionPatient=[],
            ImageOrientationPatient=[],
            PixelType=PixelValueType.CT,
        )
    )

    slices.append(
        SlicesData(
            PixelData=array_slice2,
            PixelSpacing=[],
            ImagePositionPatient=[],
            ImageOrientationPatient=[],
            PixelType=PixelValueType.CT,
        )
    )

    slices.append(
        SlicesData(
            PixelData=array_slice3,
            PixelSpacing=[],
            ImagePositionPatient=[],
            ImageOrientationPatient=[],
            PixelType=PixelValueType.CT,
        )
    )

    converter = ImageDataConverter()
    smoothed_slices = converter.smooth_data(slices, 3)

    assert np.array_equal(
        smoothed_slices[1].PixelData[1][1], np.array([3, 3, 2])
    )
