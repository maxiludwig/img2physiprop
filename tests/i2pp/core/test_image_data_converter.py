"""Test Image Data Converter Routine."""

import numpy as np
from i2pp.core.image_data_converter import (
    ImageDataConverter,
    IterationOption,
    ProcessedImageData,
)
from i2pp.core.image_reader_classes.image_reader import SlicesData
from i2pp.core.utilities import Limits


def test_gridposition_to_voxelcoord():
    """Test pxlpos_to_pxlcoord if coordinates are calculated correctly."""

    test_data = ImageDataConverter()

    slices = SlicesData(
        [], [], [0.5, 0.5], [10, 20, 30], [1, 0, 0, 0, 1, 0], ""
    )
    assert np.array_equal(
        test_data._gridposition_to_voxelcoord(slices, 0, 0), [10, 20, 30]
    )
    assert np.array_equal(
        test_data._gridposition_to_voxelcoord(slices, 10, 20), [15, 30, 30]
    )

    slices = SlicesData(
        [], [], [0.5, 0.5], [10, 20, 30], [0, 1, 0, 1, 0, 0], ""
    )

    assert np.array_equal(
        test_data._gridposition_to_voxelcoord(slices, 20, 40), [30, 30, 30]
    )


def test_ray_intersecs_rectangle():
    """Test ray_intersecs_rectangle if intersection is detected."""

    test_data = ImageDataConverter()
    coord = [1, 1]
    direction = [1, 1]
    rect_min = [2, 2]
    rect_max = [3, 3]
    assert test_data._ray_intersecs_rectangle(
        coord, direction, rect_min, rect_max
    )

    direction = [1, -1]
    assert not test_data._ray_intersecs_rectangle(
        coord, direction, rect_min, rect_max
    )


def test_iteration_irrelevant_slice():
    """Test iteration_irrelevant_slice when option "slice" is used."""

    test_data = ImageDataConverter()
    limits = Limits(max=np.array([50, 60, 70]), min=np.array([10, 20, 30]))
    coord = [10, 10, 10]
    axis = IterationOption.Slice
    orientation = [1, 0, 0, 0, 1, 0]

    assert test_data._iteration_irrelevant(limits, coord, axis, orientation)
    coord = [10, 10, 50]
    assert not test_data._iteration_irrelevant(
        limits, coord, axis, orientation
    )


def test_iteration_irrelevant_col():
    """Test iteration_irrelevant_slice when option "col" is used."""

    test_data = ImageDataConverter()
    limits = Limits(max=np.array([50, 60, 70]), min=np.array([10, 20, 30]))
    coord = [30, 10, 10]
    axis = IterationOption.Column
    orientation = [1, 0, 0, 0, 1, 0]

    assert (
        test_data._iteration_irrelevant(limits, coord, axis, orientation)
        is False
    )

    coord = [0, 10, 50]
    assert test_data._iteration_irrelevant(limits, coord, axis, orientation)

    orientation = [-0.5, 0.5, 0, 0.5, 0.5, 0]
    axis = IterationOption.Column
    coord = [1, 1, 1]
    limits = Limits(max=np.array([4, 4, 3]), min=np.array([2, 2, 0]))
    assert (
        test_data._iteration_irrelevant(limits, coord, axis, orientation)
        is False
    )

    coord = [1, 5, 1]
    assert test_data._iteration_irrelevant(limits, coord, axis, orientation)


def test_iteration_irrelevant_all():
    """Test iteration_irrelevant_slice when option "all" is used."""

    test_data = ImageDataConverter()
    limits = Limits(max=np.array([50, 60, 70]), min=np.array([10, 20, 30]))
    coord = [30, 30, 50]
    axis = IterationOption.All
    orientation = [1, 0, 0, 0, 1, 0]

    assert (
        test_data._iteration_irrelevant(limits, coord, axis, orientation)
        is False
    )

    coord = [0, 30, 60]
    assert test_data._iteration_irrelevant(limits, coord, axis, orientation)


def test_slice_2_mat_dicom():
    """Test slice_2_mat if pixelvalue is float"""

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
        img_shape = [3, 3]
        orientation = [1, 0, 0, 0, 1, 0]
        slices_new.append(
            SlicesData(
                PixelData=PixelData,
                image_shape=img_shape,
                PixelSpacing=PixelSpacing,
                ImagePositionPatient=ImagePositionPatient,
                ImageOrientationPatient=orientation,
                Modality="CT",
            )
        )

    dicom_data = ImageDataConverter()

    processed_data = dicom_data.slices_2_mat(slices_new, limits)
    # self.assertEqual(mesh_data.find_mins_maxs(input)[0],-200)
    expected_coordarray = [
        [1.5, 0.5, 0.0],
        [1.5, 1.0, 0.0],
        [2.0, 0.5, 0.0],
        [2.0, 1.0, 0.0],
    ]

    expected_pxlarray = [4, 6, 3, 10]

    assert np.array_equal(processed_data.coord_array, expected_coordarray)
    assert np.array_equal(processed_data.pxl_value, expected_pxlarray)


def test_slice_2_mat_png():
    """Test slice_2_mat if pixel value is rgb"""

    volume = np.array(
        [[[0, 0, 1], [1, 2, 3], [2, 5, 2]], [[2, 4, 9], [5, 2, 3], [3, 4, 3]]],
        dtype=np.uint16,
    )

    limits = Limits(max=np.array([0.5, 1, 1]), min=np.array([0, 0, 0]))

    PixelSpacing = [1, 1]
    ImagePositionPatient = [0, 0, 0]
    PixelData = volume
    img_shape = [3, 2]
    orientation = [1, 0, 0, 0, 1, 0]
    slices = SlicesData(
        PixelData=PixelData,
        image_shape=img_shape,
        PixelSpacing=PixelSpacing,
        ImagePositionPatient=ImagePositionPatient,
        ImageOrientationPatient=orientation,
        Modality="CT",
    )

    png_data = ImageDataConverter()

    png_data.slices_2_mat(slices, limits)
    # self.assertEqual(mesh_data.find_mins_maxs(input)[0],-200)
    expected_coordarray = [[0, 0, 0.0], [0, 1.0, 0.0]]

    expected_pxlarray = [[0, 0, 1], [2, 4, 9]]

    processed_data = png_data.slices_2_mat(slices, limits)

    assert np.array_equal(processed_data.coord_array, expected_coordarray)
    assert np.array_equal(processed_data.pxl_value, expected_pxlarray)


def test_smoothing_dicom():
    """Test smooth_data if pxl_values are floats."""
    coord1 = np.array([0, 0, 0])
    coord2 = np.array([1, 0, 0])
    coord3 = np.array([1, 1, 0])
    coord4 = np.array([3, 3, 0])

    coord_array = np.array([coord1, coord2, coord3, coord4])
    pxl_value = np.array([10, 20, 30, 40])

    processed_data = ProcessedImageData(coord_array, pxl_value)

    converter = ImageDataConverter()
    new_processed_data = converter.smooth_data(3, processed_data)

    assert new_processed_data.pxl_value[0] == 20


def test_smoothing_rgb():
    """Test smooth_data if pxl_values are arrays (e.g. RGB)"""
    coord1 = np.array([0, 0, 0])
    coord2 = np.array([1, 0, 0])
    coord3 = np.array([1, 1, 0])
    coord4 = np.array([3, 3, 0])

    coord_array = np.array([coord1, coord2, coord3, coord4])

    pxl_value = np.array(
        [[10, 0, 0], [20, 30, 30], [60, 0, 30], [0, 100, 250]]
    )

    processed_data = ProcessedImageData(coord_array, pxl_value)

    converter = ImageDataConverter()
    new_processed_data = converter.smooth_data(3, processed_data)

    assert np.array_equal(
        new_processed_data.pxl_value[0], np.array([30, 10, 20])
    )
