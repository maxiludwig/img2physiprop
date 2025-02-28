"""Test Image Data Converter Routine."""

import numpy as np
from i2pp.core.discretization_reader_classes.discretization_reader import (
    Limits,
)
from i2pp.core.image_data_converter import ImageDataConverter
from i2pp.core.image_reader_classes.image_reader import (
    ImageMetaData,
    PixelValueType,
    Slice,
    SlicesAndMetadata,
)


def test_gridposition_to_voxelcoord():
    """Test pxlpos_to_pxlcoord if coordinates are calculated correctly."""

    test_data = ImageDataConverter()

    metadata = ImageMetaData(
        pixel_spacing=[0.5, 0.5],
        orientation=[0, 1, 0, 1, 0, 0],
        pixel_type=PixelValueType.CT,
    )
    slice = Slice(pixel_data=np.array([]), position=[10, 20, 30])

    assert np.array_equal(
        test_data._gridposition_to_voxelcoord(slice, metadata, 0, 0),
        [10, 20, 30],
    )
    assert np.array_equal(
        test_data._gridposition_to_voxelcoord(slice, metadata, 20, 10),
        [15, 30, 30],
    )

    metadata = ImageMetaData(
        pixel_spacing=[0.5, 0.5],
        orientation=[1, 0, 0, 0, 1, 0],
        pixel_type=PixelValueType.CT,
    )

    assert np.array_equal(
        test_data._gridposition_to_voxelcoord(slice, metadata, 40, 20),
        [30, 30, 30],
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


def test_slices_to_3D_data_dicom():
    """Test slices_to_3D_data if pixelvalue is float."""

    volume = np.array(
        [
            [[10, 2, 3], [3, 4, 3], [5, 6, 10]],
            [[1, 2, 3], [3, 4, 3], [5, 6, 3]],
            [[1, 2, 3], [3, 4, 3], [5, 5, 3]],
        ],
        dtype=np.uint16,
    )

    limits = Limits(max=np.array([2, 1, 0]), min=np.array([1.5, 0.5, 0]))

    slices = []

    metadata = ImageMetaData(
        pixel_spacing=[0.5, 0.5],
        orientation=[0, 1, 0, 1, 0, 0],
        pixel_type=PixelValueType.CT,
    )

    for i in range(0, 3):

        slices.append(
            Slice(
                pixel_data=volume[i],
                position=[1, 0, i],
            )
        )

    dicom_data = ImageDataConverter()
    slices_and_metadata = SlicesAndMetadata(slices, metadata)

    processed_data = dicom_data.slices_to_3D_data(slices_and_metadata, limits)

    expected_coordarray = [
        [1.5, 0.5, 0.0],
        [2.0, 0.5, 0.0],
        [1.5, 1.0, 0.0],
        [2.0, 1.0, 0.0],
    ]

    expected_pxlarray = [4, 3, 6, 10]

    assert np.array_equal(processed_data.coord_array, expected_coordarray)
    assert np.array_equal(processed_data.pixel_values, expected_pxlarray)


def test_slices_to_3D_data_png():
    """Test slices_to_3D_data if pixel value is rgb."""

    volume = np.array(
        [[[0, 0, 1], [1, 2, 3], [2, 5, 2]], [[2, 4, 9], [5, 2, 3], [3, 4, 3]]],
        dtype=np.uint16,
    )

    limits = Limits(max=np.array([0.5, 1, 1]), min=np.array([0, 0, 0]))

    metadata = ImageMetaData(
        pixel_spacing=[1, 1],
        orientation=[0, 1, 0, 1, 0, 0],
        pixel_type=PixelValueType.RGB,
    )
    slice = Slice(pixel_data=volume, position=[0, 0, 0])

    slices_and_metadata = SlicesAndMetadata(slice, metadata)

    png_data = ImageDataConverter()

    expected_coordarray = [[0, 0, 0.0], [0, 1.0, 0.0]]

    expected_pxlarray = [[0, 0, 1], [2, 4, 9]]

    processed_data = png_data.slices_to_3D_data(slices_and_metadata, limits)

    assert np.array_equal(processed_data.coord_array, expected_coordarray)
    assert np.array_equal(processed_data.pixel_values, expected_pxlarray)


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
        Slice(
            pixel_data=array_slice1,
            position=[],
        )
    )

    slices.append(
        Slice(
            pixel_data=array_slice2,
            position=[],
        )
    )

    slices.append(
        Slice(
            pixel_data=array_slice3,
            position=[],
        )
    )

    converter = ImageDataConverter()
    smoothed_slices = converter.smooth_data(slices, 3)

    assert smoothed_slices[1].pixel_data[1][1] == 5


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
        Slice(
            pixel_data=array_slice1,
            position=[],
        )
    )

    slices.append(
        Slice(
            pixel_data=array_slice2,
            position=[],
        )
    )

    slices.append(
        Slice(
            pixel_data=array_slice3,
            position=[],
        )
    )

    converter = ImageDataConverter()
    smoothed_slices = converter.smooth_data(slices, 3)

    assert np.array_equal(
        smoothed_slices[1].pixel_data[1][1], np.array([3, 3, 2])
    )
