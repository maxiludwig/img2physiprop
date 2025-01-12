"""Test Image Data Converter Routine."""

import numpy as np
from i2pp.core.Image_Data_Converter import ImageData, ImageDataConverter
from i2pp.core.Image_Reader import SlicesData


def test_pxlpos_to_pxlcoord():
    """Test pxlpos_to_pxlcoord if coordinates are calculated correctly."""

    test_data = ImageDataConverter([])

    slices = SlicesData(
        [], [], [0.5, 0.5], [10, 20, 30], [1, 0, 0, 0, 1, 0], ""
    )
    assert np.array_equal(
        test_data._pxlpos_to_pxlcoord(slices, 0, 0), [10, 20, 30]
    )
    assert np.array_equal(
        test_data._pxlpos_to_pxlcoord(slices, 10, 20), [15, 30, 30]
    )

    slices = SlicesData(
        [], [], [0.5, 0.5], [10, 20, 30], [0, 1, 0, 1, 0, 0], ""
    )

    assert np.array_equal(
        test_data._pxlpos_to_pxlcoord(slices, 20, 40), [30, 30, 30]
    )


def test_ray_intersecs_rectangle():
    """Test ray_intersecs_rectangle if intersection is detected."""

    test_data = ImageDataConverter([])
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

    image_data = ImageData(coord_array=[], pxl_value=[], modality="")
    test_data = ImageDataConverter(image_data)
    limits = [10, 20, 30, 50, 60, 70]
    coord = [10, 10, 10]
    axis = "slice"
    orientation = [1, 0, 0, 0, 1, 0]

    assert test_data._iteration_irrelevant(limits, coord, axis, orientation)
    coord = [10, 10, 50]
    assert (
        test_data._iteration_irrelevant(limits, coord, axis, orientation)
        is False
    )


def test_iteration_irrelevant_col():
    """Test iteration_irrelevant_slice when option "col" is used."""

    image_data = ImageData(coord_array=[], pxl_value=[], modality="")
    test_data = ImageDataConverter(image_data)
    limits = [10, 20, 30, 50, 60, 70]
    coord = [30, 10, 10]
    axis = "col"
    orientation = [1, 0, 0, 0, 1, 0]

    assert (
        test_data._iteration_irrelevant(limits, coord, axis, orientation)
        is False
    )

    coord = [0, 10, 50]
    assert test_data._iteration_irrelevant(limits, coord, axis, orientation)

    orientation = [-0.5, 0.5, 0, 0.5, 0.5, 0]
    axis = "col"
    coord = [1, 1, 1]
    limits = [2, 2, 0, 4, 4, 3]
    assert (
        test_data._iteration_irrelevant(limits, coord, axis, orientation)
        is False
    )

    coord = [1, 5, 1]
    assert test_data._iteration_irrelevant(limits, coord, axis, orientation)


def test_iteration_irrelevant_all():
    """Test iteration_irrelevant_slice when option "all" is used."""

    image_data = ImageData(coord_array=[], pxl_value=[], modality="")
    test_data = ImageDataConverter(image_data)
    limits = [10, 20, 30, 50, 60, 70]
    coord = [30, 30, 50]
    axis = "all"
    orientation = [1, 0, 0, 0, 1, 0]

    assert (
        test_data._iteration_irrelevant(limits, coord, axis, orientation)
        is False
    )

    coord = [0, 30, 60]
    assert test_data._iteration_irrelevant(limits, coord, axis, orientation)


def test_slice_2_mat():
    """Test slice_2_mat if slice are converted correctly."""

    volume = np.array(
        [
            [[10, 2, 3], [3, 4, 3], [5, 6, 10]],  # Ebene 1
            [[1, 2, 3], [3, 4, 3], [5, 6, 3]],  # Ebene 2
            [[1, 2, 3], [3, 4, 3], [5, 5, 3]],
        ],
        dtype=np.uint16,
    )

    limits = [1.5, 0.5, 0, 2, 1, 0]

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

    image_data = ImageData(coord_array=[], pxl_value=[], modality="")
    dicom_data = ImageDataConverter(image_data)

    dicom_data.slices_2_mat(slices_new, limits)
    # self.assertEqual(mesh_data.find_mins_maxs(input)[0],-200)
    expected_coordarray = [
        [1.5, 0.5, 0.0],
        [1.5, 1.0, 0.0],
        [2.0, 0.5, 0.0],
        [2.0, 1.0, 0.0],
    ]

    expected_pxlarray = [4, 6, 3, 10]

    assert np.array_equal(
        dicom_data.image_data.coord_array, expected_coordarray
    )
    assert np.array_equal(dicom_data.image_data.pxl_value, expected_pxlarray)


def test_smoothing_dicom():
    """Test smooth_data if pxl_values are floats."""
    coord1 = np.array([0, 0, 0])
    coord2 = np.array([1, 0, 0])
    coord3 = np.array([1, 1, 0])
    coord4 = np.array([3, 3, 0])

    coord_array = np.array([coord1, coord2, coord3, coord4])
    pxl_value = np.array([10, 20, 30, 40])

    dicom_data = ImageData(coord_array, pxl_value, "CT")
    converter = ImageDataConverter(dicom_data)
    converter.smooth_data(3)

    assert converter.image_data.pxl_value[0] == 20


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

    dicom_data = ImageData(coord_array, pxl_value, "CT")
    converter = ImageDataConverter(dicom_data)
    converter.smooth_data(3)

    assert np.array_equal(
        converter.image_data.pxl_value[0], np.array([30, 10, 20])
    )
