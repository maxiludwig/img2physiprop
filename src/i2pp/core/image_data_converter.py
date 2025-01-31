"""Convertes slices of the image data into an array with coordinates and an
array with pixel values."""

from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum

import numpy as np
from i2pp.core.image_reader_classes.image_reader import SlicesData
from i2pp.core.utilities import Limits
from scipy.spatial import cKDTree
from tqdm import tqdm


class IterationOption(Enum):
    """Options for function _iteration_irrelevant."""

    Slice = 1
    Column = 2
    All = 3


@dataclass
class ProcessedImageData:
    """Dataclass for Processed image-data."""

    coord_array: np.ndarray
    pxl_value: np.ndarray


class ImageDataConverter:
    """Class to convert pixel values from a grid in an array
    with absolute coordinates."""

    def __init__(self):
        """Init ImageDataConverter."""

    def _gridposition_to_voxelcoord(
        self, slices: SlicesData, row: int, col: int
    ) -> np.ndarray:
        """Convertes grid position in SliceData to absolute x-y-z coordinate.

        Arguments:
            slices {object} -- Slices Data
            row {int} -- Row of the voxel in the grid
            col {int} -- Column of the voxel in the grid

        Returns:
            np.ndarray -- x-y-z coordinate of the voxel
        """

        image_position = np.array(slices.ImagePositionPatient)
        image_orientation = np.array(slices.ImageOrientationPatient)
        pixel_spacing = np.array(slices.PixelSpacing)

        row_vector = image_orientation[:3]
        col_vector = image_orientation[3:]

        coord = (
            image_position
            + row * pixel_spacing[0] * row_vector
            + col * pixel_spacing[1] * col_vector
        )

        return coord

    def _ray_intersecs_rectangle(
        self,
        coord: np.ndarray,
        direction: np.ndarray,
        rect_min: np.ndarray,
        rect_max: np.ndarray,
    ) -> bool:
        """Checks if a ray intersecs a rectangle.

        Arguments:
            coord {np.ndarray} -- Starting point of the ray
            direction {np.ndarray} -- Direction of the ray
            rect_min {np.ndarray} -- Minimum values of the rectangle
            rect_max {np.ndarray} -- Maximum values of the rectangle

            Returns:
            bool -- True if the ray intersecs the rectangle
        """

        t_min = float("-inf")
        t_max = float("inf")

        for i in range(2):
            if direction[i] != 0:
                length_ray1 = (rect_min[i] - coord[i]) / direction[i]
                length_ray2 = (rect_max[i] - coord[i]) / direction[i]

                length_ray1, length_ray2 = min(length_ray1, length_ray2), max(
                    length_ray1, length_ray2
                )

                t_min = max(t_min, length_ray1)
                t_max = min(t_max, length_ray2)
            else:

                if not (rect_min[i] <= coord[i] <= rect_max[i]):
                    return False

        return t_min <= t_max and t_max >= 0

    def _iteration_irrelevant(
        self,
        limits: Limits,
        coord: np.ndarray,
        check: IterationOption,
        orientation: np.ndarray,
    ) -> bool:
        """Checks if the iteration is irrelevant.

        Arguments:
            limits {object} -- Limits of the model
            coord {np.ndarray} -- Coordinate of the voxel
            check {Enum} -- Type of check
            orientation {np.ndarray} -- Orientation of the image

        Returns:
            bool -- True if the iteration is irrelevant

        Options:
            -slice: Orientation vector has no component in
                    z-direction and z-coord is not in
                    limits of the crop
                    -> slice is irrelevant

            -col: A Ray from the current position in direction
                    of the column vector does not
                    intersec the crop
                    -> no further columns iteration needed

            -all: Verifies if x-,y- and z- coordinate are inside the crop
        """

        if check == IterationOption.Slice:

            return orientation[2] == orientation[5] == 0 and (
                coord[2] < limits.min[2] or coord[2] > limits.max[2]
            )

        elif check == IterationOption.Column:

            col_vector = orientation[3:]
            rect_min = [limits.min[0], limits.min[1]]
            rect_max = [limits.max[0], limits.max[1]]

            return not self._ray_intersecs_rectangle(
                coord, col_vector, rect_min, rect_max
            )

        elif check == IterationOption.All:
            return not (
                limits.min[0] <= coord[0] <= limits.max[0]
                and limits.min[1] <= coord[1] <= limits.max[1]
                and limits.min[2] <= coord[2] <= limits.max[2]
            )

    def slices_2_mat(
        self, slices: list[SlicesData], limits: Limits
    ) -> ProcessedImageData:
        """Converts slices of the image-data in an array with pixel-coordinates
        and an array with pixel values. For Perfomance Optimization only the
        Pixels inside the Mesh-limits are processed.

        Arguments:
            slices {object} -- Slice data
            limits {np.ndarray} -- Limits of the model

        Raises:
            Runtimerror: If Mesh is not in Image_data
        """

        coord_array = []
        pxl_value = []

        if not isinstance(slices, Iterable) or isinstance(
            slices, (str, bytes)
        ):
            slices = [slices]

        for slice in tqdm(slices, desc="Process slices"):
            # coordinate top-left corner
            starting_coordinate = slice.ImagePositionPatient

            # skip all cols and rows if the vectors dont have a component
            # in z-value direction an z-value is not in limits
            # => pxl coordinates can't be in mesh
            if self._iteration_irrelevant(
                limits,
                starting_coordinate,
                IterationOption.Slice,
                slice.ImageOrientationPatient,
            ):
                continue

            # iterate over all columns of the slice
            for j in range(0, slice.image_shape[1]):

                # skip all rows-iterations if no coordinate
                # will be inside the crop
                # => pxl coordinates can't be in mesh
                coord = self._gridposition_to_voxelcoord(slice, j, 0)

                if self._iteration_irrelevant(
                    limits,
                    coord,
                    IterationOption.Column,
                    slice.ImageOrientationPatient,
                ):
                    continue

                for m in range(0, slice.image_shape[0]):

                    # checks if point is in limits
                    coord = self._gridposition_to_voxelcoord(slice, j, m)
                    if (
                        self._iteration_irrelevant(
                            limits,
                            coord,
                            IterationOption.All,
                            slice.ImageOrientationPatient,
                        )
                        is False
                    ):

                        coord_array.append([coord[0], coord[1], coord[2]])

                        pxl_value.append(slice.PixelData[m][j])

        np_coord_array = np.array(coord_array)

        if not np_coord_array.any():
            raise RuntimeError(
                "Mesh coordinates are not in image data!"
                "img2physiprop can not be executed!"
            )

        processed_data = ProcessedImageData(
            np_coord_array, np.array(pxl_value)
        )

        return processed_data

    def smooth_data(
        self, k: int, processed_data: ProcessedImageData
    ) -> ProcessedImageData:
        """Find the k nearest neighbors for each point and calculate the
        average -> Prevent measurement errors.

        Arguments:
            k {int} -- Number of nearest neighbors
            processed_data {object} -- Processed image data

        Returns:
            object -- Processed image data with smoothed pixel values"""

        tree = cKDTree(processed_data.coord_array)
        smoothed_pxl_value = []

        for i, (x, y, z) in tqdm(
            enumerate(processed_data.coord_array), desc="Smooth Data"
        ):

            _, indices = tree.query((x, y, z), k=k)
            smoothed_value = np.mean(processed_data.pxl_value[indices], axis=0)
            smoothed_pxl_value.append((smoothed_value))

        processed_data.pxl_value = np.array(smoothed_pxl_value)

        return processed_data


def convert_imagedata(
    slices: list[SlicesData], limits: Limits, config_i2pp
) -> ProcessedImageData:
    """Calls Image Data Converter.

    Arguments:
        slices {object} -- Slice data
        limits {object} -- Limits of the model
        config_i2pp {object} -- User Configuration

    Returns:
        object -- Processed image data
    """

    if config_i2pp["Further customizations"]["Smoothing"] is None:
        smoothing_bool = False
    else:
        smoothing_bool = config_i2pp["Further customizations"]["Smoothing"]

    if config_i2pp["Further customizations"]["Smoothing_area"] is None:
        smoothing_area = 27
    else:
        smoothing_area = config_i2pp["Further customizations"][
            "Smoothing_area"
        ]

    image_converter = ImageDataConverter()

    processed_data = image_converter.slices_2_mat(slices, limits)

    if smoothing_bool is True:

        image_converter.smooth_data(smoothing_area, processed_data)

    return processed_data
