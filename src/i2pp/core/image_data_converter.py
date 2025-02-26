"""Convertes slices of the image data into an array with coordinates and an
array with pixel values."""

import logging
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
from i2pp.core.image_reader_classes.image_reader import SlicesData
from i2pp.core.model_reader_classes.model_reader import Limits
from i2pp.core.visualization import plot_slice
from scipy.ndimage import uniform_filter
from tqdm import tqdm


@dataclass
class ProcessedImageData:
    """Dataclass for processed image data.

    This class is used to store the processed information of an image,
    including the coordinates and pixel values. It forms the 3D
    representation of image slices by encapsulating the 2D slice data
    into the respective pixel coordinates and the associated pixel
    values.

    Attributes:
        coord_array (np.ndarray): The array of pixel coordinates representing
            the positions of pixels in the image.
        pxl_value (np.ndarray): The array of pixel values corresponding to the
            pixels at the given coordinates.
    """

    coord_array: np.ndarray
    pxl_value: np.ndarray


class ImageDataConverter:
    """Class to convert 2D grids of image slices into a 3D array with world
    coordinates.

    This class provides methods to process image data by converting the
    2D grids of pixel data from each slice into a 3D representation. The
    3D array consists of voxel coordinates in the world space (x, y, z),
    with each voxel associated with a specific pixel value. This
    conversion allows the image data to be used in 3D space for further
    analysis and processing.
    """

    def __init__(self):
        """Init ImageDataConverter."""

    def smooth_data(
        self,
        slices: list[SlicesData],
        k: int,
    ) -> list[SlicesData]:
        """Smooths the image data by averaging the pixel values in a k-point
        neighborhood around each point.

        This method applies a smoothing filter to the 3D image data, reducing
        the impact of noise or measurement errors by calculating the average
        pixel value in a neighborhood defined by the parameter `k`.

        Arguments:
            slices (list[SlicesData]): The collection of slice data containing
                the pixel information to be smoothed.
            k (int): The size of the neighborhood (in points) used to
                calculate the average. Higher values result in smoother data
                but may blur fine details.

        Returns:
            list[SlicesData]: The smoothed image data, where each slice has
                updated `PixelData` with the average values from its
                neighboring points.
        """

        logging.info("Smooth data!")
        array_3d = np.stack(
            [slice.PixelData for slice in slices], dtype=np.float32
        )
        smoothed_array_3d = uniform_filter(
            array_3d, size=k, mode="nearest", axes=(0, 1, 2)
        )

        for i, _ in enumerate(slices):
            slices[i].PixelData = smoothed_array_3d[i]

        return slices

    def _gridposition_to_voxelcoord(
        self, slice: SlicesData, row: int, col: int
    ) -> np.ndarray:
        """Converts the grid position of a voxel in the slice to its
        corresponding absolute x-y-z coordinate.

        This method calculates the absolute 3D coordinates of a voxel based on
        its position within the 2D grid (specified by row and column) in the
        given slice, using the image's orientation and pixel spacing.

        Arguments:
            slice (SlicesData): The slice data containing information about
                image orientation, position, and spacing.
            row (int): The row index of the voxel within the 2D grid.
            col (int): The column index of the voxel within the 2D grid.

        Returns:
            np.ndarray: The absolute x-y-z coordinates of the voxel in the
            3D space.
        """

        row_vector = np.array(
            slice.ImageOrientationPatient[0:3], dtype=float
        ).flatten()
        col_vector = np.array(
            slice.ImageOrientationPatient[3:6], dtype=float
        ).flatten()

        image_position = np.array(
            slice.ImagePositionPatient, dtype=float
        ).flatten()

        coords = (
            image_position
            + row * slice.PixelSpacing[0] * row_vector
            + col * slice.PixelSpacing[1] * col_vector
        )

        return np.array(coords)

    def _ray_intersects_rectangle(
        self, ray_start: np.ndarray, ray_direction: np.ndarray, limits: Limits
    ) -> bool:
        """Checks if a ray intersects a rectangle within given bounds.

        This function performs an intersection test between a ray and a
        rectangle. The ray is defined by its starting point and direction,
        and the rectangle is described by the minimal and maximal coordinates
        in the 2D plane (as provided in the `limits`). This function is used
        to check if the voxels of a row lie within the model's boundaries. If
        the voxels are outside the model's boundaries, the entire row can be
        skipped.

        Arguments:
            ray_start (np.ndarray): The starting point of the ray, typically
                in the form of (x, y).
            ray_direction (np.ndarray): The direction vector of the ray,
                typically in the form of (dx, dy).
            limits (Limits): The bounds of the rectangle, providing the minimum
                and maximum coordinates.

        Returns:
            bool: True if the ray intersects the rectangle, False otherwise.
        """

        rect_min = limits.min[0:2]
        rect_max = limits.max[0:2]

        t_min = float("-inf")
        t_max = float("inf")

        for i in range(2):
            if ray_direction[i] != 0:
                length_ray1 = (rect_min[i] - ray_start[i]) / ray_direction[i]
                length_ray2 = (rect_max[i] - ray_start[i]) / ray_direction[i]

                length_ray1, length_ray2 = min(length_ray1, length_ray2), max(
                    length_ray1, length_ray2
                )

                t_min = max(t_min, length_ray1)
                t_max = min(t_max, length_ray2)
            else:

                if not (rect_min[i] <= ray_start[i] <= rect_max[i]):
                    return False

        return t_min <= t_max and t_max >= 0

    def slices_to_3D_data(
        self, slices: list[SlicesData], limits: Limits
    ) -> ProcessedImageData:
        """Converts a list of 2D image slices into 3D data by extracting pixel
        coordinates and values.

        This method processes the slices of image data, converting them into a
        3D representation by creating an array of pixel coordinates and an
        array of corresponding pixel values. For performance optimization, only
        pixels that lie within the specified model limits are processed. If no
        valid mesh coordinates are found within the image data, a runtime error
        will be raised.

        Arguments:
            slices (list[SlicesData]): List of SliceData objects containing
                pixel data and metadata.
            limits (Limits): The model boundaries to filter out pixels that
                lie outside these limits.

        Returns:
            ProcessedImageData: A dataclass containing the 3D array of pixel
                coordinates and corresponding pixel values from the processed
                slices.

        Raises:
            RuntimeError: If no mesh coordinates are found within the image
                data or if the mesh is not contained in the image data.
        """

        coord_array = []
        pxl_value = []

        if not isinstance(slices, Iterable) or isinstance(
            slices, (str, bytes)
        ):
            slices = [slices]

        for slice in tqdm(slices, desc="Process slices"):

            for j in range(0, slice.PixelData.shape[0]):

                coord = self._gridposition_to_voxelcoord(slice, j, 0)

                if not self._ray_intersects_rectangle(
                    coord, slice.ImageOrientationPatient[3:6], limits
                ):
                    continue

                for m in range(0, slice.PixelData.shape[1]):

                    coord = self._gridposition_to_voxelcoord(slice, j, m)

                    if np.any(coord < limits.min) or np.any(
                        coord > limits.max
                    ):
                        continue

                    coord_array.append(coord)

                    pxl_value.append(slice.PixelData[j][m])

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


def convert_imagedata(
    slices: list[SlicesData],
    limits: Limits,
    config_i2pp: dict,
    pxl_range: np.ndarray,
) -> ProcessedImageData:
    """Processes the image data by applying optional smoothing and converting
    it to 3D format.

    This function orchestrates the workflow for processing image slices by
    first applying optional smoothing to the slice data, if enabled in the
    configuration. After smoothing, the function converts the 2D slices into
    3D image data, within the specified model limits. The processed data is
    returned as a `ProcessedImageData` object containing both pixel
    coordinates and values.

    Arguments:
        slices (list[SlicesData]): List of SliceData objects containing pixel
            data and metadata.
        limits (Limits): The model boundaries to filter out pixels that lie
            outside these limits.
        config_i2pp (dict): User configuration that includes custom settings,
            such as smoothing parameters.
        pxl_range (np.ndarray): The pixel value range used for plotting and
            processing.

    Returns:
        ProcessedImageData: A dataclass containing the 3D array of pixel
            coordinates and corresponding pixel values after optional
            smoothing and conversion.
    """

    processing_options: dict = config_i2pp["Processing options"]

    smoothing_bool = bool(processing_options.get("Smoothing") or False)
    smoothing_area = int(processing_options.get("Smoothing_area") or 3)

    image_converter = ImageDataConverter()

    if smoothing_bool:

        plot_slice(
            slices[0].PixelData,
            slices[0].PixelType,
            pxl_range,
            "not_smoothed_slice",
        )
        slices = image_converter.smooth_data(slices, smoothing_area)
        plot_slice(
            slices[0].PixelData,
            slices[0].PixelType,
            pxl_range,
            "smoothed_slice",
        )

    processed_data = image_converter.slices_to_3D_data(slices, limits)

    return processed_data
