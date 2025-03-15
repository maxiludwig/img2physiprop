"""Interpolates pixel values from image-data to mesh-data."""

import logging
from abc import abstractmethod

import numpy as np
from i2pp.core.discretization_reader_classes.discretization_reader import (
    Discretization,
    Element,
)
from i2pp.core.image_reader_classes.image_reader import ImageData
from scipy.interpolate import RegularGridInterpolator


class Interpolator:
    """Class to interpolate pixel data from 3D image data to FEM Discretization
    elements.

    This class provides methods to interpolate pixel values from
    processed 3D image data onto the finite element Discretization (FEM)
    by associating the image data with Discretization elements.
    Interpolation is performed at various locations such as the nodes,
    element centers, or based on all voxels inside the element. The
    class supports multiple interpolation strategies based on user
    configuration.
    """

    def __init__(self):
        """Initialize the InterpolatorClass."""
        pass

    def world_to_grid_coords(
        self,
        world_coords: np.ndarray,
        orientation: np.ndarray,
        grid_origin: np.ndarray,
    ) -> np.ndarray:
        """Converts world coordinates to grid coordinates.

        This function transforms world coordinates into grid coordinates by
        applying the inverse of the orientation matrix and adjusting for the
        grid origin. The resulting coordinates are ordered such that the first
        dimension represents height, the second represents rows, and the third
        represents columns.

        Arguments:
            world_coords (np.ndarray): An (N, 3) array representing N points
                in world coordinates (x, y, z).
            orientation (np.ndarray): A 3×3 matrix defining the spatial mapping
                between world and grid coordinates.
            grid_origin (np.ndarray): A (3,) array representing the world
                coordinates of the first pixel in the grid.

        Returns:
            np.ndarray: An (N, 3) array of transformed grid coordinates,
                ordered as (depth, row, column).
        """

        orientation_inv = np.linalg.inv(orientation)

        return (orientation_inv @ (world_coords - grid_origin).T).T

    def interpolate_image_values_to_points(
        self, target_points: np.ndarray, image_data: ImageData
    ) -> np.ndarray:
        """Interpolates pixel values from the image data onto specified target
        points.

        This function uses scattered data interpolation to estimate pixel
        values at given target points based on the known pixel data from the
        image. Linear interpolation is used to ensure a smooth transition of
        values.

        Arguments:
            target_points (np.ndarray): An array of grid coordinates where
                pixel values should be interpolated.
            image_data (ProcessedImageData): 3D image data containing voxel
                coordinates and values

        Returns:
            np.ndarray: An array of interpolated pixel values at the target
                points.
        """

        logging.info("Start Interpolation!")
        interpolator = RegularGridInterpolator(
            (
                image_data.grid_coords.slice,
                image_data.grid_coords.row,
                image_data.grid_coords.col,
            ),
            image_data.pixel_data,
            method="linear",
            bounds_error=False,
            fill_value=np.nan,
        )

        interpolated_values = interpolator(target_points)
        logging.info("Finished Interpolation!")

        return np.array(interpolated_values)

    @abstractmethod
    def compute_element_data(
        self, dis: Discretization, image_data: ImageData
    ) -> list[Element]:
        """Computes data for each FEM element based on image voxel values.

        This abstract method should be implemented in subclasses to compute
        specific data for each FEM element in the Discretization. The type of
        computation depends on the subclass implementation and may involve
        calculating mean pixel values, centroids, or other element-specific
        data based on the provided image data.

        Arguments:
            dis (Discretization): The Discretization object containing FEM
                elements and node coordinates.
            image_data (ImageData): The 3D image data containing voxel
                coordinates and intensity values.

        Returns:
            list[Element]: A list of FEM elements with their computed data
                assigned.
        """
        pass
