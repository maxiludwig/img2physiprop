"""Interpolates pixel values from image-data to mesh-data."""

import logging
from enum import Enum
from typing import Callable

import numpy as np
from i2pp.core.discretization_reader_classes.discretization_reader import (
    Discretization,
    Element,
)
from i2pp.core.image_data_converter import ProcessedImageData
from i2pp.core.utilities import get_node_position_of_element
from scipy.interpolate import griddata
from scipy.spatial import ConvexHull, Delaunay, KDTree
from tqdm import tqdm


class Interpolator:
    """Class to interpolate pixel values from 3D image data to FEM
    Discretization elements.

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

    def compute_element_centers(self, dis: Discretization) -> Discretization:
        """Calculates the center (centroid) of each element in the
        Discretization.

        This function iterates through all elements in the given Discretization
        object, calculates the centroid of each element by averaging the
        coordinates of its associated nodes, and then adds the calculated
        center as a new attribute ('center_coord') for each element.

        Arguments:
            dis (Discretization): The Discretization object containing elements
                and nodes.

        Returns:
            Discretization: The updated Discretization object with a new
                'center_coords' attribute for each element, representing the
                calculated centroid.
        """

        for i, ele in tqdm(
            enumerate(dis.elements),
            total=len(dis.elements),
            desc="Calculate element center",
        ):

            node_position = get_node_position_of_element(
                ele.node_ids, dis.nodes.ids
            )
            element_coords = dis.nodes.coords[node_position]
            centroid = np.mean(element_coords, axis=0)

            dis.elements[i].center_coords = np.array(centroid)

        return dis

    def _point_in_hull(self, point: np.ndarray, hull: ConvexHull) -> bool:
        """Determines whether a given point is inside a convex hull.

        This function checks if a specified point lies within the convex hull
        using Delaunay triangulation. It returns `True` if the point is inside
        the hull and `False` otherwise.

        Arguments:
            point (np.ndarray): The point to check (1D array of coordinates).
            hull (ConvexHull): The convex hull object defining the boundary.

        Returns:
            bool: True if the point is inside the convex hull, False
                otherwise.
        """

        deln = Delaunay(hull.points[hull.vertices])
        return deln.find_simplex(point) >= 0

    def _get_voxels_in_element(
        self, element_points: np.ndarray, image_data: ProcessedImageData
    ) -> np.ndarray:
        """Identifies and returns the pixel values of all voxels inside a give
        mesh element.

        This function determines which points from the image data fall within
        the convex hull of a specified element. It first checks if a point is
        within the element's bounding box for an initial filter, then verifies
        if the point is inside the convex hull.

        Arguments:
            element_points (np.ndarray): Coordinates of the element nodes.
            image_data (ProcessedImageData): 3D image data containing voxel
                coordinates and values

        Returns:
            np.ndarray: An array containing pixel values of the voxels inside
                the element.
        """

        hull = ConvexHull(element_points)
        bbox_min, bbox_max = element_points.min(axis=0), element_points.max(
            axis=0
        )

        tree = KDTree(image_data.coord_array)
        candidates = tree.query_ball_point(
            hull.points[hull.vertices].mean(axis=0),
            r=np.linalg.norm(bbox_max - bbox_min),
        )

        values_in_mesh = [
            image_data.pixel_values[i]
            for i in candidates
            if np.all(image_data.coord_array[i] >= bbox_min)
            and np.all(image_data.coord_array[i] <= bbox_max)
            and self._point_in_hull(image_data.coord_array[i], hull)
        ]

        return np.array(values_in_mesh)

    def interpolate_image_values_to_points(
        self, target_points: np.ndarray, image_data: ProcessedImageData
    ) -> np.ndarray:
        """Interpolates pixel values from the image data onto specified target
        points.

        This function uses scattered data interpolation to estimate pixel
        values at given target points based on the known pixel data from the
        image. Linear interpolation is used to ensure a smooth transition of
        values.

        Arguments:
            target_points (np.ndarray): An array of coordinates where pixel
                values should be interpolated.
            image_data (ProcessedImageData): 3D image data containing voxel
                coordinates and values

        Returns:
            np.ndarray: An array of interpolated pixel values at the target
                points.
        """

        points_image = image_data.coord_array
        values_image = image_data.pixel_values

        logging.info("Start Interpolation!")
        target_points_values = griddata(
            points_image, values_image, target_points, method="linear"
        )
        logging.info("Finished Interpolation!")

        return np.array(target_points_values)

    def get_elementvalues_nodes(
        self, dis: Discretization, image_data: ProcessedImageData
    ) -> list[Element]:
        """Computes the mean pixel value for each FEM element in the
        Discretization based on its node values.

        This function interpolates pixel values at the coordinates of each
        element's nodes and calculates the mean value for the element. It
        applies when the `calculation_type` is set to "nodes".

        Arguments:
            dis (Discretization): The Discretization object containing FEM
                elements and node coordinates.
            image_data (ProcessedImageData):  3D image data containing voxel
                coordinates and values

        Returns:
            list[Element]: A list of FEM elements with their mean pixel values
                assigned.
        """

        node_values = self.interpolate_image_values_to_points(
            dis.nodes.coords, image_data
        )

        node_positions = np.array(
            [
                get_node_position_of_element(ele.node_ids, dis.nodes.ids)
                for ele in dis.elements
            ]
        )

        for i, ele in tqdm(
            enumerate(dis.elements),
            total=len(dis.elements),
            desc="Processing Elements",
        ):

            ele.value = np.mean(node_values[node_positions[i]], axis=0)

        return dis.elements

    def get_elementvalues_center(
        self, dis: Discretization, image_data: ProcessedImageData
    ) -> list[Element]:
        """Computes the pixel value for each FEM element in the Discretization
        based on its center coordinate.

        This function interpolates pixel values at the center coordinates of
        each element and assigns the interpolated value to the element. It
        applies when the `calculation_type` is set to "elementcenter".

        Arguments:
            dis (Discretization): The Discretization object containing FEM
                elements and node coordinates.
            image_data (ProcessedImageData):  3D image data containing voxel
                coordinates and values

        Returns:
            list[Element]: A list of FEM elements with their pixel values
                assigned.
        """

        dis = self.compute_element_centers(dis)

        center_coords_array = np.array(
            [ele.center_coords for ele in dis.elements]
        )

        ele_center_values = self.interpolate_image_values_to_points(
            center_coords_array, image_data
        )

        for i, ele in tqdm(
            enumerate(dis.elements),
            total=len(dis.elements),
            desc="Processing Elements",
        ):
            ele.value = ele_center_values[i]

        return dis.elements

    def get_elementvalues_all_voxels(
        self, dis: Discretization, image_data: ProcessedImageData
    ) -> list[Element]:
        """Computes the mean pixel value for each FEM element based on all
        voxels inside the element.

        This function retrieves all voxel values contained within each FEM
        element and calculates their mean. It applies when the
        `calculation_type` is set to "allVoxel".

        Arguments:
            dis (Discretization): The Discretization object containing FEM
                elements and node coordinates.
            image_data (ProcessedImageData):  3D image data containing voxel
                coordinates and values

        Returns:
            list[Element]: A list of FEM elements with their pixel values
                assigned.
        """

        node_positions = np.array(
            [
                get_node_position_of_element(ele.node_ids, dis.nodes.ids)
                for ele in dis.elements
            ]
        )

        for i, ele in tqdm(
            enumerate(dis.elements),
            total=len(dis.elements),
            desc="Element values",
        ):

            element_coords = dis.nodes.coords[node_positions[i]]

            voxels_in_mesh = self._get_voxels_in_element(
                element_coords, image_data
            )
            ele.value = np.mean(voxels_in_mesh, axis=0)

        return dis.elements


class CalculationType(Enum):
    """Enum representing different calculation types for element value
    determination.

    This enum defines the available calculation types for mapping image data
    to finite element Discretization elements. The calculation type determines
    how the image pixel values are assigned to the elements in the
    Discretization.

    Attributes:
        NODES (str): Represents the calculation method where the pixel value
            is averaged over the nodes of the element.
        CENTER (str): Represents the calculation method where the pixel value
            is based on the center of the element.
        ALLVOXELS (str): Represents the calculation method where the pixel
            value is averaged over all voxels inside the element.
    """

    NODES = "nodes"
    CENTER = "elementcenter"
    ALLVOXELS = "allvoxels"

    def get_method(
        self, interpol: Interpolator
    ) -> Callable[[Discretization, ProcessedImageData], list[Element]]:
        """Returns the interpolation method corresponding to the current
        calculation type.

        Depending on the value of the enum, this method selects the correct
        interpolation function from the given interpolator instance, which
        performs pixel value assignment for FEM elements.

        Arguments:
            interpol (InterpolatorClass): An instance of the interpolator
                class that contains the interpolation methods for elements
                (e.g., for nodes, center, or all voxels).

        Returns:
            Callable[[Discretization,ProcessedImageData], list[Element]]:
                A function corresponding to the selected calculation method
                (nodes, element center, or all voxels).
        """
        return {
            CalculationType.NODES: interpol.get_elementvalues_nodes,
            CalculationType.ALLVOXELS: interpol.get_elementvalues_all_voxels,
            CalculationType.CENTER: interpol.get_elementvalues_center,
        }[self]


def interpolate_image_to_discretization(
    dis: Discretization, image_data: ProcessedImageData, config: dict
) -> list[Element]:
    """Performs interpolation of image data onto the FEM Discretization based
    on the specified calculation type.

    This function applies different interpolation methods depending on the
    user configuration. The pixel values are assigned to the FEM elements
    using one of the following approaches:

    - "nodes": Computes the mean pixel value for each element based on its
        node values.
    - "allVoxel": Computes the mean pixel value for each element based on
        all voxels inside it.
    - "elementcenter": Assigns pixel values based on the center of each
        element.

    Arguments:
        dis (Discretization): The Discretization object containing FEM
            elements and node coordinates.
        image_data (ProcessedImageData):  3D image data containing voxel
            coordinates and values
        config (dict): User-defined configuration settings.

    Returns:
        list[Element]: A list of FEM elements with interpolated pixel values.
    """

    calculation_type = CalculationType(
        config["processing options"]["calculation_type"]
    )

    interpolator = Interpolator()

    return calculation_type.get_method(interpolator)(dis, image_data)
