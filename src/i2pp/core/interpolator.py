"""Interpolates pixel values from image-data to mesh-data."""

from dataclasses import dataclass

import numpy as np
from i2pp.core.image_data_converter import ProcessedImageData
from i2pp.core.model_reader_classes.model_reader import ModelData
from scipy.interpolate import griddata
from scipy.spatial import ConvexHull, Delaunay
from tqdm import tqdm


@dataclass
class InterpolData:
    """Dataclass for Interpolation-Data."""

    points: list
    element_ids: list
    element_values: list


class InterpolatorClass:
    """Class to interpolate between Image-data and Mesh-data."""

    def __init__(self, image_data: ProcessedImageData, model_data: ModelData):
        """Init InterpolatorClass."""
        self.image_data = image_data
        self.model_data = model_data

    def _point_in_hull(self, point: np.ndarray, hull: ConvexHull) -> bool:
        """Check if point is inside a hull

        Arguments:
            point {np.ndarray} -- Point to check
            hull {object} -- ConvexHull object

        Returns:
            bool -- True if point is inside the hull"""

        deln = Delaunay(hull.points[hull.vertices])
        return deln.find_simplex(point) >= 0

    def _get_voxels_in_element(self, element_points: np.ndarray) -> np.ndarray:
        """Returns all points from the image-data, that are inside an element

        Arguments:
            element_points {np.ndarray} -- Points of the element

        Returns:
            np.ndarray -- Pixel values of the voxels inside the element"""

        values_in_mesh = []
        hull = ConvexHull(element_points)

        bbox_min = element_points.min(axis=0)
        bbox_max = element_points.max(axis=0)

        for i, point in enumerate(self.image_data.coord_array):

            if np.any(point < bbox_min) or np.any(point > bbox_max):
                continue

            if self._point_in_hull(point, hull):
                values_in_mesh.append(self.image_data.pxl_value[i])

        return np.array(values_in_mesh)

    def interpolate_imagevalues_to_points(
        self, target_points: np.ndarray
    ) -> np.ndarray:
        """Interpolates the pixel values to target_points

        Arguments:
            target_points {np.ndarray} -- Points to interpolate the values to

        Returns:
            np.ndarray -- Interpolated pixel values"""

        points_image = self.image_data.coord_array
        values_image = self.image_data.pxl_value

        print("Starting Interpolation!")

        interpolated_values = griddata(
            points_image, values_image, target_points, method="linear"
        )

        print("Interpolation done!")

        return interpolated_values

    def get_value_of_elements_nodes(
        self, node_values: np.ndarray
    ) -> np.ndarray:
        """Calculates the mean pixel-value for each FEM-element in the model.
        For calculation_type: nodes (mean of all element nodes)

        Arguments:
            node_values {np.ndarray} -- Interpolated values for each model node

        Returns:
            np.ndarray -- Mean pixel values for each element"""

        element_values = []

        for ele_ids in tqdm(
            self.model_data.element_ids, desc="Element values"
        ):

            pxl_value = node_values[ele_ids]

            element_values.append(np.mean(pxl_value, axis=0))

        return np.array(element_values)

    def get_value_of_elements_all_Voxels(self) -> np.ndarray:
        """Calculates the mean pixel-value for each FEM-element in the mesh.
        For calculation_type: allVoxel (mean of all voxels in the element)

        Returns:
            np.ndarray -- Mean pixel values for each element"""

        element_values = []

        for ele_ids in tqdm(
            self.model_data.element_ids, desc="Element values"
        ):

            element_points = self.model_data.nodes[ele_ids]

            voxels_in_mesh = self._get_voxels_in_element(element_points)

            element_values.append(np.mean(voxels_in_mesh, axis=0))

        return np.array(element_values)


def interpolate_image_2_mesh(
    image_data: ProcessedImageData, model_data: ModelData, config
) -> InterpolData:
    """Call Interpolation functions.

    Arguments:
        image_data {object} -- Processed image data
        model_data {object} -- Model data
        config {object} -- User configuration

    Returns:
        object -- Interpolated data"""

    calculation_type = config["Further customizations"]["calculation_type"]

    interpol_data = InterpolData(
        points=model_data.nodes,
        element_ids=model_data.element_ids,
        element_values=[],
    )
    interpolator = InterpolatorClass(image_data, model_data)

    match calculation_type:
        case "nodes":
            node_values = interpolator.interpolate_imagevalues_to_points(
                model_data.nodes
            )
            interpol_data.element_values = (
                interpolator.get_value_of_elements_nodes(node_values)
            )

        case "allVoxel":
            interpol_data.element_values = (
                interpolator.get_value_of_elements_all_Voxels()
            )

        case "elementcenter":
            interpol_data.element_values = (
                interpolator.interpolate_imagevalues_to_points(
                    model_data.element_center
                )
            )

    return interpol_data
