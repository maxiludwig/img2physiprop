"""Interpolates pixel values from image-data to mesh-data."""

import numpy as np
from i2pp.core.discretization_reader_classes.discretization_reader import (
    Discretization,
    Element,
)
from i2pp.core.image_reader_classes.image_reader import ImageData
from i2pp.core.interpolator_classes.interpolator import Interpolator
from i2pp.core.utilities import get_node_position_of_element
from tqdm import tqdm


class InterpolatorCenter(Interpolator):
    """Subclass of Interpolator for mapping 3D image data to finite element
    mesh centers.

    This class extends the Interpolator and specializes in assigning pixel
    values from 3D image data to finite element mesh elements by interpolating
    at their centers. This approach is used when `calculation_type` is set to
    "elementcenter".
    """

    def compute_element_centers(self, dis: Discretization) -> Discretization:
        """Computes the centroid of each element in a finite element mesh.

        This method calculates the centroid of each element in a given
        Discretization by averaging the coordinates of its associated nodes.
        The centroid is then stored as an attribute (`center_coords`) for each
        element.

        Arguments:
            dis (Discretization): A finite element mesh containing elements
                and their associated nodes.

        Returns:
            Discretization: The input Discretization object with updated
                centroid coordinates for each element.
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

            ele.center_coords = np.array(centroid)

        return dis

    def compute_element_data(
        self, dis: Discretization, image_data: ImageData
    ) -> list[Element]:
        """Interpolates pixel values to finite element mesh elements using
        element centers.

        This method assigns interpolated pixel values to each element in the
        Discretization based on its centroid. It first computes the centroid
        for each element, maps those coordinates to the image grid, and then
        retrieves the corresponding pixel values through interpolation.

        Arguments:
            dis (Discretization): A finite element mesh containing elements
                and their associated nodes.
            image_data (ImageData): A structured representation of 3D image
                data, including pixel intensities, grid coordinates, and
                orientation.

        Returns:
            list[Element]: A list of elements with interpolated pixel values
                assigned.
        """

        dis = self.compute_element_centers(dis)

        element_center_world_coords = np.array(
            [ele.center_coords for ele in dis.elements]
        )
        element_center_grid_coords = self.world_to_grid_coords(
            element_center_world_coords,
            image_data.orientation,
            image_data.position,
        )

        ele_center_values = self.interpolate_image_values_to_points(
            element_center_grid_coords, image_data
        )

        for i, ele in tqdm(
            enumerate(dis.elements),
            total=len(dis.elements),
            desc="Processing Elements",
        ):
            ele.data = ele_center_values[i]

        self.log_interpolation_warnings()

        return dis.elements
