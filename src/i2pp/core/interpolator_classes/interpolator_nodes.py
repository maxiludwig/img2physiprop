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


class InterpolatorNodes(Interpolator):
    """Subclass of Interpolator for mapping 3D image data to finite element
    mesh nodes.

    This class extends Interpolator and specializes in assigning pixel values
    from 3D image data to finite element mesh elements by interpolating at
    their nodes. This approach is used when `calculation_type` is set to
    "elementcenter".
    """

    def compute_element_data(
        self, dis: Discretization, image_data: ImageData
    ) -> list[Element]:
        """Calculates the mean interpolated pixel value for each FEM element
        based on its node values.

        This method determines the pixel values at the nodes of each element
        in the discretization by first transforming their coordinates into
        the image grid coordinate system. It then interpolates the pixel values
        at these transformed positions and computes their mean to assign a
        representative value to the element.

        Arguments:
            dis (Discretization): The FEM discretization containing elements
                and node coordinate data.
            image_data (ImageData): The 3D image dataset, including voxel
                values, spatial positioning, and metadata.

        Returns:
            list[Element]: A list of FEM elements, each assigned a mean
            interpolated pixel value.
        """

        node_grid_coords = self.world_to_grid_coords(
            dis.nodes.coords, image_data.orientation, image_data.position
        )

        node_values = self.interpolate_image_values_to_points(
            node_grid_coords, image_data
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
            ele_nodes = node_values[node_positions[i]]

            ele.data = (
                np.nanmean(ele_nodes, axis=0)
                if not np.all(np.isnan(ele_nodes))
                else np.full(image_data.pixel_type.num_values, np.nan)
            )

        self.log_interpolation_warnings()

        return dis.elements
