"""Functions for visualizations."""

import numpy as np
import pyvista as pv
from i2pp.core.image_reader_classes.image_reader import (
    ImageData,
    PixelValueType,
)
from i2pp.core.visualization_classes.visualization import Visualizer


class ImageVisualizer(Visualizer):
    """A specialized visualizer for image-based structured grids.

    This class extends `Visualizer` to handle image data and generate a
    structured grid representation using PyVista. It supports RGB and scalar
    pixel values, applying the appropriate transformations for visualization.
    """

    def compute_grid(self, image_data: ImageData) -> None:
        """Creates a structured grid from image data.

        This function constructs a PyVista `StructuredGrid` using the
        predefined grid coordinates from the `ImageData` object. It assigns
        pixel values to the corresponding cell data. If the image data
        contains RGB values, they are reshaped accordingly; otherwise, scalar
        values are stored.

        Additionally, a transformation matrix is applied to adjust the grid's
        orientation and position in 3D space.

        All necessary information for plotting (specifically `self.grid` and
        `self.grid_visible`) is stored in the instance attributes for later use

        Arguments:
            image_data (ImageData): The image data containing precomputed grid
                coordinates, pixel values, orientation, and position.

        Returns:
            None
        """

        x, y, z = np.meshgrid(
            image_data.grid_coords.slice,
            image_data.grid_coords.row,
            image_data.grid_coords.col,
            indexing="ij",
        )
        structured_grid = pv.StructuredGrid(x, y, z)

        if image_data.pixel_type == PixelValueType.RGB:

            structured_grid.point_data["rgb_values"] = (
                image_data.pixel_data.reshape(-1, 3, order="F")
            )
        else:
            structured_grid.point_data["ScalarValues"] = (
                image_data.pixel_data.flatten(order="F")
            )

        transform_matrix = np.eye(4)
        transform_matrix[:3, :3] = image_data.orientation
        transform_matrix[:3, 3] = image_data.position

        structured_grid.transform(transform_matrix)

        self.grid = structured_grid
        self.grid_visible = structured_grid

        return None
