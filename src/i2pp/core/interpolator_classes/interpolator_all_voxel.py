"""Interpolates pixel values from image-data to mesh-data."""

from typing import Tuple

import numpy as np
from i2pp.core.discretization_reader_classes.discretization_reader import (
    Discretization,
    Element,
)
from i2pp.core.image_reader_classes.image_reader import GridCoords, ImageData
from i2pp.core.interpolator_classes.interpolator import Interpolator
from i2pp.core.utilities import find_mins_maxs, get_node_position_of_element
from scipy.spatial import ConvexHull
from tqdm import tqdm


class InterpolatorAllVoxel(Interpolator):
    """Subclass of Interpolator for mapping 3D image data to finite element
    mesh elements.

    This class extends Interpolator and specializes in assigning pixel values
    from 3D image data to finite element mesh elements by computing the mean
    of all voxels contained within each element. This approach is used when
    `calculation_type` is set to "allvoxels".
    """

    def _search_bounding_box(
        self, grid_coords: GridCoords, element_grid_coords: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Searches for the indices within the grid that correspond to the
        bounding box of an element.

        This function computes the bounding box of the given element in grid
        coordinates and finds the indices in the grid that correspond to the
        minimum and maximum bounds along each axis (slice, row, and column).
        It uses the `np.searchsorted` method to efficiently find the
        appropriate index ranges that enclose the element's bounding box.

        The bounding box is optionally enlarged by a specified amount (default
        is 0) to ensure proper inclusion of all relevant voxels.

        Arguments:
            grid_coords (GridCoords): The grid coordinates containing the full
                range of slices, rows, and columns.
            element_grid_coords (np.ndarray): The coordinates of the element
                in the grid, used to compute the bounding box.

        Returns:
            (Tuple[np.ndarray, np.ndarray]):
                - slice_indices: An array of indices for slices within the
                    bounding box.
                - row_indices: An array of indices for rows within the
                    bounding box.
                - col_indices: An array of indices for columns within the
                    bounding box.
        """

        mins, maxs = find_mins_maxs(points=element_grid_coords, enlargement=0)

        slice_min_idx, slice_max_idx = np.searchsorted(
            grid_coords.slice, [mins[0], maxs[0]], side="left"
        ), np.searchsorted(grid_coords.slice, [mins[0], maxs[0]], side="right")
        row_min_idx, row_max_idx = np.searchsorted(
            grid_coords.row, [mins[1], maxs[1]], side="left"
        ), np.searchsorted(grid_coords.row, [mins[1], maxs[1]], side="right")
        col_min_idx, col_max_idx = np.searchsorted(
            grid_coords.col, [mins[2], maxs[2]], side="left"
        ), np.searchsorted(grid_coords.col, [mins[2], maxs[2]], side="right")

        slice_indices = np.arange(slice_min_idx[0], slice_max_idx[1])
        row_indices = np.arange(row_min_idx[0], row_max_idx[1])
        col_indices = np.arange(col_min_idx[0], col_max_idx[1])

        return slice_indices, row_indices, col_indices

    def _is_inside_element(self, point: np.ndarray, hull: ConvexHull):
        """Checks if a point is inside a convex element.

        This function determines whether a given point is inside a convex hull
        defined by the `ConvexHull` object. It does this by evaluating the
        inequalities that define the convex region, using the hull's equations.
        If the point satisfies all of these inequalities, it is considered to
        be inside the element.

        Arguments:
            point (np.ndarray): A 3D point in space, represented as a NumPy
                array.
            hull (ConvexHull): A `ConvexHull` object that defines the
                boundaries of the element.

        Returns:
            bool: True if the point is inside the convex element, False
                otherwise.
        """

        A = hull.equations[:, :-1]
        b = hull.equations[:, -1]

        return np.all(A @ point + b <= 0)

    def _get_data_of_element(
        self, element_node_grid_coords: np.ndarray, image_data: ImageData
    ) -> np.ndarray:
        """Computes the representative pixel value for a given element based on
        its nodes in grid coordinates.

        This function identifies voxels within the element by checking whether
        their grid coordinates fall inside the convex hull formed by the
        element's node coordinates. It then extracts the corresponding pixel
        values and returns their mean.

        If no voxels are found, it estimates the pixel value via interpolation
        at the element's center. If the center of the element is outside the
        grid, it returns `np.nan`.

        Arguments:
            element_node_grid_coords (np.ndarray): The grid coordinates of
                the element's nodes.
            image_data (ImageData): Image data containing voxel coordinates
                and pixel values.

        Returns:
            np.ndarray: The mean pixel value of all voxels inside the element.
                If no voxels are found but at least one node is inside the
                grid, an interpolated value is returned.
                If all nodes are outside the grid, returns `np.nan`.
        """

        data = []

        slice_indices, row_indices, col_indices = self._search_bounding_box(
            image_data.grid_coords, element_node_grid_coords
        )

        hull = ConvexHull(element_node_grid_coords)

        for i in slice_indices:
            for j in row_indices:
                for k in col_indices:
                    grid_coord = np.array(
                        [
                            image_data.grid_coords.slice[i],
                            image_data.grid_coords.row[j],
                            image_data.grid_coords.col[k],
                        ]
                    )

                    if self._is_inside_element(grid_coord, hull):
                        data.append(image_data.pixel_data[i, j, k])

        if data:
            return np.mean(data, axis=0)
        else:
            self.backup_interpolation += 1

            element_center = np.mean(element_node_grid_coords, axis=0)

            return self.interpolate_image_values_to_points(
                element_center, image_data
            )[0]

    def compute_element_data(
        self, dis: Discretization, image_data: ImageData
    ) -> list[Element]:
        """Converts FEM node coordinates to grid coordinates and computes the
        mean pixel value for each element.

        This function first converts the world coordinates of the FEM nodes
        into grid coordinates. It then checks which voxels are located within
        each element by using the grid coordinates of the nodes. Finally, it
        calculates the mean value of all voxels inside the element and assigns
        this mean value to the element's data.

        Arguments:
            dis (Discretization): The Discretization object containing FEM
                elements and node coordinates.
            image_data (ImageData): A structured representation containing 3D
                pixel data, grid coordinates, orientation, and metadata.

        Returns:
            list[Element]: A list of FEM elements with their pixel values
                assigned.
        """

        node_grid_coords = self.world_to_grid_coords(
            dis.nodes.coords, image_data.orientation, image_data.position
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
            desc="Element values",
        ):

            element_node_grid_coords = node_grid_coords[node_positions[i]]

            ele.data = self._get_data_of_element(
                element_node_grid_coords, image_data
            )

        self.log_interpolation_warnings()

        return dis.elements
