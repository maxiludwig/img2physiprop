"""Interpolates pixel values from image-data to mesh-data."""

from typing import Tuple, cast

import numpy as np
from i2pp.core.discretization_readers.discretization_reader import (
    Discretization,
    Element,
)
from i2pp.core.image_readers.image_reader import GridCoords, ImageData
from i2pp.core.interpolators.interpolator import Interpolator
from i2pp.core.utilities import find_mins_maxs, get_node_position_of_element
from scipy.spatial import ConvexHull
from tqdm import tqdm


class InterpolatorAllVoxel(Interpolator):
    """Interpolator for mapping 3D image data to finite element mesh elements.

    This class supports both unweighted and node-weighted voxel mean
    calculations, controlled via the 'mode' parameter. It assigns pixel
    values from 3D image data to finite element mesh elements by computing
    the mean of all voxels within each element.
    This functionality is used when the `interpolation_method` is set to
    "allvoxels" or "allvoxels_weighted".
    """

    def __init__(
        self,
        *args,
        filter_outliers: bool = False,
        mode: str = "allvoxels",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._filter_outliers_enabled = filter_outliers
        self._mode = mode  # "allvoxels" or "allvoxels_weighted"

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

    def _filter_outliers_modified_zscore(
        self, values: np.ndarray, threshold: float = 3.5
    ) -> np.ndarray:
        """Identify outliers using the Modified Z-Score method.

        Args:
            values (np.ndarray):
                Array of voxel values (1D or 2D, axis=0 = voxels)
            threshold (float):
                Cutoff for Modified Z-Score, default 3.5

        Returns:
            mask (np.ndarray): Boolean array, True = value is not an outlier
        """
        values = np.asarray(values)
        median = np.median(values, axis=0)

        # Median Absolute Deviation (MAD)
        mad = np.median(np.abs(values - median), axis=0)
        mad = np.maximum(mad, 1e-10)  # avoid division by zero

        # Modified Z-Score
        modified_z = 0.6745 * (values - median) / mad

        # Mask: True = not an outlier
        mask = np.abs(modified_z) <= threshold

        return mask

    def _weighted_voxel_mean(
        self,
        voxels_phys: np.ndarray,
        values: np.ndarray,
        element_node_phys: np.ndarray,
        node_weights: np.ndarray,
    ) -> float:
        """Calculate a node-weighted mean of voxel values, with optional
        outlier filtering.

        Computes the weighted mean of voxel values within an element,
        where the weight is determined based on the distances between the voxel
        and the element's nodes, scaled by the node weights. Outliers can be
        excluded using the Modified Z-Score method.

        Args:
            voxels_phys (np.ndarray):
                Physical coordinate of voxels within the element (N_voxel x 3)
            values (np.ndarray):
                Corresponding voxel values (N_voxels x ...).
            element_node_phys (np.ndarray):
                Physical coordinates of the element's nodes (N_nodes x 3).
            node_weights (np.ndarray):
                Weights assigned to the nodes (N_nodes,).

        Returns:
            float: The weighted mean of the voxel values.
        """

        distances = np.linalg.norm(
            element_node_phys[:, np.newaxis, :]
            - voxels_phys[np.newaxis, :, :],
            axis=2,
        )
        distances = np.maximum(distances, 1e-10)

        voxel_weights = np.sum(node_weights[:, np.newaxis] / distances, axis=0)

        if len(values) > 5:
            mask = self._filter_outliers_modified_zscore(values)
            filtered_values = values[mask]
            filtered_weights = voxel_weights[mask]

            # Fallback if all voxels are filtered out
            if len(filtered_values) == 0:
                filtered_values = values
                filtered_weights = voxel_weights

        else:
            # No outlier filtering for small voxel counts
            return np.average(values, weights=voxel_weights, axis=0)

        # Compute weighted mean
        return np.average(filtered_values, weights=filtered_weights, axis=0)

    def _get_data_of_element(
        self,
        element_node_grid_coords: np.ndarray,
        image_data: ImageData,
        node_weights_current: np.ndarray | None = None,
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
            node_weights_current (np.ndarray | None):
                Weights assigned to the nodes.

        Returns:
            np.ndarray: The mean pixel value of all voxels inside the element.
                If no voxels are found but at least one node is inside the
                grid, an interpolated value is returned.
                If all nodes are outside the grid, returns `np.nan`.
        """

        # Lists for collection (typed for mypy)
        voxels_phys: list[np.ndarray] = []
        data_list: list[np.ndarray] = []

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
                        voxels_phys.append(
                            np.array(
                                [
                                    image_data.grid_coords.slice[i],
                                    image_data.grid_coords.row[j],
                                    image_data.grid_coords.col[k],
                                ]
                            )
                        )
                        data_list.append(image_data.pixel_data[i, j, k])

        if len(voxels_phys) > 0:
            voxels_phys_np = np.asarray(voxels_phys)
            data = np.asarray(
                data_list
            )  # ensure ndarray for boolean masks and vector ops

            if self._mode == "allvoxels":
                if self._filter_outliers_enabled and len(data) > 5:
                    mask = self._filter_outliers_modified_zscore(data)
                    filtered = data[mask]
                    mean_val = (
                        np.mean(filtered, axis=0)
                        if len(filtered) > 0
                        else np.mean(data, axis=0)
                    )
                else:
                    mean_val = np.mean(data, axis=0)
                return (
                    np.array([mean_val])
                    if image_data.pixel_type.num_values == 1
                    else np.full(image_data.pixel_type.num_values, mean_val)
                )

            if self._mode == "allvoxels_weighted":
                element_node_phys = element_node_grid_coords
                if node_weights_current is None:
                    weighted = np.mean(data, axis=0)
                else:
                    weighted = (
                        self._weighted_voxel_mean(
                            voxels_phys_np,
                            data,
                            element_node_phys,
                            node_weights_current,
                        )
                        if self._filter_outliers_enabled
                        else np.average(
                            data,
                            weights=np.sum(
                                node_weights_current[:, np.newaxis]
                                / np.maximum(
                                    np.linalg.norm(
                                        element_node_phys[:, np.newaxis, :]
                                        - voxels_phys_np[np.newaxis, :, :],
                                        axis=2,
                                    ),
                                    1e-10,
                                ),
                                axis=0,
                            ),
                            axis=0,
                        )
                    )

                return (
                    np.array([weighted])
                    if image_data.pixel_type.num_values == 1
                    else np.full(image_data.pixel_type.num_values, weighted)
                )

            # Unknown mode fallback
            mean_val = np.mean(data, axis=0)
            return (
                np.array([mean_val])
                if image_data.pixel_type.num_values == 1
                else np.full(image_data.pixel_type.num_values, mean_val)
            )

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
                surfaces, elements and node coordinates.
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
            # Cast weights to ndarray when present to satisfy type checker
            node_weights_current = None
            if getattr(dis.nodes, "weights", None) is not None:
                weights_nd = cast(np.ndarray, dis.nodes.weights)
                node_weights_current = weights_nd[node_positions[i]]

            ele.data = self._get_data_of_element(
                element_node_grid_coords,
                image_data,
                node_weights_current=node_weights_current,
            )

            if np.all(np.isnan(ele.data)):
                self.nan_elements += 1

        self._log_interpolation_warnings()
        return dis.elements
