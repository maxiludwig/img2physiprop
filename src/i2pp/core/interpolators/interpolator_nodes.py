"""Interpolates pixel values from image-data to mesh-data."""

from typing import Optional, Union, cast

import numpy as np
from i2pp.core.discretization_readers.discretization_reader import (
    Discretization,
    Element,
)
from i2pp.core.image_readers.image_reader import ImageData
from i2pp.core.interpolators.interpolator import Interpolator
from i2pp.core.utilities import get_node_position_of_element
from tqdm import tqdm


class InterpolatorNodes(Interpolator):
    """Subclass of Interpolator for mapping 3D image data to finite element
    mesh nodes.

    This class extends Interpolator and specializes in assigning pixel values
    from 3D image data to finite element mesh elements by interpolating at
    their nodes. This approach is used when `interpolation_method` is set to
    "nodes".
    """

    # Add mode to control whether to use node weights
    def __init__(
        self,
        *args,
        mode: str = "nodes",
        surf_node_val: Optional[Union[float, np.ndarray]] = None,
        **kwargs,
    ):
        """Initializes the InterpolatorNodes."""
        super().__init__()
        self._mode = mode  # "nodes" or "nodes_weighted"
        self.set_node_value = surf_node_val

    # Helpers for readability and error handling
    def _compute_unweighted_mean(
        self, ele_nodes: np.ndarray, num_values: int
    ) -> np.ndarray:
        """Computes the unweighted mean of pixel values for an element's nodes.

        This method calculates the arithmetic mean of pixel values associated
        with an element's nodes. It handles both single-channel and multi-
        channel image data. A key feature is its handling of NaN values: if a
        node has a NaN value in any channel, that node is excluded from the
        mean calculation. If all nodes for an element have NaN values, the
        element's resulting value will be NaN.

        Args:
            ele_nodes (np.ndarray): An array of interpolated pixel values for
                the nodes of a single element. Shape can be (num_nodes,) for
                single-channel data or (num_nodes, num_values) for multi-
                channel data.
            num_values (int): The number of values per pixel (e.g., 1 for
                grayscale, 3 for RGB).

        Returns:
            np.ndarray: An array containing the mean pixel value(s) for the
            element. The shape is (num_values,).
        """
        # Mask out nodes that contain any NaN across channels
        if ele_nodes.ndim == 1:
            nan_mask = ~np.isnan(ele_nodes)
        else:
            nan_mask = ~np.isnan(ele_nodes).any(axis=1)

        if not np.any(nan_mask):
            return np.full(num_values, np.nan)

        valid = ele_nodes[nan_mask]
        mean_val = np.mean(valid, axis=0)
        return mean_val if num_values > 1 else np.array([mean_val])

    def _compute_weighted_mean(
        self,
        ele_nodes: np.ndarray,
        weights_current: np.ndarray,
        num_values: int,
    ) -> np.ndarray:
        """Computes the weighted mean of pixel values for an element's nodes.

        This method calculates the weighted average of pixel values from an
        element's nodes. It is used when a more nuanced contribution of each
        node is desired, based on pre-assigned weights. Nodes with NaN values
        are excluded from the calculation. The weights of the valid nodes are
        normalized to sum to 1 before computing the average to prevent scaling
        issues.

        Args:
            ele_nodes (np.ndarray): An array of interpolated pixel values for
                the nodes of a single element. Shape can be (num_nodes,) for
                single-channel data or (num_nodes, num_values) for multi-
                channel data.
            weights_current (np.ndarray): An array of weights corresponding to
                each node in `ele_nodes`. Shape is (num_nodes,).
            num_values (int): The number of values per pixel (e.g., 1 for
                grayscale, 3 for RGB).

        Returns:
            np.ndarray: An array containing the weighted mean pixel value(s)
            for the element. The shape is (num_values,).

        Raises:
            ValueError: If `weights_current` is None or if the shape of
                `ele_nodes` and `weights_current` are incompatible.
        """
        if weights_current is None:
            raise ValueError("Node weights are required for weighted mode.")
        if ele_nodes.shape[0] != weights_current.shape[0]:
            raise ValueError(
                "Incompatible shapes:"
                f" ele_nodes has {ele_nodes.shape[0]} nodes, "
                f"weights_current has {weights_current.shape[0]}."
            )

        # Mask out nodes that contain any NaN across channels
        if ele_nodes.ndim == 1:
            nan_mask = ~np.isnan(ele_nodes)
        else:
            nan_mask = ~np.isnan(ele_nodes).any(axis=1)

        if not np.any(nan_mask):
            return np.full(num_values, np.nan)

        vals = ele_nodes[nan_mask]
        w = weights_current[nan_mask]

        wsum = float(np.sum(w))
        if wsum == 0:
            if num_values > 1:
                return np.mean(vals, axis=0)
            else:
                return np.array([np.mean(vals)])

        w = w / wsum  # Normalize weights

        if num_values == 1:
            val = np.average(vals.reshape(-1), weights=w)
            return np.array([val])
        else:
            return np.average(vals, weights=w, axis=0)

    def _override_surface_nodes(
        self,
        *,
        node_values: np.ndarray,
        dis: Discretization,
        surf_node_value: Union[np.ndarray, float],
        num_values: int,
    ) -> None:
        """Overrides the pixel values at surface nodes with a specified value.

        This method modifies the `node_values` array in place, setting the
        values of nodes that belong to any surface in the discretization to
        the provided `surf_node_value`.

        Arguments:
            node_values (np.ndarray): Array of pixel values at each node.
            dis (Discretization): The FEM discretization containing surfaces
                and node data.
            surf_node_value (np.ndarray | float): The value to assign to
                surface nodes. Can be a single float or an array matching
                the number of pixel values.
            num_values (int): The number of pixel values per node.
        """

        surf_node_val = np.asarray(surf_node_value)

        if surf_node_val.size != num_values:
            raise ValueError(
                f"set_surf_node_value must have {num_values} value(s), "
                f"got {surf_node_val.size}"
            )

        surface_node_ids = {
            nid for surf in dis.surfaces for nid in surf.node_ids
        }

        node_id_to_index = {nid: i for i, nid in enumerate(dis.nodes.ids)}

        surface_indices = [
            node_id_to_index[nid]
            for nid in surface_node_ids
            if nid in node_id_to_index
        ]

        if surface_indices:
            node_values[surface_indices] = surf_node_value

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
            dis (Discretization): The FEM discretization containing surfaces,
                elements and node coordinate data.
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

        if self.set_node_value is not None and dis.surfaces:
            self._override_surface_nodes(
                node_values=node_values,
                dis=dis,
                surf_node_value=self.set_node_value,
                num_values=image_data.pixel_type.num_values,
            )

        node_positions = np.array(
            [
                get_node_position_of_element(ele.node_ids, dis.nodes.ids)
                for ele in dis.elements
            ]
        )

        # Only prepare node weights if we're in weighted mode
        node_weights = None
        if (
            self._mode == "nodes_weighted"
            and getattr(dis.nodes, "weights", None) is not None
        ):
            node_weights = cast(np.ndarray, dis.nodes.weights)

        for i, ele in tqdm(
            enumerate(dis.elements),
            total=len(dis.elements),
            desc="Processing Elements",
        ):
            ele_nodes = node_values[node_positions[i]]
            num_values = image_data.pixel_type.num_values

            if node_weights is not None:
                weights_current = node_weights[node_positions[i]]
                ele.data = self._compute_weighted_mean(
                    ele_nodes, weights_current, num_values
                )
            else:
                ele.data = self._compute_unweighted_mean(ele_nodes, num_values)

            if np.all(np.isnan(ele.data)):
                self.nan_elements += 1

        self._log_interpolation_warnings()

        return dis.elements
