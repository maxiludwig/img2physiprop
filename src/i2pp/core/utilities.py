"""Useful functions that are used in other modules."""

from typing import Tuple

import numpy as np


def find_mins_maxs(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Calculates the minimum and maximum coordinate values of a given set of
    3D points.

    This function determines the bounding box for a set of points by finding
    the minimum and maximum values along each axis (X, Y, Z). A constant
    enlargement factor is applied to extend the bounding box slightly.

    Arguments:
        points (np.ndarray): A NumPy array of shape (N, 3) representing N
            points in 3D space.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Two NumPy arrays representing the
            minimum and maximum coordinates of the bounding box. The first
            array contains the minimum values [min_x, min_y, min_z], and the
            second array contains the maximum values [max_x, max_y, max_z].
    """

    CONST_ENLARGE = 2

    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    min_coords = np.array(
        [
            np.min(x) - CONST_ENLARGE,
            np.min(y) - CONST_ENLARGE,
            np.min(z) - CONST_ENLARGE,
        ]
    )
    max_coords = np.array(
        [
            np.max(x) + CONST_ENLARGE,
            np.max(y) + CONST_ENLARGE,
            np.max(z) + CONST_ENLARGE,
        ]
    )

    return min_coords, max_coords


def normalize_values(data: np.ndarray, pxl_range: np.ndarray) -> np.ndarray:
    """Normalizes the input data to a range between 0 and 1 based on the
    provided pixel range.

    This function shifts the data by subtracting the minimum pixel value and
    then scales it by dividing it by the total range (max - min). The
    resulting values will be in the range [0, 1].

    Arguments:
        data (np.ndarray): The array of data values to normalize.
        pxl_range (np.ndarray): A NumPy array containing the minimum and
            maximum pixel values [min, max] used for normalization.

    Returns:
        np.ndarray: The normalized data with values scaled between 0 and 1.
    """

    normalized_data = (data - pxl_range[0]) / (pxl_range[1] - pxl_range[0])

    return normalized_data


def get_node_position_of_element(
    element_node_ids: np.ndarray, node_ids: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Retrieves the positions (indices) of element nodes in the global node
    list.

    This function maps each node ID in `element_node_ids` to its corresponding
    index in the `node_ids` array, allowing efficient lookup for finite element
    analysis.

    Arguments:
        element_node_ids (np.ndarray): An array of node IDs belonging to a
            specific element.
        node_ids (np.ndarray): An array of all node IDs in the Discretization.

    Returns:
        np.ndarray: An array of indices representing the positions of element
            nodes in `node_ids`.
    """

    id_to_index = {node_id: idx for idx, node_id in enumerate(node_ids)}

    return np.array([id_to_index[nid] for nid in element_node_ids])
