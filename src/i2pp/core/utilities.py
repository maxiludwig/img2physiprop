"""Useful functions that are used in other modules."""

import logging
from typing import Any, Optional, Tuple

import numpy as np
from scipy.ndimage import uniform_filter


def find_mins_maxs(
    points: np.ndarray, enlargement: Optional[float] = 0
) -> Tuple[np.ndarray, np.ndarray]:
    """Computes the axis-aligned bounding box for a set of 3D points.

    This function calculates the minimum and maximum coordinate values along
    each axis (X, Y, Z) to determine the bounding box of the input points.
    An optional enlargement factor can be applied to expand the bounding box
    in all directions.

    Args:
        points (np.ndarray): A NumPy array of shape (N, 3) representing N
            points in 3D space.
        enlargement (Optional[float]): An optional value to expand the bounding
            box equally along all axes. Defaults to 0.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Two NumPy arrays representing the
            minimum and maximum coordinates of the bounding box. The first
            array contains the minimum values [min_x, min_y, min_z], and the
            second array contains the maximum values [max_x, max_y, max_z].
    """

    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    min_coords = np.array(
        [
            np.min(x) - enlargement,
            np.min(y) - enlargement,
            np.min(z) - enlargement,
        ]
    )
    max_coords = np.array(
        [
            np.max(x) + enlargement,
            np.max(y) + enlargement,
            np.max(z) + enlargement,
        ]
    )

    return min_coords, max_coords


def normalize_values(data: np.ndarray, pxl_range: np.ndarray) -> np.ndarray:
    """Normalizes  data to a range between 0 and 1 based on the provided pixel
    range.

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

    return np.searchsorted(node_ids, element_node_ids)


def smooth_data(
    data: np.ndarray,
    smoothing_window: int,
) -> np.ndarray:
    """Applies a smoothing filter to 3D image data by averaging pixel values.

    This function reduces noise or measurement errors in the image data by
    applying a smoothing filter. The filter calculates the average pixel value
    within a neighborhood defined by the `smoothing_window` parameter, which
    helps to smooth out irregularities in the data.

    Args:
        data (np.ndarray): A 3D array containing the pixel data to be smoothed.
        smoothing_window (int): The size of the neighborhood (in points) used
            to compute the average. Larger values result in smoother data,
            but may reduce fine details.

    Returns:
        np.ndarray: The smoothed image data as a 3D array, where each pixel's
        value has been replaced by the average of its neighbors within the
        defined smoothing window.
    """

    logging.info("Smooth data!")

    return uniform_filter(
        data, size=smoothing_window, mode="nearest", axes=(0, 1, 2)
    )


def make_json_serializable(obj: Any) -> Any:
    """Converts NumPy data types to standard Python types for JSON
    serialization. This function recursively processes NumPy arrays and scalar
    types to ensure compatibility with JSON serialization.

    Args:
        obj: The object to convert, which can be a NumPy array, scalar, or
            other data type.
    Returns:
        Any: The converted object, where NumPy arrays are converted to lists,
        and NumPy scalars are converted to standard Python types (int, float).
    """
    if isinstance(obj, np.ndarray):
        return [make_json_serializable(item) for item in obj]
    elif isinstance(
        obj,
        (
            np.int64,
            np.int32,
            np.int16,
            np.int8,
            np.uint64,
            np.uint32,
            np.uint16,
            np.uint8,
        ),
    ):
        return int(obj)
    elif isinstance(obj, (np.float64, np.float32, np.float16)):
        return float(obj)
    return obj
