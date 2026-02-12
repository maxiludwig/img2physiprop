"""Interpolate image data to FEM-Elements."""

import numpy as np
from i2pp.core.configuration_validator.validator import Interpolation
from i2pp.core.discretization_readers.discretization_reader import (
    Discretization,
    Element,
)
from i2pp.core.image_readers.image_reader import ImageData
from i2pp.core.interpolators.interpolator_types import InterpolationType


def _set_surf_element_value(
    dis: Discretization, elements: list[Element], value: float | list[float]
) -> None:
    """Sets a specified value to all elements containing at least one surface
    node.

    Applies the provided scalar or vector value to all elements that
    touch any surface, regardless of whether element.data was already
    computed.
    """
    surface_node_ids: set[int] = set()
    for surface in dis.surfaces:
        surface_node_ids.update(surface.node_ids)

    if not surface_node_ids:
        return

    # Coerce provided value: scalar -> float; array-like -> np.ndarray
    val_arr = np.asarray(value)
    if val_arr.shape == ():
        coerced_scalar = float(val_arr)
        for ele in elements:
            if not surface_node_ids.isdisjoint(ele.node_ids):
                ele.data = coerced_scalar
    else:
        coerced_vec = val_arr.astype(val_arr.dtype, copy=False)
        for ele in elements:
            if not surface_node_ids.isdisjoint(ele.node_ids):
                ele.data = coerced_vec


def interpolate_image_to_discretization(
    dis: Discretization,
    image_data: ImageData,
    interpolation: Interpolation,
) -> list[Element]:
    """Performs interpolation of image data onto the FEM Discretization.

    This function uses the settings provided in the `interpolation` object
    to control the interpolation process. The pixel values are assigned to
    the FEM elements using one of the following approaches:

    - "nodes": Computes the mean pixel value for each element based on its
        node values.
    - "allvoxels": Computes the mean pixel value for each element based on
        all voxels inside it.
    - "elementcenter": Assigns pixel values based on the center of each
        element.

    Arguments:
        dis: The Discretization object containing FEM
            surfaces, elements and node coordinates.
        image_data: A structured representation containing 3D
            pixel data, grid coordinates, orientation, and metadata.
        interpolation: The interpolation configuration object, containing
            settings like the interpolation method, outlier filtering, and
            surface value assignments.

    Returns:
        A list of FEM elements with interpolated pixel data.
    """

    enum_interpolation_method = InterpolationType(interpolation.method)

    interpolator = enum_interpolation_method.create_interpolator(
        filter_outliers=interpolation.filter_outliers,
        set_node_value=interpolation.set_node_value,
    )

    elements = interpolator.compute_element_data(dis, image_data)

    # apply fixed value to elements at boundary if configured
    if interpolation.set_ele_value is not None:
        _set_surf_element_value(dis, elements, interpolation.set_ele_value)

    return elements
