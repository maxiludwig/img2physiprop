"""Interpolate image data to FEM-Elements."""

from i2pp.core.configuration_validator.validator import Interpolation
from i2pp.core.discretization_readers.discretization_reader import (
    Discretization,
    Element,
)
from i2pp.core.image_readers.image_reader import ImageData
from i2pp.core.interpolators.interpolator_types import InterpolationType


def interpolate_image_to_discretization(
    dis: Discretization,
    image_data: ImageData,
    interpolation: Interpolation,
) -> list[Element]:
    """Performs interpolation of image data onto the FEM Discretization based
    on the specified interpolation method.

    This function applies different interpolation methods depending on the
    user configuration. The pixel values are assigned to the FEM elements
    using one of the following approaches:

    - "nodes": Computes the mean pixel value for each element based on its
        node values.
    - "allvoxels": Computes the mean pixel value for each element based on
        all voxels inside it.
    - "elementcenter": Assigns pixel values based on the center of each
        element.

    Arguments:
        dis (Discretization): The Discretization object containing FEM
            surfaces, elements and node coordinates.
        image_data (ImageData): A structured representation containing 3D
            pixel data, grid coordinates, orientation, and metadata.
        interpolation (Interpolation): The interpolation configuration object.

    Returns:
        list[Element]: A list of FEM elements with interpolated pixel data.
    """
    enum_interpolation_method = InterpolationType(interpolation.method)

    interpolator = enum_interpolation_method.create_interpolator(
        filter_outliers=interpolation.filter_outliers,
        set_surf_node_value=interpolation.set_surface_node_value,
    )

    return interpolator.compute_element_data(dis, image_data)
