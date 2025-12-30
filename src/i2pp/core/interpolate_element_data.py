"""Interpolate image data to FEM-Elements."""

from i2pp.core.discretization_readers.discretization_reader import (
    Discretization,
    Element,
)
from i2pp.core.image_readers.image_reader import ImageData
from i2pp.core.interpolators.interpolator_types import InterpolationType


def interpolate_image_to_discretization(
    dis: Discretization,
    image_data: ImageData,
    interpolation_method: str,
    *,
    filter_outliers: bool = False,
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
        interpolation_method (str): The type of interpolation to perform for
            assigning pixel values to the elements. This should match one of
            the `InterpolationType` enum values (e.g., "nodes",
            "elementcenter", "allvoxels").
        filter_outliers (bool, optional): Whether to filter outliers during
            interpolation. Defaults to False.

    Returns:
        list[Element]: A list of FEM elements with interpolated pixel data.
    """
    enum_interpolation_method = InterpolationType(interpolation_method)

    # Use factory to configure weighted vs unweighted and outlier filtering
    interpolator = enum_interpolation_method.create_interpolator(
        filter_outliers=filter_outliers
    )

    return interpolator.compute_element_data(dis, image_data)
