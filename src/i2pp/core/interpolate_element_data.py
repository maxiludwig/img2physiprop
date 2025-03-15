"""Interpolate image data to FEM-Elements."""

from enum import Enum
from typing import Type, cast

from i2pp.core.discretization_reader_classes.discretization_reader import (
    Discretization,
    Element,
)
from i2pp.core.image_reader_classes.image_reader import ImageData
from i2pp.core.interpolator_classes.interpolator import Interpolator
from i2pp.core.interpolator_classes.interpolator_all_voxel import (
    InterpolatorAllVoxel,
)
from i2pp.core.interpolator_classes.interpolator_center import (
    InterpolatorCenter,
)
from i2pp.core.interpolator_classes.interpolator_nodes import InterpolatorNodes


class CalculationType(Enum):
    """Enum representing different calculation types for element value
    determination.

    This enum defines the available calculation types for mapping image data
    to finite element Discretization elements. The calculation type determines
    how the image pixel data is assigned to the elements in the
    Discretization.

    Attributes:
        NODES (str): Represents the calculation method where the pixel value
            is averaged over the nodes of the element.
        CENTER (str): Represents the calculation method where the pixel value
            is based on the center of the element.
        ALLVOXELS (str): Represents the calculation method where the pixel
            value is averaged over all voxels inside the element.
    """

    NODES = "nodes"
    CENTER = "elementcenter"
    ALLVOXELS = "allvoxels"

    def get_interpolator(self) -> Type[Interpolator]:
        """Retrieves the appropriate interpolation class based on the selected
        calculation type.

        This method returns the corresponding interpolator class for assigning
        pixel values to FEM elements, depending on the current calculation
        type. Supported methods include interpolation at element nodes, element
        centers, or averaging all voxels within an element.

        Returns:
            Type[Interpolator]: The interpolator class that matches the
                specified calculation type.
        """

        return {
            CalculationType.NODES: InterpolatorNodes,
            CalculationType.ALLVOXELS: InterpolatorAllVoxel,
            CalculationType.CENTER: InterpolatorCenter,
        }[self]


def interpolate_image_to_discretization(
    dis: Discretization, image_data: ImageData, config: dict
) -> list[Element]:
    """Performs interpolation of image data onto the FEM Discretization based
    on the specified calculation type.

    This function applies different interpolation methods depending on the
    user configuration. The pixel values are assigned to the FEM elements
    using one of the following approaches:

    - "nodes": Computes the mean pixel value for each element based on its
        node values.
    - "allVoxel": Computes the mean pixel value for each element based on
        all voxels inside it.
    - "elementcenter": Assigns pixel values based on the center of each
        element.

    Arguments:
        dis (Discretization): The Discretization object containing FEM
            elements and node coordinates.
        image_data (ImageData): A structured representation containing 3D
            pixel data, grid coordinates, orientation, and metadata.
        config (dict): User-defined configuration settings.

    Returns:
        list[Element]: A list of FEM elements with interpolated pixel data.
    """

    calculation_type = CalculationType(
        config["processing options"]["calculation_type"]
    )

    interpolator = cast(Interpolator, calculation_type.get_interpolator()())

    return interpolator.compute_element_data(dis, image_data)
