"""Interpolation type definitions and handling."""

from enum import Enum
from typing import Type

from i2pp.core.interpolators.interpolator import Interpolator
from i2pp.core.interpolators.interpolator_all_voxel import InterpolatorAllVoxel
from i2pp.core.interpolators.interpolator_center import InterpolatorCenter
from i2pp.core.interpolators.interpolator_nodes import InterpolatorNodes


class InterpolationType(Enum):
    """Enum representing different interpolation methods for element value
    determination.

    This enum defines the available interpolation methods for mapping image
    data to finite element Discretization elements. The interpolation method
    determines how the image pixel data is assigned to the elements in the
    Discretization.

    Attributes:
        NODES (str): Represents the interpolation method where the pixel value
            is averaged over the nodes of the element.
        CENTER (str): Represents the interpolation method where the pixel value
            is based on the center of the element.
        ALLVOXELS (str): Represents the interpolation method where the pixel
            value is averaged over all voxels inside the element.
    """

    NODES = "nodes"
    CENTER = "elementcenter"
    ALLVOXELS = "allvoxels"

    def get_interpolator(self) -> Type[Interpolator]:
        """Retrieves the appropriate interpolation class based on the selected
        interpolation method.

        This method returns the corresponding interpolator class for assigning
        pixel values to FEM elements, depending on the current interpolation
        method. Supported methods include interpolation at element nodes,
        element centers, or averaging all voxels within an element.

        Returns:
            Type[Interpolator]: The interpolator class that matches the
                specified interpolation method.

        Raises:
            ValueError: If the interpolation method is not supported.
        """

        interpolator_map = {
            InterpolationType.NODES: InterpolatorNodes,
            InterpolationType.ALLVOXELS: InterpolatorAllVoxel,
            InterpolationType.CENTER: InterpolatorCenter,
        }

        if self not in interpolator_map:
            raise ValueError(f"Unsupported interpolation method: {self}")

        return interpolator_map[self]
