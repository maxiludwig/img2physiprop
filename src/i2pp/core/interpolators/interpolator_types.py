"""Interpolation type definitions and handling."""

from enum import Enum

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
        NODES_WEIGHTED (str): Represents the interpolation method where the
            pixel value is weighted over the nodes of the element.
        CENTER (str): Represents the interpolation method where the pixel value
            is based on the center of the element.
        ALLVOXELS (str): Represents the interpolation method where the pixel
            value is averaged over all voxels inside the element.
        ALLVOXELS_WEIGHTED (str): Represents the interpolation method where the
            pixel value is weighted over all voxels inside the element.

    Use create_interpolator to obtain a configured interpolator instance.
    """

    NODES = "nodes"
    NODES_WEIGHTED = "nodes_weighted"
    CENTER = "elementcenter"
    ALLVOXELS = "allvoxels"
    ALLVOXELS_WEIGHTED = "allvoxels_weighted"

    def create_interpolator(
        self, *, filter_outliers: bool = False
    ) -> Interpolator:
        """Factory that returns a configured interpolator instance.

        For ALLVOXELS/ALLVOXELS_WEIGHTED, this method configures the
        unified InterpolatorAllVoxel with the appropriate mode and
        optional outlier filtering.
        """
        if self == InterpolationType.ALLVOXELS:
            return InterpolatorAllVoxel(
                mode="allvoxels", filter_outliers=filter_outliers
            )
        if self == InterpolationType.ALLVOXELS_WEIGHTED:
            return InterpolatorAllVoxel(
                mode="allvoxels_weighted", filter_outliers=filter_outliers
            )
        if self in (InterpolationType.NODES, InterpolationType.NODES_WEIGHTED):
            return InterpolatorNodes()
        if self == InterpolationType.CENTER:
            return InterpolatorCenter()
        raise ValueError(f"Unsupported interpolation method: {self}")
