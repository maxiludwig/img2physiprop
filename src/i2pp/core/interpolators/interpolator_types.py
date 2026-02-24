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
        NODES_SCALED (str): Represents the interpolation method where the
            pixel value is scaled over the nodes of the element.
        CENTER (str): Represents the interpolation method where the pixel value
            is based on the center of the element.
        ALLVOXELS (str): Represents the interpolation method where the pixel
            value is averaged over all voxels inside the element.
        ALLVOXELS_SCALED (str): Represents the interpolation method where the
            pixel value is scaled over all voxels inside the element.

    Use create_interpolator to obtain a configured interpolator instance.
    """

    NODES = "nodes"
    NODES_SCALED = "nodes_scaled"
    CENTER = "elementcenter"
    ALLVOXELS = "allvoxels"
    ALLVOXELS_SCALED = "allvoxels_scaled"

    def create_interpolator(
        self,
        *,
        filter_outliers: bool = False,
        set_node_value: float | list[float] | None = None,
        idw_power: int = 2,
    ) -> Interpolator:
        """Creates and returns a configured interpolator instance based on the
        selected interpolation method.

        This method returns a configured interpolator instance for assigning
        pixel values to FEM elements, depending on the current interpolation
        method. Supported methods include interpolation at element nodes,
        element centers, or averaging all voxels within an element.

        Args:
            filter_outliers (bool): If True, outliers will be filtered during
                interpolation. Defaults to False.
            set_node_value (float | None): Value to set for surface nodes.
                Defaults to None.
            idw_power (int): Power parameter for inverse distance weighting.
                Only applicable for ALLVOXELS_SCALED method. Defaults to 2.

        Returns:
            Interpolator: An instance of the interpolator that matches the
                specified interpolation method.

        Raises:
            ValueError: If the interpolation method is not supported.
        """
        if self == InterpolationType.ALLVOXELS:
            return InterpolatorAllVoxel(
                mode="allvoxels", filter_outliers=filter_outliers
            )
        if self == InterpolationType.ALLVOXELS_SCALED:
            return InterpolatorAllVoxel(
                mode="allvoxels_scaled",
                filter_outliers=filter_outliers,
                idw_power=idw_power,
            )
        if self == InterpolationType.NODES:
            return InterpolatorNodes(
                mode="nodes", surf_node_val=set_node_value
            )
        if self == InterpolationType.NODES_SCALED:
            return InterpolatorNodes(
                mode="nodes_scaled", surf_node_val=set_node_value
            )
        if self == InterpolationType.CENTER:
            return InterpolatorCenter()
        raise ValueError(f"Unsupported interpolation method: {self}")
