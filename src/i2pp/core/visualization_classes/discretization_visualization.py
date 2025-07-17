"""Functions for visualizations."""

import numpy as np
from i2pp.core.discretization_helpers import initialize_unstructured_grid
from i2pp.core.discretization_reader_classes.discretization_reader import (
    Discretization,
    Element,
)
from i2pp.core.visualization_classes.visualization import Visualizer


class DiscretizationVisualizer(Visualizer):
    """A visualizer for finite element discretization grids.

    This class extends `Visualizer` to handle unstructured grids based on
    finite element discretization data. It generates PyVista `UnstructuredGrid`
    representations from element and node data, allowing visualization with RGB
    or scalar values.
    """

    def compute_grid(
        self,
        config: dict,
        elements_with_values: list[Element],
        dis: Discretization,
    ) -> None:
        """Generates an unstructured grid from discretization data.

        This function constructs a pyvista unstructured grid using the nodes
        and elements from a discretization file. It assigns pixel values (RGB
        or scalar) to the grid cells based on the provided
        `elements_with_values` list.

        If no value can be assigned to an element (e.g., due to filtering or
        if the element is outside the grid), the element value is set to white
        (255 for RGB, the upper bound of `pixel_range` for scalar values).

        All necessary information for plotting (such as the grid, extracted
        grid, value presence mask, and metadata) is stored in the class
        instance attributes (`self.grid`, `self.extracted_grid`,
        `self.grid_visible`, `self.ele_has_value`, and `self.nan_exist`) for
        later use.

        Arguments:
            config (dict): Configuration dictionary containing processing
                options.
            elements_with_values (list[Element]): List of elements with
                assigned values.
            dis (Discretization): The discretization object containing nodes
                and elements.
        Returns:
            None
        """

        config["processing options"]["material_ids"] = None
        unstructured_grid, ele_has_value = initialize_unstructured_grid(
            elements_with_values, self.pixel_type, dis
        )

        self.nan_exist = not np.all(ele_has_value)
        self.ele_has_value = ele_has_value
        self.grid = unstructured_grid
        self.extracted_grid = (
            unstructured_grid.extract_cells(ele_has_value)
            if self.nan_exist
            else unstructured_grid
        )
        self.grid_visible = unstructured_grid

        return None
