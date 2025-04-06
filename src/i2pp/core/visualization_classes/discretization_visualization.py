"""Functions for visualizations."""

import numpy as np
import pyvista as pv
from i2pp.core.discretization_reader_classes.discretization_reader import (
    Element,
)
from i2pp.core.image_reader_classes.image_reader import (
    PixelValueType,
)
from i2pp.core.import_discretization import verify_and_load_discretization
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
        pixel_range: np.ndarray,
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
            pixel_type (PixelValueType): Specifies whether pixel values are
                RGB or scalar.
            pixel_range (np.ndarray): A 2-element array specifying the valid
                pixel value range [min, max].

        Returns:
            None
        """

        config["processing options"]["material_ids"] = None
        unfiltered_dis = verify_and_load_discretization(config)

        cells = []
        cell_types = []

        node_id_to_index = {
            node_id: i for i, node_id in enumerate(unfiltered_dis.nodes.ids)
        }

        for ele in unfiltered_dis.elements:

            node_indices = [node_id_to_index[nid] for nid in ele.node_ids]

            if len(node_indices) == 4:
                cell_types.append(pv.CellType.TETRA)
            elif len(node_indices) == 8:
                cell_types.append(pv.CellType.HEXAHEDRON)
            elif len(node_indices) == 10:
                cell_types.append(pv.CellType.QUADRATIC_TETRA)
            else:
                raise ValueError(
                    f"Unsupported element with {len(node_indices)} nodes."
                )

            cells.append([len(node_indices)] + node_indices)

        cells = np.array(
            [item for sublist in cells for item in sublist], dtype=np.int64
        )
        cell_types = np.array(cell_types, dtype=np.uint8)

        unstructured_grid = pv.UnstructuredGrid(
            cells, cell_types, unfiltered_dis.nodes.coords
        )

        id_to_value = {ele.id: ele.data for ele in elements_with_values}

        num_cells = unstructured_grid.n_cells
        ele_has_value = np.full(num_cells, False, dtype=bool)

        if self.pixel_type == PixelValueType.RGB:
            values = np.full((num_cells, 3), pixel_range[1], dtype=np.uint8)
        else:
            values = np.full(num_cells, pixel_range[1], dtype=np.float32)

        for i, ele in enumerate(unfiltered_dis.elements):
            if ele.id in id_to_value:

                value = id_to_value[ele.id]
                if not np.all(np.isnan(value)):

                    ele_has_value[i] = True

                    if self.pixel_type.num_values > 1:

                        values[i, :] = value
                    else:
                        values[i] = value

        if self.pixel_type == PixelValueType.RGB:
            unstructured_grid.cell_data["rgb_values"] = values
        else:
            unstructured_grid.cell_data["ScalarValues"] = values

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
