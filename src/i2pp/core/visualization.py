"""Functions for visualizations."""

import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
import vtk
from i2pp.core.discretization_reader_classes.discretization_reader import (
    Element,
)
from i2pp.core.image_reader_classes.image_reader import (
    ImageData,
    PixelValueType,
)
from i2pp.core.import_discretization import verify_and_load_discretization
from i2pp.core.utilities import normalize_values


def plot_slice(
    grid: np.ndarray,
    pxl_value_type: PixelValueType,
    pxl_range: np.ndarray,
    name_plot: str,
) -> None:
    """Plot a 2D slice of image data and save it as a PNG file.

    Arguments:
        grid (np.ndarray): The 2D array representing pixel values.
        pxl_value_type (PixelValueType): The type of pixel values (e.g., RGB
            or grayscale).
        pxl_range (np.ndarray): The range of pixel values for scaling
            grayscale images.
        name_plot (str): The name of the output PNG file.

    Returns:
        None
    """
    if pxl_value_type == PixelValueType.RGB:
        plt.imshow(grid)
    else:
        plt.imshow(grid, cmap="gray", vmin=pxl_range[0], vmax=pxl_range[1])

    plt.axis("off")

    title = name_plot + ".png"
    plt.savefig(title, dpi=300, bbox_inches="tight")


def create_vtk_from_unfiltered_discretizazion(
    config: dict,
    elements_with_values: list[Element],
    pixel_type: PixelValueType,
    pixel_range: np.ndarray,
) -> vtk.vtkUnstructuredGrid:
    """Create a VTK unstructured grid from unfiltered discretization data.

    Arguments:
        config (dict): Configuration dictionary containing processing options.
        elements_with_values (list[Element]): A list of elements with
            associated values.
        pixel_type (PixelValueType): The type of pixel values (RGB or scalar).
        pixel_range (np.ndarray): The range of pixel values for normalization.

    Returns:
        vtk.vtkUnstructuredGrid: The generated VTK unstructured grid.
    """

    config["processing options"]["material_ids"] = None
    unfiltered_dis = verify_and_load_discretization(config)
    unstructured_grid = vtk.vtkUnstructuredGrid()

    points = vtk.vtkPoints()
    node_id_to_vtk_id = {}
    for i, (node_id, coord) in enumerate(
        zip(unfiltered_dis.nodes.ids, unfiltered_dis.nodes.coords)
    ):
        vtk_id = points.InsertNextPoint(coord[0], coord[1], coord[2])
        node_id_to_vtk_id[node_id] = vtk_id

    unstructured_grid.SetPoints(points)

    if pixel_type == PixelValueType.RGB:
        value_array = vtk.vtkUnsignedCharArray()
        value_array.SetNumberOfComponents(3)
        value_array.SetName("RGB_Values")
    else:
        value_array = vtk.vtkFloatArray()
        value_array.SetNumberOfComponents(1)
        value_array.SetName("ScalarValues")

    id_to_value = {elem.id: elem.data for elem in elements_with_values}

    for element in unfiltered_dis.elements:
        num_nodes = len(element.node_ids)

        if num_nodes == 4:
            cell = vtk.vtkTetra()
        elif num_nodes == 8:
            cell = vtk.vtkHexahedron()
        else:
            raise ValueError(
                f"Element with {num_nodes} nodes is not supported"
            )

        for i, node_id in enumerate(element.node_ids):
            cell.GetPointIds().SetId(i, node_id_to_vtk_id[node_id])

        unstructured_grid.InsertNextCell(
            cell.GetCellType(), cell.GetPointIds()
        )

        if element.id in id_to_value:
            value = id_to_value[element.id]
            value_norm = normalize_values(value, pixel_range)

            if pixel_type == PixelValueType.RGB:

                value_array.InsertNextTuple3(
                    value_norm[0], value_norm[1], value_norm[2]
                )
            else:
                value_array.InsertNextValue(float(value_norm))

    unstructured_grid.GetCellData().SetScalars(value_array)

    return unstructured_grid


def plot_vtk_slice(
    unstructured_grid: vtk.vtkUnstructuredGrid,
    slice_z: float,
    pixel_type: PixelValueType,
) -> None:
    """Plot a 2D slice of a VTK unstructured grid at a specified Z-coordinate.

    Arguments:
        unstructured_grid (vtk.vtkUnstructuredGrid): The VTK unstructured grid
            to slice.
        slice_z (float): The Z-coordinate at which to extract the slice.
        pixel_type (PixelValueType): The type of pixel values for
            visualization.

    Returns:
        None
    """

    plotter = pv.Plotter()

    sliced_vtk = unstructured_grid.slice(normal="z", origin=(0, 0, slice_z))

    if pixel_type == PixelValueType.RGB:

        plotter.add_mesh(sliced_vtk, show_edges=True, rgb=True)

    elif pixel_type == PixelValueType.CT or pixel_type == PixelValueType.MRT:

        plotter.add_mesh(sliced_vtk, show_edges=True, cmap="gray")

    plotter.view_isometric()

    plotter.show()


def plot_vtk_3D(
    unstructured_grid: vtk.vtkUnstructuredGrid, pixel_type: PixelValueType
) -> None:
    """Visualize a 3D VTK unstructured grid using PyVista.

    Arguments:
        unstructured_grid (vtk.vtkUnstructuredGrid): The VTK unstructured grid
            to visualize.
        pixel_type (PixelValueType): The type of pixel values for
            visualization.

    Returns:
        None
    """

    pv.start_xvfb()

    plotter = pv.Plotter()

    if pixel_type == PixelValueType.RGB:

        plotter.add_mesh(unstructured_grid, show_edges=True, rgb=True)

    elif pixel_type == PixelValueType.CT or pixel_type == PixelValueType.MRT:

        plotter.add_mesh(unstructured_grid, show_edges=True, cmap="gray")

    plotter.view_isometric()

    plotter.show()


def visualize_discretization(
    config: dict, elements_with_values: list[Element], image_data=ImageData
) -> None:
    """Generate a 3D visualization of the discretization data using PyVista.

    This function creates a 3D plot of the discretized mesh, displaying nodes,
    elements, and their associated values. It first converts the
    discretization data into a VTK unstructured grid and then visualizes it in
    both full 3D and a sliced 2D view.

    Arguments:
        config (dict): Configuration dictionary containing paths and
            processing options.
        elements_with_values (list[Element]): A list of elements with
            associated values.
        image_data (ImageData): Object containing pixel type and pixel range
            for visualization.

    Raises:
        RuntimeError: If the discretization data is invalid or cannot be
            processed.

    Returns:
        None
    """

    unstructured_grid = create_vtk_from_unfiltered_discretizazion(
        config,
        elements_with_values,
        image_data.pixel_type,
        image_data.pixel_range,
    )
    plot_vtk_3D(unstructured_grid, image_data.pixel_type)
    plot_vtk_slice(unstructured_grid, 0.0, image_data.pixel_type)

    return None
