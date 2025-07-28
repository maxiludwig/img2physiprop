"""Export data to a file using user function."""

import logging
from pathlib import Path

import numpy as np
from i2pp.core.discretization_helpers import initialize_unstructured_grid
from i2pp.core.discretization_readers.discretization_reader import (
    Discretization,
    Element,
)
from i2pp.core.exporters.export_format import ExportFormat
from i2pp.core.exporters.exporter import Exporter
from i2pp.core.image_readers.image_reader import PixelValueType
from i2pp.core.user_function_transformer.user_function_transformer import (
    UserFunctionTransformer,
)


def export_vtk(
    output_file: Path,
    elements: list[Element],
    pixel_type: PixelValueType,
    exported_data: dict,
    dis: Discretization,
):
    """Exports the interpolated physical property data to a VTK file for the
    verification of the i2pp output.

    Arguments:
        output_file (Path): The path to the output vtk file.
        elements (list[Element]): List of elements with IDs and data.
        pixel_type (PixelValueType): Type of pixel values used in the
            discretization.
        exported_data (dict): Data to be exported, typically containing
            interpolated physical properties.
    """
    unstructured_grid, _ = initialize_unstructured_grid(
        elements, pixel_type, dis
    )
    for key, value in exported_data.items():
        names = exported_data[key].dtype.names
        if names is None:
            raise RuntimeError(
                "The exported data must be a structured numpy array "
                "with named fields."
            )
        assert np.array_equal(
            np.array([ele.id for ele in dis.elements]), value[names[0]] - 1
        ), "The keys of the exported data must match the element IDs."
        for name in names[1:]:  # Skip the 'index' field
            if np.issubdtype(value[name].dtype, np.number):
                # only add if the data is transferable to a VTK file
                unstructured_grid.cell_data[f"{key}_{name}"] = value[name]

    # Ensure the output file ends with the correct suffix
    if not output_file.suffix:
        logging.warning(
            "Output file has no suffix. Appending the export format " "suffix."
        )
        output_file = output_file.with_suffix(".vtu")

    unstructured_grid.save(output_file)


def export_data(
    elements: list[Element],
    dis: Discretization,
    user_script_path: Path,
    user_function_name: str,
    export_format: str,
    property_output_file: Path,
    name_of_output_property: str,
    normalize: bool,
    vtk_output_file: Path,
    pixel_range: np.ndarray,
    pixel_type: PixelValueType,
) -> None:
    """Exports element data to a file using a user-defined function.

    This function collects element IDs and values, applies normalization if
    specified in the user configuration, and then exports the data using a
    custom user-defined function.

    Workflow:
        - Extracts element values and IDs.
        - Loads the user function.
        - If normalization is enabled, normalizes the element values.
        - Calls the user function to generate the export string.
        - Writes the export string to a file using `Exporter.write_data()`.
        - Depending on the export format, it also exports the data to a VTK
          file. This is not possible if the user function returns a string.

    Arguments:
        elements (List[Element]): List of elements with IDs and data.
        dis (Discretization): The discretization object containing nodes
            and elements.
        user_script_path (Path): Path to the user script containing the
            user-defined function.
        user_function_name (str): Name of the user-defined function to call
            for exporting data.
        export_format (str): The format in which the data will be
            exported (e.g., "json", "txt").
        property_output_file (Path): Path to the output file where the
            exported data will be written.
        name_of_output_property (str): Name of the property to be exported.
        normalize (bool): Whether to normalize the element values
            before exporting.
        vtk_output_file (Path): Path to the output file for VTK export.
        pixel_range (np.ndarray): Pixel range for normalization if enabled.
        pixel_type (PixelValueType): Type of pixel values.

    Raises:
        RuntimeError: If the user function cannot be loaded.
    """
    logging.info("Exporting file.")

    # Apply transformation
    transformer = UserFunctionTransformer(
        normalize=normalize, pixel_range=pixel_range
    )
    result = transformer.apply_transformation(
        elements, user_script_path, user_function_name
    )

    # Export data
    export_format_enum = ExportFormat(export_format)
    exporter: Exporter = export_format_enum.get_exporter()()
    exported_data = exporter.write_data(
        result, property_output_file, name_of_output_property
    )

    # if we have a json exporter, we can export to vtk
    if exporter.export_format == ExportFormat.JSON.value:
        logging.info("Exporting data to VTK file {vtk_output_file}.")
        export_vtk(vtk_output_file, elements, pixel_type, exported_data, dis)
