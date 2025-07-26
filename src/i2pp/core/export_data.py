"""Export data to a file using user function."""

import importlib.util
import json
import logging
from enum import Enum
from pathlib import Path
from typing import Any, Callable

import numpy as np
from i2pp.core.discretization_helpers import initialize_unstructured_grid
from i2pp.core.discretization_reader_classes.discretization_reader import (
    Discretization,
    Element,
)
from i2pp.core.image_reader_classes.image_reader import PixelValueType
from i2pp.core.utilities import make_json_serializable, normalize_values


class ExportFormat(Enum):
    """Enum for export formats."""

    JSON = "json"
    TXT = "txt"


class Exporter:
    """Class to handle the export of data by loading user-defined functions and
    writing output.

    The `Exporter` class provides methods to load user-defined functions
    dynamically from a script and use them to process data. Additionally,
    it handles the saving of the processed data to a file according to user
    configuration.

    Attributes:
        export_format (ExportFormat): The format in which the data will be
        exported.
    """

    def __init__(self):
        """Init ExportClass."""
        pass

    def load_user_function(
        self, script_path: Path, function_name: str
    ) -> Callable[..., Any]:
        """Dynamically loads a user-defined function from a script file.

        This function attempts to import a Python script as a module and
        retrieve a specified function from it. If the script or function is
        not found, an error is raised.

        Arguments:
            script_path (str): The file path to the user script.
            function_name (str): The name of the function to load.

        Returns:
            Callable: The loaded user-defined function.

        Raises:
            RuntimeError: If the script file does not exist.
            RuntimeError: If the script cannot be loaded as a module.
            RuntimeError: If the function is not found or is not callable.
        """

        if not Path(script_path).is_file():
            raise RuntimeError(
                f"User script '{script_path}' not found!"
                "img2physiprop can not be executed!"
            )

        spec = importlib.util.spec_from_file_location(
            "user_module", script_path
        )

        if spec is None:
            raise RuntimeError(f"Failed to load module spec for {script_path}")

        user_module = importlib.util.module_from_spec(spec)

        if spec.loader is None:
            raise RuntimeError(f"Failed to load module from {script_path}")

        spec.loader.exec_module(user_module)

        user_function = getattr(user_module, function_name, None)

        if not callable(user_function):
            raise RuntimeError("Userfunction not found")

        return user_function

    def parse_export_format(self, export_format: str):
        """Parses the export format from the user configuration.

        This function retrieves the export format from the user configuration
        and returns it as an `ExportFormat` enum value. If the format is not
        specified or is invalid, it raises an error.

        Arguments:
            export_format (str): The export format as a string.
        Raises:
            RuntimeError: If the export format is not supported or not
            specified.
        """
        try:
            self.export_format = ExportFormat(export_format)
        except ValueError:
            supported_formats = ", ".join(fmt.value for fmt in ExportFormat)
            raise RuntimeError(
                (
                    f"Export format '{export_format}' is not supported! "
                    f"Supported formats are: {supported_formats}."
                )
            )

    def write_data(
        self, data: Any, output_file: Path, name_of_output_property: str = ""
    ) -> dict:
        """Writes the provided data to a file based on user configuration.

        This function saves data resulting from the evaluation of the user
        function to a file in a specified directory with a user-defined or
        default filename. Depending on the user function and the export
        format it writes either a JSON or TXT file.

        Arguments:
            data (Any): The data to be written to the file. It can be a
                string for TXT format or a numpy array for JSON format.
            element_ids (np.ndarray): Array of element IDs corresponding
                to the data.
            output_file (Path): Path to the output file.

        Returns:
            dict: A dictionary containing the exported data. For JSON
                format, it has the output property name as the key and the
                processed data as the value. For TXT format, it returns an
                empty dictionary.
        """

        if not output_file.suffix:
            logging.warning(
                "Output file has no suffix. Appending the export format "
                "suffix."
            )
            output_file = output_file.with_suffix(
                f".{self.export_format.value}"
            )
        if not output_file.parent.exists():
            logging.info(
                f"Creating directory {output_file.parent} for output file."
            )
            output_file.parent.mkdir(parents=True, exist_ok=True)
        logging.info(f"Writing data to {output_file}")

        if self.export_format == ExportFormat.JSON:
            assert isinstance(data, np.ndarray), (
                "You specified a JSON export format. In this case, the user "
                "function must return a structured numpy array. First field "
                "must be integer-valued and named 'index'. The other fields "
                "can be of any type, but must be JSON serializable."
            )
            assert data.dtype.names is not None, (
                "The structured numpy array must have named fields. "
                "Adapt the user function."
            )
            assert np.issubdtype(data.dtype[0], np.integer), (
                "The first field of the structured numpy array must be "
                "integer-valued. Adapt the user function."
            )
            assert data.dtype.names[0] == "index", (
                "The first field of the structured numpy array must be named "
                "'index'. Adapt the user function."
            )
            assert len(data.dtype.names) > 1, (
                "The structured numpy array must have at least one additional "
                "field. Adapt the user function."
            )
            if name_of_output_property == "":
                raise RuntimeError(
                    "You specified a JSON export format. In this case, you "
                    "must also specify the 'name_of_output_property' in the "
                    "configuration."
                )

            field_names = data.dtype.names

            # Convert the structured numpy array to a dictionary
            json_dump_data = {
                name_of_output_property: {
                    str(entry[field_names[0]]): (
                        make_json_serializable(entry[field_names[1]])
                        if len(field_names) == 2
                        else [
                            make_json_serializable(entry[field])
                            for field in field_names[1:]
                        ]
                    )
                    for entry in data
                }
            }

            with open(output_file, "w") as json_file:
                try:
                    json.dump(json_dump_data, json_file, indent=4)
                except TypeError as e:
                    logging.error(f"Error writing JSON data: {e}")
                    raise RuntimeError(
                        "Failed to write JSON data. Ensure all data is "
                        "JSON serializable."
                    )

            return {name_of_output_property: data}

        elif self.export_format == ExportFormat.TXT:
            err_msg = (
                "You specified a TXT export format. In this case, the user "
                "function must return a string."
            )
            assert isinstance(data, str), err_msg
            with open(output_file, "w") as txt_file:
                txt_file.write(data)
            return {}
        else:
            raise RuntimeError(
                f"Export format '{self.export_format.value}' is not supported!"
            )

    def export_vtk(
        self,
        output_file: Path,
        elements: list[Element],
        pixel_type: PixelValueType,
        exported_data: dict,
        dis: Discretization,
    ):
        """Exports the interpolated physical property data to a VTK file for
        the verification of the i2pp output.

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
                "Output file has no suffix. Appending the export format "
                "suffix."
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

    element_ids = np.array([ele.id + 1 for ele in elements])
    element_data = np.array([ele.data for ele in elements])

    exporter = Exporter()
    exporter.parse_export_format(export_format=export_format)

    if normalize:
        element_data = normalize_values(element_data, pixel_range)

    user_function = exporter.load_user_function(
        user_script_path, user_function_name
    )

    result = user_function(element_ids, element_data)

    exported_data = exporter.write_data(
        result, property_output_file, name_of_output_property
    )

    if exporter.export_format == ExportFormat.JSON:
        exporter.export_vtk(
            vtk_output_file, elements, pixel_type, exported_data, dis
        )
