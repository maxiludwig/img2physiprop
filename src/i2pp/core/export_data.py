"""Export data to a file using user function."""

import importlib.util
import logging
from pathlib import Path
from typing import Any, Callable

import numpy as np
from i2pp.core.model_reader_classes.model_reader import Element
from i2pp.core.utilities import normalize_values


class Exporter:
    """Class to handle the export of data by loading user-defined functions and
    writing output.

    The `Exporter` class provides methods to load user-defined functions
    dynamically from a script and use them to process data. Additionally,
    it handles the saving of the processed data to a file according to user
    configuration.
    """

    def __init__(self):
        """Init ExportClass."""

    def load_user_function(
        self, script_path: str, function_name: str
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

    def write_data(self, export_string: str, config: dict):
        """Writes the provided data to a file based on user configuration.

        This function saves the given string content to a file in a specified
        directory with a user-defined or default filename.

        Arguments:
            export_string (str): The string to write to the file.
            config (dict): User configuration with output settings.

        File location and name are determined by `config["Output options"]`:
            - "Output path": Directory to save the file (defaults to CWD).
            - "Output name": Filename (defaults to "Output").

        The function writes the content to a file with a `.pattern` extension.
        """

        user_config: dict = config["Output options"]

        directory = Path(user_config.get("Output_path") or Path.cwd())
        output_name = str(user_config.get("Output_name") or "Output")

        path = directory / f"{output_name}.pattern"

        with open(path, "w", encoding="utf-8") as file:
            file.write(export_string)


def export_data(elements: list[Element], config: dict, pxl_range: np.ndarray):
    """Exports element data to a file using a user-defined function.

    This function collects element IDs and values, applies normalization if
    specified in the user configuration, and then exports the data using a
    custom user-defined function.

    Workflow:
        - Extracts element values and IDs.
        - Loads the user function from `config["Processing options"]`.
        - If normalization is enabled in `config["Processing options"]`,
          it normalizes the element values.
        - Calls the user function to generate the export string.
        - Writes the export string to a file using `Exporter.write_data()`.

    Arguments:
        elements (List[Element]): List of elements with IDs and values.
        config (dict): User configuration containing export settings.
        pxl_range (np.ndarray): Pixel range for normalization if enabled.

    Raises:
        RuntimeError: If the user function cannot be loaded.
    """

    logging.info("Export File!")

    element_values = []
    element_ids = []
    for ele in elements:
        element_values.append(ele.value)
        element_ids.append(ele.id + 1)

    np.array(element_values)
    np.array(element_ids)

    processing_options = config["Processing options"]

    script_path = processing_options["user_script"]
    function_name = processing_options["user_function"]

    exporter = Exporter()

    if processing_options["normalize_values"]:
        element_values = normalize_values(elements, pxl_range)

    user_function = exporter.load_user_function(script_path, function_name)

    export_string = user_function(element_ids, element_values)

    exporter.write_data(export_string, config)
