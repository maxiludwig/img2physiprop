"""Export data to a file using user function."""

import importlib.util
import os
from typing import Any, Callable

import numpy as np


class Exporter:
    """Class to export data."""

    def __init__(self):
        """Init ExportClass."""

    def load_user_function(
        self, script_path: str, function_name: str
    ) -> Callable[..., Any]:
        """Load user function from script.

        Arguments:
            script_path {str} -- Path to the user script.
            function_name {str} -- Name of the function to load.

        Retruns:
            function -- User function.
        """

        if not os.path.isfile(script_path):
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

        if spec.loader is None:  # Falls kein Loader existiert, Fehler werfen
            raise RuntimeError(f"Failed to load module from {script_path}")

        spec.loader.exec_module(user_module)

        user_function = getattr(user_module, function_name, None)

        if not callable(user_function):
            raise RuntimeError("Userfunction not found")

        return user_function

    def normalize_values(
        self, data: np.ndarray, pxl_range: np.ndarray
    ) -> np.ndarray:
        """Normalize values to a range from 0 to 1.

        Arguments:
            data {np.ndarray} -- Data to normalize.
            pxl_range {np.ndarray} -- Pixel range.

        Returns:
            np.ndarray -- Normalized data.
        """

        data = data - pxl_range[0]

        data = data / (pxl_range[1] - pxl_range[0])

        return data

    def write_data(self, data: np.ndarray, output_path: str):
        """Write data to file.

        Arguments:
            data {np.ndarray} -- Data to export.
            filename {str} -- Name of the file.
            path {str} -- Path to the file.
        """
        pass


def export_data(element_values: np.ndarray, config, pxl_range: np.ndarray):
    """Export data to a file using user function.

    Arguments:
        interpolation_data {np.ndarray} -- Interpolated data.
        config {object} -- User configuration.
    """

    script_path = config["general"]["user_script"]
    function_name = config["general"]["user_function"]
    output_path = config["general"]["output_directory"]

    exporter = Exporter()

    norm_element_values = exporter.normalize_values(element_values, pxl_range)

    user_function = exporter.load_user_function(script_path, function_name)

    export_data = user_function(norm_element_values)

    exporter.write_data(export_data, output_path)
