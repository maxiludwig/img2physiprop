"""Transform pixel values according to user-defined transformation function."""

import importlib.util
from pathlib import Path
from typing import Any, Callable

import numpy as np
from i2pp.core.discretization_readers.discretization_reader import Element
from i2pp.core.utilities import normalize_values


class UserFunctionTransformer:
    """Handles the application of user-defined transformation functions."""

    def __init__(
        self, normalize: bool = False, pixel_range: np.ndarray = None
    ):
        self.normalize = normalize
        self.pixel_range = pixel_range

    def load_user_function(
        self, script_path: Path, function_name: str
    ) -> Callable:
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
            raise RuntimeError(f"User script '{script_path}' not found!")

        spec = importlib.util.spec_from_file_location(
            "user_module", script_path
        )
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Failed to load module spec for {script_path}")

        user_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(user_module)

        user_function = getattr(user_module, function_name, None)
        if not callable(user_function):
            raise RuntimeError(
                f"User function {function_name} not found or not callable"
            )

        return user_function

    def apply_transformation(
        self, elements: list[Element], script_path: Path, function_name: str
    ) -> Any:
        """Applies user-defined transformation to element data."""
        element_ids = np.array([ele.id + 1 for ele in elements])
        element_data = np.array([ele.data for ele in elements])

        if self.normalize:
            element_data = normalize_values(element_data, self.pixel_range)

        user_function = self.load_user_function(script_path, function_name)
        return user_function(element_ids, element_data)
