"""Transform data using a user-defined function."""

from pathlib import Path
from typing import Any

import numpy as np
from i2pp.core.discretization_readers.discretization_reader import Element
from i2pp.core.user_function_transformer.user_function_transformer import (
    UserFunctionTransformer,
)


def transform_data(
    elements: list[Element],
    user_script_path: Path,
    user_function_name: str,
    normalize: bool,
    pixel_range: np.ndarray,
) -> Any:
    """Transforms element data using a user-defined function.

    Applies a user-defined transformation to the element values,
    optionally normalizing them beforehand.

    Arguments:
        elements (List[Element]): List of elements with IDs and data.
            The data of each element will be transformed.
        user_script_path (Path): Path to the user script containing the user-
            defined function.
        user_function_name (str): Name of the user-defined function to call.
        normalize (bool): Whether to normalize the element values before
            exporting.
        pixel_range (np.ndarray): Pixel range for normalization if enabled.

    Returns:
        Any: Transformed data result.
    """
    transformer = UserFunctionTransformer(
        normalize=normalize, pixel_range=pixel_range
    )
    return transformer.apply_transformation(
        elements, user_script_path, user_function_name
    )
