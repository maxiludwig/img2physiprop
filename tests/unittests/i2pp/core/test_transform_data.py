"""Test transform_data functions."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
from i2pp.core.discretization_readers.discretization_reader import Element
from i2pp.core.transform_data import transform_data


def test_transform_data_calls_user_function():
    """Test transform_data calls the user-defined function."""
    elements = [
        Element(id=1, node_ids=np.array([0, 0, 0]), data=np.array([1, 2, 3])),
        Element(id=2, node_ids=np.array([1, 1, 1]), data=np.array([4, 5, 6])),
    ]
    user_script_path = Path("user_script.py")
    user_function_name = "process_image_data"
    normalize = False
    pixel_range = np.array([0, 255])

    mock_uft = MagicMock()
    mock_uft.apply_transformation.return_value = np.array([10, 20, 30])
    with patch(
        "i2pp.core.transform_data.UserFunctionTransformer",
        return_value=mock_uft,
    ):
        result = transform_data(
            elements,
            user_script_path,
            user_function_name,
            normalize,
            pixel_range,
        )
        mock_uft.apply_transformation.assert_called_once_with(
            elements, user_script_path, user_function_name
        )
        assert np.array_equal(result, np.array([10, 20, 30]))
