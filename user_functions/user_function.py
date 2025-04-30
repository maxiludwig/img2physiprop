"""User function for the user function example."""

import numpy as np


def user_function(volume_ids: np.ndarray, data: np.ndarray) -> str:
    """User function."""
    data = data

    output_string = "\n".join(
        [f"{volume_ids[i]}: {data[i]}" for i in range(len(volume_ids))]
    )

    return output_string
