"""User function for the user function example."""

import numpy as np


def user_function(volume_ids: np.ndarray, data: np.ndarray) -> str:
    """User function."""

    mass_frac = [(ele_data / 255) for ele_data in data]

    output_string = "\n".join(
        [
            f"{volume_ids[i]}:{mass_frac[i]},{1-mass_frac[i]}"
            for i in range(len(volume_ids))
        ]
    )

    return output_string
