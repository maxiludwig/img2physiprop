"""User function for the user function example."""

import numpy as np


def user_function_array(
    volume_ids: np.ndarray, data: np.ndarray
) -> np.ndarray:
    """Dummy user function returning a structured array.

    The first field is 'index', second is 'fractions'.
    """
    # Assume data is a 1D array of values (e.g. grayscale or similar)
    # Normalize data to [0, 1]
    norm_data = (data - np.min(data)) / (np.max(data) - np.min(data))

    # Fractions: [1-norm_data, norm_data]
    fractions = np.vstack((1 - norm_data, norm_data)).T

    # Structured array definition
    output_data = np.zeros(
        len(volume_ids), dtype=[("index", "i4"), ("fractions", "f8", 2)]
    )
    output_data["index"] = volume_ids.astype(int)
    output_data["fractions"] = fractions

    return output_data


def user_function_string(volume_ids: np.ndarray, data: np.ndarray) -> str:
    """Dummy user function returning a string.

    For 4C it is better to use a structured array, as seen above. For
    more specific use cases, you might want to use a different data
    format that you can define here.
    """

    mass_frac = [(ele_data / 255) for ele_data in data]

    output_string = "\n".join(
        [
            f"{volume_ids[i]}:{mass_frac[i]},{1-mass_frac[i]}"
            for i in range(len(volume_ids))
        ]
    )

    return output_string
