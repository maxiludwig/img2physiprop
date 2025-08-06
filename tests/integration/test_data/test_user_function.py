"""User function for the user function example."""

import numpy as np


def user_function(element_ids: np.ndarray, data: np.ndarray) -> str:
    """User function."""
    # the data represents an array of rgb colors
    transformed_data = np.zeros(len(data))

    print(data)
    for i, d in enumerate(data):
        # if the pixel is blue-ish, it is set to 0
        if d[2] > 0.6:
            transformed_data[i] = 0
        else:
            # otherwise, it is set to 1
            transformed_data[i] = np.mean(d)

    # sort data and element_ids together according to element_ids
    sorted_indices = np.argsort(element_ids)
    data = data[sorted_indices]
    element_id = element_ids[sorted_indices]

    # we create a structured numpy array with the first field as 'index'
    output_array = np.zeros(
        (transformed_data.shape[0],),
        dtype=[("index", np.int32), ("data", np.float32)],
    )

    output_array["index"] = element_id
    output_array["data"] = transformed_data

    return output_array
