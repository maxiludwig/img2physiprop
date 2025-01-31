"""Test Utilities Routine."""

import numpy as np
from i2pp.core.utilities import Limits, find_mins_maxs


def test_find_mins_maxs():
    """Test find_mins_maxs."""

    points = np.array([[0, 4, 1], [4, 9, 100], [-200, 39, 1], [20, 50, -93]])

    expected_output = Limits(
        max=np.array([22, 52, 102]), min=np.array([-202, 2, -95])
    )
    limit = find_mins_maxs(points)

    assert np.array_equal(limit.max, expected_output.max)
    assert np.array_equal(limit.min, expected_output.min)
