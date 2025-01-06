"""Test Utilities Routine."""

import numpy as np
from i2pp.core.utilities import find_ids, find_mins_maxs


def test_find_mins_maxs():
    """Test find_mins_maxs."""

    points = [[0, 4, 1], [4, 9, 100], [-200, 39, 1], [20, 50, -93]]

    expected_output = np.array([-202, 2, -95, 22, 52, 102])

    assert np.array_equal(find_mins_maxs(points), expected_output)


def test_find_ids():
    """Test find_ids."""

    points = [np.array([0, 0, 0]), np.array([1, 1, 1])]
    nodes = [np.array([-1, 2, 3]), np.array([1, 1, 1]), np.array([0, 0, 0])]

    assert find_ids(points, nodes) == [2, 1]
