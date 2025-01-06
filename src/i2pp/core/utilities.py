"""Useful function that are used in other modules."""

import numpy as np


def find_mins_maxs(points):
    """Find maxima and minuma x-,y-,z-coordinate for a set of points."""
    minx = maxx = miny = maxy = minz = maxz = None
    CONST_ENLARGE = 2

    for p in points:
        # p contains (x, y, z)

        if minx is None:
            minx = p[0]
            maxx = p[0]  # maxx
            miny = p[1]  # miny
            maxy = p[1]  # maxy
            minz = p[2]  # minz
            maxz = p[2]  # maxz
        else:
            minx = min(p[0], minx)  # minx
            maxx = max(p[0], maxx)  # maxx
            miny = min(p[1], miny)  # miny
            maxy = max(p[1], maxy)  # maxy
            minz = min(p[2], minz)  # minz
            maxz = max(p[2], maxz)  # maxz

    limits = np.array(
        [
            minx - CONST_ENLARGE,
            miny - CONST_ENLARGE,
            minz - CONST_ENLARGE,
            maxx + CONST_ENLARGE,
            maxy + CONST_ENLARGE,
            maxz + CONST_ENLARGE,
        ],
        dtype=float,
    )

    return limits


def find_ids(points_to_find, nodes):
    """Searches Points_to_find in a set of nodes and returns the position in
    the array "nodes"."""

    nodes_array_as_tuples = [tuple(p) for p in nodes]
    points_to_find_as_tuples = [tuple(p) for p in points_to_find]

    point_set = set(nodes_array_as_tuples)

    indices = [
        nodes_array_as_tuples.index(point) if point in point_set else -1
        for point in points_to_find_as_tuples
    ]
    return indices
