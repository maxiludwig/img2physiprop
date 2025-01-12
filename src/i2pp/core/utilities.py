"""Useful function that are used in other modules."""

import matplotlib.pyplot as plt
import numpy as np


def find_mins_maxs(points):
    """Find maxima and minuma x-,y-,z-coordinate for a set of points."""
    minx = maxx = miny = maxy = minz = maxz = None
    CONST_ENLARGE = 2

    x = [p[0] for p in points]
    y = [p[1] for p in points]
    z = [p[2] for p in points]

    minx = min(x)
    miny = min(y)
    minz = min(z)

    maxx = max(x)
    maxy = max(y)
    maxz = max(z)

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


def save_plot(coords, values, name):
    """Save plot to verify results."""

    x = [p[0] for p in coords]
    y = [p[1] for p in coords]
    z = [p[2] for p in coords]

    print(set(z))

    dynamic_min = min(values)
    dynamic_max = max(values)

    print(dynamic_min)
    print(dynamic_max)

    normalized_values = [
        (v - dynamic_min) / (dynamic_max - dynamic_min) for v in values
    ]
    normalized_values = np.clip(normalized_values, 0, 1)

    plt.scatter(
        x, y, c=normalized_values, cmap="gray", s=1, edgecolors="black"
    )
    plt.gca().invert_yaxis()

    plt.xlabel("X-Koordinate")
    plt.ylabel("Y-Koordinate")
    plt.title("2D-Punkte mit Grauwerten")

    plt.savefig(name, dpi=300)
