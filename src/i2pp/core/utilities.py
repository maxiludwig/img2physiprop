"""Useful function that are used in other modules."""

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np


@dataclass
class Limits:
    """Dataclass for Limits of a set of points."""

    max: np.ndarray
    min: np.ndarray


def find_mins_maxs(points: np.ndarray) -> Limits:
    """Find the minimum and maximum values of a set of points.

    Arguments:
        points {np.ndarray} -- Set of points

    Returns:
        object -- Limits of the points
    """
    minx = maxx = miny = maxy = minz = maxz = None
    CONST_ENLARGE = 2

    x = [p[0] for p in points]
    y = [p[1] for p in points]
    z = [p[2] for p in points]

    minx = min(x) - CONST_ENLARGE
    miny = min(y) - CONST_ENLARGE
    minz = min(z) - CONST_ENLARGE

    maxx = max(x) + CONST_ENLARGE
    maxy = max(y) + CONST_ENLARGE
    maxz = max(z) + CONST_ENLARGE

    return Limits(
        max=np.array([maxx, maxy, maxz]), min=np.array([minx, miny, minz])
    )


def plot_png(
    coords: np.ndarray, values: np.ndarray, range: np.ndarray
) -> None:
    """Plot the points in a 2D-Plot."""

    colors_normalized = values / range[1]

    x = [p[0] for p in coords]
    y = [p[1] for p in coords]

    plt.scatter(x, y, c=colors_normalized, s=100)
    plt.savefig("not_smoothed", dpi=300)


def save_plot(coords: np.ndarray, values: np.ndarray, name: str) -> None:
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
