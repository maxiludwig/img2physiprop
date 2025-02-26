"""Runner which executes the main routine of img2physiprop."""

import matplotlib.pyplot as plt
import numpy as np
from i2pp.core.image_reader_classes.image_reader import PixelValueType


def plot_slice(
    grid: np.ndarray,
    pxl_value_type: PixelValueType,
    pxl_range: np.ndarray,
    name_plot: str,
) -> None:
    """Plot the points in a 2D-Plot."""
    if pxl_value_type == PixelValueType.RGB:
        plt.imshow(grid)
    else:
        plt.imshow(grid, cmap="gray", vmin=pxl_range[0], vmax=pxl_range[1])

    plt.axis("off")

    title = name_plot + ".png"
    plt.savefig(title, dpi=300, bbox_inches="tight")
