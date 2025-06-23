"""Runner which executes the main routine of img2physiprop."""

import copy
import time

from i2pp.core.export_data import export_data
from i2pp.core.import_discretization import verify_and_load_discretization
from i2pp.core.import_image import verify_and_load_imagedata
from i2pp.core.interpolate_element_data import (
    interpolate_image_to_discretization,
)
from i2pp.core.utilities import smooth_data
from i2pp.core.visualize_results import visualize_results, visualize_smoothing


def run_i2pp(config_i2pp):
    """Executes the img2physiprop (i2pp) workflow by processing image data and
    mapping it to a finite element Discretization.

    This function performs the following steps:
    1. Loads and verifies the finite element Discretization data.
    2. Loads and verifies the image data within the Discretization's bounding
        box.
    3. Optionally applying smoothing to the image data before interpolation.
    4. Interpolates the image data onto the mesh elements based on the
        user-defined calculation type.
    5. Exports the processed data using a user-specified function.
    6. Visualizes the results if enabled in the configuration.

    Arguments:
        config_i2pp(dict): User configuration containing paths, settings,
            and processing options.
    """

    start_time = time.time()

    dis = verify_and_load_discretization(config_i2pp)

    image_data = verify_and_load_imagedata(config_i2pp, dis.bounding_box)

    processing_options: dict = config_i2pp["processing options"]
    smoothing_bool = processing_options.get("smoothing", False)

    if smoothing_bool:

        smoothing_area = int(processing_options.get("smoothing_area", 3))

        visulaization_options: dict = config_i2pp["visualization_options"]
        bool_show_smoothing = visulaization_options.get(
            "plot_smoothing", False
        )

        if bool_show_smoothing:
            image_unsmoothed = copy.deepcopy(image_data)

        image_data.pixel_data = smooth_data(
            image_data.pixel_data, smoothing_area
        )

        if bool_show_smoothing:

            time_pre_smoothing = time.time()
            visualize_smoothing(image_data, image_unsmoothed)
            time_after_smoothing = time.time()
            start_time = start_time + (
                time_after_smoothing - time_pre_smoothing
            )

    elements = interpolate_image_to_discretization(
        dis, image_data, config_i2pp
    )

    export_data(elements, config_i2pp, image_data.pixel_range)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Execution time of run_i2pp: {elapsed_time:.2f} seconds")

    visulaization_options: dict = config_i2pp["visualization_options"]

    if bool(visulaization_options["plot_results"]):
        visualize_results(config_i2pp, elements, image_data)
