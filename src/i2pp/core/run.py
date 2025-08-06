"""Runner which executes the main routine of img2physiprop."""

import copy
import logging
import time
from pathlib import Path

from i2pp.core.configuration_validator.validator import I2PPConfig
from i2pp.core.discretization_helpers import verify_and_load_discretization
from i2pp.core.export_data import export_data
from i2pp.core.import_image import verify_and_load_imagedata
from i2pp.core.interpolate_element_data import (
    interpolate_image_to_discretization,
)
from i2pp.core.transform_data import transform_data
from i2pp.core.utilities import smooth_data
from i2pp.core.visualize_results import visualize_results, visualize_smoothing


def run_i2pp(config_i2pp):
    """Executes the img2physiprop (i2pp) workflow by processing image data and
    mapping it to a finite element discretization.

    This function performs the following steps:
    1. Loads and verifies the finite element discretization data.
    2. Loads and verifies the image data within the discretization's bounding
        box.
    3. Optionally applying smoothing to the image data before interpolation.
    4. Interpolates the image data onto the mesh elements based on the
        user-defined interpolation method.
    5. Exports the processed data using a user-specified function.
    6. Visualizes the results if enabled in the configuration.

    Arguments:
        config_i2pp(dict): User configuration containing paths, settings,
            and processing options.
    """

    start_time = time.time()

    # Load and validate the configuration
    config = I2PPConfig.from_dict(config_i2pp)

    # Load the discretization data
    discretization = verify_and_load_discretization(
        config.import_.discretization.path,
        config.import_.discretization.options,
    )

    # Load the image data
    image = verify_and_load_imagedata(
        config.import_.image.path,
        config.import_.image.options,
        discretization.bounding_box,
    )

    # If smoothing is enabled, smooth the image data
    if config.processing.smoothing:
        # Create a copy of the image data for visualization purposes
        if config.processing.smoothing.visualize:
            image_raw = copy.deepcopy(image)

        image.pixel_data = smooth_data(
            image.pixel_data, config.processing.smoothing.smoothing_area
        )

        if config.processing.smoothing.visualize:
            visualize_smoothing(image, image_raw)

    # Interpolate the image data onto the discretization
    elements = interpolate_image_to_discretization(
        discretization,
        image,
        interpolation_method=config.processing.interpolation_method,
    )

    # Transform the data using the user-defined python function
    transformed_data = transform_data(
        elements=elements,
        user_script_path=config.processing.transformation.user_script,
        user_function_name=config.processing.transformation.user_function,
        normalize=config.processing.transformation.normalize_values,
        pixel_range=image.pixel_range,
    )

    # If visualization is enabled, visualize the results
    if config.processing.transformation.visualize:
        visualize_results(elements, image, discretization)

    # Retrieve export options from the configuration
    property_output_path: Path = (
        config.export.folder_path
        / f"{config.export.file_name}.{config.export.type}"
    )
    vtk_output_path: Path = (
        config.export.folder_path / f"{config.export.file_name}.vtu"
    )

    # Export the data
    export_data(
        transformed_data=transformed_data,
        elements=elements,
        discretization=discretization,
        export_format=config.export.type,
        property_output_file=property_output_path,
        name_of_output_property=config.export.output_parameter_name,
        vtk_output_file=vtk_output_path,
        pixel_type=image.pixel_type,
    )

    # Log the execution time
    end_time = time.time()
    elapsed_time = end_time - start_time

    logging.info(f"Execution time of run_i2pp: {elapsed_time:.2f} seconds")
