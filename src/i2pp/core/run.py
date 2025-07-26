"""Runner which executes the main routine of img2physiprop."""

import copy
import os
import time
from pathlib import Path

from i2pp.core.discretization_helpers import verify_and_load_discretization
from i2pp.core.export_data import export_data
from i2pp.core.import_image import verify_and_load_imagedata
from i2pp.core.interpolate_element_data import (
    interpolate_image_to_discretization,
)
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
        user-defined calculation type.
    5. Exports the processed data using a user-specified function.
    6. Visualizes the results if enabled in the configuration.

    Arguments:
        config_i2pp(dict): User configuration containing paths, settings,
            and processing options.
    """

    start_time = time.time()

    relative_path = Path(config_i2pp["discretization"]["path"])
    discretization_path = Path.cwd() / relative_path

    options = dict()
    options["material_ids"] = config_i2pp["processing options"].get(
        "material_ids", None
    )
    dis = verify_and_load_discretization(discretization_path, options)

    # Retrieve information from configuration for loading image data
    try:
        relative_path = Path(config_i2pp["image"]["path"])
    except KeyError as e:
        raise ValueError(f"Missing required configuration key: {e}") from e
    image_path = Path.cwd() / relative_path

    image_options = dict()
    image_options["image_metadata"] = config_i2pp["image"].get("metadata", {})

    # Load the image data
    image_data = verify_and_load_imagedata(
        image_path, image_options, dis.bounding_box
    )

    processing_options: dict = config_i2pp["processing options"]
    smoothing_bool = processing_options.get("smoothing", False)

    if smoothing_bool:

        smoothing_area = int(processing_options.get("smoothing_area", 3))

        visualization_options: dict = config_i2pp["visualization_options"]
        bool_show_smoothing = visualization_options.get(
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

    # Retrieve the interpolation method from the configuration
    interpolation_method = (
        config_i2pp["processing options"]["interpolation_method"],
    )
    # Interpolate the image data onto the mesh elements
    elements = interpolate_image_to_discretization(
        dis, image_data, interpolation_method=interpolation_method
    )

    # Retrieve export/output options from the configuration
    export_format = config_i2pp["output options"].get(
        "export_format", "Not specified"
    )
    directory = Path(
        config_i2pp["output options"].get("output_path") or Path.cwd()
    )
    output_name = str(
        config_i2pp["output options"].get("output_name") or "i2pp_output"
    )
    property_output_path = os.path.join(
        directory, f"{output_name}.{export_format}"
    )
    name_of_output_property = config_i2pp["output options"].get(
        "name_of_output_property", None
    )
    vtk_output_path = os.path.join(
        config_i2pp["output options"]["output_path"] or Path.cwd(),
        f"{config_i2pp['output options']['output_name'] or 'i2pp_output'}.vtu",
    )
    processing_options = config_i2pp["processing options"]
    script_path = processing_options["user_script"]
    function_name = processing_options["user_function"]
    normalize = processing_options.get("normalize_values", False)

    # Export the data
    export_data(
        elements=elements,
        dis=dis,
        user_script_path=Path(script_path),
        user_function_name=function_name,
        export_format=export_format,
        property_output_file=Path(property_output_path),
        name_of_output_property=name_of_output_property,
        normalize=normalize,
        vtk_output_file=Path(vtk_output_path),
        pixel_range=image_data.pixel_range,
        pixel_type=image_data.pixel_type,
    )

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Execution time of run_i2pp: {elapsed_time:.2f} seconds")

    visualization_options: dict = config_i2pp["visualization_options"]

    if bool(visualization_options["plot_results"]):
        visualize_results(elements, image_data, dis)
