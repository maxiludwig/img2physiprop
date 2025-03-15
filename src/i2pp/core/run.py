"""Runner which executes the main routine of img2physiprop."""

from i2pp.core.export_data import export_data
from i2pp.core.import_discretization import verify_and_load_discretization
from i2pp.core.import_image import verify_and_load_imagedata
from i2pp.core.interpolate_element_data import (
    interpolate_image_to_discretization,
)
from i2pp.core.utilities import smooth_data


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

    Arguments:
        config_i2pp(dict): User configuration containing paths, settings,
            and processing options.
    """

    dis = verify_and_load_discretization(config_i2pp)

    image_data = verify_and_load_imagedata(config_i2pp, dis.bounding_box)

    processing_options: dict = config_i2pp["processing options"]
    smoothing_bool = bool(processing_options.get("smoothing") or False)

    if smoothing_bool:

        smoothing_area = int(processing_options.get("smoothing_area") or 3)
        image_data.pixel_data = smooth_data(
            image_data.pixel_data, smoothing_area
        )

    elements = interpolate_image_to_discretization(
        dis, image_data, config_i2pp
    )

    export_data(elements, config_i2pp, image_data.pixel_range)
