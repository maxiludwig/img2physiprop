"""Runner which executes the main routine of img2physiprop."""

from i2pp.core.export_data import export_data
from i2pp.core.image_data_converter import convert_imagedata
from i2pp.core.import_image import verify_and_load_imagedata
from i2pp.core.import_model import verify_and_load_model
from i2pp.core.interpolator import interpolate_image_to_mesh


def run_i2pp(config_i2pp):
    """Executes the img2physiprop (i2pp) workflow by processing image data and
    mapping it to a finite element model.

    This function performs the following steps:
    1. Loads and verifies the finite element model data.
    2. Loads and verifies the image data within the model's limits.
    3. Converts the image slices into 3D data while optimizing performance.
    4. Interpolates the image data onto the mesh elements based on the
        user-defined calculation type.
    5. Exports the processed data using a user-specified function.

    Arguments:
        config_i2pp(dict): User configuration containing paths, settings,
            and processing options.
    """

    model_data = verify_and_load_model(config_i2pp)
    slices, pxl_range = verify_and_load_imagedata(
        config_i2pp, model_data.limits
    )

    image_data = convert_imagedata(
        slices, model_data.limits, config_i2pp, pxl_range
    )

    elements = interpolate_image_to_mesh(image_data, model_data, config_i2pp)

    export_data(elements, config_i2pp, pxl_range)
