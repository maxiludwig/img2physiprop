"""Runner which executes the main routine of img2physiprop."""

from i2pp.core.export_data import export_data
from i2pp.core.image_data_converter import convert_imagedata
from i2pp.core.import_discretization import verify_and_load_discretization
from i2pp.core.import_image import verify_and_load_imagedata
from i2pp.core.interpolator import interpolate_image_to_discretization


def run_i2pp(config_i2pp):
    """Executes the img2physiprop (i2pp) workflow by processing image data and
    mapping it to a finite element Discretization.

    This function performs the following steps:
    1. Loads and verifies the finite element Discretization data.
    2. Loads and verifies the image data within the Discretization's limits.
    3. Converts the image slices into 3D data while optimizing performance.
    4. Interpolates the image data onto the mesh elements based on the
        user-defined calculation type.
    5. Exports the processed data using a user-specified function.

    Arguments:
        config_i2pp(dict): User configuration containing paths, settings,
            and processing options.
    """

    dis = verify_and_load_discretization(config_i2pp)
    slices_and_metadata = verify_and_load_imagedata(config_i2pp, dis.limits)

    image_data = convert_imagedata(
        slices_and_metadata, dis.limits, config_i2pp
    )

    elements = interpolate_image_to_discretization(
        dis, image_data, config_i2pp
    )

    export_data(
        elements, config_i2pp, slices_and_metadata.metadata.pixel_range
    )
