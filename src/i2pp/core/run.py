"""Runner which executes the main routine of img2physiprop."""

# import matplotlib.pyplot as plt
from i2pp.core.export_data import export_data
from i2pp.core.image_data_converter import convert_imagedata
from i2pp.core.import_image import verify_and_load_imagedata
from i2pp.core.import_model import verify_and_load_mesh
from i2pp.core.interpolator import interpolate_image_2_mesh


def run_i2pp(config_i2pp):
    """Executes Runner.

    Arguments:
        config_i2pp {object} -- User Configuration.
    """

    mesh_data = verify_and_load_mesh(config_i2pp)

    slices, pxl_range = verify_and_load_imagedata(config_i2pp)

    image_data = convert_imagedata(slices, mesh_data.limits, config_i2pp)

    interpol_data = interpolate_image_2_mesh(
        image_data, mesh_data, config_i2pp
    )

    export_data(interpol_data.element_values, config_i2pp, pxl_range)
