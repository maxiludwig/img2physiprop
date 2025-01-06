"""Runner which executes the main routine of img2physiprop."""

from pathlib import Path

# import matplotlib.pyplot as plt
from i2pp.core.Image_Data_Converter import convert_imagedata
from i2pp.core.Image_Reader import verify_and_load_imagedata
from i2pp.core.Interpolator import interpolate_image_2_mesh
from i2pp.core.Mesh_Data import process_mesh
from i2pp.core.Mesh_Reader import verify_and_load_mesh


def run_i2pp(config_i2pp):
    """Executes Runner."""

    # define all directories
    directory_Mesh = Path(config_i2pp["general"]["input_mesh_directory"])
    directory_imagedata = config_i2pp["general"]["input_data_directory"]
    # output_directory = Path(config_i2pp["general"]["output_directory"])

    # load mesh
    input_mesh = verify_and_load_mesh(directory_Mesh)
    # Assign mesh to class Mesh_data

    mesh_data = process_mesh(input_mesh)

    # mesh_data.limits=[-1000,-1000,0,1000,10000,3]
    slices = verify_and_load_imagedata(directory_imagedata, config_i2pp)
    print("test")
    image_data = convert_imagedata(slices, mesh_data.limits)

    interpolate_image_2_mesh(image_data, mesh_data)
    """X=[p[0] for p in mesh_data.nodes] y=[p[1] for p in mesh_data.nodes]
    z=[p[2] for p in mesh_data.nodes]

    fig = plt.figure(figsize=(10, 7)) ax = fig.add_subplot(111,
    projection='3d')

    scatter = ax.scatter(x,y,z, c=interpol_values, cmap='gray', s=10)
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.5, pad=0.1)
    cbar.set_label('Grauwert')

    # Achsentitel ax.set_xlabel("X") ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    plt.show()
    """
