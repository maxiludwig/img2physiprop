"""Module for visualization of smoothing method."""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import yaml
from i2pp.core.Image_Data_Converter import convert_imagedata
from i2pp.core.Image_Reader import verify_and_load_imagedata
from munch import munchify

directory = "/scratch/bayer/Git_respository/img2physiprop/Test_png/"
parser = argparse.ArgumentParser(description="Process some integers.")
parser.add_argument(
    "--config_file_path",
    "-cfp",
    help="Path to config file.",
    type=str,
    default="src/i2pp/main_example_config.yaml",
)
args = parser.parse_args()

if not os.path.isfile(args.config_file_path):
    raise RuntimeError(
        "Config file not found! img2physiprop can not be executed!"
    )

# load config and convert to simple namespace for easier access
with open(args.config_file_path, "r") as file:
    config = munchify(yaml.safe_load(file))

slices = verify_and_load_imagedata(directory, config)

limits = [-1000, -1000, 0, 1000, 1000, 0]

image_data = convert_imagedata(slices, limits)

x = [p[0] for p in image_data.coord_array]
y = [p[1] for p in image_data.coord_array]


colors_normalized = np.array(image_data.pxl_value) / 255
plt.scatter(x, y, c=colors_normalized, s=100)
plt.savefig("smoothed", dpi=300)
