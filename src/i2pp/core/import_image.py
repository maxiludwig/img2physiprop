"""Import image data and convert it in slices."""

import glob
import os
from enum import Enum
from typing import Union

import numpy as np
from i2pp.core.image_reader_classes.dicom_reader import DicomReader
from i2pp.core.image_reader_classes.image_reader import (
    PixelValueType,
    SlicesData,
)
from i2pp.core.image_reader_classes.png_reader import PngReader


class ImageFormat(Enum):
    """Options for image-data format."""

    Dicom = 1
    PNG = 2


def verify_input(directory: str) -> ImageFormat:
    """Verifies if data exists, is readable and determines fomat-type.

    Arguments:
        directory {str} -- Path to the image-data folder.

    Raises:
        RuntimeError: If Path not exists.
        RuntimeError: If Path has no readable data.
        RuntimeError: If Path has two or more formats of readable data.
    """

    if not os.path.exists(directory):
        raise RuntimeError(
            "Imagedata file not found!" "img2physiprop can not be executed!"
        )

    png_exist = False
    directory_png = directory + "*.png"
    if glob.glob(directory_png, recursive=False):
        png_exist = True

    dicom_exist = False
    directory_dicom = directory + "*.dcm"
    if glob.glob(directory_dicom, recursive=False):
        dicom_exist = True

    if not png_exist and dicom_exist:
        format_input = ImageFormat.Dicom

    elif png_exist and not dicom_exist:
        format_input = ImageFormat.PNG

    elif not png_exist and not dicom_exist:
        raise RuntimeError(
            "Input data file is empty or has no readible data!"
            "Please make sure the input file has the correct format"
            "(dicom/png). Img2physiprop can not be executed!"
        )

    elif png_exist and dicom_exist:
        raise RuntimeError(
            "Input data file has two different format types!"
            "Img2physiprop can not be executed!"
        )

    return format_input


def verify_and_load_imagedata(config) -> tuple[list[SlicesData], np.ndarray]:
    """Calls Image Reader.

    Arguments:
        config {object} -- User Configuration"""

    directory = config["general"]["input_data_directory"]

    format_image = verify_input(directory)

    image_reader: Union[DicomReader, PngReader]

    if format_image == ImageFormat.Dicom:

        image_reader = DicomReader(config)

    elif format_image == ImageFormat.PNG:

        image_reader = PngReader(config)
        image_reader.verify_additional_informations

    raw_image = image_reader.load_image(directory)
    slices = image_reader.image_2_slices(raw_image)

    if slices[0].Modality == PixelValueType.CT:

        pxl_range = np.array([-1024, 3071])

    elif slices[0].Modality == PixelValueType.MRT:

        all_pxls = np.array([s.PixelData for s in slices])

        pxl_range = np.array([np.min(all_pxls), np.max(all_pxls)])

    else:
        pxl_range = np.array([0, 255])

    return slices, pxl_range
