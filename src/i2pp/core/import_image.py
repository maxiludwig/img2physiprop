"""Import image data and convert it into slices."""

from pathlib import Path
from typing import cast

import numpy as np
from i2pp.core.image_reader_classes.dicom_reader import DicomReader
from i2pp.core.image_reader_classes.image_reader import (
    ImageFormat,
    ImageReader,
    PixelValueType,
    SlicesData,
)
from i2pp.core.image_reader_classes.png_reader import PngReader
from i2pp.core.model_reader_classes.model_reader import Limits


def verify_input(directory: Path) -> ImageFormat:
    """Verifies the existence and readability of image data and determines the
    format type.

    This function checks whether the provided directory exists, contains
    readable image data, and determines the format of the image data (either
    DICOM or PNG). If the directory is empty, contains both DICOM and PNG
    files, or is otherwise invalid, an appropriate error is raised.

    Arguments:
        directory (Path): Path to the image-data folder.

    Raises:
        RuntimeError: If the specified path does not exist.
        RuntimeError: If the path has no readable data.
        RuntimeError: If the path contains more than one type of readable data
            (both PNG and DICOM).

    Returns:
        ImageFormat: The format of the image data (either DICOM or PNG).
    """
    if not Path(directory).is_dir():
        raise RuntimeError(
            "Imagedata file not found! img2physiprop cannot be executed!"
        )

    png_exist = any(directory.glob("*.png"))
    dicom_exist = any(directory.glob("*.dcm"))

    if not png_exist and dicom_exist:
        format_input = ImageFormat.Dicom

    elif png_exist and not dicom_exist:
        format_input = ImageFormat.PNG

    elif not png_exist and not dicom_exist:
        raise RuntimeError(
            "Input data file is empty or has no readable data! "
            "Please make sure the input file has the correct format "
            "(dicom/png). Img2physiprop cannot be executed!"
        )

    elif png_exist and dicom_exist:
        raise RuntimeError(
            "Input data file has two different format types! "
            "Img2physiprop cannot be executed!"
        )

    return format_input


def verify_and_load_imagedata(
    config: dict, limits: Limits
) -> tuple[list[SlicesData], np.ndarray]:
    """Verifies input data format and loads the image data.

    This function first checks the input directory for valid image data,
    verifies the format (either DICOM or PNG), and then uses the corresponding
    image reader to load the image data. The data is then converted into a
    list of slices, and the pixel range is determined based on the pixel type
    (MRT or other).

    Arguments:
        config (dict): User configuration containing the directory for input
            data and other settings.
        limits (Limits): Model boundaries used for processing image data.

    Returns:
        tuple (list[SlicesData], np.ndarray): A tuple containing a list of
            SlicesData objects representing the image slices, and an ndarray
            containing the pixel range.

    Raises:
        RuntimeError: If the input data directory is invalid or contains
            unsupported data formats.
    """
    directory = Path(config["Input Informations"]["image_folder_path"])

    format_image = verify_input(directory)

    readers = {ImageFormat.Dicom: DicomReader, ImageFormat.PNG: PngReader}

    image_reader = cast(ImageReader, readers[format_image](config, limits))

    raw_image = image_reader.load_image(directory)
    slices = image_reader.image_to_slices(raw_image)

    if slices[0].PixelType == PixelValueType.MRT:
        all_pxls = np.concatenate([s.PixelData.flatten() for s in slices])
        pxl_range = np.array([all_pxls.min(), all_pxls.max()])
    else:
        pxl_range = slices[0].PixelType.pxl_range

    return slices, pxl_range
