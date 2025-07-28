"""Import image data and convert it into 3D data."""

import logging
from pathlib import Path

import numpy as np
from i2pp.core.discretization_readers.discretization_reader import BoundingBox
from i2pp.core.image_readers.image_format import ImageFormat
from i2pp.core.image_readers.image_reader import (
    ImageData,
    PixelValueType,
)


def _detect_and_append_suffixes(folder_path: Path) -> None:
    """Scans the folder and appends correct suffixes to image files without
    extensions based on format-specific logic.

    Arguments:
        folder_path (Path): Path to folder with possibly suffix-less files.

    Returns:
        None
    """
    for file in folder_path.iterdir():
        if not file.is_file() or file.suffix:
            continue

        for fmt in ImageFormat:
            if fmt.is_file_of_format(file):
                new_file = file.with_suffix(fmt.value)
                logging.info(
                    f"Appending suffix: {file.name} -> {new_file.name}"
                )
                file.rename(new_file)
                break


def determine_image_format(folder_path: Path) -> ImageFormat:
    """Verifies the existence and readability of image data and determines the
    format type.

    This function checks whether the provided folder exists, contains
    readable image data, and determines the format of the image data (either
    DICOM or PNG). If the directory is empty, contains both DICOM and PNG
    files, or is otherwise invalid, an appropriate error is raised.

    Arguments:
        folder_path (Path): Path to the image-data folder.

    Raises:
        RuntimeError: If the specified path does not exist.
        RuntimeError: If the path has no readable data.
        RuntimeError: If the path contains more than one type of readable data
            (e.g. PNG and DICOM).

    Returns:
        ImageFormat: The format of the image data.
    """
    if not Path(folder_path).is_dir():
        raise RuntimeError(
            f"Path {folder_path} to the image data cannot be found!"
        )

    supported_formats = {
        fmt: any(folder_path.glob(f"*{fmt.value}")) for fmt in ImageFormat
    }

    detected_formats = {
        fmt for fmt, exists in supported_formats.items() if exists
    }

    if len(detected_formats) == 1:
        return detected_formats.pop()

    if not detected_formats:
        raise RuntimeError(
            "Image data folder is empty or has no readable data! "
            "Please make sure the input file has the correct format "
            f"({', '.join(fmt.value for fmt in ImageFormat)})."
        )

    raise RuntimeError(
        "Image data folder contains multiple format types! "
        "Img2physiprop cannot be executed!"
    )


def verify_and_load_imagedata(
    folder_path: Path, options: dict, bounding_box: BoundingBox
) -> ImageData:
    """Validates input data format and loads 3D image data.

    This function checks the specified input folder for valid image files,
    determines the format (DICOM or PNG), and loads the data using the
    appropriate image reader. The image is then converted into a structured
    format containing pixel data and metadata. Additionally, the function sets
    the pixel intensity range based on the pixel type.

    Arguments:
        folder_path (Path): Path to the image-data folder.
        options (dict): Options for loading image data (e.g., metadata
            settings).
        bounding_box (BoundingBox): The spatial region defining the area of
            interest for image processing.

    Returns:
        ImageData: An object containing structured pixel data, metadata, and
            spatial information.

    Raises:
        RuntimeError: If the input data folder is invalid or contains
            unsupported data formats.
    """
    _detect_and_append_suffixes(folder_path)

    image_format = determine_image_format(folder_path)

    image_reader = image_format.get_reader()(options, bounding_box)

    raw_image = image_reader.load_image(folder_path)

    image_data = image_reader.convert_to_image_data(raw_image)

    if image_data.pixel_type == PixelValueType.MRT:

        image_data.pixel_range = np.array(
            [image_data.pixel_data.min(), image_data.pixel_data.max()]
        )
    else:
        image_data.pixel_range = image_data.pixel_type.pixel_range

    return image_data
