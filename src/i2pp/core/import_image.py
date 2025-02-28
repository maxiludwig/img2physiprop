"""Import image data and convert it into slices."""

from enum import Enum
from pathlib import Path
from typing import Type, cast

import numpy as np
from i2pp.core.discretization_reader_classes.discretization_reader import (
    Limits,
)
from i2pp.core.image_reader_classes.dicom_reader import DicomReader
from i2pp.core.image_reader_classes.image_reader import (
    ImageReader,
    PixelValueType,
    SlicesAndMetadata,
)
from i2pp.core.image_reader_classes.png_reader import PngReader


class ImageFormat(Enum):
    """ImageFormat (Enum): Represents the supported formats for image data.

    Attributes:
        Dicom: Represents the DICOM image format, commonly used in medical
            imaging.
        PNG: Represents the PNG (Portable Network Graphics) image format,
            typically used for color images.

    This enum is used to define the input format of raw image data and helps
    in determining how the image data should be processed based on its format
    (e.g., DICOM vs. PNG).
    """

    DICOM = ".dcm"
    PNG = ".png"

    def get_reader(self) -> Type[ImageReader]:
        """Returns the appropriate image reader class based on the image
        format.

        Returns:
            Type[ImageReader]: A class that is a subclass of `ImageReader`,
                either `DicomReader` or `PngReader`.
        """
        return {
            ImageFormat.DICOM: DicomReader,
            ImageFormat.PNG: PngReader,
        }[self]


def determine_image_format(directory: Path) -> ImageFormat:
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
            (e.g. PNG and DICOM).

    Returns:
        ImageFormat: The format of the image data.
    """
    if not Path(directory).is_dir():
        raise RuntimeError(
            f"Path {directory} to the image data cannot be found!"
        )

    supported_formats = {
        fmt: any(directory.glob(f"*{fmt.value}")) for fmt in ImageFormat
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
    config: dict, limits: Limits
) -> SlicesAndMetadata:
    """Verifies input data format and loads the image data.

    This function first checks the input directory for valid image data,
    verifies the format (either DICOM or PNG), and then uses the corresponding
    image reader to load the image data. The data is then converted into a
    list of slices, and the pixel range is determined based on the pixel type
    (MRT or other).

    Arguments:
        config (dict): User configuration containing the directory for input
            data and other settings.
        limits (Limits): Discretization boundaries used for processing image
            data.

    Returns:
        SlicesAndMetadata: A list of `SlicesAndMetadata` objects containing
            structured pixel data and metadata for each valid slice.

    Raises:
        RuntimeError: If the input data directory is invalid or contains
            unsupported data formats.
    """
    relative_path = Path(config["input informations"]["image_folder_path"])

    directory = directory = Path.cwd() / relative_path

    image_format = determine_image_format(directory)

    image_reader = cast(ImageReader, image_format.get_reader()(config, limits))

    raw_image = image_reader.load_image(directory)
    slices_and_metadata = image_reader.image_to_slices(raw_image)

    if slices_and_metadata.metadata.pixel_type == PixelValueType.MRT:
        all_pxls = np.concatenate(
            [s.pixel_data.flatten() for s in slices_and_metadata.slices]
        )
        slices_and_metadata.metadata.pixel_range = np.array(
            [all_pxls.min(), all_pxls.max()]
        )
    else:
        slices_and_metadata.metadata.pixel_range = (
            slices_and_metadata.metadata.pixel_type.pxl_range
        )

    return slices_and_metadata
