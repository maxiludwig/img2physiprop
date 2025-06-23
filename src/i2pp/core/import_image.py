"""Import image data and convert it into 3D data."""

from enum import Enum
from pathlib import Path
from typing import Type, cast

import numpy as np
import pydicom
from i2pp.core.discretization_reader_classes.discretization_reader import (
    BoundingBox,
)
from i2pp.core.image_reader_classes.dicom_reader import DicomReader
from i2pp.core.image_reader_classes.image_reader import (
    ImageData,
    ImageReader,
    PixelValueType,
)
from i2pp.core.image_reader_classes.png_reader import PngReader
from PIL import Image


class ImageFormat(Enum):
    """ImageFormat (Enum): Represents the supported formats for image data.

    Attributes:
        Dicom: Represents the DICOM image format, commonly used in medical
            imaging.
        PNG: Represents the PNG (Portable Network Graphics) image format,
            typically used for color images.

    This enum is used to define the format of th input data and helps
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

    def is_file_of_format(self, path: Path) -> bool:
        """Checks if a file matches this image format.

        Arguments:
            path (Path): The file path to check.

        Returns:
            bool: True if the file matches this image format, False otherwise.
        """
        if not path.is_file():
            return False

        try:
            if self == ImageFormat.DICOM:
                pydicom.dcmread(path, stop_before_pixels=True)
                return True

            elif self == ImageFormat.PNG:
                with Image.open(path) as img:
                    img.verify()
                return True

        except Exception:
            return False

        return False


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
                print(f"Appending suffix: {file.name} -> {new_file.name}")
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
    config: dict, bounding_box: BoundingBox
) -> ImageData:
    """Validates input data format and loads 3D image data.

    This function checks the specified input folder for valid image files,
    determines the format (DICOM or PNG), and loads the data using the
    appropriate image reader. The image is then converted into a structured
    format containing pixel data and metadata. Additionally, the function sets
    the pixel intensity range based on the pixel type.

    Arguments:
        config (dict): User configuration containing the directory for input
            data and other settings.
        bounding_box (BoundingBox): The spatial region defining the area of
            interest for image processing.

    Returns:
        ImageData: An object containing structured pixel data, metadata, and
            spatial information.

    Raises:
        RuntimeError: If the input data folder is invalid or contains
            unsupported data formats.
    """
    relative_path = Path(config["input informations"]["image_folder_path"])

    folder_path = Path.cwd() / relative_path

    _detect_and_append_suffixes(folder_path)

    image_format = determine_image_format(folder_path)

    image_reader = cast(
        ImageReader, image_format.get_reader()(config, bounding_box)
    )

    raw_image = image_reader.load_image(folder_path)

    image_data = image_reader.convert_to_image_data(raw_image)

    if image_data.pixel_type == PixelValueType.MRT:

        image_data.pixel_range = np.array(
            [image_data.pixel_data.min(), image_data.pixel_data.max()]
        )
    else:
        image_data.pixel_range = image_data.pixel_type.pxl_range

    return image_data
