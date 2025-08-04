"""Image format detection and handling."""

from enum import Enum
from pathlib import Path
from typing import Type

import pydicom
from i2pp.core.image_readers.dicom_reader import DicomReader
from i2pp.core.image_readers.image_reader import ImageReader
from i2pp.core.image_readers.png_reader import PngReader
from PIL import Image


class ImageFormat(Enum):
    """ImageFormat (Enum): Represents the supported formats for image data.

    Attributes:
        DICOM: Represents the DICOM image format, commonly used in medical
            imaging.
        PNG: Represents the PNG (Portable Network Graphics) image format,
            typically used for color images.

    This enum is used to define the format of the input data and helps
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

        Raises:
            ValueError: If the image format is not supported.
        """
        readers = {
            ImageFormat.DICOM: DicomReader,
            ImageFormat.PNG: PngReader,
        }

        if self not in readers:
            raise ValueError(f"Unsupported image format: {self}")

        return readers[self]

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
