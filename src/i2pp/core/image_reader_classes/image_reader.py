"""Import image data and convert it into slices."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Union

import numpy as np
from i2pp.core.model_reader_classes.model_reader import Limits
from pydicom.dataset import FileDataset


class PixelValueType(Enum):
    """Enum for pixel value types used in image data.

    This enum defines the possible types of pixel values that a slice can
    have, determining how pixel values should be interpreted:

    Attributes:
        CT: Computed tomography pixel values.
        RGB: Red, Green, Blue color pixel values.
        MRT: Magnetic resonance tomography pixel values.

    Properties:
        pxl_range: Returns the default pixel range for each pixel value type.
            - For CT: The range is between -1024 and 3071.
            - For RGB: The range is between 0 and 255.
            - For MRT: The pixel range of MRT can vary and must be calculated
              separately.
    """

    CT = "CT"
    RGB = "RGB"
    MRT = "MR"

    @property
    def pxl_range(self) -> np.ndarray:
        """Returns the default pixel range for each pixel value type."""
        if self == PixelValueType.CT:
            return np.array([-1024, 3071])
        elif self == PixelValueType.MRT:
            return None
        elif self == PixelValueType.RGB:
            return np.array([0, 255])
        else:
            raise ValueError(f"Unsupported PixelValueType: {self}")


@dataclass
class SlicesData:
    """This dataclass holds the necessary information for a single slice in a
    3D medical image.

    Attributes:
        PixelData (np.ndarray): The pixel data of the slice as a 2D array.
        PixelSpacing (np.ndarray): The spacing between pixels in millimeters,
            typically in the x and y directions.
        ImagePositionPatient (np.ndarray): The position of the slice in the
            patient's coordinate system.
        ImageOrientationPatient (np.ndarray): The orientation of the slice
            relative to the patient's body. The first three values represent
            the direction of the rows, while the next three values represent
            the direction of the columns in the image.
        PixelType (PixelValueType): The type of pixel values.
            - CT: Computed tomography pixel values.
            - RGB: Red, Green, Blue pixel values for color images.
            - MRT: Magnetic resonance tomography pixel values.
    """

    PixelData: np.ndarray
    PixelSpacing: np.ndarray
    ImagePositionPatient: np.ndarray
    ImageOrientationPatient: np.ndarray
    PixelType: PixelValueType


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

    Dicom = ".dcm"
    PNG = ".png"


class ImageReader(ABC):
    """Abstract base class for reading and processing image data.

    This class defines a common interface for loading image data from
    different file formats (e.g., PNG and DICOM) and converting it into
    structured slice representations. It ensures a consistent workflow
    for handling both medical imaging (DICOM) and standard 2D images
    (PNG).
    """

    def __init__(self, config: dict, limits: Limits):
        """Init ImageReader."""
        self.config = config
        self.limits = limits

    @abstractmethod
    def load_image(self, directory: Path) -> Union[FileDataset, np.ndarray]:
        """Loads raw image data from a specified directory containing either
        PNG or DICOM files.

        This method reads 2D image files (either in PNG or DICOM format) from
        the provided directory, processes them as needed, and returns a list
        of the raw data as a 3D image representation.

        Arguments:
            directory (Path): The path to the directory containing the image
                files.

        Returns:
            Union(FileDataset, np.ndarray): Returns either a FileDataset
                containing the DICOM data or a NumPy array representing the RGB
                values of the PNG (depending on the file format).
        """

        pass

    @abstractmethod
    def image_to_slices(
        self, raw_image: Union[FileDataset, np.ndarray]
    ) -> list[SlicesData]:
        """Converts raw image data into a list of SlicesData.

        This method processes the raw image data (either from a DICOM
        FileDataset or a PNG NumPy array) and transforms it into a list of
        SlicesData. Each `SlicesData` object represents a two-dimensional
        slice of the 3D image, which is constructed from the input data.
        The conversion ensures that the subsequent calculations and operations
        are independent of the image format, making it compatible with both
        DICOM and PNG imports.

        Arguments:
            raw_image (Union(FileDataset, np.ndarray)): The raw image data
                from DICOM or PNG files to be converted into slices.

        Returns:
            list[SlicesData]: A list of `SlicesData` objects, where each
                object represents a single slice from the raw image data. The
                combined slices form the full 3D image.
        """

        pass
