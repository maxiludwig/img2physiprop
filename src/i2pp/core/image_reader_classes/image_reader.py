"""Import image data and convert it into slices."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional, Union

import numpy as np
from i2pp.core.discretization_reader_classes.discretization_reader import (
    Limits,
)
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
class ImageMetaData:
    """Stores metadata related to a image slice.

    This class contains essential metadata about a single image slice,
    including pixel spacing, orientation, and pixel type.

    Attributes:
        pixel_spacing (np.ndarray): The spacing between pixels in millimeters.
        orientation (np.ndarray): The orientation of the image slice in space,
            defining how the image is aligned.
        pixel_type (PixelValueType): The type of pixel values in the image
            (e.g., CT, MRI, RGB).
    """

    pixel_spacing: np.ndarray
    orientation: np.ndarray
    pixel_type: PixelValueType
    pixel_range: Optional[np.ndarray] = None


@dataclass
class Slice:
    """Represents an individual image slice with pixel data and position.

    This class stores the actual pixel data of an image slice along with its
    position in 3D space.

    Attributes:
        pixel_data (np.ndarray): The pixel data of the slice stored as a 2D
            NumPy array.
        position (np.ndarray): The spatial position of the slice, typically
            representing the coordinates of the first pixel.
    """

    pixel_data: np.ndarray
    position: np.ndarray


@dataclass
class SlicesAndMetadata:
    """Holds a collection of slices and their associated metadata.

    This dataclass organizes multiple slices of a 3D image along with
    the necessary metadata required for processing.

    Attributes:
        slices (list[Slices]): A list of individual image slices that make up
            the full dataset.
        metadata (ImageMetaData): The metadata associated with the image,
            including pixel spacing, orientation, and pixel type.
    """

    slices: list[Slice]
    metadata: ImageMetaData


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
    ) -> SlicesAndMetadata:
        """Converts raw image data into a list of SlicesData.

        This method processes the raw image data (either from a DICOM
        FileDataset or a PNG NumPy array) and transforms it into a list of
        SlicesAndMetadata. Each `SlicesAndMetadata` object represents a 2D
        slice of the 3D image, which is constructed from the input data.
        The conversion ensures that the subsequent calculations and operations
        are independent of the image format, making it compatible with both
        DICOM and PNG imports.

        Arguments:
            raw_image (Union(FileDataset, np.ndarray)): The raw image data
                from DICOM or PNG files to be converted into slices.

        Returns:
            SlicesAndMetadata: A SlicesAndMetadata object, where each
                object represents a single slice from the raw image data. The
                combined slices form the full 3D image.
        """

        pass
