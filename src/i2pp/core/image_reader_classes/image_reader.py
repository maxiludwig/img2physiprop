"""Import image data and convert it into 3D data."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional, Union

import numpy as np
from i2pp.core.discretization_reader_classes.discretization_reader import (
    BoundingBox,
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

    @property
    def num_values(self) -> int:
        """Return the number of values per pixel for the given pixel type.

        Returns:
            int:
                - 1 for scalar types such as CT or MRT
                - 3 for RGB images

        Raises:
            ValueError: If the PixelValueType is unsupported.
        """
        if self == PixelValueType.CT or self == PixelValueType.MRT:
            return 1
        elif self == PixelValueType.RGB:
            return 3
        else:
            raise ValueError(f"Unsupported PixelValueType: {self}")


class SliceOrientation(Enum):
    """Enum representing different slice orientations in a 3D volume."""

    XY = "XY"
    YZ = "YZ"
    XZ = "XZ"
    UNKNOWN = "Unkown"

    def get_axis_index(self) -> int | None:
        """Returns the corresponding axis index for the slice orientation.

        Returns:
            (int | None): The index of the axis perpendicular to the slice
                plane (0 for YZ, 1 for XZ, 2 for XY), or None if unknown.
        """

        return {
            SliceOrientation.XY: 2,
            SliceOrientation.YZ: 0,
            SliceOrientation.XZ: 1,
        }.get(self, None)

    def is_within_crop(
        self, position: np.ndarray, bounding_box: BoundingBox
    ) -> bool:
        """Checks if a slice with a given position is within the bounding box.

        Args:
            position (np.ndarray): The spatial position of the slice.
            bounding_box (BoundingBox): The bounding box defining the cropping
                region.

        Returns:
            bool: True if the position is within the bounding box limits;
                otherwise, False.
        """

        axis_index = self.get_axis_index()
        if axis_index is None:
            return True
        return (
            bounding_box.min[axis_index]
            <= position[axis_index]
            <= bounding_box.max[axis_index]
        )


@dataclass
class GridCoords:
    """Represents the spatial coordinates of a 3D grid in grid space.

    Attributes:
        slice (np.ndarray): Coordinates along the slice (depth) dimension.
        row (np.ndarray): Coordinates along the row (height) dimension.
        col (np.ndarray): Coordinates along the column (width) dimension.
    """

    slice: np.ndarray
    row: np.ndarray
    col: np.ndarray


@dataclass
class ImageData:
    """Represents a 3D medical or scientific image with spatial and intensity
    metadata.

    Attributes:
        pixel_data (np.ndarray): A 3D NumPy array representing pixel
            intensity values.
        grid_coords (GridCoords): Spatial coordinates along the slice,
            row, and column dimensions in grid space.
        orientation (np.ndarray):
            A (3,3)-shaped NumPy array defining the image orientation in world
                coordinates:
            - The first column represents the slice (depth) direction.
            - The second column represents the row (height) direction.
            - The third column represents the column (width) direction.
        position (np.ndarray): The world coordinate position of the first
            pixel.
        pixel_type (PixelValueType): The type of pixel data (e.g., CT, MRI,
            RGB).
        pixel_range (Optional[np.ndarray]):
            A 2-element array specifying the valid pixel value range
                [min, max] (optional).
    """

    pixel_data: np.ndarray
    grid_coords: GridCoords
    orientation: np.ndarray
    position: np.ndarray
    pixel_type: PixelValueType
    pixel_range: Optional[np.ndarray] = None


class ImageReader(ABC):
    """Abstract base class for reading and processing image data.

    This class defines a common interface for loading image data from
    different file formats (e.g., PNG and DICOM) and converting it into
    structured ImageData representations. It ensures a consistent
    workflow for handling both medical imaging (DICOM) and standard 2D
    images (PNG).
    """

    def __init__(self, config: dict, bounding_box: BoundingBox):
        """Init ImageReader."""
        self.config = config
        self.bounding_box = bounding_box

    def get_slice_orientation(
        self, row_direction: np.ndarray, col_direction: np.ndarray
    ) -> SliceOrientation:
        """Determines the slice orientation based on the row and column
        direction vectors.

        Args:
            row_direction (np.ndarray): A 3-element array representing the row
                direction in 3D space.
            col_direction (np.ndarray): A 3-element array representing the
                column direction in 3D space.

        Returns:
            SliceOrientation: The identified slice orientation (XY, YZ, XZ, or
                UNKNOWN).

        Notes:
            - XY: If both row and column directions are perpendicular to the
                Z-axis.
            - YZ: If both row and column directions are perpendicular to the
                X-axis.
            - XZ: If both row and column directions are perpendicular to the
                Y-axis.
            - UNKNOWN: If the orientation does not match any of the predefined
                categories.
        """
        TOLERANCE = 1e-6
        if np.isclose(
            np.dot(row_direction, [0, 0, 1]), 0, atol=TOLERANCE
        ) and np.isclose(np.dot(col_direction, [0, 0, 1]), 0, atol=TOLERANCE):
            return SliceOrientation.XY
        elif np.isclose(
            np.dot(row_direction, [1, 0, 0]), 0, atol=TOLERANCE
        ) and np.isclose(np.dot(col_direction, [1, 0, 0]), 0, atol=TOLERANCE):
            return SliceOrientation.YZ
        elif np.isclose(
            np.dot(row_direction, [0, 1, 0]), 0, atol=TOLERANCE
        ) and np.isclose(np.dot(col_direction, [0, 1, 0]), 0, atol=TOLERANCE):
            return SliceOrientation.XZ

        return SliceOrientation.UNKNOWN

    @abstractmethod
    def load_image(self, folder_path: Path) -> Union[FileDataset, np.ndarray]:
        """Loads raw image data from a specified folder containing either PNG
        or DICOM files.

        This method reads 2D image files (either in PNG or DICOM format) from
        the provided folder, processes them as needed, and returns a list
        of the raw data as a 3D image representation.

        Arguments:
            folder_path (Path): The path to the folder containing the image
                files.

        Returns:
            Union(FileDataset, np.ndarray): Returns either a FileDataset
                containing the DICOM data or a NumPy array representing the RGB
                values of the PNG (depending on the file format).
        """

        pass

    @abstractmethod
    def convert_to_image_data(
        self, raw_image: Union[FileDataset, np.ndarray]
    ) -> ImageData:
        """Converts raw image data into an ImageData object.

        This method processes the raw input data, which can be either a
        DICOM FileDataset or a PNG NumPy array, and transforms it into a
        structured `ImageData` object. The conversion extracts pixel values,
        grid coordinates, and orientation information to ensure consistency in
        downstream processing, regardless of the input format.

        Arguments:
            raw_image (Union[FileDataset, np.ndarray]):
                The raw image data from DICOM or PNG to be converted.

        Returns:
            ImageData:
                A structured representation of the 3D image, including pixel
                data, grid coordinates, orientation, and metadata.
        """

        pass
