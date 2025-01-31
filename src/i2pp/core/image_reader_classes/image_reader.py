"""Import image data and convert it in slices."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

import numpy as np


@dataclass
class SlicesData:
    """Dataclass for slice-data."""

    PixelData: np.ndarray
    image_shape: np.ndarray
    PixelSpacing: np.ndarray
    ImagePositionPatient: np.ndarray
    ImageOrientationPatient: np.ndarray
    Modality: Enum


class PixelValueType(Enum):
    """Enum for pixel value types."""

    CT = 1
    RGB = 2
    MRT = 3


class ImageReader(ABC):
    """Class to read image-data."""

    def __init__(self, config):
        """Init ImageReader."""
        self.config = config

    @abstractmethod
    def load_image(self, directory):
        """Load raw-image-data."""
        pass

    @abstractmethod
    def image_2_slices(self, image):
        """Turn raw-Image-data data into slices."""
        pass
