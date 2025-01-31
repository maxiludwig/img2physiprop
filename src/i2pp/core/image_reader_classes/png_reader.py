"""Import image data and convert it in slices."""

import glob
from typing import Tuple, Union

import numpy as np
from i2pp.core.image_reader_classes.image_reader import (
    ImageReader,
    PixelValueType,
    SlicesData,
)
from PIL import Image


class PngReader(ImageReader):
    """Class to read PNG-data."""

    def _check_array(
        self, array: Union[list, np.ndarray], needed_size: Tuple[int]
    ) -> bool:
        """Private funtion to check if array has the correct size or is none.
        This Function is used to verify the 'Additional Informations'.

        Arguments:
            array {Union[list, np.ndarray]} -- Array to check
            needed_size {Tuple[int]} -- Needed size of the array

        Returns:
            bool -- True if the array has not the correct size or is none"""

        if array is None:
            return False

        elif isinstance(array, (list, np.ndarray)):
            array = np.array(array)
            if array.shape == needed_size:
                return False
            else:
                return True
        else:
            return True

    def verify_additional_informations(self, config) -> None:
        """Checks if all additional informations have the correct format and
        dimension.

        Arguments:
            config {dict} -- Config with the additional informations.

        Raises:
           RuntimeError: If Pixel_Spacing is not an 2x1 Array and not none.
           RuntimeError: If Slice_Thickness is not an int/float and not none.
           RuntimeError: If Image_Position is not an 3x1 Array and not none.
           RuntimeError: If Image_Orientation is not an 6x1 Array and not none.
           RuntimeError: If Modality is not in the list of the
                            accepted Modalities.
        """

        # Check Pixel_Spacing
        spacing = config["Additional Information"]["Pixel_Spacing"]

        if self._check_array(spacing, (2,)):
            raise RuntimeError(
                "Parameter 'Spacing' not readable."
                "Spacing has to be an Array with size 2x1"
            )

        # Check Slice_Thickness
        allowed_types_thickness = (int, float)
        thickness = config["Additional Information"]["Slice_Thickness"]
        if (
            not isinstance(thickness, allowed_types_thickness)
            and thickness is not None
        ):

            raise RuntimeError(
                "Parameter 'Slice_Thickness' not readable."
                "Slice_Thickness has to be a float"
            )

        # Check Image_Position
        start_pos = config["Additional Information"]["Image_Position"]

        if self._check_array(start_pos, (3,)):

            raise RuntimeError(
                "Parameter 'Image_Position' not readable."
                "Image_Position has to be an Array with size 3x1"
            )

        # Check Image_Orientation
        orientation = config["Additional Information"]["Image_Orientation"]

        if self._check_array(orientation, (6,)):
            raise RuntimeError(
                "Parameter 'Image_Orientation' not readable. "
                "Image_Orientation has to be an Array with size 6x1"
            )

    def load_image(self, directory: str) -> np.ndarray:
        """Load raw-image-data for png format.

        Arguments:
            directory {str} -- Path to the png folder.

        Returns:
            np.ndarray -- Raw image data"""

        print("Load image data!")

        raw_png = []
        directory_png = directory + "*.png"
        for fname in glob.glob(directory_png, recursive=False):
            image_png = Image.open(fname)
            rgb_image = image_png.convert("RGB")
            raw_png.append(rgb_image)

        return raw_png

    def image_2_slices(self, raw_png: np.ndarray) -> list[SlicesData]:
        """Import 'Additional Informations'.
        If 'Additional Informations' are not given -> default values.
        Then turn raw-image-data into slices.

        Arguments:
            raw_png {np.ndarray} -- Raw image data

        Returns:
            object -- SlicesData of the image data"""

        slices = []

        for i in range(0, len(raw_png)):

            pxl_data = np.array(raw_png[i])
            img_shape = np.array(pxl_data.shape)

            if self.config["Additional Information"]["Pixel_Spacing"] is None:
                spacing = np.array([1, 1])
            else:
                spacing = np.array(
                    self.config["Additional Information"]["Pixel_Spacing"]
                )

            if (
                self.config["Additional Information"]["Slice_Thickness"]
                is None
            ):
                slice_thickness = float(1)
            else:
                slice_thickness = float(
                    self.config["Additional Information"]["Slice_Thickness"]
                )

            if self.config["Additional Information"]["Image_Position"] is None:
                start_pos = np.array([0, 0, 0])

            else:
                start_pos = np.array(
                    self.config["Additional Information"]["Image_Position"]
                )

            pos = start_pos + np.array([0, 0, i * slice_thickness])

            if (
                self.config["Additional Information"]["Image_Orientation"]
                is None
            ):
                orientation = np.array([0, 1, 0, 1, 0, 0])
            else:
                orientation = np.array(
                    self.config["Additional Information"]["Image_Orientation"]
                )

            slices.append(
                SlicesData(
                    PixelData=pxl_data,
                    image_shape=img_shape,
                    PixelSpacing=spacing,
                    ImagePositionPatient=pos,
                    ImageOrientationPatient=orientation,
                    Modality=PixelValueType.RGB,
                )
            )

        return slices
