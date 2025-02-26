"""Import image data and convert it into slices."""

import logging
from pathlib import Path

import numpy as np
from i2pp.core.image_reader_classes.image_reader import (
    ImageReader,
    PixelValueType,
    SlicesData,
)
from PIL import Image


class PngReader(ImageReader):
    """Class for reading and processing PNG image data.

    This class extends `ImageReader` to handle 2D PNG images and convert them
    into structured 3D slice representations. It validates additional metadata,
    loads PNG files from a specified directory, and processes them into
    standardized `SlicesData` objects.
    """

    def _verify_additional_informations(self, additional_info: dict) -> None:
        """Validates the format and dimensions of the additional information
        provided.

        This method checks that each required parameter in the dictionary
        config["additional_info"] is present and has the correct shape or data
        type. It performs validation for:
        - Pixel_Spacing: Must be a 2x1 array.
        - Slice_Thickness: Must be an integer or float.
        - Image_Position: Must be a 3x1 array.
        - Image_Orientation: Must be a 6x1 array or None.

        Arguments:
            additional_info (dict): Dictionary containing additional
                information with keys: 'Pixel_Spacing', 'Slice_Thickness',
                'Image_Position' and 'Image_Orientation'.

        Raises:
            RuntimeError: If any of the parameters ('Pixel_Spacing',
                'Slice_Thickness', 'Image_Position' and 'Image_Orientation')
                are missing, of incorrect data type, or have an incorrect
                shape.
        """

        expected_shapes = {
            "Pixel_Spacing": (2,),
            "Image_Position": (3,),
            "Image_Orientation": (6,),
        }

        for key, shape in expected_shapes.items():
            value = additional_info.get(key)

            if key == "Image_Orientation" and value is None:
                continue

            if value is None:
                raise RuntimeError(
                    f"Missing parameter '{key}' in additional information."
                )

            if not isinstance(value, (list, tuple, np.ndarray)):
                raise RuntimeError(
                    f"Parameter '{key}' has the wrong type. Expected: list, "
                    "tuple, or np.ndarray."
                )

            array_value = np.array(value)
            if array_value.shape != shape:
                raise RuntimeError(
                    f"Parameter '{key}' has the wrong shape. Expected: "
                    "{shape}, but got: {array_value.shape}."
                )

        thickness = additional_info.get("Slice_Thickness")
        if not isinstance(thickness, (int, float)):
            raise RuntimeError(
                "Parameter 'Slice_Thickness' must be a number (int or float)."
            )

    def load_image(self, directory: Path) -> list[np.ndarray]:
        """Loads and processes PNG image data from a specified directory.

        This function reads all PNG files in the given directory, verifies
        the format of the additional information in the configuration, and
        converts the images to RGB format. The 2-dimensional PNG images
        together represent a 3D image.

        Arguments:
            directory (Path): The directory containing the PNG image files.

        Returns:
            list[np.ndarray]: A list of RGB images as NumPy arrays, each
                representing a loaded image.

        Raises:
            RuntimeError: If the additional information does not meet the
                expected format or dimension.
        """

        logging.info("Load image data!")

        self._verify_additional_informations(
            self.config["Additional Information"]
        )

        raw_png = []

        for fname in directory.glob("*.png"):
            image_png = Image.open(fname)
            rgb_image = image_png.convert("RGB")
            raw_png.append(rgb_image)

        return raw_png

    def image_to_slices(self, raw_pngs: list[np.ndarray]) -> list[SlicesData]:
        """Converts a list of 2D PNG images into structured slice data.

        This function processes the provided PNG images, converting each one
        into a slice based on the slice thickness and the start position
        defined in the additional information. It filters out slices that
        fall outside the defined Z-axis limits of the 3D model. Relevant
        metadata, including pixel spacing, image position, and orientation,
        is extracted from the configuration, and each slice is stored as a
        `SlicesData` object.

        Arguments:
            raw_png (list[np.ndarray]): A list of 2D PNG images, each
                representing a slice in a 3D volume.

        Returns:
            list[SlicesData]: A list of `SlicesData` objects, each containing
                processed slice data with associated metadata.
        """

        slices = []
        additional_info: dict = self.config["Additional Information"]

        spacing = np.array(additional_info["Pixel_Spacing"])
        slice_thickness = float(additional_info["Slice_Thickness"])
        start_pos = np.array(additional_info["Image_Position"])
        orientation = np.array(
            additional_info.get("Image_Orientation") or [0, -1, 0, 1, 0, 0]
        )

        for i, png in enumerate(raw_pngs):

            pxl_data = np.array(png)

            pos = start_pos + np.array([0, 0, i * slice_thickness])

            if self.limits.min[2] <= pos[2] <= self.limits.max[2]:

                slices.append(
                    SlicesData(
                        PixelData=pxl_data,
                        PixelSpacing=spacing,
                        ImagePositionPatient=pos,
                        ImageOrientationPatient=orientation,
                        PixelType=PixelValueType.RGB,
                    )
                )

        return slices
