"""Import image data and convert it into slices."""

import logging
from pathlib import Path

import numpy as np
from i2pp.core.image_reader_classes.image_reader import (
    ImageMetaData,
    ImageReader,
    PixelValueType,
    Slice,
    SlicesAndMetadata,
)
from PIL import Image


class PngReader(ImageReader):
    """Handles reading and processing of PNG image data.

    This class extends `ImageReader` to process 2D PNG images and convert them
    into structured slices. It validates image metadata, loads PNG files
    from a directory, and processes them into `SlicesAndMetadata` objects.
    """

    def _verify_image_metadata(self, image_metadata: dict) -> None:
        """Validates the format and dimensions of the image_metadata provided.

        This method checks that each required parameter in the dictionary
        config["image_metadata"] is present and has the correct shape or data
        type. It performs validation for:
        - pixel_spacing: Must be a 2x1 array.
        - slice_thickness: Must be an integer or float.
        - image_position: Must be a 3x1 array.
        - image_orientation: Must be a 6x1 array or None.

        Arguments:
            image_metadata (dict): Dictionary containing image_metadata
                with keys: 'pixel_spacing', 'slice_thickness',
                'image_position' and 'image_orientation'.

        Raises:
            RuntimeError: If any of the parameters are missing, of incorrect
                data type, or have an incorrect shape.
        """

        expected_shapes = {
            "pixel_spacing": (2,),
            "image_position": (3,),
            "image_orientation": (6,),
        }

        for key, shape in expected_shapes.items():
            value = image_metadata.get(key)

            if key == "image_orientation" and value is None:
                continue

            if value is None:
                raise RuntimeError(
                    f"Missing parameter '{key}' in image_metadata."
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

        thickness = image_metadata.get("slice_thickness")
        if not isinstance(thickness, (int, float)):
            raise RuntimeError(
                "Parameter 'slice_thickness' must be a number (int or float)."
            )

    def load_image(self, directory: Path) -> list[np.ndarray]:
        """Loads and processes PNG image data from a specified directory.

        This function reads all PNG files in the given directory, verifies
        the format of the image_metadata in the configuration, and
        converts the images to RGB format. The 2-dimensional PNG images
        together represent a 3D image.

        Arguments:
            directory (Path): The directory containing the PNG image files.

        Returns:
            list[np.ndarray]: A list of RGB images as NumPy arrays, each
                representing a loaded image.

        Raises:
            RuntimeError: If the image_metadata does not meet the
                expected format or dimension.
        """

        logging.info("Load image data!")

        self._verify_image_metadata(self.config["image_metadata"])

        raw_png = []

        for fname in directory.glob("*.png"):
            image_png = Image.open(fname)
            rgb_image = image_png.convert("RGB")
            raw_png.append(rgb_image)

        return raw_png

    def image_to_slices(self, raw_pngs: list[np.ndarray]) -> SlicesAndMetadata:
        """Converts 2D PNG images into structured slice data.

        Processes the provided PNG images and extracts relevant metadata,
        including pixel spacing, image position, and orientation. The images
        are then filtered based on the defined Z-axis limits to ensure valid
        slices are included.

        Arguments:
            raw_png (list[np.ndarray]): A list of 2D PNG images, each
                representing a slice in a 3D volume.

        Returns:
            SlicesAndMetadata: A structured representation of slices with
                metadata.
        """

        image_metadata: dict = self.config["image_metadata"]

        spacing = np.array(image_metadata["pixel_spacing"])
        orientation = np.array(
            image_metadata.get("image_orientation") or [0, -1, 0, 1, 0, 0]
        )

        metadata = ImageMetaData(
            pixel_spacing=spacing,
            orientation=orientation,
            pixel_type=PixelValueType.RGB,
        )

        slice_thickness = float(image_metadata["slice_thickness"])
        start_pos = np.array(image_metadata["image_position"])

        slices = []

        for i, png in enumerate(raw_pngs):

            pxl_data = np.array(png)

            pos = start_pos + np.array([0, 0, i * slice_thickness])

            if self.limits.min[2] <= pos[2] <= self.limits.max[2]:

                slices.append(
                    Slice(
                        pixel_data=pxl_data,
                        position=pos,
                    )
                )

        return SlicesAndMetadata(slices, metadata)
