"""Import PNG data and convert it into 3D data."""

import logging
import re
from pathlib import Path

import numpy as np
from i2pp.core.image_reader_classes.image_reader import (
    GridCoords,
    ImageData,
    ImageReader,
    PixelValueType,
)
from PIL import Image


class PngReader(ImageReader):
    """Handles reading and processing of PNG image data.

    This class extends `ImageReader` to process 2D PNG images and convert them
    into structured slices. It validates image metadata, loads PNG files
    from a folder, and processes them into `ImageData` objects.
    """

    def _verify_image_metadata(self, image_metadata: dict) -> None:
        """Validates the format and dimensions of the image_metadata provided.

        This method checks that each required parameter in the dictionary
        config["image_metadata"] is present and has the correct shape or data
        type. It performs validation for:
        - pixel_spacing: Must be a 3x1 array.
        - image_position: Must be a 3x1 array.
        - row_orientation: Must be a 3x1 array or None.
        - column_orientation: Must be a 3x1 array or None.
        - slice_orientation: Must be a 3x1 array or None.

        Arguments:
            image_metadata (dict): Dictionary containing image_metadata
                with keys: 'pixel_spacing', 'image_position',
                'row_orientation' and 'column_orientation','slice_orientation'.

        Raises:
            RuntimeError: If any of the parameters are missing, of incorrect
                data type, or have an incorrect shape.
        """

        expected_shapes = {
            "pixel_spacing": (3,),
            "image_position": (3,),
            "row_orientation": (3,),
            "column_orientation": (3,),
            "slice_orientation": (3,),
        }

        for key, shape in expected_shapes.items():
            value = image_metadata.get(key)

            if (
                key == "row_orientation"
                or key == "column_orientation"
                or key == "slice_orientation"
            ) and value is None:
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

    def _extract_number(self, path: Path) -> float:
        """Extracts the first number found in the filename (without extension).

        Arguments:
            path (Path): The file path to PNG-folder.

        Returns:
            float: The first number found in the filename. If no number is
                found, returns float('inf') to ensure non-numeric filenames
                are sorted last.
        """
        match = re.search(r"\d+", path.stem)

        return int(match.group()) if match else float("inf")

    def load_image(self, folder_path: Path) -> list[np.ndarray]:
        """Loads and processes PNG image data from a specified directory.

        This function reads all PNG files in the given folder, verifies
        the format of the image_metadata in the configuration, and
        converts the images to RGB format. The 2-dimensional PNG images
        together represent a 3D image.

        Arguments:
            folder_path (Path): The path to the folder containing the PNG
                image files.

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

        for fname in sorted(
            folder_path.glob("*.png"), key=self._extract_number
        ):
            print(f"Loading image: {fname.name}")
            image_png = Image.open(fname)
            rgb_image = image_png.convert("RGB")
            raw_png.append(np.array(rgb_image))

        return raw_png

    def convert_to_image_data(self, raw_pngs: list[np.ndarray]) -> ImageData:
        """Converts a list of 2D PNG images into a structured 3D volume.

        This method processes PNG images as individual slices, importing
        metadata such as pixel spacing, position, and orientation. It filters
        slices based on a specified bounding box, ensuring only relevant
        slices are included.

        Args:
            raw_pngs (list[np.ndarray]): A list of 2D NumPy arrays, each
                representing a slice in a 3D volume.

        Raises:
            RuntimeError: If PNGs are not in bounding box.

        Returns:
            ImageData: A structured representation containing 3D pixel data,
                grid coordinates, orientation, and metadata.
        """

        image_metadata = self.config["image_metadata"]

        row_direction = np.array(
            image_metadata.get("row_direction") or [0, -1, 0]
        )

        column_direction = np.array(
            image_metadata.get("column_direction") or [1, 0, 0]
        )

        slice_direction = np.array(
            image_metadata.get("slice_direction") or [0, 0, 1]
        )

        slice_orientation = self._get_slice_orientation(
            row_direction, column_direction
        )

        spacing = np.array(image_metadata["pixel_spacing"])
        start_coords = np.array(image_metadata["image_position"])

        pixel_data_list = []
        coords_in_crop = []

        for i, png in enumerate(raw_pngs):

            coords_slice = start_coords + i * slice_direction * spacing[0]

            if slice_orientation.is_within_crop(
                coords_slice, self.bounding_box
            ):

                pixel_data_list.append(png)
                coords_in_crop.append(np.array(coords_slice))
            else:
                continue

        if not pixel_data_list:
            raise RuntimeError(
                "No slice images found within the volume of the imported mesh."
            )

        pixel_data = np.array(pixel_data_list)

        N_slice, N_row, N_col, _ = pixel_data.shape

        slice_coords = np.arange(N_slice) * spacing[0]
        row_coords = np.arange(N_row) * spacing[1]
        col_coords = np.arange(N_col) * spacing[2]

        return ImageData(
            pixel_data=pixel_data,
            grid_coords=GridCoords(slice_coords, row_coords, col_coords),
            orientation=np.column_stack(
                (slice_direction, row_direction, column_direction)
            ),
            position=coords_in_crop[0],
            pixel_type=PixelValueType.RGB,
        )
