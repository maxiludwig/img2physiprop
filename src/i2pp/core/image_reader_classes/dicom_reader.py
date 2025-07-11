"""Import dicom data and convert it into 3D data."""

import logging
from pathlib import Path

import numpy as np
import pydicom
from i2pp.core.image_reader_classes.image_reader import (
    GridCoords,
    ImageData,
    ImageReader,
    PixelValueType,
)
from pydicom.dataset import FileDataset
from tqdm import tqdm


class DicomReader(ImageReader):
    """A class for reading, processing, and organizing DICOM image data.

    This class is designed to load DICOM image slices from a specified
    folder, filter and sort them based on spatial attributes, and
    convert them into structured data for 3D image reconstruction. It
    preserves critical metadata such as orientation, position, and
    pixeltype.
    """

    def _sort_dicoms(
        self, dicoms: list[FileDataset], slice_direction: np.ndarray
    ) -> list[FileDataset]:
        """Sorts a list of DICOM datasets in ascending order along a specified
        direction vector. Sorting is based on the projection of each slice's
        position onto the direction vector.

        Arguments:
            dicoms (List[FileDataset]): A list of DICOM datasets, each
                containing the `ImagePositionPatient` attribute representing
                the slice's position in space.
            slice_direction (np.ndarray): A 3D vector defining the direction
                along which the slices should be sorted.

        Returns:
            List[FileDataset]: The sorted list of DICOM datasets.
        """

        projections = [
            np.dot(np.array(ds.ImagePositionPatient), slice_direction)
            for ds in dicoms
        ]

        sorted_indices = np.argsort(projections)

        sorted_dicoms = [dicoms[i] for i in sorted_indices]

        return sorted_dicoms

    def load_image(self, folder_path: Path) -> list[FileDataset]:
        """Loads and sorts DICOM image data from the specified folder.

        This function searches for all `.dcm` files in the given folder,
        reads them using `pydicom`, and filters out any files that do not
        contain the `ImagePositionPatient` attribute.

        Arguments:
            folder_path (Path): Path to the folder containing DICOM files.

        Returns:
            list[FileDataset]: A list of DICOM datasets with valid
                image data, representing a 3D image when combined.
        """

        logging.info("Load image data!")

        files = []
        for fname in folder_path.glob("*.dcm"):
            files.append(pydicom.dcmread(fname))

        raw_dicoms = []

        skipcount = 0
        for f in files:
            if hasattr(f, "ImagePositionPatient"):
                raw_dicoms.append(f)
            else:
                skipcount += 1

        return raw_dicoms

    def convert_to_image_data(
        self, raw_dicoms: list[FileDataset]
    ) -> ImageData:
        """Processes and structures DICOM datasets into a 3D image
        representation.

        This function extracts pixel data and metadata from a list of DICOM
        slices, sorts them along the slice direction, and filters slices
        within a specified bounding box. It constructs a structured 3D image
        with pixel data, grid coordinates, orientation, position, and pixel
        type.

        Arguments:
            raw_dicoms (list[FileDataset]): A list of DICOM datasets
                representing 2D slices of a 3D image.

        Returns:
            ImageData: A structured representation of the 3D image, including
                pixel data, grid coordinates, orientation, and metadata.


        Raises:
            RuntimeError: If the DICOM modality is not supported.
            RuntimeError: If Dicom is not in bounding box.

        Notes:
            - The function assumes all slices share the same orientation,
                pixel spacing, and slice thickness.
        """

        try:
            pixel_type = PixelValueType(raw_dicoms[0].Modality)

        except ValueError:
            raise RuntimeError("Modality not supported")

        row_direction = np.array(raw_dicoms[0].ImageOrientationPatient[3:])
        column_direction = np.array(raw_dicoms[0].ImageOrientationPatient[:3])
        slice_direction = np.cross(column_direction, row_direction)

        slice_orientation = self._get_slice_orientation(
            row_direction, column_direction
        )

        sorted_dicoms = self._sort_dicoms(raw_dicoms, slice_direction)

        raw_pixel_data = []
        coords_in_crop = []

        for dicom in tqdm(sorted_dicoms, desc="Processing Elements"):

            if slice_orientation.is_within_crop(
                dicom.ImagePositionPatient, self.bounding_box
            ):

                raw_pixel_data.append(dicom.pixel_array)
                coords_in_crop.append(dicom.ImagePositionPatient)

            else:
                continue

        if not raw_pixel_data:
            raise RuntimeError(
                "No slice images found within the volume of the imported mesh."
            )

        slope = float(getattr(sorted_dicoms[0], "RescaleSlope", 1.0))
        intercept = float(getattr(sorted_dicoms[0], "RescaleIntercept", 0.0))
        pixel_data = np.array(raw_pixel_data) * slope + intercept

        slice_spacing = float(
            getattr(
                sorted_dicoms[0],
                "SpacingBetweenSlices",
                sorted_dicoms[0].SliceThickness,
            )
        )

        N_slice, N_col, N_row = pixel_data.shape

        slice_coords = np.arange(N_slice) * slice_spacing
        row_coords = np.arange(N_row) * sorted_dicoms[0].PixelSpacing[0]
        col_coords = np.arange(N_col) * sorted_dicoms[0].PixelSpacing[1]

        return ImageData(
            pixel_data=np.array(pixel_data),
            grid_coords=GridCoords(slice_coords, row_coords, col_coords),
            orientation=np.column_stack(
                (slice_direction, row_direction, column_direction)
            ),
            position=coords_in_crop[0],
            pixel_type=pixel_type,
        )
