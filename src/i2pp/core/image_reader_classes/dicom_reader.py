"""Import image data and convert it into slices."""

import logging
from pathlib import Path

import numpy as np
import pydicom
from i2pp.core.image_reader_classes.image_reader import (
    ImageMetaData,
    ImageReader,
    PixelValueType,
    Slice,
    SlicesAndMetadata,
)
from pydicom.dataset import FileDataset
from pydicom.pixels import pixel_array


class DicomReader(ImageReader):
    """Handles reading and processing of DICOM image data.

    This class loads DICOM image slices from a directory, filters and
    sorts them based on their spatial attributes, and converts them into
    structured slice data. It facilitates the reconstruction of a 3D
    image from 2D DICOM slices while preserving essential metadata such
    as pixel spacing, position, and orientation.
    """

    def load_image(self, directory: Path) -> list[FileDataset]:
        """Loads and sorts DICOM image data from the specified directory.

        This function searches for all `.dcm` files in the given directory,
        reads them using `pydicom`, and filters out any files that do not
        contain the `ImagePositionPatient` attribute. The remaining DICOM
        files, which represent 2D slices of a 3D image, are then sorted
        based on their Z-axis position to reconstruct the correct order of
        the 3D volume.

        Arguments:
            directory (Path): Path to the directory containing DICOM files.

        Returns:
            list[FileDataset]: A sorted list of DICOM datasets with valid
                image data, representing a 3D image when combined.
        """

        logging.info("Load image data!")

        files = []
        for fname in directory.glob("*.dcm"):
            files.append(pydicom.dcmread(fname))

        raw_dicom = []

        skipcount = 0
        for f in files:
            if hasattr(f, "ImagePositionPatient"):
                raw_dicom.append(f)
            else:
                skipcount += 1

        raw_dicoms = sorted(raw_dicom, key=lambda s: s.ImagePositionPatient[2])

        return raw_dicoms

    def image_to_slices(
        self, raw_dicoms: list[FileDataset]
    ) -> SlicesAndMetadata:
        """Converts DICOM datasets into structured slice data.

        This method extracts pixel data and metadata from DICOM slices while
        ensuring that only slices within the specified Z-axis limits are
        included. Metadata such as pixel spacing, orientation, and pixel type
        is stored alongside the extracted slice data.

        Args:
            raw_dicoms (list[FileDataset]): A list of DICOM datasets
                representing 2D slices of a 3D image.

        Returns:
            SlicesAndMetadata: Structured pixel data and metadata for valid
                slices.

        Raises:
            RuntimeError: If the modality of a DICOM file is not supported.
        """

        slices = []

        try:
            pxl_type = PixelValueType(raw_dicoms[0].Modality)

        except ValueError:
            raise RuntimeError("Modality not supported")

        metadata = ImageMetaData(
            pixel_spacing=np.array(raw_dicoms[0].PixelSpacing),
            orientation=np.array(raw_dicoms[0].ImageOrientationPatient),
            pixel_type=pxl_type,
        )

        for dicom in raw_dicoms:
            if (
                self.limits.min[2]
                <= dicom.ImagePositionPatient[2]
                <= self.limits.max[2]
            ):

                pxl_data = np.array(pixel_array(dicom))
                pos = np.array(dicom.ImagePositionPatient)

                slices.append(Slice(pixel_data=pxl_data, position=pos))
            else:
                continue

        return SlicesAndMetadata(slices, metadata)
