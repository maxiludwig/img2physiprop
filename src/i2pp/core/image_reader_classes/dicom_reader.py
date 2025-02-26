"""Import image data and convert it into slices."""

import logging
from pathlib import Path

import numpy as np
import pydicom
from i2pp.core.image_reader_classes.image_reader import (
    ImageReader,
    PixelValueType,
    SlicesData,
)
from pydicom.dataset import FileDataset
from pydicom.pixels import pixel_array


class DicomReader(ImageReader):
    """A class for reading and processing DICOM image data.

    This class provides functionality to load DICOM image slices from a
    specified directory, filter and sort them based on their spatial
    attributes, and convert them into structured slice data. It enables
    the reconstruction of a 3D image from 2D DICOM slices while
    preserving important metadata such as pixel spacing, position, and
    orientation.
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
    ) -> list[SlicesData]:
        """Converts a list of DICOM files into structured slice data.

        This function processes DICOM image slices, filtering out those that
        are outside the defined Z-axis limits of the 3D model. It extracts
        relevant metadata, including pixel spacing, position, and orientation,
        and assigns a pixel value type based on the modality (CT or MR).
        The processed slices are stored as `SlicesData` objects.

        Arguments:
            raw_dicoms (list[FileDataset]): A list of DICOM datasets
                representing 2D slices of a 3D image.

        Returns:
            list[SlicesData]: A list of `SlicesData` objects containing
                structured pixel data and metadata for each valid slice.

        Raises:
            RuntimeError: If the modality of a DICOM file is not supported.
        """

        slices = []

        for dicom in raw_dicoms:
            if (
                self.limits.min[2]
                <= dicom.ImagePositionPatient[2]
                <= self.limits.max[2]
            ):

                pxl_data = np.array(pixel_array(dicom))
                spacing = np.array(dicom.PixelSpacing)
                pos = np.array(dicom.ImagePositionPatient)
                orientation = np.array(dicom.ImageOrientationPatient)

                try:
                    pxl_type = PixelValueType(dicom.Modality)

                except ValueError:
                    raise RuntimeError("Modality not supported")

                slices.append(
                    SlicesData(
                        PixelData=pxl_data,
                        PixelSpacing=spacing,
                        ImagePositionPatient=pos,
                        ImageOrientationPatient=orientation,
                        PixelType=pxl_type,
                    )
                )
            else:
                continue

        return slices
