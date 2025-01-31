"""Import image data and convert it in slices."""

import glob

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
    """Class to read dicom-data."""

    def load_image(self, directory: str) -> FileDataset:
        """Load raw-image-data for dicom format and sort it by Slice-Position.

        Arguments:
            directory {str} -- Path to the dicom folder.

        Returns:
                object -- Raw image data
        """

        print("Load image data!")
        directory_Dicom_new = directory + "*.dcm"
        files = []
        for fname in glob.glob(directory_Dicom_new, recursive=False):
            files.append(pydicom.dcmread(fname))

        raw_dicom = []

        skipcount = 0
        for f in files:
            if hasattr(f, "ImagePositionPatient"):
                raw_dicom.append(f)
            else:
                skipcount = skipcount + 1

        raw_dicom = sorted(raw_dicom, key=lambda s: s.ImagePositionPatient[2])

        return raw_dicom

    def image_2_slices(self, raw_dicom: FileDataset) -> list[SlicesData]:
        """Turns the raw-image-data in SlicesData

        Arguments:
            raw_dicom {object} -- Raw image data

        Returns:
            object -- SlicesData of the image data"""

        slices = []

        for i in range(0, len(raw_dicom)):
            pxl_data = np.array(pixel_array(raw_dicom[i]))
            img_shape = np.array((pxl_data.shape))
            spacing = np.array(raw_dicom[i].PixelSpacing)
            pos = np.array(raw_dicom[i].ImagePositionPatient)
            orientation = np.array(raw_dicom[i].ImageOrientationPatient)

            if raw_dicom[i].Modality == "CT":
                mod = PixelValueType.CT
            elif raw_dicom[i].Modality == "MR":
                mod = PixelValueType.MRT
            else:
                RuntimeError("Modality not supported")

            slices.append(
                SlicesData(
                    PixelData=pxl_data,
                    image_shape=img_shape,
                    PixelSpacing=spacing,
                    ImagePositionPatient=pos,
                    ImageOrientationPatient=orientation,
                    Modality=mod,
                )
            )

        return slices
