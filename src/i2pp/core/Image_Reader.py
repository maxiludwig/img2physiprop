"""Import image data and convert it in slices."""

import glob
import os

# from abc import abstractmethod
from dataclasses import dataclass

import numpy as np
import pydicom
from PIL import Image
from pydicom.pixels import pixel_array


@dataclass
class SlicesData:
    """Dataclass for slice-data."""

    PixelData: list
    image_shape: list
    PixelSpacing: list
    ImagePositionPatient: list
    ImageOrientationPatient: list
    Modality: list


class ImageReader:
    """Class to read image-data."""

    def __init__(self, format_input, slices):
        """Init ImageReader."""
        self.slices = slices
        self.format_input = format_input

    def verify_input(self, directory):
        """Verifies if data exists, is readable and determines fomat-type.

        Raises:
            RuntimeError: If Path not exists.
            RuntimeError: If Path has no readable data.
            RuntimeError: If Path has two or more formats of readable data.
        """

        if not os.path.exists(directory):
            raise RuntimeError(
                "Imagedata file not found!"
                "img2physiprop can not be executed!"
            )

        # check if imagedata is png-format
        files_check_png = []
        directory_png = directory + "*.png"
        for fname in glob.glob(directory_png, recursive=False):
            files_check_png.append(Image.open(fname))

        # check if imagedata is dicom-format
        files_check_dicom = []
        directory_dicom = directory + "*.dcm"

        for fname in glob.glob(directory_dicom, recursive=False):
            files_check_dicom.append(pydicom.dcmread(fname))

        if files_check_png == [] and files_check_dicom != []:
            self.format_input = "dicom"

        elif files_check_png != [] and files_check_dicom == []:
            self.format_input = "png"

        elif files_check_png == [] and files_check_dicom == []:
            raise RuntimeError(
                "Input data file is empty or has no readible data!"
                "Please make sure the input file has the correct format"
                "(dicom/png). Img2physiprop can not be executed!"
            )

        elif files_check_png != [] and files_check_dicom != []:
            raise RuntimeError(
                "Input data file has two different format types!"
                "Img2physiprop can not be executed!"
            )

        return self.format_input

    """
    @abstractmethod
    def load_image(self, directory):

        raise NotImplementedError()

    @abstractmethod
    def image_2_slices(self, image):

        raise NotImplementedError()
"""


class DicomReader(ImageReader):
    """Class to read dicom-data."""

    def load_image(self, directory):
        """Load image-data for dicom format."""

        print("Load image data!")
        directory_Dicom_new = directory + "*.dcm"
        files = []
        for fname in glob.glob(directory_Dicom_new, recursive=False):
            files.append(pydicom.dcmread(fname))

        # skip files with no SliceLocation (eg scout views)
        dicom = []

        skipcount = 0
        for f in files:
            if hasattr(f, "ImagePositionPatient"):
                dicom.append(f)
            else:
                skipcount = skipcount + 1

        dicom = sorted(dicom, key=lambda s: s.ImagePositionPatient[2])

        return dicom

    def image_2_slices(self, dicom):
        """Turns input-data into usable data (for dicom)"""

        for i in range(0, len(dicom)):
            pxl_data = pixel_array(dicom[i])
            img_shape = list(pxl_data.shape)
            spacing = dicom[i].PixelSpacing
            pos = dicom[i].ImagePositionPatient
            orientation = dicom[i].ImageOrientationPatient
            mod = dicom[i].Modality

            self.slices.append(
                SlicesData(
                    PixelData=pxl_data,
                    image_shape=img_shape,
                    PixelSpacing=spacing,
                    ImagePositionPatient=pos,
                    ImageOrientationPatient=orientation,
                    Modality=mod,
                )
            )

        return self.slices


class PngReader(ImageReader):
    """Class to read PNG-data."""

    def verify_additional_informations(self, config):
        """Checks if all additional information have the correct format an
        dimension.

        Raises:
           RuntimeError: If Pixel_Spacing is not an 2x1 Array.
           RuntimeError: If Slice_Thickness is not an int/float.
           RuntimeError: If Image_Position is not an 3x1 Array.
           RuntimeError: If Image_Orientation is not an 6x1 Array.
           RuntimeError: If Modality is not in the list of the
                            accepted Modalities.
        """

        # Check Pixel_Spacing
        spacing = np.array(config["Additional Information"]["Pixel_Spacing"])
        if not isinstance(spacing, np.ndarray) or not spacing.shape == (2,):
            raise RuntimeError(
                "Parameter 'Spacing' not readable."
                "Spacing has to be an Array with size 2x1"
            )

        # Check Slice_Thickness
        allowed_types_thickness = (int, float)
        thickness = config["Additional Information"]["Slice_Thickness"]
        if not isinstance(thickness, allowed_types_thickness):

            raise RuntimeError(
                "Parameter 'Slice_Thickness' not readable."
                "Slice_Thickness has to be a float"
            )

        # Check Image_Position
        start_pos = np.array(
            config["Additional Information"]["Image_Position"]
        )
        if not isinstance(start_pos, np.ndarray) or not start_pos.shape == (
            3,
        ):
            raise RuntimeError(
                "Parameter 'Image_Position' not readable."
                "Image_Position has to be an Array with size 3x1"
            )

        # Check Image_Orientation
        orientation = np.array(
            config["Additional Information"]["Image_Orientation"]
        )
        if not isinstance(
            orientation, np.ndarray
        ) or not orientation.shape == (6,):
            raise RuntimeError(
                "Parameter 'Image_Orientation' not readable. "
                "Image_Orientation has to be an Array with size 6x1"
            )

        # Check Modality
        mod = config["Additional Information"]["Modality"]
        if mod not in ["CT", "MR"]:
            raise RuntimeError(
                "Parameter 'Modality' not readable. "
                "Modality has to be a string"
            )

    def load_image(self, directory):
        """Load image-data for png format."""

        print("Load image data!")

        png = []
        directory_png = directory + "*.png"
        for fname in glob.glob(directory_png, recursive=False):
            image_png = Image.open(fname)
            rgb_image = image_png.convert("RGB")
            png.append(rgb_image)

        return png

    def image_2_slices(self, png, config):
        """Turns input-data into usable data (for png)"""

        for i in range(0, len(png)):

            pxl_data = np.array(png[i])
            img_shape = np.array(pxl_data.shape)
            spacing = np.array(
                config["Additional Information"]["Pixel_Spacing"]
            )
            slice_thickness = float(
                config["Additional Information"]["Slice_Thickness"]
            )
            start_pos = np.array(
                config["Additional Information"]["Image_Position"]
            )
            pos = start_pos + np.array([0, 0, i * slice_thickness])
            orientation = np.array(
                config["Additional Information"]["Image_Orientation"]
            )
            mod = str(config["Additional Information"]["Modality"])

            self.slices.append(
                SlicesData(
                    PixelData=pxl_data,
                    image_shape=img_shape,
                    PixelSpacing=spacing,
                    ImagePositionPatient=pos,
                    ImageOrientationPatient=orientation,
                    Modality=mod,
                )
            )

        return self.slices


def verify_and_load_imagedata(directory, config):
    """Calls Image Reader."""

    # Check Format of Imagedata
    input_data = ImageReader("", [])
    input_data.verify_input(directory)

    # Import Image-data
    if input_data.format_input == "dicom":

        image_data = DicomReader(input_data.format_input, [])
        dicom = image_data.load_image(directory)
        slices = image_data.image_2_slices(dicom)

        return slices

    elif image_data.format_input == "png":

        image_data = PngReader(input_data.format_input, [])
        image_data.verify_additional_informations
        png = image_data.load_image(directory)
        slices = image_data.image_2_slices(png, config)
    return slices
