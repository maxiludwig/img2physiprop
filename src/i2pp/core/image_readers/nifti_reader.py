"""Import NIfTI data and convert it into 3D data."""

import itertools
import logging
from pathlib import Path

import nibabel as nib
import numpy as np
from i2pp.core.image_readers.image_reader import (
    GridCoords,
    ImageData,
    ImageReader,
    PixelValueType,
)

# NIfTI stores world coordinates in RAS, whereas DICOM and the
# discretizations use LPS.
RAS_TO_LPS = np.diag([-1.0, -1.0, 1.0, 1.0])


class NiftiReader(ImageReader):
    """A class for reading 3D NIfTI images (.nii / .nii.gz).

    The image is converted from the RAS coordinate system of NIfTI to
    the LPS coordinate system used by DICOM. Only the voxels within the
    bounding box of the discretization are loaded. Since NIfTI files do
    not store the imaging modality, the pixel type is taken from the
    image option `pixel_type` ("CT" or "MR", default "CT").
    """

    def _get_pixel_type(self) -> PixelValueType:
        """Determines the pixel type from the image options.

        Returns:
            PixelValueType: The pixel type of the image.

        Raises:
            RuntimeError: If the pixel type is not supported for NIfTI
                images.
        """
        name = (self.options or {}).get("pixel_type", "CT")
        try:
            pixel_type = PixelValueType(name)
        except ValueError:
            pixel_type = None

        if pixel_type not in (PixelValueType.CT, PixelValueType.MRT):
            raise RuntimeError(
                f"Unsupported pixel type '{name}' for NIfTI images. "
                "Supported pixel types are: CT, MR."
            )
        return pixel_type

    def _crop_indices(
        self,
        linear: np.ndarray,
        origin: np.ndarray,
        shape: tuple[int, ...],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Computes the voxel index ranges covering the bounding box.

        Arguments:
            linear (np.ndarray): (3, 3) matrix mapping voxel indices to LPS
                world coordinates.
            origin (np.ndarray): LPS world coordinates of voxel (0, 0, 0).
            shape (tuple[int, ...]): Shape of the image.

        Returns:
            tuple[np.ndarray, np.ndarray]: Lower (inclusive) and upper
                (exclusive) voxel indices along each image axis.

        Raises:
            RuntimeError: If the image does not overlap the bounding box.
        """
        shape_arr = np.array(shape[:3])
        if self.bounding_box is None:
            return np.zeros(3, dtype=int), shape_arr

        corners = np.array(
            list(
                itertools.product(
                    *zip(self.bounding_box.min, self.bounding_box.max)
                )
            ),
            dtype=float,
        )
        indices = np.linalg.solve(linear, (corners - origin).T).T

        lower = np.clip(np.floor(indices.min(axis=0)).astype(int), 0, None)
        upper = np.minimum(
            np.ceil(indices.max(axis=0)).astype(int) + 1, shape_arr
        )

        if np.any(upper <= lower):
            raise RuntimeError(
                "The NIfTI image does not overlap the volume of the "
                "imported mesh."
            )
        return lower, upper

    def load_image(self, file_path: Path) -> nib.Nifti1Image:
        """Loads a NIfTI image without reading its voxel data yet.

        Arguments:
            file_path (Path): Path to the .nii or .nii.gz file.

        Returns:
            nib.Nifti1Image: The NIfTI image with lazily loaded voxel data.

        Raises:
            RuntimeError: If the image is not three-dimensional.
        """
        logging.info("Load image data!")

        image = nib.load(file_path)
        if len(image.shape) != 3:
            raise RuntimeError(
                "Only 3D NIfTI images are supported, but the image has the "
                f"shape {image.shape}."
            )
        return image

    def convert_to_image_data(self, image: nib.Nifti1Image) -> ImageData:
        """Converts a NIfTI image into an ImageData object.

        The voxel axes (i, j, k) of the NIfTI image are mapped to the
        (column, row, slice) axes of `ImageData`, i.e. the pixel data is
        stored as [k, j, i]. Scaling factors of the NIfTI header are
        applied to the voxel values.

        Arguments:
            image (nib.Nifti1Image): The NIfTI image.

        Returns:
            ImageData: A structured representation of the 3D image, including
                pixel data, grid coordinates, orientation, and metadata.

        Raises:
            RuntimeError: If the affine of the image is sheared.
        """
        pixel_type = self._get_pixel_type()

        affine = RAS_TO_LPS @ image.affine
        linear = affine[:3, :3]
        origin = affine[:3, 3]

        spacing = np.linalg.norm(linear, axis=0)
        directions = linear / spacing
        if not np.allclose(directions.T @ directions, np.eye(3), atol=1e-4):
            raise RuntimeError(
                "NIfTI images with sheared (non-orthogonal) axes are not "
                "supported."
            )

        lower, upper = self._crop_indices(linear, origin, image.shape)
        crop = tuple(slice(lo, up) for lo, up in zip(lower, upper))
        data = np.asarray(image.dataobj[crop], dtype=np.float64)

        n_i, n_j, n_k = data.shape

        return ImageData(
            pixel_data=data.transpose(2, 1, 0),
            grid_coords=GridCoords(
                slice=np.arange(n_k) * spacing[2],
                row=np.arange(n_j) * spacing[1],
                col=np.arange(n_i) * spacing[0],
            ),
            orientation=np.column_stack(
                (directions[:, 2], directions[:, 1], directions[:, 0])
            ),
            position=origin + linear @ lower,
            pixel_type=pixel_type,
        )
