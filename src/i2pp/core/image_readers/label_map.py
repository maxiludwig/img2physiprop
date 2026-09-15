"""Label maps from NIfTI files."""

import itertools
from functools import lru_cache
from pathlib import Path
from typing import Optional

import nibabel as nib
import numpy as np
from i2pp.core.discretization_readers.discretization_reader import BoundingBox
from i2pp.core.image_readers.nifti_reader import RAS_TO_LPS


class LabelMap:
    """An integer label map read from a NIfTI file (.nii / .nii.gz), e.g. the
    terminal unit regions of a lung tree.

    Label 0 is background. Labels are looked up at LPS world coordinates
    using the nearest voxel, so the label map may have a different
    resolution than the image data, as long as both share the same world
    coordinate system.
    """

    def __init__(self, file_path: Path):
        """Loads the label map.

        Arguments:
            file_path (Path): Path to the NIfTI label map.

        Raises:
            RuntimeError: If the label map is not three-dimensional or does
                not contain integer labels.
        """
        image = nib.load(file_path)
        if len(image.shape) != 3:
            raise RuntimeError(
                "Only 3D NIfTI label maps are supported, but the label map "
                f"has the shape {image.shape}."
            )

        self.data = np.asanyarray(image.dataobj)
        if not np.issubdtype(self.data.dtype, np.integer):
            raise RuntimeError(
                "Label maps must contain integer labels, but the label map "
                f"has the data type {self.data.dtype}."
            )

        self.voxel_to_world = RAS_TO_LPS @ image.affine
        self.world_to_voxel = np.linalg.inv(self.voxel_to_world)

    def labels_at(self, points: np.ndarray) -> np.ndarray:
        """Looks up the labels at the given points.

        Arguments:
            points (np.ndarray): (N, 3) array of LPS world coordinates.

        Returns:
            np.ndarray: (N,) array with the label of the nearest voxel of
                each point, 0 for points outside of the label map.
        """
        points = np.atleast_2d(points)
        indices = np.rint(
            points @ self.world_to_voxel[:3, :3].T + self.world_to_voxel[:3, 3]
        ).astype(int)

        in_map = np.all((indices >= 0) & (indices < self.data.shape), axis=1)

        labels = np.zeros(len(points), dtype=self.data.dtype)
        labels[in_map] = self.data[tuple(indices[in_map].T)]
        return labels

    def bounding_box(self, labels: np.ndarray) -> Optional[BoundingBox]:
        """Computes the world bounding box of all voxels with the given labels.

        Arguments:
            labels (np.ndarray): The labels of interest.

        Returns:
            Optional[BoundingBox]: The bounding box including the full
                extent of the voxels, or None if none of the labels occurs.
        """
        indices = np.argwhere(np.isin(self.data, labels))
        if len(indices) == 0:
            return None

        corners = np.array(
            list(
                itertools.product(
                    *zip(indices.min(axis=0) - 0.5, indices.max(axis=0) + 0.5)
                )
            )
        )
        world = (
            corners @ self.voxel_to_world[:3, :3].T
            + self.voxel_to_world[:3, 3]
        )
        return BoundingBox(min=world.min(axis=0), max=world.max(axis=0))


@lru_cache(maxsize=1)
def load_label_map(file_path: Path) -> LabelMap:
    """Loads a label map, reusing the last loaded one for the same path.

    Arguments:
        file_path (Path): Path to the NIfTI label map.

    Returns:
        LabelMap: The label map.
    """
    return LabelMap(file_path)
