"""Interpolates pixel values within the regions of a label map."""

from pathlib import Path

import numpy as np
from i2pp.core.discretization_readers.discretization_reader import (
    BoundingBox,
    Discretization,
    Element,
)
from i2pp.core.image_readers.image_reader import ImageData
from i2pp.core.image_readers.label_map import load_label_map
from i2pp.core.interpolators.interpolator import Interpolator
from i2pp.core.utilities import get_node_position_of_element


def element_labels(dis: Discretization, label_field: str) -> np.ndarray:
    """Reads the label of each element from its data field.

    Arguments:
        dis (Discretization): The discretization.
        label_field (str): Name of the element field holding the label.

    Returns:
        np.ndarray: The label of each element.

    Raises:
        ValueError: If an element does not have the label field.
    """
    labels = []
    for ele in dis.elements:
        if label_field not in ele.fields:
            available = ", ".join(ele.fields) or "none"
            raise ValueError(
                f"Element {ele.id} has no field '{label_field}' holding its "
                f"label. Available fields: {available}."
            )
        labels.append(int(ele.fields[label_field]))
    return np.array(labels, dtype=np.int64)


def enlarge_bounding_box_to_labels(
    dis: Discretization, label_map_path: Path, label_field: str
) -> None:
    """Enlarges the bounding box of the discretization to cover all labelled
    regions of its elements, so that the loaded image data covers them.

    Arguments:
        dis (Discretization): The discretization with a bounding box.
        label_map_path (Path): Path to the NIfTI label map.
        label_field (str): Name of the element field holding the label.
    """
    labels = element_labels(dis, label_field)
    box = load_label_map(label_map_path).bounding_box(labels[labels != 0])
    if box is None or dis.bounding_box is None:
        return

    dis.bounding_box = BoundingBox(
        min=np.minimum(dis.bounding_box.min, box.min),
        max=np.maximum(dis.bounding_box.max, box.max),
    )


class InterpolatorLabelMap(Interpolator):
    """Interpolator assigning each element the mean of all image voxels within
    its region of a label map.

    The label of each element is read from an element data field (e.g. the
    VTU cell data `tu_label` of the terminal units of a lung tree). Each
    image voxel gets the label of the nearest voxel of the label map, so the
    label map may have a different resolution than the image. This
    functionality is used when the `interpolation_method` is set to
    "labelmap".
    """

    def __init__(
        self,
        label_map_path: Path,
        label_field: str,
        filter_outliers: bool = False,
    ):
        """Initialize the InterpolatorLabelMap.

        Arguments:
            label_map_path (Path): Path to the NIfTI label map.
            label_field (str): Name of the element field holding the label.
            filter_outliers (bool): If True, outliers are removed using the
                modified Z-score before averaging.
        """
        super().__init__()
        self._label_map_path = label_map_path
        self._label_field = label_field
        self._filter_outliers_enabled = filter_outliers

    def _collect_voxels(
        self, image_data: ImageData, labels: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Collects the labels and values of all image voxels carrying one of
        the given labels.

        The image is processed slice by slice to limit the memory usage.

        Arguments:
            image_data (ImageData): The image data.
            labels (np.ndarray): The labels of interest.

        Returns:
            tuple[np.ndarray, np.ndarray]: The labels (N_voxels,) and values
                (N_voxels, N_channels) of the voxels.
        """
        label_map = load_label_map(self._label_map_path)
        grid = image_data.grid_coords
        position = np.asarray(image_data.position, dtype=float)

        rows, cols = np.meshgrid(grid.row, grid.col, indexing="ij")
        slice_points = np.column_stack(
            (np.zeros(rows.size), rows.ravel(), cols.ravel())
        )

        voxel_labels = []
        voxel_values = []
        for k, slice_coord in enumerate(grid.slice):
            slice_points[:, 0] = slice_coord
            world = position + slice_points @ image_data.orientation.T
            slice_labels = label_map.labels_at(world)

            keep = np.isin(slice_labels, labels)
            voxel_labels.append(slice_labels[keep])
            voxel_values.append(
                image_data.pixel_data[k].reshape(rows.size, -1)[keep]
            )

        return np.concatenate(voxel_labels), np.concatenate(voxel_values)

    def _mean(self, values: np.ndarray) -> np.ndarray:
        """Computes the mean of voxel values with optional outlier
        filtering."""
        if self._filter_outliers_enabled and len(values) > 5:
            keep = self._filter_outliers_modified_zscore(values)
            if np.any(keep):
                values = values[keep]
        return np.mean(values, axis=0)

    def _value_at_element_center(
        self, ele: Element, dis: Discretization, image_data: ImageData
    ) -> np.ndarray:
        """Interpolates the image at the center of an element."""
        center = dis.nodes.coords[
            get_node_position_of_element(ele.node_ids, dis.nodes.ids)
        ].mean(axis=0)
        center_grid = self.world_to_grid_coords(
            center[np.newaxis],
            image_data.orientation,
            np.asarray(image_data.position, dtype=float),
        )
        return np.atleast_1d(
            self.interpolate_image_values_to_points(center_grid, image_data)[0]
        )

    def compute_element_data(
        self, dis: Discretization, image_data: ImageData
    ) -> list[Element]:
        """Computes the mean pixel value within the labelled region of each
        element.

        Label 0 is background and marks elements without a region. If no
        image voxel carries the label of an element, the image is
        interpolated at the element center instead.

        Arguments:
            dis (Discretization): The discretization whose elements have the
                label field.
            image_data (ImageData): A structured representation containing 3D
                pixel data, grid coordinates, orientation, and metadata.

        Returns:
            list[Element]: A list of elements with their pixel values
                assigned.
        """
        labels = element_labels(dis, self._label_field)
        voxel_labels, voxel_values = self._collect_voxels(
            image_data, labels[labels != 0]
        )

        order = np.argsort(voxel_labels, kind="stable")
        voxel_labels = voxel_labels[order]
        voxel_values = voxel_values[order]
        starts = np.searchsorted(voxel_labels, labels, side="left")
        ends = np.searchsorted(voxel_labels, labels, side="right")

        for ele, start, end in zip(dis.elements, starts, ends):
            if start == end:
                self.backup_interpolation += 1
                ele.data = self._value_at_element_center(ele, dis, image_data)
            else:
                ele.data = self._mean(voxel_values[start:end])

            if np.all(np.isnan(ele.data)):
                self.nan_elements += 1

        self._log_interpolation_warnings()

        return dis.elements
