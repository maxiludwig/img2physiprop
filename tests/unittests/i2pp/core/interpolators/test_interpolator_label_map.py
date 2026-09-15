"""Test Label Map Interpolator Routine."""

from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
from i2pp.core.discretization_readers.discretization_reader import (
    BoundingBox,
    Discretization,
    Element,
    Nodes,
)
from i2pp.core.image_readers.image_reader import (
    GridCoords,
    ImageData,
    PixelValueType,
)
from i2pp.core.image_readers.label_map import LabelMap, load_label_map
from i2pp.core.image_readers.nifti_reader import RAS_TO_LPS
from i2pp.core.interpolators.interpolator_label_map import (
    InterpolatorLabelMap,
    element_labels,
    enlarge_bounding_box_to_labels,
)

N = 10
# CT voxel (s, r, c) is located at LPS world (s, r, c) + 0.25
ORIGIN = np.full(3, 0.25)


def _image(pixel_data: np.ndarray) -> ImageData:
    """Create CT image data on a grid with 1 mm spacing."""
    coords = np.arange(N, dtype=float)
    return ImageData(
        pixel_data=pixel_data,
        grid_coords=GridCoords(coords, coords, coords),
        orientation=np.eye(3),
        position=ORIGIN,
        pixel_type=PixelValueType.CT,
    )


def _world_x() -> np.ndarray:
    """CT image whose value is the world x coordinate of the voxel."""
    s, _, _ = np.meshgrid(*(np.arange(N),) * 3, indexing="ij")
    return ORIGIN[0] + s.astype(float)


def _write_label_map(path: Path, labels: np.ndarray, lps_affine) -> Path:
    """Write a NIfTI label map with the given LPS affine."""
    affine = RAS_TO_LPS @ lps_affine
    nib.save(nib.Nifti1Image(labels.astype(np.int32), affine), path)
    return path


def _same_grid_label_map(path: Path) -> Path:
    """Label map on the CT grid: label 1 for world x < 3, else 2."""
    labels = np.full((N, N, N), 2)
    labels[:3] = 1
    affine = np.eye(4)
    affine[:3, 3] = ORIGIN
    return _write_label_map(path, labels, affine)


def _coarse_label_map(path: Path) -> Path:
    """Label map with 2 mm voxels: label 1 for world x < 3, else 2."""
    labels = np.full((6, 6, 6), 2)
    labels[:2] = 1
    return _write_label_map(path, labels, np.diag([2.0, 2.0, 2.0, 1.0]))


def _dis(*labels) -> Discretization:
    """Line elements with the given labels and nodes at (4.25..6.25)."""
    coords = np.array([[4.25] * 3, [6.25] * 3] * len(labels))
    elements = [
        Element(
            node_ids=np.array([2 * i, 2 * i + 1]),
            id=i,
            fields={"tu_label": label},
        )
        for i, label in enumerate(labels)
    ]
    return Discretization(
        nodes=Nodes(coords=coords, ids=np.arange(len(coords))),
        elements=elements,
        surfaces=[],
    )


@pytest.mark.parametrize(
    "write_label_map", [_same_grid_label_map, _coarse_label_map]
)
def test_mean_within_labelled_regions(tmp_path: Path, write_label_map):
    """Test the mean per label for a label map on the CT grid and on a coarser
    grid."""
    path = write_label_map(tmp_path / "labels.nii.gz")
    interpolator = InterpolatorLabelMap(path, "tu_label")

    elements = interpolator.compute_element_data(
        _dis(1, 2), _image(_world_x())
    )

    np.testing.assert_allclose(elements[0].data, [1.25])  # x = 0.25..2.25
    np.testing.assert_allclose(elements[1].data, [6.25])  # x = 3.25..9.25
    assert interpolator.backup_interpolation == 0


def test_fallback_to_element_center(tmp_path: Path):
    """Test the fallback for labels without voxels in the image."""
    path = _same_grid_label_map(tmp_path / "labels.nii.gz")
    interpolator = InterpolatorLabelMap(path, "tu_label")

    elements = interpolator.compute_element_data(_dis(3), _image(_world_x()))

    np.testing.assert_allclose(elements[0].data, [5.25])
    assert interpolator.backup_interpolation == 1


def test_label_zero_has_no_region(tmp_path: Path):
    """Test that label 0 (background) is not averaged and does not enlarge the
    bounding box."""
    labels = np.zeros((N, N, N))
    labels[:3] = 1
    affine = np.eye(4)
    affine[:3, 3] = ORIGIN
    path = _write_label_map(tmp_path / "labels.nii.gz", labels, affine)
    dis = _dis(0)
    dis.bounding_box = BoundingBox(min=np.full(3, 1.0), max=np.full(3, 2.0))

    enlarge_bounding_box_to_labels(dis, path, "tu_label")
    interpolator = InterpolatorLabelMap(path, "tu_label")
    elements = interpolator.compute_element_data(dis, _image(_world_x()))

    np.testing.assert_allclose(dis.bounding_box.min, [1.0, 1.0, 1.0])
    np.testing.assert_allclose(elements[0].data, [5.25])
    assert interpolator.backup_interpolation == 1


def test_filter_outliers_removes_spike(tmp_path: Path):
    """Test that outlier filtering removes a single extreme voxel."""
    path = _same_grid_label_map(tmp_path / "labels.nii.gz")
    pixel_data = np.full((N, N, N), 5.0)
    pixel_data[5, 5, 5] = 10000.0

    unfiltered = InterpolatorLabelMap(path, "tu_label").compute_element_data(
        _dis(2), _image(pixel_data)
    )
    filtered = InterpolatorLabelMap(
        path, "tu_label", filter_outliers=True
    ).compute_element_data(_dis(2), _image(pixel_data))

    assert np.asarray(unfiltered[0].data)[0] > 5.0
    np.testing.assert_allclose(filtered[0].data, [5.0])


def test_element_labels_missing_field():
    """Test that elements without the label field raise an error."""
    dis = _dis(1)
    dis.elements[0].fields = {"block_id": 2}

    with pytest.raises(ValueError, match="no field 'tu_label'"):
        element_labels(dis, "tu_label")


def test_enlarge_bounding_box_to_labels(tmp_path: Path):
    """Test that the bounding box covers the full extent of the labelled
    voxels."""
    path = _same_grid_label_map(tmp_path / "labels.nii.gz")
    dis = _dis(1)
    dis.bounding_box = BoundingBox(min=np.full(3, 1.0), max=np.full(3, 2.0))

    enlarge_bounding_box_to_labels(dis, path, "tu_label")

    np.testing.assert_allclose(dis.bounding_box.min, [-0.25, -0.25, -0.25])
    np.testing.assert_allclose(dis.bounding_box.max, [2.75, 9.75, 9.75])


def test_label_map_lookup(tmp_path: Path):
    """Test the nearest voxel lookup at LPS world coordinates."""
    label_map = LabelMap(_coarse_label_map(tmp_path / "labels.nii.gz"))

    points = np.array(
        [
            [0.9, 0.0, 0.0],  # nearest voxel (0, 0, 0)
            [3.1, 4.0, 4.0],  # nearest voxel (2, 2, 2)
            [-5.0, 0.0, 0.0],  # outside of the label map
        ]
    )
    np.testing.assert_array_equal(label_map.labels_at(points), [1, 2, 0])
    assert label_map.bounding_box(np.array([99])) is None


def test_label_map_rejects_invalid_images(tmp_path: Path):
    """Test that non-integer and 4D label maps are rejected."""
    float_path = tmp_path / "float.nii.gz"
    nib.save(nib.Nifti1Image(np.zeros((2, 2, 2)), np.eye(4)), float_path)
    with pytest.raises(RuntimeError, match="integer labels"):
        LabelMap(float_path)

    path_4d = tmp_path / "4d.nii.gz"
    nib.save(
        nib.Nifti1Image(np.zeros((2, 2, 2, 2), dtype=np.int32), np.eye(4)),
        path_4d,
    )
    with pytest.raises(RuntimeError, match="Only 3D NIfTI label maps"):
        LabelMap(path_4d)


def test_load_label_map_is_cached(tmp_path: Path):
    """Test that the label map is only loaded once per path."""
    path = _same_grid_label_map(tmp_path / "labels.nii.gz")

    assert load_label_map(path) is load_label_map(path)
