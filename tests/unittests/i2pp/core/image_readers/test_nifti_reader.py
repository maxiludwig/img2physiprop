"""Test NIfTI Reader Routine."""

from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
from i2pp.core.discretization_readers.discretization_reader import BoundingBox
from i2pp.core.image_readers.image_format import ImageFormat
from i2pp.core.image_readers.image_reader import PixelValueType
from i2pp.core.image_readers.nifti_reader import RAS_TO_LPS, NiftiReader
from i2pp.core.import_image import verify_and_load_imagedata

# RAS affine with flipped x and y axes, as written by most DICOM converters
AFFINE = np.array(
    [
        [-2.0, 0.0, 0.0, 10.0],
        [0.0, -3.0, 0.0, 20.0],
        [0.0, 0.0, 4.0, 30.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
)
LARGE_BOX = BoundingBox(min=np.full(3, -1000.0), max=np.full(3, 1000.0))


def _write_nifti(path: Path, data: np.ndarray, **header) -> Path:
    """Write a NIfTI image with the test affine."""
    image = nib.Nifti1Image(data, AFFINE)
    if "slope" in header:
        image.header.set_slope_inter(header["slope"], header["inter"])
    nib.save(image, path)
    return path


def _read(path: Path, options=None, bounding_box=LARGE_BOX):
    """Read a NIfTI image with the NiftiReader."""
    reader = NiftiReader(options or {}, bounding_box)
    return reader.convert_to_image_data(reader.load_image(path))


def _voxel_world_coords(image_data, k, j, i) -> np.ndarray:
    """World coordinates of an ImageData voxel with pixel index [k, j, i]."""
    grid = np.array(
        [
            image_data.grid_coords.slice[k],
            image_data.grid_coords.row[j],
            image_data.grid_coords.col[i],
        ]
    )
    return image_data.position + image_data.orientation @ grid


def test_convert_to_image_data_maps_voxels_to_lps(tmp_path: Path):
    """Test that voxel values and LPS world coordinates are consistent with the
    NIfTI affine."""
    data = np.arange(4 * 5 * 6, dtype=np.int16).reshape(4, 5, 6)
    image_data = _read(_write_nifti(tmp_path / "ct.nii.gz", data))

    assert image_data.pixel_data.shape == (6, 5, 4)
    assert image_data.pixel_type == PixelValueType.CT
    np.testing.assert_allclose(image_data.position, [-10.0, -20.0, 30.0])

    lps_affine = RAS_TO_LPS @ AFFINE
    for i, j, k in [(0, 0, 0), (3, 1, 2), (2, 4, 5)]:
        assert image_data.pixel_data[k, j, i] == data[i, j, k]
        np.testing.assert_allclose(
            _voxel_world_coords(image_data, k, j, i),
            (lps_affine @ [i, j, k, 1])[:3],
        )


def test_convert_to_image_data_applies_scaling(tmp_path: Path):
    """Test that the scaling factors of the header are applied."""
    data = np.full((2, 2, 2), 10, dtype=np.int16)
    path = _write_nifti(tmp_path / "ct.nii", data, slope=2.0, inter=-1024.0)

    np.testing.assert_allclose(_read(path).pixel_data, -1004.0)


def test_convert_to_image_data_crops_to_bounding_box(tmp_path: Path):
    """Test that only the voxels required for interpolation within the bounding
    box are loaded."""
    data = np.arange(4 * 5 * 6, dtype=np.int16).reshape(4, 5, 6)
    path = _write_nifti(tmp_path / "ct.nii.gz", data)

    # small LPS box around the voxel (i, j, k) = (2, 3, 1): the voxel and
    # its direct neighbors are needed to interpolate within the box
    center = (RAS_TO_LPS @ AFFINE @ [2, 3, 1, 1])[:3]
    box = BoundingBox(min=center - 0.1, max=center + 0.1)
    image_data = _read(path, bounding_box=box)

    assert image_data.pixel_data.shape == (3, 3, 3)
    assert image_data.pixel_data[1, 1, 1] == data[2, 3, 1]
    np.testing.assert_allclose(
        _voxel_world_coords(image_data, 1, 1, 1), center
    )


def test_convert_to_image_data_outside_bounding_box(tmp_path: Path):
    """Test that an image outside of the bounding box raises an error."""
    path = _write_nifti(tmp_path / "ct.nii.gz", np.zeros((2, 2, 2)))
    box = BoundingBox(min=np.full(3, 500.0), max=np.full(3, 600.0))

    with pytest.raises(RuntimeError, match="does not overlap"):
        _read(path, bounding_box=box)


def test_convert_to_image_data_pixel_type_option(tmp_path: Path):
    """Test the pixel type option."""
    path = _write_nifti(tmp_path / "mr.nii.gz", np.zeros((2, 2, 2)))

    assert _read(path, {"pixel_type": "MR"}).pixel_type == PixelValueType.MRT
    for invalid in ("RGB", "PET"):
        with pytest.raises(RuntimeError, match="Unsupported pixel type"):
            _read(path, {"pixel_type": invalid})


def test_load_image_rejects_4d_images(tmp_path: Path):
    """Test that 4D images are rejected."""
    path = tmp_path / "4d.nii.gz"
    nib.save(nib.Nifti1Image(np.zeros((2, 2, 2, 2)), AFFINE), path)

    with pytest.raises(RuntimeError, match="Only 3D NIfTI images"):
        NiftiReader({}, LARGE_BOX).load_image(path)


def test_convert_to_image_data_rejects_sheared_affine(tmp_path: Path):
    """Test that sheared affines are rejected."""
    path = tmp_path / "sheared.nii.gz"
    affine = AFFINE.copy()
    affine[0, 1] = 1.0
    nib.save(nib.Nifti1Image(np.zeros((2, 2, 2)), affine), path)

    with pytest.raises(RuntimeError, match="sheared"):
        _read(path)


def test_image_format_nifti(tmp_path: Path):
    """Test NIfTI detection by suffix and the reader of the format."""
    for name in ("ct.nii", "ct.nii.gz"):
        path = tmp_path / name
        path.write_bytes(b"")
        assert ImageFormat.NIFTI.is_file_of_format(path)
    assert not ImageFormat.NIFTI.is_file_of_format(tmp_path / "missing.nii")
    assert ImageFormat.NIFTI.get_reader() is NiftiReader


def test_verify_and_load_imagedata_nifti_file(tmp_path: Path):
    """Test loading a NIfTI file sets the pixel range of the pixel type."""
    path = _write_nifti(tmp_path / "ct.nii.gz", np.zeros((2, 2, 2)))

    image_data = verify_and_load_imagedata(path, {}, LARGE_BOX)

    assert image_data.pixel_type == PixelValueType.CT
    np.testing.assert_array_equal(image_data.pixel_range, [-1024, 3071])


def test_verify_and_load_imagedata_folder_with_nifti(tmp_path: Path):
    """Test that NIfTI files must be passed as file, not as folder."""
    _write_nifti(tmp_path / "ct.nii.gz", np.zeros((2, 2, 2)))

    with pytest.raises(RuntimeError, match="pass the path of the NIfTI"):
        verify_and_load_imagedata(tmp_path, {}, LARGE_BOX)
