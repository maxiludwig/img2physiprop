"""Tests for the Interpolator types module."""

from pathlib import Path

import pytest
from i2pp.core.interpolators.interpolator_all_voxel import InterpolatorAllVoxel
from i2pp.core.interpolators.interpolator_center import InterpolatorCenter
from i2pp.core.interpolators.interpolator_label_map import (
    InterpolatorLabelMap,
)
from i2pp.core.interpolators.interpolator_nodes import InterpolatorNodes
from i2pp.core.interpolators.interpolator_types import InterpolationType


@pytest.mark.parametrize(
    "enum_value, expected_class",
    [
        (InterpolationType.NODES, InterpolatorNodes),
        (InterpolationType.NODES_SCALED, InterpolatorNodes),
        (InterpolationType.CENTER, InterpolatorCenter),
        (InterpolationType.ALLVOXELS, InterpolatorAllVoxel),
        (InterpolationType.ALLVOXELS_SCALED, InterpolatorAllVoxel),
    ],
)
def test_create_interpolator_returns_expected_class(
    enum_value, expected_class
):
    """Test create_interpolator returns instance of the expected class."""
    inst = enum_value.create_interpolator()
    assert isinstance(inst, expected_class)


def test_enum_values():
    """Enum values are correctly defined."""
    assert InterpolationType.NODES.value == "nodes"
    assert InterpolationType.NODES_SCALED.value == "nodes_scaled"
    assert InterpolationType.CENTER.value == "elementcenter"
    assert InterpolationType.ALLVOXELS.value == "allvoxels"
    assert InterpolationType.ALLVOXELS_SCALED.value == "allvoxels_scaled"
    assert InterpolationType.LABELMAP.value == "labelmap"


def test_create_interpolator_labelmap():
    """The labelmap interpolator requires a label map and a label field."""
    inst = InterpolationType.LABELMAP.create_interpolator(
        label_map_path=Path("labels.nii.gz"), label_field="tu_label"
    )
    assert isinstance(inst, InterpolatorLabelMap)

    with pytest.raises(ValueError, match="requires a label map path"):
        InterpolationType.LABELMAP.create_interpolator()


@pytest.mark.parametrize(
    "enum_value, expected_mode",
    [
        (InterpolationType.ALLVOXELS, "allvoxels"),
        (InterpolationType.ALLVOXELS_SCALED, "allvoxels_scaled"),
    ],
)
def test_create_interpolator_configures_mode(enum_value, expected_mode):
    """InterpolatorAllVoxel is configured with correct mode."""
    inst = enum_value.create_interpolator()
    assert isinstance(inst, InterpolatorAllVoxel)
    assert getattr(inst, "_mode") == expected_mode


def test_create_interpolator_filter_outliers_flag():
    """filter_outliers flag is propagated to InterpolatorAllVoxel."""
    inst = InterpolationType.ALLVOXELS.create_interpolator(
        filter_outliers=True
    )
    assert isinstance(inst, InterpolatorAllVoxel)
    assert getattr(inst, "_filter_outliers_enabled") is True


def test_create_interpolator_unsupported_member():
    """Test create_interpolator raises ValueError for unsupported enum
    members."""

    class FakeEnum:
        """Fake enum class to simulate an unsupported interpolation type."""

        pass

    with pytest.raises(ValueError, match="Unsupported interpolation method"):
        # Directly call method on enum class to mimic bad usage
        InterpolationType.create_interpolator(FakeEnum())
