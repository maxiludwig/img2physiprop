"""Tests for the Interpolator types module."""

import pytest
from i2pp.core.interpolators.interpolator_all_voxel import InterpolatorAllVoxel
from i2pp.core.interpolators.interpolator_center import InterpolatorCenter
from i2pp.core.interpolators.interpolator_nodes import InterpolatorNodes
from i2pp.core.interpolators.interpolator_types import InterpolationType


@pytest.mark.parametrize(
    "enum_value, expected_class",
    [
        (InterpolationType.NODES, InterpolatorNodes),
        (InterpolationType.CENTER, InterpolatorCenter),
        (InterpolationType.ALLVOXELS, InterpolatorAllVoxel),
    ],
)
def test_get_interpolator(enum_value, expected_class):
    """Test that get_interpolator returns the correct interpolator class."""
    assert enum_value.get_interpolator() == expected_class


def test_enum_values():
    """Test that enum values are correctly defined."""

    assert InterpolationType.NODES.value == "nodes"
    assert InterpolationType.CENTER.value == "elementcenter"
    assert InterpolationType.ALLVOXELS.value == "allvoxels"


def test_get_interpolator_unsupported_member():
    """Test get_interpolator raises ValueError for unsupported enum members."""

    class FakeEnum:
        """Fake enum class to simulate an unsupported interpolation type."""

        pass

    with pytest.raises(ValueError, match="Unsupported interpolation method"):
        InterpolationType.get_interpolator(FakeEnum())
