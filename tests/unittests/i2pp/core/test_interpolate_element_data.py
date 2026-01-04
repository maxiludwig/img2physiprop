"""Test setting fixed values for elements at the boundary."""

import numpy as np
from i2pp.core.configuration_validator.validator import Interpolation
from i2pp.core.discretization_readers.discretization_reader import (
    Discretization,
    Element,
    Nodes,
    Surface,
)
from i2pp.core.image_readers.image_reader import (
    GridCoords,
    ImageData,
    PixelValueType,
)
from i2pp.core.interpolate_element_data import (
    interpolate_image_to_discretization,
)


def _make_simple_discretization():
    """Create a minimal discretization with two elements touching a boundary
    surface."""
    # Nodes: 0..3, with coordinates forming a unit square in grid space
    node_coords = np.array(
        [
            [0, 0, 0],  # node 0
            [1, 0, 0],  # node 1
            [0, 1, 0],  # node 2
            [1, 1, 0],  # node 3
        ]
    )
    node_ids = np.arange(4)
    nodes = Nodes(coords=node_coords, ids=node_ids)

    # Elements: E0 uses nodes [0,1]; E1 uses nodes [2,3]
    elements = [
        Element(node_ids=[0, 1], id=0),
        Element(node_ids=[2, 3], id=1),
    ]

    # Surface contains node 1 and 2 -> E0 and E1 touch boundary
    surfaces = [Surface(node_ids=np.array([1, 2], dtype=int), id=0)]

    return Discretization(nodes=nodes, elements=elements, surfaces=surfaces)


def _make_ct_image():
    """Create a small 3D scalar CT image with zero values for interpolation
    tests."""
    # Simple 3x3x3 scalar image with zeros
    pixel_data = np.zeros((3, 3, 3), dtype=float)
    grid_coords = GridCoords(
        slice=np.arange(3), row=np.arange(3), col=np.arange(3)
    )
    return ImageData(
        pixel_data=pixel_data,
        grid_coords=grid_coords,
        orientation=np.eye(3),
        position=np.zeros(3),
        pixel_type=PixelValueType.CT,
    )


def _make_rgb_image():
    """Create a small 3D RGB image with zero values for interpolation tests."""
    # Simple 3x3x3x3 RGB image with zeros
    pixel_data = np.zeros((3, 3, 3, 3), dtype=np.uint8)
    grid_coords = GridCoords(
        slice=np.arange(3), row=np.arange(3), col=np.arange(3)
    )
    return ImageData(
        pixel_data=pixel_data,
        grid_coords=grid_coords,
        orientation=np.eye(3),
        position=np.zeros(3),
        pixel_type=PixelValueType.RGB,
    )


def test_boundary_elements_receive_fixed_scalar_value():
    """Verify boundary-touching elements receive the configured fixed scalar
    value."""
    dis = _make_simple_discretization()
    image = _make_ct_image()

    # Interpolation: nodes, set surface element value to 999.0
    interp_cfg = Interpolation(
        method="nodes",
        filter_outliers=False,
        set_node_value=None,
        set_ele_value=999.0,
        node_weight=None,
    )

    elements = interpolate_image_to_discretization(dis, image, interp_cfg)

    # Both elements touch the boundary -> both should receive fixed value
    assert float(elements[0].data) == 999.0
    assert float(elements[1].data) == 999.0


def test_boundary_elements_receive_fixed_rgb_value_vector():
    """Verify boundary-touching elements receive the configured fixed RGB
    vector."""
    dis = _make_simple_discretization()
    image = _make_rgb_image()

    fixed_rgb = [10, 20, 30]
    interp_cfg = Interpolation(
        method="nodes",
        filter_outliers=False,
        set_node_value=None,
        set_ele_value=fixed_rgb,
        node_weight=None,
    )

    elements = interpolate_image_to_discretization(dis, image, interp_cfg)

    # Both elements touch the boundary -> both should receive fixed RGB vector
    assert np.array_equal(np.asarray(elements[0].data), np.asarray(fixed_rgb))
    assert np.array_equal(np.asarray(elements[1].data), np.asarray(fixed_rgb))


def test_boundary_elements_receive_fixed_value_without_prior_data():
    """Boundary-touching elements get fixed values even if element.data was not
    set before."""
    dis = _make_simple_discretization()
    image = _make_ct_image()

    # Configure interpolation with method that will compute,
    # but we simulate unset data by overwriting after
    interp_cfg = Interpolation(
        method="nodes",
        filter_outliers=False,
        set_node_value=None,
        set_ele_value=123.0,
        node_weight=None,
    )

    # Perform interpolation
    elements = interpolate_image_to_discretization(dis, image, interp_cfg)

    # Verify fixed scalar applied
    assert float(elements[0].data) == 123.0
    assert float(elements[1].data) == 123.0

    # Now test vector assignment on RGB image
    image_rgb = _make_rgb_image()
    interp_cfg_vec = Interpolation(
        method="nodes",
        filter_outliers=False,
        set_node_value=None,
        set_ele_value=[7, 8, 9],
        node_weight=None,
    )
    elements_vec = interpolate_image_to_discretization(
        dis, image_rgb, interp_cfg_vec
    )
    assert np.array_equal(
        np.asarray(elements_vec[0].data), np.array([7, 8, 9])
    )
    assert np.array_equal(
        np.asarray(elements_vec[1].data), np.array([7, 8, 9])
    )
