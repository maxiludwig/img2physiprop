"""Test Interpolator Routine."""

import numpy as np
from i2pp.core.discretization_readers.discretization_reader import (
    Discretization,
)
from i2pp.core.discretization_readers.discretization_reader import (
    Element as DisElement,
)
from i2pp.core.discretization_readers.discretization_reader import (
    Nodes,
    Surface,
)
from i2pp.core.image_readers.image_reader import (
    GridCoords,
    ImageData,
    PixelValueType,
)
from i2pp.core.interpolators.interpolator_all_voxel import InterpolatorAllVoxel
from scipy.spatial import ConvexHull


def test__search_bounding_box_all_inside():
    """Test _search_bounding_box if all points are inside the grid."""
    slice_coords = np.arange(20) * 0.5
    row_coords = np.arange(20) * 1
    col_coords = np.arange(20) * 2

    grid_coords = GridCoords(slice_coords, row_coords, col_coords)

    ele_grid_coords = np.array(
        [[3, 12, 22], [0.9, 4, 3], [5, 19, 16], [7, 9, 11]]
    )

    interpolator = InterpolatorAllVoxel()
    i_slice, i_row, i_col = interpolator._search_bounding_box(
        grid_coords, ele_grid_coords
    )

    print(slice_coords[i_slice])
    assert [min(i_slice), max(i_slice)] == [2, 14]
    assert [min(i_row), max(i_row)] == [4, 19]
    assert [min(i_col), max(i_col)] == [2, 11]


def test__search_bounding_box_element_outside():
    """Test _search_bounding_box if part of the element is outside of the
    grid."""
    slice_coords = np.arange(10) * 1
    row_coords = np.arange(10) * 1
    col_coords = np.arange(10) * 1

    grid_coords = GridCoords(slice_coords, row_coords, col_coords)

    ele_grid_coords = np.array(
        [[-1, -1, 0], [1, 4, 3], [5, -5, 16], [7, 8, -7]]
    )

    interpolator = InterpolatorAllVoxel()
    i_slice, i_row, i_col = interpolator._search_bounding_box(
        grid_coords, ele_grid_coords
    )

    assert [min(i_slice), max(i_slice)] == [0, 7]
    assert [min(i_row), max(i_row)] == [0, 8]
    assert [min(i_col), max(i_col)] == [0, 9]


def test__search_bounding_box_element_not_in_gird():
    """Test _search_bounding_box if element is not in grid."""
    slice_coords = np.arange(10) * 1
    row_coords = np.arange(10) * 1
    col_coords = np.arange(10) * 1

    grid_coords = GridCoords(slice_coords, row_coords, col_coords)

    ele_grid_coords = np.array(
        [[-1, -1, -1], [-2, -4, -7], [-10, -5, -16], [-7, -8, -7]]
    )

    interpolator = InterpolatorAllVoxel()
    i_slice, i_row, i_col = interpolator._search_bounding_box(
        grid_coords, ele_grid_coords
    )

    assert len(i_slice) == 0
    assert len(i_row) == 0
    assert len(i_col) == 0


def test__is_inside_element_in_element():
    """Test _is_inside_element_in_element for a tetraeder if the point is
    inside."""

    element = np.array([[0, 0, 0], [2, 0, 0], [0, 2, 0], [0, 0, 2]])

    point = np.array([0.5, 0.5, 0.5])

    hull = ConvexHull(element)
    interpolator = InterpolatorAllVoxel()
    assert interpolator._is_inside_element(point, hull)


def test__is_inside_element_on_element():
    """Test _is_inside_element_in_element for a tetraeder if the point is on
    the surface."""

    element = np.array([[0, 0, 0], [2, 0, 0], [0, 2, 0], [0, 0, 2]])

    point = np.array([0.5, 0.5, 0])

    hull = ConvexHull(element)
    interpolator = InterpolatorAllVoxel()
    assert interpolator._is_inside_element(point, hull)


def test__is_inside_element_outside():
    """Test _is_inside_element_in_element for a tetraeder if the point is not
    in element."""

    element = np.array([[0, 0, 0], [2, 0, 0], [0, 2, 0], [0, 0, 2]])

    point = np.array([0.5, 0.5, -0.01])

    hull = ConvexHull(element)
    interpolator = InterpolatorAllVoxel()
    assert interpolator._is_inside_element(point, hull) is not True


def test_get_data_of_element_element_in_grid_scalar():
    """get_data_of_element if the element is in grid for scalar data."""
    element = np.array(
        [
            [0, 2, 1],
            [0, 3, 1],
            [1, 3, 1],
            [1, 2, 1],
            [0, 2, 2],
            [0, 3, 2],
            [1, 3, 2],
            [1, 2, 2],
        ]
    )

    N_slice, N_row, N_col = 5, 5, 5
    pixel_data = (
        np.arange(N_slice * N_row * N_col)
        .reshape((N_slice, N_row, N_col))
        .astype(float)
    )

    slice_coords = np.arange(5)
    row_coords = np.arange(5)
    col_coords = np.arange(5)

    grid_coords = GridCoords(slice_coords, row_coords, col_coords)
    image_data = ImageData(
        pixel_data, grid_coords, np.eye(3), np.zeros(3), PixelValueType.CT
    )

    interpolator = InterpolatorAllVoxel()

    data_inside = [11, 12, 16, 17, 36, 37, 41, 42]
    assert interpolator._get_data_of_element(element, image_data) == np.mean(
        data_inside
    )


def test_get_data_of_element_element_in_grid_RGB():
    """get_data_of_element if the element is in grid for RGB data."""

    pixel_data = np.random.randint(0, 256, size=(4, 4, 4, 3))
    grid_coords = GridCoords(
        slice=np.array([0, 1, 2, 3]),
        row=np.array([0, 1, 2, 3]),
        col=np.array([0, 1, 2, 3]),
    )

    element_node_grid_coords = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [1, 1, 0],
            [0, 0, 1],
            [1, 0, 1],
            [0, 1, 1],
            [1, 1, 1],
        ]
    )

    interpolator = InterpolatorAllVoxel()
    image_data = ImageData(
        pixel_data, grid_coords, np.eye(3), np.zeros(3), PixelValueType.RGB
    )

    data = interpolator._get_data_of_element(
        element_node_grid_coords, image_data
    )

    assert data is not None
    assert data.shape == (3,)
    assert np.all(data >= 0) and np.all(data <= 255)


def test_get_data_of_element_element_not_in_grid():
    """get_data_of_element if the element is not in grid."""

    element = np.array(
        [
            [0, 2, -1],
            [0, 3, -1],
            [1, 3, -1],
            [1, 2, -1],
            [0, 2, -2],
            [0, 3, -2],
            [1, 3, -2],
            [1, 2, -2],
        ]
    )

    N_slice, N_row, N_col = 5, 5, 5
    pixel_data = (
        np.arange(N_slice * N_row * N_col)
        .reshape((N_slice, N_row, N_col))
        .astype(float)
    )

    slice_coords = np.arange(5)
    row_coords = np.arange(5)
    col_coords = np.arange(5)

    grid_coords = GridCoords(slice_coords, row_coords, col_coords)
    image_data = ImageData(
        pixel_data, grid_coords, np.eye(3), np.zeros(3), PixelValueType.CT
    )

    interpolator = InterpolatorAllVoxel()

    assert np.isnan(interpolator._get_data_of_element(element, image_data))


def test_get_data_of_element_element_low_resolution_image():
    """get_data_of_element if the element is in grid, but no points are inside
    the element."""

    element = np.array(
        [
            [1, 1, 1],
            [2, 1, 1],
            [1, 2, 1],
            [2, 2, 1],
            [1, 1, 2],
            [2, 1, 2],
            [1, 2, 2],
            [2, 2, 2],
        ]
    )

    N_slice, N_row, N_col = 5, 5, 5
    pixel_data = (
        np.arange(N_slice * N_row * N_col)
        .reshape((N_slice, N_row, N_col))
        .astype(np.uint16)
    )

    slice_coords = np.arange(5) * 3

    row_coords = np.arange(5) * 3
    col_coords = np.arange(5) * 3

    grid_coords = GridCoords(slice_coords, row_coords, col_coords)
    image_data = ImageData(
        pixel_data, grid_coords, np.eye(3), np.zeros(3), PixelValueType.CT
    )

    interpolator = InterpolatorAllVoxel()

    interpol_point = np.array([0, 1, 5, 6, 25, 26, 30, 31])

    assert np.equal(
        interpolator._get_data_of_element(element, image_data),
        np.mean(interpol_point),
    )


def _make_simple_discretization(
    node_coords, elements_node_ids, node_weights=None
):
    """Helper to create a minimal Discretization with nodes/elements and
    weights."""
    node_ids = np.arange(len(node_coords))
    nodes = Nodes(ids=node_ids, coords=np.asarray(node_coords))
    if node_weights is None:
        node_weights = np.ones(len(node_coords), dtype=float)
    setattr(nodes, "weights", np.asarray(node_weights))

    # Create Elements with required 'id' field;
    # center_coords/data left as defaults
    elements = [
        DisElement(node_ids=np.asarray(nids), id=idx)
        for idx, nids in enumerate(elements_node_ids)
    ]

    # Single Surface referencing all element ids
    surfaces = [Surface(node_ids=np.array([], dtype=int), id=0)]

    return Discretization(nodes=nodes, elements=elements, surfaces=surfaces)


def test_unweighted_vs_weighted_mode_mean_differs_when_weights_bias():
    """Weighted mode should differ from unweighted mean when node weights bias
    proximity."""
    # Image: small grid 3x3x3 with deterministic values
    pixel_data = np.arange(27, dtype=float).reshape((3, 3, 3))
    grid_coords = GridCoords(
        slice=np.arange(3), row=np.arange(3), col=np.arange(3)
    )
    image_data = ImageData(
        pixel_data, grid_coords, np.eye(3), np.zeros(3), PixelValueType.CT
    )

    # Element: cube corners spanning [0,1] in each axis
    ele_nodes = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [1, 1, 0],
            [0, 0, 1],
            [1, 0, 1],
            [0, 1, 1],
            [1, 1, 1],
        ]
    )

    # Discretization with 8 nodes, one element using all nodes
    dis = _make_simple_discretization(
        node_coords=ele_nodes,
        elements_node_ids=[np.arange(8)],
        node_weights=np.array(
            [10, 10, 10, 10, 1, 1, 1, 1], dtype=float
        ),  # bias towards lower z nodes
    )

    # Unweighted
    interpolator_unweighted = InterpolatorAllVoxel(
        mode="allvoxels", filter_outliers=False
    )
    elems_unweighted = interpolator_unweighted.compute_element_data(
        dis, image_data
    )
    val_unweighted = elems_unweighted[0].data

    # Weighted
    interpolator_weighted = InterpolatorAllVoxel(
        mode="allvoxels_weighted", filter_outliers=False
    )
    elems_weighted = interpolator_weighted.compute_element_data(
        dis, image_data
    )
    val_weighted = elems_weighted[0].data

    assert np.isfinite(val_unweighted).all()
    assert np.isfinite(val_weighted).all()
    # Expect a difference due to weight bias toward voxels near lower-z nodes
    assert not np.allclose(val_unweighted, val_weighted)


def test_filter_outliers_reduces_extreme_values_in_unweighted_mode():
    """Outlier filtering should reduce influence of extreme voxel values."""
    # Construct image with an element that will include voxels;
    # insert an extreme outlier
    N = 5
    pixel_data = np.ones((N, N, N), dtype=float)
    pixel_data[2, 2, 2] = 1e6  # extreme outlier inside the element bbox
    grid_coords = GridCoords(
        slice=np.arange(N), row=np.arange(N), col=np.arange(N)
    )
    image_data = ImageData(
        pixel_data, grid_coords, np.eye(3), np.zeros(3), PixelValueType.CT
    )

    element = np.array(
        [
            [1, 1, 1],
            [3, 1, 1],
            [1, 3, 1],
            [3, 3, 1],
            [1, 1, 3],
            [3, 1, 3],
            [1, 3, 3],
            [3, 3, 3],
        ]
    )

    # Unfiltered
    interp_no_filter = InterpolatorAllVoxel(
        mode="allvoxels", filter_outliers=False
    )
    val_no_filter = interp_no_filter._get_data_of_element(element, image_data)

    # Filtered
    interp_filter = InterpolatorAllVoxel(
        mode="allvoxels", filter_outliers=True
    )
    val_filter = interp_filter._get_data_of_element(element, image_data)

    # Without filtering, mean should be much larger due to the outlier
    assert val_no_filter > 10  # arbitrary threshold above normal mean of ones
    # With filtering, mean should be close to 1
    assert np.isclose(val_filter, 1.0, rtol=1e-3)


def test_filter_outliers_applies_in_weighted_mode():
    """Outlier filtering should apply in weighted mode as well."""
    N = 5
    pixel_data = np.ones((N, N, N), dtype=float)
    pixel_data[2, 2, 2] = 1e6
    grid_coords = GridCoords(
        slice=np.arange(N), row=np.arange(N), col=np.arange(N)
    )
    image_data = ImageData(
        pixel_data, grid_coords, np.eye(3), np.zeros(3), PixelValueType.CT
    )

    element = np.array(
        [
            [1, 1, 1],
            [3, 1, 1],
            [1, 3, 1],
            [3, 3, 1],
            [1, 1, 3],
            [3, 1, 3],
            [1, 3, 3],
            [3, 3, 3],
        ]
    )

    # Node weights biased toward corners near
    # the outlier but filtering should mitigate
    weights = np.array([5, 5, 5, 5, 5, 5, 5, 5], dtype=float)

    interp_w_no_filter = InterpolatorAllVoxel(
        mode="allvoxels_weighted", filter_outliers=False
    )
    val_w_no_filter = interp_w_no_filter._get_data_of_element(
        element, image_data, node_weights_current=weights
    )

    interp_w_filter = InterpolatorAllVoxel(
        mode="allvoxels_weighted", filter_outliers=True
    )
    val_w_filter = interp_w_filter._get_data_of_element(
        element, image_data, node_weights_current=weights
    )

    # Unfiltered should be larger than ~1 due to outlier influence
    assert np.all(val_w_no_filter > 1.0)
    # Filtered result should be close to baseline of ones
    assert np.allclose(val_w_filter, 1.0, rtol=1e-3)
    # And filtering should reduce the value relative to unfiltered
    assert np.all(val_w_filter < val_w_no_filter)


def test_unknown_mode_defaults_to_unweighted_mean():
    """Unknown mode should fall back to unweighted mean."""
    element = np.array(
        [
            [0, 2, 1],
            [0, 3, 1],
            [1, 3, 1],
            [1, 2, 1],
            [0, 2, 2],
            [0, 3, 2],
            [1, 3, 2],
            [1, 2, 2],
        ]
    )
    N_slice, N_row, N_col = 5, 5, 5
    pixel_data = (
        np.arange(N_slice * N_row * N_col)
        .reshape((N_slice, N_row, N_col))
        .astype(float)
    )
    grid_coords = GridCoords(np.arange(5), np.arange(5), np.arange(5))
    image_data = ImageData(
        pixel_data, grid_coords, np.eye(3), np.zeros(3), PixelValueType.CT
    )

    # Expected unweighted mean of voxels inside the element
    data_inside = [11, 12, 16, 17, 36, 37, 41, 42]
    expected_mean = np.mean(data_inside)

    # Use unknown mode to trigger fallback
    interpolator = InterpolatorAllVoxel(mode="unknown", filter_outliers=False)
    result = interpolator._get_data_of_element(element, image_data)

    # Compare scalar value robustly regardless of return shape
    result_val = float(np.asarray(result).mean())
    assert np.isclose(result_val, expected_mean)


def test_weighted_small_voxel_count_uses_simple_weighted_average():
    """When voxel count is small (<=5), _weighted_voxel_mean should return
    simple weighted average."""
    # Grid 2x2x2
    grid_coords = GridCoords(
        slice=np.array([0.0, 1.0]),
        row=np.array([0.0, 1.0]),
        col=np.array([0.0, 1.0]),
    )
    # Deterministic pixel values: s + 10*r + 100*c
    s = grid_coords.slice[:, None, None]
    r = grid_coords.row[None, :, None]
    c = grid_coords.col[None, None, :]
    pixel_data = (s + 10 * r + 100 * c).astype(float)
    image_data = ImageData(
        pixel_data, grid_coords, np.eye(3), np.zeros(3), PixelValueType.CT
    )

    # Element hull tightly around x,y ~ 0 and spanning z from 0 to 1
    element_nodes = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.1, 0.0, 0.5],
            [0.0, 0.1, 0.5],
        ]
    )
    # One element uses the 4 nodes, with explicit node weights
    node_ids = np.arange(len(element_nodes))
    nodes = Nodes(
        ids=node_ids,
        coords=element_nodes,
        weights=np.array([2.0, 2.0, 1.0, 1.0]),
    )
    elements = [DisElement(node_ids=node_ids, id=0)]
    surfaces = [Surface(node_ids=np.array([], dtype=int), id=0)]
    dis = Discretization(nodes=nodes, elements=elements, surfaces=surfaces)

    interpolator = InterpolatorAllVoxel(
        mode="allvoxels_weighted", filter_outliers=True
    )
    elems = interpolator.compute_element_data(dis, image_data)
    result = float(np.asarray(elems[0].data).mean())

    # Manually compute included voxels and expected weighted average
    hull = ConvexHull(element_nodes)
    included_points = []
    included_vals = []
    for zi, zv in enumerate(grid_coords.slice):
        for yi, yv in enumerate(grid_coords.row):
            for xi, xv in enumerate(grid_coords.col):
                p = np.array([zv, yv, xv])
                A, b = hull.equations[:, :-1], hull.equations[:, -1]
                if np.all(A @ p + b <= 0):
                    included_points.append(p)
                    included_vals.append(pixel_data[zi, yi, xi])

    voxels_phys = np.asarray(included_points)
    values = np.asarray(included_vals)

    # Expect small voxel count (<=5) to use simple weighted average
    assert len(values) <= 5

    # Recompute voxel weights exactly as in implementation
    distances = np.linalg.norm(
        element_nodes[:, np.newaxis, :] - voxels_phys[np.newaxis, :, :], axis=2
    )
    distances = np.maximum(distances, 1e-10)
    voxel_weights = np.sum(nodes.weights[:, np.newaxis] / distances, axis=0)

    expected = np.average(values, weights=voxel_weights, axis=0)
    assert np.isclose(result, expected)
