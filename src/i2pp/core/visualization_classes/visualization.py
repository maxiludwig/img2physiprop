"""Functions for visualizations."""

from abc import ABC, abstractmethod
from typing import Union

import numpy as np
import pyvista as pv
from i2pp.core.image_reader_classes.image_reader import (
    PixelValueType,
)


class Visualizer(ABC):
    """A class for visualizing structured and unstructured grids using PyVista.

    Provides interactive controls for adjusting opacity, slicing the
    grid, and toggling NaN element visibility.
    """

    def __init__(
        self,
        pixel_type: PixelValueType,
        pixel_range: np.ndarray,
        title: str | None = None,
    ):
        """Initialize the Visualizer.

        Arguments:
            pixel_type (PixelValueType): Specifies whether the pixel data is
                RGB or scalar.
            title (str | None, optional): The title of the plot window.
                Defaults to None.
        """
        self.plotter = pv.Plotter()
        self.pixel_type = pixel_type
        self.pixel_range = pixel_range
        self.title = title
        self.slice_height = 0.00
        self.opacity = 1.00
        self.show_3d = True
        self.nan_exist = False
        self.grid = pv.UnstructuredGrid()
        self.extracted_grid = None
        self.grid_visible = pv.UnstructuredGrid()
        self.current_actor = None
        self.show_edges = False

    @abstractmethod
    def compute_grid(self, *args, **kwargs) -> None:
        """Abstract method to compute the grid to be visualized.

        This method is intended to be implemented by subclasses to
        compute the grid to be visualized based on their specific
        requirements. The implementation of this method should modify
        the relevant attributes of the class, such as the grid data or
        any other internal state that is necessary for visualization.
        """

        pass

    def _update_display(
        self, grid: Union[pv.StructuredGrid, pv.UnstructuredGrid], **kwargs
    ) -> None:
        """Updates the currently displayed mesh in the plotter.

        Removes the previous mesh actor (if any) and adds the provided mesh
        with the given visualization settings.

        Arguments:
            grid (Union[pv.StructuredGrid, pv.UnstructuredGrid]): The new
                grid to display.
            **kwargs: Additional keyword arguments for PyVista's `add_mesh`.

        Returns:
            None
        """

        if self.current_actor is not None:
            self.plotter.remove_actor(self.current_actor)
        self.current_actor = self.plotter.add_mesh(grid, **kwargs)

    def _toggle_slicer(self) -> None:
        """Toggles between 3D model visualization and 2D slice visualization.

        Clears any existing slider widgets and re-adds the appropriate slider
        depending on the current visualization mode (opacity for 3D, slice
        height for 2D).

        Returns:
            None
        """

        self.plotter.clear_slider_widgets()

        if self.show_3d:
            self.plotter.add_slider_widget(
                self._update_opacity,
                rng=[0, 1],
                value=self.opacity,
                title="Opacity",
                pointa=(0.4, 0.9),
                pointb=(0.9, 0.9),
            )

        else:

            self.plotter.add_slider_widget(
                self._update_slice_height,
                rng=[self.grid_visible.bounds[4], self.grid_visible.bounds[5]],
                value=self.slice_height,
                title="Slice Height",
                pointa=(0.4, 0.9),
                pointb=(0.9, 0.9),
            )

    def _update_grid(self, is_checked: bool) -> None:
        """Updates the displayed grid based on the NaN elements toggle state.

        If checked, all elements are displayed. If unchecked, only elements
        with assigned values are shown.

        Arguments:
            is_checked (bool): Whether to show all elements or only those with
                values.

        Returns:
            None
        """
        self.grid_visible = (
            self.extracted_grid if not is_checked else self.grid
        )

    def _show_3d_model(self) -> None:
        """Displays the 3D model of the grid with the specified opacity.

        Arguments:
            opacity (float): The opacity level (0 to 1).

        Returns:
            None
        """

        self.show_3d = True

        if self.pixel_type == PixelValueType.RGB:

            self._update_display(
                self.grid_visible,
                scalars="rgb_values",
                rgb=True,
                opacity=self.opacity,
            )

        else:

            self._update_display(
                self.grid_visible,
                scalars="ScalarValues",
                cmap="viridis",
                opacity=self.opacity,
                clim=(self.pixel_range[0], self.pixel_range[1]),
            )

        self._update_edges()
        self.plotter.add_axes()
        self.plotter.render()

    def _slice_at_height(self) -> None:
        """Displays a 2D slice of the grid at the specified height.

        Extracts a slice along the Z-axis and updates the display.

        Arguments:
            height (float): The Z-coordinate where the slice is taken.

        Returns:
            None
        """

        self.show_3d = False

        slice_mesh = self.grid_visible.slice(
            normal=[0, 0, 1], origin=(0, 0, self.slice_height)
        )

        if self.pixel_type == PixelValueType.RGB:
            self._update_display(slice_mesh, scalars="rgb_values", rgb=True)
        else:
            self._update_display(
                slice_mesh,
                scalars="ScalarValues",
                cmap="viridis",
                clim=(self.pixel_range[0], self.pixel_range[1]),
            )

        self._update_edges()
        self.plotter.add_axes()
        self.plotter.render()

    def _toggle_nan_view(self, is_checked: bool = True) -> None:
        """Toggles the visibility of elements without assigned values.

        Updates the displayed grid and maintains the current view mode
        (3D model or slice).

        Arguments:
            is_checked (bool, optional): Whether to show or hide NaN
                elements. Defaults to True.

        Returns:
            None
        """

        self._update_grid(is_checked)

        if self.show_3d:
            self._show_3d_model()
        else:
            self._slice_at_height()

        self._toggle_slicer()

    def _toggle_view(self, is_checked: bool = True) -> None:
        """Toggles between 3D and 2D slice visualization.

        Resets the visualization mode and updates the sliders accordingly.

        Arguments:
            is_checked (bool, optional): Whether to switch to 3D mode
                (True) or 2D slice mode (False). Defaults to True.

        Returns:
            None
        """
        if is_checked:
            self._show_3d_model()

        else:
            self._slice_at_height()

        self._toggle_slicer()

    def _update_opacity(self, value: float) -> None:
        """Updates the opacity of the 3D model.

        If the current view mode is 3D, the displayed mesh is updated.

        Arguments:
            value (float): The new opacity level (0 to 1).

        Returns:
            None
        """

        self.opacity = value

        if self.show_3d:
            self._show_3d_model()

    def _update_slice_height(self, value: float) -> None:
        """Updates the height of the 2D slice.

        If the current view mode is a 2D slice, the displayed slice is
        updated.

        Arguments:
            value (float): The new Z-coordinate for the slice.

        Returns:
            None
        """

        self.slice_height = value

        if not self.show_3d:
            self._slice_at_height()

    def _toggle_edges(self, is_checked: bool = False):
        """Toggle the visibility of mesh edges in the plotter.

        This method updates the internal `show_edges` attribute based on the
        input checkbox state (`is_checked`) and triggers an update of the edge
        visibility in the currently displayed actor.

        Args:
            is_checked (bool, optional): Indicates whether edges should be
                shown. Defaults to False.
        """

        self.show_edges = is_checked
        self._update_edges()

    def _update_edges(self):
        """Update the visibility of edges for the currently active actor.

        If a mesh actor is currently displayed in the plotter
        (`self.current_actor`), this method applies the value of
        `self.show_edges` to control whether the mesh edges are visible.
        """

        if self.current_actor is not None:
            self.current_actor.GetProperty().SetEdgeVisibility(self.show_edges)

    def plot_grid(self) -> None:
        """Visualizes a structured or unstructured grid using PyVista with
        interactive controls for opacity adjustment, slice visualization, and
        NaN element toggling.

        This function initializes a PyVista plotter, displays the grid, and
        provides UI elements to switch between a full 3D model and a 2D slice,
        adjust opacity, and optionally hide elements without assigned values.

        Arguments:
            grid (Union[pv.StructuredGrid, pv.UnstructuredGrid]): The grid to
                be visualized.
            pixel_type (PixelValueType): Specifies whether the pixel data is
                RGB or scalar.
            ele_has_value (np.ndarray | None, optional): A boolean array
                indicating which elements have assigned values. If None, all
                elements are assumed to have values. Defaults to None.
            title (str | None, optional): The title of the plot window.
                Defaults to None.

        Returns:
            None
        """

        self.plotter.window_size = [1024, 768]
        self.plotter.title = self.title

        self.slice_height = (self.grid.bounds[4] + self.grid.bounds[5]) / 2

        self._toggle_slicer()
        self._show_3d_model()

        if self.nan_exist:
            self.plotter.add_text(
                "Toggle Nan-Elements:",
                position="upper_left",
                font_size=8,
                color="black",
            )
            self.plotter.add_checkbox_button_widget(
                self._toggle_nan_view, size=30, value=True, position=(210, 740)
            )

        if isinstance(self.grid, pv.UnstructuredGrid):

            self.plotter.add_text(
                "Toggle Edges:", position=(0, 600), font_size=8, color="black"
            )

            self.plotter.add_checkbox_button_widget(
                self._toggle_edges, size=30, value=False, position=(210, 600)
            )

        self.plotter.add_checkbox_button_widget(
            self._toggle_view, value=True, position=(0.85, 0.95)
        )

        self.plotter.show()
