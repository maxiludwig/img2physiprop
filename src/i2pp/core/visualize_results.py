"""Functions for visualizations."""

import logging
from multiprocessing import Process

from i2pp.core.discretization_readers.discretization_reader import (
    Discretization,
    Element,
)
from i2pp.core.image_readers.image_reader import ImageData
from i2pp.core.visualizers.discretization_visualizer import (
    DiscretizationVisualizer,
)
from i2pp.core.visualizers.image_visualizer import ImageVisualizer


def _run_processes_safely(*processes: Process) -> None:
    """Safely run multiple processes with error handling.

    Args:
        *processes: Variable number of Process objects to run.
    """
    try:
        # Start all processes
        for process in processes:
            process.start()

        # Wait for all processes to complete
        for process in processes:
            process.join()

    except Exception as e:
        logging.error(f"Error in visualization process: {e}")
        # Ensure all processes are terminated
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join()


def visualize_results(
    elements_with_values: list[Element],
    image_data: ImageData,
    dis: Discretization,
) -> None:
    """Visualizes both the finite element discretization and the image data.

    This function launches two separate processes:
    - One for visualizing the finite element mesh as an unstructured grid.
    - One for visualizing the image data as a structured grid.

    Arguments:
        elements_with_values (list[Element]): List of elements with their
            assigned values.
        image_data (ImageData): Image data containing pixel values and grid
            coordinates.
        dis (Discretization): The discretization object containing nodes,
            elements and surfaces.
    Returns:
        None
    """

    def plot_discretization():
        """Creates and visualizes an unstructured grid from the discretization
        data.

        This function generates a PyVista UnstructuredGrid based on the
        provided element data and plots it.
        """

        visualizer_dis = DiscretizationVisualizer(
            pixel_type=image_data.pixel_type,
            pixel_range=image_data.pixel_range,
            title="Mesh Visualization",
        )

        visualizer_dis.compute_grid(elements_with_values, dis)

        visualizer_dis.plot_grid()

    def plot_image():
        """Creates and visualizes a structured grid from the image data.

        This function generates a PyVista StructuredGrid using the image
        pixel data and displays it.
        """

        visualizer_image = ImageVisualizer(
            pixel_type=image_data.pixel_type,
            pixel_range=image_data.pixel_range,
            title="Image Visualization",
        )

        visualizer_image.compute_grid(image_data)

        visualizer_image.plot_grid()

    process_discretization = Process(target=plot_discretization)
    process_image = Process(target=plot_image)

    _run_processes_safely(process_discretization, process_image)


def visualize_smoothing(
    image_data_smoothed: ImageData, image_data_unsmoothed: ImageData
) -> None:
    """Visualizes smoothed and unsmoothed image data.

    This function launches two separate processes:
    - One for visualizing the smoothed image data.
    - One for visualizing the original (unsmoothed) image data.

    Arguments:
        image_data_smoothed (ImageData): Image data after smoothing.
        image_data_unsmoothed (ImageData): Original image data before
            smoothing.
    """

    def plot_smoothed():
        """Creates and visualizes a structured grid from the smoothed image
        data.

        This function generates a PyVista StructuredGrid using the image
        pixel data and displays it.
        """

        visualizer_smoothed = ImageVisualizer(
            pixel_type=image_data_smoothed.pixel_type,
            pixel_range=image_data_smoothed.pixel_range,
            title="Smoothed Image Visualization",
        )
        visualizer_smoothed.compute_grid(image_data_smoothed)
        visualizer_smoothed.plot_grid()

    def plot_unsmoothed():
        """Creates and visualizes a structured grid from the unsmoothed image
        data.

        This function generates a PyVista StructuredGrid using the image
        pixel data and displays it.
        """

        visualizer_unsmoothed = ImageVisualizer(
            pixel_type=image_data_unsmoothed.pixel_type,
            pixel_range=image_data_unsmoothed.pixel_range,
            title="Original Image Visualization",
        )
        visualizer_unsmoothed.compute_grid(image_data_unsmoothed)
        visualizer_unsmoothed.plot_grid()

    process_smoothed = Process(target=plot_smoothed)
    process_unsmoothed = Process(target=plot_unsmoothed)

    _run_processes_safely(process_smoothed, process_unsmoothed)
