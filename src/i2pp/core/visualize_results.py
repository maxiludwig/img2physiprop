"""Functions for visualizations."""

from multiprocessing import Process

from i2pp.core.discretization_reader_classes.discretization_reader import (
    Element,
)
from i2pp.core.image_reader_classes.image_reader import (
    ImageData,
)
from i2pp.core.visualization_classes.discretization_visualization import (
    DiscretizationVisualizer,
)
from i2pp.core.visualization_classes.image_visualization import ImageVisualizer


def visualize_results(
    config: dict, elements_with_values: list[Element], image_data: ImageData
) -> None:
    """Visualizes both the finite element discretization and the image data.

    This function launches two separate processes:
    - One for visualizing the finite element mesh as an unstructured grid.
    - One for visualizing the image data as a structured grid.

    Arguments:
        config (dict): Configuration dictionary containing visualization
            options.
        elements_with_values (list[Element]): List of elements with their
            assigned values.
        image_data (ImageData): Image data containing pixel values and grid
            coordinates.

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
            image_data.pixel_type,
            image_data.pixel_range,
            title="Mesh Visualization",
        )

        visualizer_dis.compute_grid(
            config, elements_with_values, image_data.pixel_range
        )

        visualizer_dis.plot_grid()

    def plot_image():
        """Creates and visualizes a structured grid from the image data.

        This function generates a PyVista StructuredGrid using the image
        pixel data and displays it.
        """

        visualizer_image = ImageVisualizer(
            image_data.pixel_type,
            image_data.pixel_range,
            title="Image Visualization",
        )

        visualizer_image.compute_grid(image_data)

        visualizer_image.plot_grid()

    thread1 = Process(target=plot_discretization)
    thread2 = Process(target=plot_image)

    thread1.start()
    thread2.start()

    thread1.join()
    thread2.join()


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

    Returns:
        None
    """

    def plot_smoothed():
        """Creates and visualizes a structured grid from the smoothed image
        data.

        This function generates a PyVista StructuredGrid using the image
        pixel data and displays it.
        """

        visualizer_smoothed = ImageVisualizer(
            image_data_smoothed.pixel_type,
            image_data_smoothed.pixel_range,
            "Smoothed Image Visualization",
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
            image_data_unsmoothed.pixel_type,
            image_data_unsmoothed.pixel_range,
            "Original Image Visualization",
        )
        visualizer_unsmoothed.compute_grid(image_data_unsmoothed)
        visualizer_unsmoothed.plot_grid()

    thread1 = Process(target=plot_smoothed)
    thread2 = Process(target=plot_unsmoothed)

    thread1.start()
    thread2.start()

    thread1.join()
    thread2.join()
