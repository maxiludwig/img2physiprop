"""Import Discretization."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass
class BoundingBox:
    """Dataclass for storing the boundaries (BoundingBox) of a set of points.

    Attributes:
        min (np.ndarray): An array representing the minimum values along
            each axis (e.g., x, y, z) of the set of points.
        max (np.ndarray): An array representing the maximum values along
            each axis (e.g., x, y, z) of the set of points.
    """

    min: np.ndarray
    max: np.ndarray


@dataclass
class Nodes:
    """Class for storing information about nodes in a Discretization.

    Attributes:
        coords (np.ndarray): An (N, 3) array containing the (x, y, z)
            world coordinates of each node.
        ids (np.ndarray): An array of IDs for each node.
    """

    coords: np.ndarray
    ids: np.ndarray


@dataclass
class Element:
    """Class for storing information about an individual element in the
    Discretization.

    This class represents a single element in the mesh Discretization, defined
    by its node IDs, an element ID, the coordinates of its center, and
    associated data.

    Attributes:
        node_ids (np.ndarray): An array of node IDs that define the nodes of
            the element.
        id (int): A unique identifier for the element.
        world_coords (Optional[np.ndarray]): An (N, 3) array containing the
            (x, y, z) world coordinates of the center of the element.
        data (Optional[np.ndarray]): An array representing the data associated
            with the element, such as RGB colors or grayscale intensities
    """

    node_ids: np.ndarray
    id: int
    center_coords: Optional[np.ndarray] = None
    data: Optional[np.ndarray] = None


@dataclass
class Discretization:
    """Class for storing Discretization data.

    This class is used to represent the Discretization of a mesh,
    including information about the nodes' coordinates, the elements that
    form the mesh, and the bounding box that define the boundaries of the
    Discretization.

    Attributes:
        nodes (Nodes): The nodes of the Discretization, which include the
            coordinates and IDs of each node.
        elements (list[Element]): A list of elements, each representing a part
            of the Discretization, containing information such as node IDs,
            element ID, center coordinates, and element data.
        bounding_box (Optional[np.ndarray]): Boundary limits of the
            Discretization.
    """

    nodes: Nodes
    elements: list[Element]
    bounding_box: Optional[BoundingBox] = None


class DiscretizationReader(ABC):
    """Abstract base class for reading and processing finite element
    Discretizations.

    The `DiscretizationReader` class defines the interface for reading and
    processing Discretization data from different file formats (e.g., `.dat`,
    `.mesh`). Subclasses must implement the `load_discretization` method to
    handle specific file formats.
    """

    def __init__(self):
        """Init DiscretizationReader."""
        pass

    @abstractmethod
    def load_discretization(
        self, file_path: Path, config: dict
    ) -> Discretization:
        """Abstract method to load Discretization data from a file path.

        This method imports the nodes and elements of a Discretization from a
        specific file format (e.g., .dat, .mesh). The method will parse the
        file, extract the relevant information, and return a Discretization
        object that encapsulates the nodes, elements, and the boundaries in
        which the Discretization nodes are located.

        Arguments:
            file_path (Path): Path to the Discretization file (e.g., .dat or
                .mesh file).
            config (dict): A dictionary containing configuration options for
                loading the Discretization.

        Returns:
            Discretization: An instance of Discretization containing
                information of the nodes and elements of the imported mesh,
                along with the boundaries in which the Discretization nodes
                are located.
        """
        pass
