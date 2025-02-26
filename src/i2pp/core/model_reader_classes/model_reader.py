"""Import Mesh data."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from i2pp.core.utilities import get_node_position_of_element
from tqdm import tqdm


@dataclass
class Limits:
    """Dataclass for storing the boundaries (limits) of a set of points.

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
    """Class for storing information about nodes in a model.

    Attributes:
        coords (np.ndarray): An array of coordinates for each node, where
            each entry represents the (x, y, z) position of a node.
        ids (np.ndarray): An array of IDs for each node.
    """

    coords: np.ndarray
    ids: np.ndarray


@dataclass
class Element:
    """Class for storing information about an individual element in the model.

    This class represents a single element in the mesh model, defined by its
    node IDs, an ID, the coordinates of its center, and associated values.

    Attributes:
        node_ids (np.ndarray): An array of node IDs that define the nodes of
            the element.
        id (int): A unique identifier for the element.
        center_coord (np.ndarray): The coordinates of the center of the
            element.
        value (np.ndarray): An array of values associated with the element.
    """

    node_ids: np.ndarray
    id: int
    center_coord: np.ndarray
    value: np.ndarray


@dataclass
class ModelData:
    """Class for storing model data.

    This class is used to represent the mesh data of a model, including
    information about the nodes' coordinates, the elements that form the mesh,
    and the limits that define the boundaries of the model.

    Attributes:
        nodes (Nodes): The nodes of the model, which include the coordinates
            and IDs of each node.
        elements (list[Element]): A list of elements, each representing a part
            of the model, containing information such as node IDs, element ID,
            center coordinates, and element value.
        limits (Limits): Boundary limits of the model.
    """

    nodes: Nodes
    elements: list[Element]
    limits: Limits


class ModelReader(ABC):
    """Abstract base class for reading and processing finite element models.

    The `ModelReader` class defines the interface for reading and processing
    model data from different file formats (e.g., `.dat`, `.mesh`). Subclasses
    must implement the `load_model` method to handle specific file formats.
    """

    def __init__(self):
        """Init ModelReader."""
        pass

    @abstractmethod
    def load_model(self, directory: Path, config: dict) -> ModelData:
        """Abstract method to load model data from a file.

        This method imports the nodes and elements of a model from a specific
        file format (e.g., .dat, .mesh). The method will parse the file,
        extract the relevant information, and return a ModelData object that
        encapsulates the nodes, elements, and the boundaries in which the model
        nodes are located.

        Arguments:
            directory (Path): Path to the model file (e.g., .dat or .mesh
                file).
            config (dict): A dictionary containing configuration options for
                loading the model.

        Returns:
            ModelData: An instance of ModelData containing information of the
                nodes and elements of the imported model, along with
                the boundaries in which the model nodes are located.
        """
        pass

    def get_center(self, model: ModelData) -> ModelData:
        """Calculates the center (centroid) of each element in the model.

        This function iterates through all elements in the given ModelData
        object, calculates the centroid of each element by averaging the
        coordinates of its associated nodes, and then adds the calculated
        center as a new attribute ('center_coord') for each element.

        Arguments:
            model (ModelData): The ModelData object containing elements and
                nodes.

        Returns:
            ModelData: The updated ModelData object with a new 'center_coord'
                attribute for each element, representing the calculated
                centroid.
        """

        for i, ele in tqdm(
            enumerate(model.elements),
            total=len(model.elements),
            desc="Calculate element center",
        ):

            node_position = get_node_position_of_element(
                ele.node_ids, model.nodes.ids
            )
            element_coords = model.nodes.coords[node_position]
            centroid = np.mean(element_coords, axis=0)

            model.elements[i].center_coord = np.array(centroid)

        return model
