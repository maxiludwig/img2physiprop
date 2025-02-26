"""Import Mesh data."""

import logging
from pathlib import Path

import numpy as np
import trimesh
from i2pp.core.model_reader_classes.model_reader import (
    Element,
    Limits,
    ModelData,
    ModelReader,
    Nodes,
)


class MeshReader(ModelReader):
    """Class for reading and processing finite element models from .mesh files.

    This class extends `ModelReader` to handle `.mesh` files, which store
    discretized finite element models. It provides functionality to import
    the model, filter elements based on material IDs, and structure the data
    into a `ModelData` object.
    """

    def load_model(self, directory: Path, config: dict) -> ModelData:
        """Loads and processes a finite element model from a .mesh file.

        This function imports the nodes and elements from a .mesh file using
        `trimesh`, organizes them into a `ModelData` object, and initializes
        necessary attributes.

        Arguments:
            directory (Path): Path to the .mesh file.

        Returns:
            ModelData: A structured representation of the finite element
                model, including nodes and elements.
        """

        logging.info("Importing Model data")

        raw_model = trimesh.load(directory)

        nodes = Nodes(raw_model.vertices, list(range(len(raw_model.faces))))

        elements = []
        for i, face in enumerate(raw_model.faces):
            elements.append(Element(face, i, np.array([]), []))

        model = ModelData(
            nodes=nodes, elements=elements, limits=Limits([], [])
        )
        return model
