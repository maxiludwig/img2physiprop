"""Import Mesh data."""

import logging
from pathlib import Path

import trimesh
from i2pp.core.discretization_reader_classes.discretization_reader import (
    Discretization,
    DiscretizationReader,
    Element,
    Nodes,
)


class MeshReader(DiscretizationReader):
    """Class for reading and processing finite element models from .mesh files.

    This class extends `DiscretizationReader` to handle `.mesh` files, which
    store discretized finite element models. It provides functionality to
    import the mesh, filter elements based on material IDs, and structure the
    data into a `Discretization` object.
    """

    def _filter_discretization(self) -> None:
        """Filters the finite element model to include only elements with
        specified material IDs.

        This function iterates through the elements in the discretized model
        and selects only those whose material ID matches one of the specified
        `mat_ids`. The corresponding nodes of these elements are also
        retained. After filtering, the nodes are sorted based on their IDs.
        """
        raise RuntimeError("This function is not implemented yet.")

    def load_discretization(
        self, file_path: Path, config: dict
    ) -> Discretization:
        """Loads and processes a finite element model from a .mesh file.

        This function imports the nodes and elements from a .mesh file using
        `trimesh`, organizes them into a `Discretization` object, and
        initializes necessary attributes.

        Arguments:
            file_path (Path): Path to the .mesh file.

        Returns:
            Discretization: A structured representation of the finite element
                model, including nodes and elements.
        """

        logging.info("Importing Model data")

        raw_dis = trimesh.load(file_path)

        if config["material_ids"] is not None:

            self._filter_discretization()

        nodes = Nodes(
            coords=raw_dis.vertices, ids=list(range(len(raw_dis.faces)))
        )

        elements = []
        for i, face in enumerate(raw_dis.faces):
            elements.append(Element(node_ids=face, id=i))

        dis = Discretization(nodes=nodes, elements=elements)

        return dis
