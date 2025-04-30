"""Import Dat data."""

import logging
from pathlib import Path

import lnmmeshio
import numpy as np
from i2pp.core.discretization_reader_classes.discretization_reader import (
    Discretization,
    DiscretizationReader,
    Element,
    Nodes,
)
from lnmmeshio import Discretization as DiscretizationLNM
from tqdm import tqdm


class DatReader(DiscretizationReader):
    """Class for reading and processing finite element models from .dat files.

    This class extends `DiscretizationReader` to handle `.dat` files, which
    store discretized finite element models. It provides functionality to
    import the Discretization, filter elements based on material IDs, and
    structure the data into a `Discretization` object.
    """

    def _filter_discretization(
        self, dis: DiscretizationLNM, mat_ids: np.ndarray
    ) -> DiscretizationLNM:
        """Filters the finite element model to include only elements with
        specified material IDs.

        This function iterates through the elements in the discretized model
        and selects only those whose material ID matches one of the specified
        `mat_ids`. The corresponding nodes of these elements are also
        retained. After filtering, the nodes are sorted based on their IDs.

        Arguments:
            dis (DiscretizationLNM): The discretized finite element model.
            mat_ids (np.ndarray): Array of material IDs to filter.

        Returns:
            DiscretizationLNM: The filtered discretized model containing only
                the selected elements and nodes.
        """

        dis.compute_ids(zero_based=True)

        filtered_nodes = set()
        filtered_elements = set()

        for mat_id in mat_ids:

            for ele in tqdm(
                dis.elements.structure, desc=f"Filtering elements MAT {mat_id}"
            ):
                if int(ele.options["MAT"]) == mat_id:
                    for n in ele.nodes:
                        filtered_nodes.add(n)
                    filtered_elements.add(ele)
                else:
                    continue

        sorted_nodes = sorted(filtered_nodes, key=lambda x: x.id)

        dis.elements.structure = list(filtered_elements)
        dis.nodes = list(sorted_nodes)

        return dis

    def load_discretization(
        self, file_path: Path, config: dict
    ) -> Discretization:
        """Loads and processes a finite element model from a .dat file.

        This function imports nodes and elements from a .dat file using
        `lnmmeshio`, applies optional material ID filtering, and organizes
        the data into a `Discretization` object.

        Arguments:
            file_path (Path): Path to the .dat file.
            config (dict): User configuration containing material ID filters.

        Returns:
            Discretization: A structured representation of the finite element
                model, including nodes and elements.
        """

        logging.info("Importing Discretization data")

        raw_dis = lnmmeshio.read(str(file_path))

        raw_dis.compute_ids(zero_based=True)

        if config["material_ids"] is not None:

            raw_dis = self._filter_discretization(
                raw_dis, np.array(config["material_ids"])
            )

        nodes_coords = []
        node_ids = []

        for node in raw_dis.nodes:
            nodes_coords.append(node.coords)
            node_ids.append(node.id)

        elements = []

        for ele in raw_dis.elements.structure:
            ele_node_ids = []
            for node in ele.nodes:
                ele_node_ids.append(node.id)

            elements.append(
                Element(node_ids=np.array(ele_node_ids), id=ele.id)
            )

        dis = Discretization(
            nodes=Nodes(coords=np.array(nodes_coords), ids=np.array(node_ids)),
            elements=elements,
        )

        return dis
