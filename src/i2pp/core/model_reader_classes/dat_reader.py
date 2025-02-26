"""Import Mesh data."""

import logging
from pathlib import Path

import lnmmeshio
import numpy as np
from i2pp.core.model_reader_classes.model_reader import (
    Element,
    Limits,
    ModelData,
    ModelReader,
    Nodes,
)
from lnmmeshio import Discretization
from tqdm import tqdm


class DatReader(ModelReader):
    """Class for reading and processing finite element models from .dat files.

    This class extends `ModelReader` to handle `.dat` files, which store
    discretized finite element models. It provides functionality to import
    the model, filter elements based on material IDs, and structure the data
    into a `ModelData` object.
    """

    def _filter_model(
        self, dis: Discretization, mat_ids: np.ndarray
    ) -> Discretization:
        """Filters the finite element model to include only elements with
        specified material IDs.

        This function iterates through the elements in the discretized model
        and selects only those whose material ID matches one of the specified
        `mat_ids`. The corresponding nodes of these elements are also
        retained.

        Arguments:
            dis (Discretization): The discretized finite element model.
            mat_ids (np.ndarray): Array of material IDs to filter.

        Returns:
            Discretization: The filtered discretized model containing only the
                selected elements and nodes.
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

        dis.elements.structure = list(filtered_elements)
        dis.nodes = list(filtered_nodes)

        return dis

    def load_model(self, directory: Path, config: dict) -> ModelData:
        """Loads and processes a finite element model from a .dat file.

        This function imports nodes and elements from a .dat file using
        `lnmmeshio`, applies optional material ID filtering, and organizes
        the data into a `ModelData` object.

        Arguments:
            directory (Path): Path to the .dat file.
            config (dict): User configuration containing material ID filters
                and other settings.

        Returns:
            ModelData: A structured representation of the finite element
                model, including nodes and elements.
        """

        logging.info("Importing Model data")

        raw_model = lnmmeshio.read(str(directory))

        raw_model.compute_ids(zero_based=True)

        if config["Matarial_IDs"] is not None:

            raw_model = self._filter_model(
                raw_model, np.array(config["Matarial_IDs"])
            )

        nodes_coords = []
        node_ids = []

        for node in raw_model.nodes:
            nodes_coords.append(node.coords)
            node_ids.append(node.id)

        elements = []

        for ele in raw_model.elements.structure:
            ele_node_ids = []
            for node in ele.nodes:
                ele_node_ids.append(node.id)

            elements.append(
                Element(
                    node_ids=np.array(ele_node_ids),
                    id=ele.id,
                    center_coord=np.array([]),
                    value=np.array([]),
                )
            )

        model = ModelData(
            nodes=Nodes(coords=np.array(nodes_coords), ids=np.array(node_ids)),
            elements=elements,
            limits=Limits([], []),
        )

        return model
