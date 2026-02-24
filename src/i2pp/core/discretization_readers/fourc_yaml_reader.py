"""Import 4C.yaml data."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import lnmmeshio
import numpy as np
from i2pp.core.discretization_readers.discretization_reader import (
    Discretization,
    DiscretizationReader,
    Element,
    Nodes,
    Surface,
)
from lnmmeshio import Discretization as FourCDiscretization
from tqdm import tqdm

if TYPE_CHECKING:
    from i2pp.core.configuration_validator.validator import Processing


class FourCYamlReader(DiscretizationReader):
    """Class for reading and processing finite element models from .4C.yaml
    files.

    This class extends `DiscretizationReader` to handle `.4C.yaml` files, which
    store discretized finite element models. It provides functionality to
    import the Discretization, filter elements based on material IDs, and
    structure the data into a `Discretization` object.
    """

    def _filter_discretization(
        self, dis: FourCDiscretization, mat_ids: np.ndarray
    ) -> FourCDiscretization:
        """Filters the discretization to include only elements with specified
        material IDs.

        This function iterates through the elements in the discretization
        and selects only those whose material ID matches one of the specified
        `mat_ids`. The corresponding nodes of these elements are also
        retained. After filtering, the nodes are sorted based on their IDs.

        Arguments:
            dis (FourCDiscretization): The discretization.
            mat_ids (np.ndarray): Array of material IDs to filter.

        Returns:
            FourCDiscretization: The filtered discretization containing only
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
        dis.compute_ids(zero_based=True)
        return dis

    def load_discretization(
        self,
        file_path: Path,
        options: dict,
        processing: Processing,
    ) -> Discretization:
        """Loads and processes a finite element discretization from a .4C.yaml
        file.

        This function imports nodes and elements from a .4C.yaml file using
        `lnmmeshio`, applies optional material ID filtering, and organizes
        the data into a `Discretization` object.

        Arguments:
            file_path (Path): Path to the .4C.yaml file.
            options (dict): Options for loading the discretization.
                Filtering for material ids can be enabled by specifying
                `material_ids` in the options dictionary.
            processing (I2PPConfig.processing):
                Processing configuration object.

        Returns:
            Discretization: The finite element discretization including nodes
            and elements.
        """

        logging.info("Importing discretization data")

        if processing is None:
            raise ValueError(
                "Processing configuration is required"
                "for loading the discretization."
            )

        raw_dis = lnmmeshio.read(str(file_path))

        raw_dis.compute_ids(zero_based=True)

        if options.get("material_ids") and options is not None:
            raw_dis = self._filter_discretization(
                raw_dis, np.array(options["material_ids"])
            )

        scaling_factors = processing.interpolation.node_scaling_factors

        interior_node_scaling = scaling_factors.interior_node_scaling
        surface_node_scaling = scaling_factors.surface_node_scaling

        nodes_coords = []
        node_ids = []
        nodes_scaling = []

        for node in raw_dis.nodes:
            nodes_coords.append(node.coords)
            node_ids.append(node.id)
            nodes_scaling.append(interior_node_scaling)

        elements = []

        for ele in raw_dis.elements.structure:
            ele_node_ids = []
            for node in ele.nodes:
                ele_node_ids.append(node.id)

            elements.append(
                Element(node_ids=np.array(ele_node_ids), id=ele.id)
            )

        surfaces = []

        node_id_to_idx = {nid: i for i, nid in enumerate(node_ids)}
        for surf in raw_dis.surfacenodesets:
            surf_node_ids = []
            for node in surf.nodes:
                surf_node_ids.append(node.id)
                position = node_id_to_idx[node.id]
                nodes_scaling[position] = surface_node_scaling

            surfaces.append(
                Surface(node_ids=np.array(surf_node_ids), id=surf.id)
            )

        dis = Discretization(
            nodes=Nodes(
                coords=np.array(nodes_coords),
                ids=np.array(node_ids),
                scaling_factors=np.array(nodes_scaling),
            ),
            elements=elements,
            surfaces=surfaces,
        )

        return dis
