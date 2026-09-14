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

    def _is_selected(self, ele, element_filter: dict) -> bool:
        """Checks whether an element passes the element filter.

        The filter field is looked up in the element data (e.g. VTU cell data
        such as `block_id`) and then in the element options (e.g. the
        material `MAT`).

        Arguments:
            ele: The lnmmeshio element.
            element_filter (dict): Dictionary with the keys `field` (name of
                the element data field or option) and `values` (list of
                values to keep).

        Returns:
            bool: True if the value of the field is one of the filter values.

        Raises:
            ValueError: If the element has no data field or option with the
                given name.
        """
        field = element_filter["field"]
        if field in ele.data:
            value = ele.data[field]
        elif field in ele.options:
            value = ele.options[field]
        else:
            available = ", ".join([*ele.data, *ele.options])
            raise ValueError(
                f"Element filter field '{field}' not found in the element "
                f"data or options. Available fields: {available}."
            )

        # options such as the material may be stored as strings
        if isinstance(value, str):
            try:
                value = int(value)
            except ValueError:
                pass

        return value in element_filter["values"]

    def _filter_discretization(
        self,
        dis: FourCDiscretization,
        element_filter: dict,
    ) -> FourCDiscretization:
        """Filters the discretization to include only elements whose filter
        field has one of the specified values.

        This function iterates through the elements in the discretization
        and selects only those whose element data field or option (e.g.
        `block_id` or `MAT`) matches one of the values of `element_filter`.
        The corresponding nodes of these elements are also retained. After
        filtering, the nodes are sorted based on their IDs.

        Arguments:
            dis (FourCDiscretization): The discretization.
            element_filter (dict): Dictionary with the keys `field` and
                `values` to filter elements by.

        Returns:
            FourCDiscretization: The filtered discretization containing only
                the selected elements and nodes.
        """

        dis.compute_ids(zero_based=True)

        filtered_nodes = set()
        filtered_elements = []

        for ele in tqdm(dis.elements.structure, desc="Filtering elements"):
            if self._is_selected(ele, element_filter):
                for n in ele.nodes:
                    filtered_nodes.add(n)
                filtered_elements.append(ele)

        sorted_nodes = sorted(filtered_nodes, key=lambda x: x.id)

        dis.elements.structure = filtered_elements
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
        `lnmmeshio`, applies optional element filtering, and organizes
        the data into a `Discretization` object.

        Arguments:
            file_path (Path): Path to the .4C.yaml file.
            options (dict): Options for loading the discretization.
                Filtering of elements (e.g. by the VTU cell data `block_id`
                or the material `MAT`) can be enabled by specifying
                `element_filter` with the keys `field` and `values`.
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

        # remember the position of each element in the input file, since
        # filtering renumbers the element ids
        source_ids = {ele: ele.id for ele in raw_dis.elements.structure}

        element_filter = (options or {}).get("element_filter")

        if element_filter is not None:
            if "field" not in element_filter or "values" not in element_filter:
                raise ValueError(
                    "The discretization option 'element_filter' requires the "
                    "keys 'field' and 'values'."
                )
            raw_dis = self._filter_discretization(raw_dis, element_filter)
            if not raw_dis.elements.structure:
                raise RuntimeError(
                    "No elements left in the discretization after filtering."
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
                Element(
                    node_ids=np.array(ele_node_ids),
                    id=ele.id,
                    source_id=source_ids[ele],
                )
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
