"""Import Mesh data."""

import lnmmeshio
import numpy as np
from i2pp.core.model_reader_classes.model_reader import ModelData, ModelReader


class DatReader(ModelReader):
    """Class to read .dat-data"""

    def load_model(self, directory: str) -> ModelData:
        """Import nodes and elements of a .dat-file.

        Arguments:
            directory {str} -- Path to the .dat-file.

        Returns:
            object -- ModelData object"""

        print("Importing Model data")

        directory = str(directory)
        raw_model = lnmmeshio.read(directory)

        raw_model.compute_ids(True)

        nodes = raw_model.get_node_coords()

        element_ids = []

        for ele in raw_model.elements.structure:
            element = []
            for node in ele.nodes:
                element.append(node.id)

            element_ids.append(element)

        model = ModelData(
            nodes=np.array(nodes),
            element_ids=np.array(element_ids),
            element_center=np.array([]),
            limits=object,
        )

        return model
