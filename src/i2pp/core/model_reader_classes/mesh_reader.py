"""Import Mesh data."""

import numpy as np
import trimesh
from i2pp.core.model_reader_classes.model_reader import ModelData, ModelReader


class MeshReader(ModelReader):
    """Class to read Mesh-data."""

    def load_model(self, directory: str) -> ModelData:
        """Import the nodes and elements of a .mesh-file.

        Arguments:
            directory {str} -- Path to the .mesh-file.

        Returns:
            object -- ModelData object"""

        print("Importing Model data")

        raw_model = trimesh.load(directory)

        model = ModelData(
            nodes=np.array(raw_model.vertices),
            element_ids=np.array(raw_model.faces),
            element_center=np.array([]),
            limits=object,
        )

        return model
