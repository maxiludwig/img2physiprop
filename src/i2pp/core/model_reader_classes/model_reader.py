"""Import Mesh data."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from tqdm import tqdm


@dataclass
class ModelData:
    """Class for Mesh-data."""

    nodes: np.array
    element_ids: np.array
    element_center: np.array
    limits: np.array


class ModelReader(ABC):
    """Class to read Mesh-data."""

    def __init__(self):
        """Init ModelReader."""
        pass

    @abstractmethod
    def load_model(self, directory: str):
        """Load raw-model-data. and turn is into a ModelData object."""
        pass

    def get_center(self, model: ModelData) -> ModelData:
        """Calculates center of each element

        Arguments:
            model {object} -- ModelData object

        Returns:
            object -- ModelData object with new Attribute 'element_center'"""
        for ele_ids in tqdm(
            model.element_ids, desc="Calculate element center"
        ):
            element_nodes = model.nodes[ele_ids]
            centroid = np.mean(element_nodes, axis=0)
            model.element_center.append(centroid)

        model.element_center = np.array(model.element_center)

        return model
