"""Discretization format detection and handling."""

from enum import Enum
from typing import Type

from i2pp.core.discretization_readers.discretization_reader import (
    DiscretizationReader,
)
from i2pp.core.discretization_readers.fourc_yaml_reader import FourCYamlReader
from i2pp.core.discretization_readers.mesh_reader import MeshReader


class DiscretizationFormat(Enum):
    """DiscretizationFormat (Enum): Defines the supported file formats for
    discretization data.

    Attributes:
        MESH: Represents the discretization data in '.mesh' format
        YAML: Represents the discretization data in the '.4C.yaml' format
    """

    MESH = ".mesh"
    YAML = ".yaml"

    def get_reader(self) -> Type[DiscretizationReader]:
        """Returns the appropriate discretization reader class based on the
        discretization format.

        Returns:
            Type[DiscretizationReader]: A class that is a subclass of
        `DiscretizationReader`, either `MeshReader` or `FourCYamlReader`.
        """
        return {
            DiscretizationFormat.MESH: MeshReader,
            DiscretizationFormat.YAML: FourCYamlReader,
        }[self]
