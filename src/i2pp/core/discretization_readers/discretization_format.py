"""Discretization format detection and handling."""

from enum import Enum
from typing import Type

from i2pp.core.discretization_readers.discretization_reader import (
    DiscretizationReader,
)
from i2pp.core.discretization_readers.lnmmeshio_reader import LnmmeshioReader
from i2pp.core.discretization_readers.trimesh_reader import TrimeshReader


class DiscretizationFormat(Enum):
    """DiscretizationFormat (Enum): Defines the supported file formats for
    discretization data.

    Attributes:
        MESH: Represents the discretization data in '.mesh' format
        YAML: Represents the discretization data in the '.4C.yaml' format
        VTU: Represents the discretization data in the '.vtu' format, read
            via lnmmeshio (e.g. line2 meshes of airway trees)
    """

    MESH = ".mesh"
    YAML = ".yaml"
    VTU = ".vtu"

    def get_reader(self) -> Type[DiscretizationReader]:
        """Returns the appropriate discretization reader class based on the
        discretization format.

        Returns:
            Type[DiscretizationReader]: A class that is a subclass of
        `DiscretizationReader`, either `TrimeshReader` or `LnmmeshioReader`.
        """
        return {
            DiscretizationFormat.MESH: TrimeshReader,
            DiscretizationFormat.YAML: LnmmeshioReader,
            DiscretizationFormat.VTU: LnmmeshioReader,
        }[self]
