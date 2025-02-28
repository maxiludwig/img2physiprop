"""Import Mesh data."""

from enum import Enum
from pathlib import Path
from typing import Type, cast

from i2pp.core.discretization_reader_classes.dat_reader import DatReader
from i2pp.core.discretization_reader_classes.discretization_reader import (
    Discretization,
    DiscretizationReader,
    Limits,
)
from i2pp.core.discretization_reader_classes.mesh_reader import MeshReader
from i2pp.core.utilities import find_mins_maxs


class DiscretizationFormat(Enum):
    """DiscretizationFormat (Enum): Defines the supported file formats for
    Discretization data.

    Attributes:
        MESH: Represents the Discretization data in the '.mesh' format,
            commonly used for 3D Models.
        DAT: Represents the Discretization data in the '.dat' format,
            typically used for storing node and element data.

    This enum helps identify the format of the Discretization data being
    processed and guides the appropriate handling of the data based on its
    format (e.g., .mesh vs. .dat).
    """

    MESH = ".mesh"
    DAT = ".dat"

    def get_reader(self) -> Type[DiscretizationReader]:
        """Returns the appropriate Discretization reader class based on the
        Discretization format.

        Returns:
            Type[DiscretizationReader]: A class that is a subclass of
                `DiscretizationReader`, either `MeshReader` or `DatReader`.
        """
        return {
            DiscretizationFormat.MESH: MeshReader,
            DiscretizationFormat.DAT: DatReader,
        }[self]


def determine_discretization_format(directory: Path) -> DiscretizationFormat:
    """Verifies if the Discretization file exists, is readable, and has a
    supported format.

    This function checks whether the provided Discretization file exists and
    if its format is valid (either `.mesh` or `.dat`). If the file is missing
    or has an unsupported format, it raises an error.

    Arguments:
        directory (Path): Path to the Discretization data file.

    Returns:
        DiscretizationFormat: The format of the Discretization file (`.mesh`
            or `.dat`).

    Raises:
        RuntimeError: If the specified file does not exist.
        RuntimeError: If the file format is not `.mesh` or `.dat`.
    """
    if not directory.is_file():
        raise RuntimeError(
            f"Path {directory} to the Discretization cannot be found!"
        )

    try:
        return DiscretizationFormat(directory.suffix)
    except ValueError:
        raise RuntimeError(
            f"{directory.suffix} not readable! Supported formats are: "
            f"{', '.join(fmt.value for fmt in DiscretizationFormat)}"
        )


def verify_and_load_discretization(config: dict) -> Discretization:
    """Loads and processes mesh data based on the user configuration.

    This function verifies the input mesh file, selects the appropriate reader
    (MeshReader or DatReader), and loads the Discretization data. If element
    center calculations are required, it processes the Discretization
    accordingly. Finally, it determines the Discretization's bounding limits.

    Arguments:
        config (dict): User configuration containing paths and processing
            options.

    Returns:
        DiscretizationData: The loaded and processed mesh data.

    Raises:
        RuntimeError: If the mesh file is not valid or in the wrong format.
    """
    relative_path = Path(
        config["input informations"]["discretization_file_path"]
    )

    directory = Path.cwd() / relative_path

    dis_format = determine_discretization_format(directory)

    dis_reader = cast(DiscretizationReader, dis_format.get_reader()())

    dis = dis_reader.load_discretization(
        directory, config["processing options"]
    )

    limits = find_mins_maxs(dis.nodes.coords)

    dis.limits = Limits(min=limits[0], max=limits[1])

    return dis
