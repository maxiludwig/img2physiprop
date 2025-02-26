"""Import Mesh data."""

from enum import Enum
from pathlib import Path
from typing import Type, cast

from i2pp.core.interpolator import CalculationType
from i2pp.core.model_reader_classes.dat_reader import DatReader
from i2pp.core.model_reader_classes.mesh_reader import MeshReader
from i2pp.core.model_reader_classes.model_reader import (
    Limits,
    ModelData,
    ModelReader,
)
from i2pp.core.utilities import find_mins_maxs


class ModelFormat(Enum):
    """ModelFormat (Enum): Defines the supported file formats for model data.

    Attributes:
        MESH: Represents the model data in the '.mesh' format, commonly used
              for 3D models.
        DAT: Represents the model data in the '.dat' format, typically used
             for storing node and element data.

    This enum helps identify the format of the model data being processed
    and guides the appropriate handling of the data based on its format
    (e.g., .mesh vs. .dat).
    """

    MESH = ".mesh"
    DAT = ".dat"

    def get_class(self) -> Type[ModelReader]:
        """Returns the appropriate model reader class based on the model
        format.

        Returns:
            Type[ModelReader]: A class that is a subclass of `ModelReader`,
                either `MeshReader` or `DatReader`.
        """
        return {ModelFormat.MESH: MeshReader, ModelFormat.DAT: DatReader}[self]


def verify_input(directory: Path) -> ModelFormat:
    """Verifies if the model file exists, is readable, and has a supported
    format.

    This function checks whether the provided model file exists and if its
    format is valid (either `.mesh` or `.dat`). If the file is missing or
    has an unsupported format, it raises an error.

    Arguments:
        directory (Path): Path to the model data file.

    Returns:
        ModelFormat: The format of the model file (`.mesh` or `.dat`).

    Raises:
        RuntimeError: If the specified file does not exist.
        RuntimeError: If the file format is not `.mesh` or `.dat`.
    """
    if not directory.is_file():
        raise RuntimeError(
            "Mesh data not found! img2physiprop cannot be executed!"
        )

    if directory.suffix not in [".mesh", ".dat"]:
        raise RuntimeError(
            f"{directory.suffix} not readable! Format has to be '.mesh'"
            "or '.dat'"
        )

    return ModelFormat(directory.suffix)


def verify_and_load_model(config: dict) -> ModelData:
    """Loads and processes mesh data based on the user configuration.

    This function verifies the input mesh file, selects the appropriate reader
    (MeshReader or DatReader), and loads the model data. If element center
    calculations are required, it processes the model accordingly. Finally,
    it determines the model's bounding limits.

    Arguments:
        config (dict): User configuration containing paths and processing
            options.

    Returns:
        ModelData: The loaded and processed mesh data.

    Raises:
        RuntimeError: If the mesh file is not valid or in the wrong format.
    """
    directory = Path(config["Input Informations"]["model_file_path"])

    model_format = verify_input(directory)

    model_reader = cast(ModelReader, model_format.get_class())

    model = model_reader.load_model(directory, config["Processing options"])

    if (
        CalculationType(config["Processing options"]["calculation_type"])
        == CalculationType.CENTER
    ):
        model = model_reader.get_center(model)

    limits = find_mins_maxs(model.nodes.coords)

    model.limits = Limits(min=limits[0], max=limits[1])

    return model
