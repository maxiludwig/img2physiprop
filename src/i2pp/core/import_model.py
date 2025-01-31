"""Import Mesh data."""

import os
from enum import Enum
from pathlib import Path
from typing import Union

from i2pp.core.model_reader_classes.dat_reader import DatReader
from i2pp.core.model_reader_classes.mesh_reader import MeshReader
from i2pp.core.model_reader_classes.model_reader import ModelData
from i2pp.core.utilities import find_mins_maxs


class ModelFormat(Enum):
    """Options for model-data format."""

    Mesh = 1
    Dat = 2


def verify_input(directory: Path) -> ModelFormat:
    """Verify if model exists and is readable.

    Arguments:
        directory {str} -- Path to the model-data file.

    Raises:
        RuntimeError: If Mesh is not a valid file
        RuntimeError: If Mesh has wrong format-type.
    """

    if not os.path.isfile(directory):
        raise RuntimeError(
            "Mesh data not found! img2physiprop can not be executed!"
        )

    file_extension = directory.suffix

    if file_extension == ".mesh":
        format_input = ModelFormat.Mesh
    elif file_extension == ".dat":
        format_input = ModelFormat.Dat
    else:
        raise RuntimeError(
            "Mesh data not readable! Format has to be '.mesh' or '.dat'"
        )

    return format_input


def verify_and_load_mesh(config) -> ModelData:
    """Calls Mesh Reader functions.

    Arguments:
        config {object} -- User Configuration.

    Returns:
        object -- ModelData object."""

    directory = Path(config["general"]["input_mesh_directory"])

    suffix = verify_input(directory)

    model_reader: Union[MeshReader, DatReader]

    if suffix == ModelFormat.Mesh:

        model_reader = MeshReader()

    elif suffix == ModelFormat.Dat:
        model_reader = DatReader()

    model = model_reader.load_model(str(directory))

    if config["Further customizations"]["calculation_type"] == "elementcenter":
        model = model_reader.get_center(model)

    model.limits = find_mins_maxs(model.nodes)

    return model
