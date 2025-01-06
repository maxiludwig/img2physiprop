"""Import Mesh data."""

import os
from dataclasses import dataclass

import lnmmeshio
import numpy as np
import trimesh


@dataclass
class MeshData:
    """Class for Mesh-data."""

    nodes: list
    element_coords: list
    element_ids: list
    limits: np.array


class MeshReader:
    """Class to read Mesh-data."""

    def __init__(self, mesh):
        self.mesh = mesh
        # self.slices=slices
        # self.pxl_array=pxl_array

    def verify_input(self, directory):
        """Verifies if mesh exists and is readable.

        Raises:
            RuntimeError: If Mesh is not a valid file
            RuntimeError: If Mesh has wrong format-type.
        """

        if not os.path.isfile(directory):
            raise RuntimeError(
                "Mesh data not found! img2physiprop can not be executed!"
            )

        allowed_format = [".mesh", ".dat"]
        file_extension = directory.suffix

        if file_extension not in allowed_format:
            raise RuntimeError(
                "Mesh data not readable! Format has to be '.mesh' or '.dat'"
            )

    def load_mesh(self, directory):
        """Load and convert mesh in usable data."""

        print("Importing Mesh data")

        file_extension = directory.suffix
        if file_extension == ".mesh":
            input_mesh = trimesh.load(directory)

            self.mesh = MeshData(
                nodes=input_mesh.vertices,
                elements=input_mesh.faces,
                limits=np.array([]),
            )

        elif file_extension == ".dat":

            self.mesh = MeshData(
                nodes=[],
                element_coords=[],
                element_ids=[],
                limits=np.array([]),
            )

            directory = str(directory)
            input_dat = lnmmeshio.read(directory)

            nodes = []
            element_coords = []

            for node in input_dat.nodes:

                nodes.append(np.array(node.coords))

            for ele in input_dat.elements.structure:
                element = []
                for node in ele.nodes:
                    element.append(node.coords)

                element_coords.append(np.array(element))

            self.mesh.nodes = nodes
            self.mesh.element_coords = element_coords

        return self.mesh


def verify_and_load_mesh(directory):
    """Calls Mesh Reader functions."""

    mesh = MeshReader([])
    mesh.verify_input(directory)
    mesh.load_mesh(directory)

    return mesh.mesh
