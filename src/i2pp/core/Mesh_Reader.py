"""Import Mesh data."""

import os
from dataclasses import dataclass

import lnmmeshio
import numpy as np
import trimesh
from i2pp.core.utilities import find_mins_maxs


@dataclass
class MeshData:
    """Class for Mesh-data."""

    nodes: list
    element_ids: np.array
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
                element_coords=[],
                element_ids=input_mesh.faces,
                limits=np.array([]),
            )

        elif file_extension == ".dat":

            self.mesh = MeshData(nodes=[], element_ids=[], limits=np.array([]))

            directory = str(directory)
            mesh_read = lnmmeshio.read(directory)
            print(mesh_read.elements.structure[0].nodes[0].id)
            mesh_read.compute_ids(True)
            self.mesh.nodes = mesh_read.get_node_coords()

            element_ids = []
            for ele in mesh_read.elements.structure:
                element = []
                for node in ele.nodes:
                    element.append(node.id)

                element_ids.append(np.array(element))

            self.mesh.element_ids = element_ids

        return self.mesh


def verify_and_load_mesh(directory):
    """Calls Mesh Reader functions."""

    mesh_data = MeshReader([])

    mesh_data.verify_input(directory)

    mesh_data.load_mesh(directory)

    mesh_data.mesh.limits = find_mins_maxs(mesh_data.mesh.nodes)

    return mesh_data.mesh
