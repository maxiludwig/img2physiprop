"""Converts Mesh-data into usable data."""

from i2pp.core.utilities import find_ids, find_mins_maxs
from tqdm import tqdm


class MeshClass:
    """Class to process Mesh-data."""

    def __init__(self, mesh):
        """Init MeshClass."""
        self.mesh = mesh

    def assign_ids_of_elements(self):
        """Assigns node ids to element-nodes.

        This means that the calculation only has to be carried out once
        for each node and can be assigned to the elements again later
        """

        for points in tqdm(
            self.mesh.element_coords, desc="Assign Node-IDs to elements"
        ):
            ids_element = find_ids(points, self.mesh.nodes)

            self.mesh.element_ids.append(ids_element)


def process_mesh(input_mesh):
    """Calls Mesh_Data functions."""

    mesh_data = MeshClass(input_mesh)
    mesh_data.assign_ids_of_elements()
    mesh_data.mesh.limits = find_mins_maxs(input_mesh.nodes)

    return mesh_data.mesh
