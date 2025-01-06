"""Interpolates pixel values from image-data to mesh-data."""

from dataclasses import dataclass

from scipy.interpolate import griddata

# from scipy.spatial import ConvexHull, Delaunay
from tqdm import tqdm


@dataclass
class InterpolData:
    """Dataclass for Interpolation-Data."""

    points: list
    points_value: list
    element_ids: list
    element_value: list


class InterpolatorClass:
    """Class to interpolate between Image-data and Mesh-data."""

    def __init__(self, image_data, mesh_data, interpol_data):
        """Init InterpolatorClass."""
        self.image_data = image_data
        self.mesh_data = mesh_data
        self.interpol_data = interpol_data

    """

    def _point_in_hull(self,point,hull):
        deln = Delaunay(hull.points[hull.vertices])
        return deln.find_simplex(point) >= 0

    def voxels_in_mesh(self):
        points_in_mesh=[]
        hull = ConvexHull(self.mesh_data.nodes)
        num=0

        for point in self.image_data.coord_array:
            num+=1
            if self._point_in_hull(point,hull):
                points_in_mesh.append(point)

            print(num)

        print(len(points_in_mesh))

        return points_in_mesh
    """

    def imagevalues_2_meshcoords(self):
        """Interpolates the pixel values form the image-data to the nodes of
        the mesh-data."""

        points_image = self.image_data.coord_array
        value_image = self.image_data.pxl_value

        target_points = self.mesh_data.nodes
        """print(points_image) print(value_image) print(target_points)"""
        print("Starting Interpolation!")

        interpolated_values = griddata(
            points_image, value_image, target_points, method="linear"
        )

        print("Interpolation done!")

        self.interpol_data.points_value = interpolated_values

        return self.interpol_data.points_value

    def get_value_of_elements(self):
        """Calculates the mean pixel-value for each FEM-element in the mesh."""

        for ele_ids in tqdm(self.mesh_data.element_ids, desc="Element values"):

            pxl_value = 0
            num_points = 0

            for id in ele_ids:

                pxl_value += self.interpol_data.points_value[id]
                num_points += 1

            self.interpol_data.element_value.append(pxl_value / num_points)

        return self.interpol_data.element_value

    def grid_smooting(self):
        """Not implemented yet."""
        pass


def interpolate_image_2_mesh(image_data, mesh_data):
    """Call Interpolation functions."""

    interpol_data = InterpolData(
        points=mesh_data.nodes,
        points_value=[],
        element_ids=mesh_data.element_ids,
        element_value=[],
    )
    interpolator = InterpolatorClass(image_data, mesh_data, interpol_data)
    interpolator.imagevalues_2_meshcoords()
    interpolator.get_value_of_elements()

    return interpolator.interpol_data
