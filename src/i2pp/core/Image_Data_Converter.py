"""Convertes slices of the image data into an array with coordinates and an
array with pixel values."""

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree
from tqdm import tqdm


@dataclass
class ImageData:
    """Dataclass for Converted image-data."""

    coord_array: np.ndarray
    pxl_value: np.ndarray
    modality: str


class ImageDataConverter:
    """Class to convert slices in usable data."""

    def __init__(self, image_data):
        """Init ImageDataConverter."""

        self.image_data = image_data

    def _pxlpos_to_pxlcoord(self, slices, row, col):
        """Convertes array-position to x-y-z coordinate for an voxel."""

        image_position = np.array(slices.ImagePositionPatient)
        image_orientation = np.array(slices.ImageOrientationPatient)
        pixel_spacing = np.array(slices.PixelSpacing)

        row_vector = image_orientation[:3]
        col_vector = image_orientation[3:]

        coord = (
            image_position
            + row * pixel_spacing[0] * row_vector
            + col * pixel_spacing[1] * col_vector
        )

        return coord

    def _ray_intersecs_rectangle(self, coord, direction, rect_min, rect_max):
        """Checks if a ray intersecs a rectangle in the plane (For data-
        optimization)"""

        t_min = float("-inf")
        t_max = float("inf")

        for i in range(2):
            if direction[i] != 0:
                length_ray1 = (rect_min[i] - coord[i]) / direction[i]
                length_ray2 = (rect_max[i] - coord[i]) / direction[i]

                length_ray1, length_ray2 = min(length_ray1, length_ray2), max(
                    length_ray1, length_ray2
                )

                t_min = max(t_min, length_ray1)
                t_max = min(t_max, length_ray2)
            else:

                if not (rect_min[i] <= coord[i] <= rect_max[i]):
                    return False

        return t_min <= t_max and t_max >= 0

    def _iteration_irrelevant(self, limits, coord, check, orientation):
        """Checks if an interation is necessary.

        Options:
            -slice: Orientation vector has no component in
                    z-direction an plane heigth is not in
                    limits of the crop
                    -> slice is irrelevant

            -col: A Ray from the current position in direction
                    of the column vector does not
                    intersecs the crop
                    -> no further columns iteration needed

            -all: Verifies if x-,y- and z- coordinates are inside the crop
        """

        match check:
            case "slice":
                if orientation[2] == orientation[5] == 0 and (
                    coord[2] < limits[2] or coord[2] > limits[5]
                ):
                    return True
                else:
                    return False

            case "col":
                col_vector = orientation[3:]
                rect_min = [limits[0], limits[1]]
                rect_max = [limits[3], limits[4]]

                if not self._ray_intersecs_rectangle(
                    coord, col_vector, rect_min, rect_max
                ):
                    return True

                else:
                    return False

            case "all":
                if (
                    limits[0] <= coord[0] <= limits[3]
                    and limits[1] <= coord[1] <= limits[4]
                    and limits[2] <= coord[2] <= limits[5]
                ):
                    return False
                else:
                    return True

    def slices_2_mat(self, slices, limits):
        """Converts slices of the image-data in an array with pixel-coordinates
        and an array with pixel values. For Perfomance Optimization only the
        Pixels inside the Mesh-limits are processed.

        Raises:
            Runtimerror: If Mesh is not in Image_data
        """

        for slice in tqdm(slices, desc="Process slices"):
            # coordinate top-left corner
            starting_coordinate = slice.ImagePositionPatient

            # skip all cols and rows if the vectors dont have a component
            # in z-value direction an z-value is not in limits
            # => pxl coordinates can't be in mesh
            if self._iteration_irrelevant(
                limits,
                starting_coordinate,
                "slice",
                slice.ImageOrientationPatient,
            ):
                continue

            # array for gray values
            gray_values_array = slice.PixelData

            # size gray-array
            img_shape = slice.image_shape

            # ortintation of image
            orientation = slice.ImageOrientationPatient

            for j in range(0, img_shape[0]):

                # skip all rows-iterations if no coordinate
                # will be inside the crop
                # => pxl coordinates can't be in mesh
                coord = self._pxlpos_to_pxlcoord(slice, j, 0)

                if self._iteration_irrelevant(
                    limits, coord, "col", orientation
                ):
                    continue

                for m in range(0, img_shape[1]):

                    # checks if point is in limits
                    coord = self._pxlpos_to_pxlcoord(slice, j, m)
                    if (
                        self._iteration_irrelevant(
                            limits, coord, "all", orientation
                        )
                        is False
                    ):

                        gray_value = gray_values_array[m][j]

                        self.image_data.coord_array.append(
                            [coord[0], coord[1], coord[2]]
                        )

                        self.image_data.pxl_value.append(gray_value)

        self.image_data.coord_array = np.array(self.image_data.coord_array)
        self.image_data.pxl_value = np.array(self.image_data.pxl_value)

        if not self.image_data.coord_array.any():
            raise RuntimeError(
                "Mesh coordinates are not in image data!"
                "img2physiprop can not be executed!"
            )

        self.image_data.modality = slices[0].Modality

        return self.image_data

    def smooth_data(self, k):
        """Find the k nearest neighbors for each point and calculate the
        average -> Prevent measurement errors."""

        coords = self.image_data.coord_array
        values = self.image_data.pxl_value
        """colors_normalized = values / 255.0.

        x = [p[0] for p in coords] y = [p[1] for p in coords]

        plt.scatter(x, y, c=colors_normalized, s=100)
        plt.savefig("not_smoothed", dpi=300)
        """

        tree = cKDTree(coords)
        smoothed_pxl_value = []

        for i, (x, y, z) in tqdm(
            enumerate(self.image_data.coord_array), desc="Smooth Data"
        ):

            _, indices = tree.query((x, y, z), k=k)
            smoothed_value = np.mean(values[indices], axis=0)
            smoothed_pxl_value.append((smoothed_value))

        self.image_data.pxl_value = smoothed_pxl_value
        return self.image_data


def convert_imagedata(slices, limits):
    """Calls Image Data Converter."""

    image_dataclass = ImageData(coord_array=[], pxl_value=[], modality="")

    image_data = ImageDataConverter(image_dataclass)

    image_data.slices_2_mat(slices, limits)

    return image_data.smooth_data(27)
