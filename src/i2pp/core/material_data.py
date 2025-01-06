"""Maps material parameters to the different elements of the model."""


class MaterialClass:
    """Class to map material parameters."""

    def __init__(self, material_type, interpolation_data):
        """Init MaterialClass."""
        self.material_type = material_type
        self.interpolation_data = interpolation_data
