"""Module for validating and managing the configuration of the I2PP
application."""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

from i2pp.core.configuration_validator.validation_helpers import (
    resolve_and_validate_path,
)
from i2pp.core.exporters.export_format import ExportFormat
from i2pp.core.interpolators.interpolator_types import InterpolationType


def _normalize_element_filter(options: Dict[str, Any]) -> Dict[str, Any]:
    """Translates the deprecated option `material_ids` into an element filter
    and validates the element filter.

    Arguments:
        options (Dict[str, Any]): Options of the discretization.

    Returns:
        Dict[str, Any]: Options in which `material_ids` is replaced by
            `element_filter: {field: MAT, values: material_ids}`.

    Raises:
        ValueError: If both `material_ids` and `element_filter` are given or
            if the element filter is malformed.
    """
    options = dict(options)
    material_ids = options.pop("material_ids", None)
    element_filter = options.get("element_filter")

    if material_ids:
        if element_filter is not None:
            raise ValueError(
                "The discretization options 'material_ids' and "
                "'element_filter' cannot be set at the same time. "
                "'material_ids' is deprecated, use 'element_filter: "
                "{field: MAT, values: [...]}' instead."
            )
        logging.warning(
            "The discretization option 'material_ids' is deprecated. Use "
            f"'element_filter: {{field: MAT, values: {list(material_ids)}}}' "
            "instead."
        )
        element_filter = {"field": "MAT", "values": list(material_ids)}
        options["element_filter"] = element_filter

    if element_filter is not None and (
        not isinstance(element_filter, dict)
        or not isinstance(element_filter.get("field"), str)
        or not isinstance(element_filter.get("values"), (list, tuple))
        or len(element_filter["values"]) == 0
    ):
        raise ValueError(
            "The discretization option 'element_filter' requires a 'field' "
            "(name of the element data field, e.g. block_id or MAT) and a "
            "non-empty list of 'values'."
        )

    return options


@dataclass
class Discretization:
    """Class representing a discretization configuration."""

    path: Path
    type: str
    options: Optional[Dict[str, Any]] = field(default_factory=dict)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "Discretization":
        """Creates a Discretization instance from a dictionary."""
        return Discretization(
            path=resolve_and_validate_path(d["path"]),
            type=d["type"],
            options=_normalize_element_filter(d.get("options") or {}),
        )


@dataclass
class Image:
    """Class representing an image configuration."""

    path: Path
    type: str
    options: Dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "Image":
        """Creates an Image instance from a dictionary."""
        return Image(
            path=resolve_and_validate_path(d["path"]),
            type=d["type"],
            options=(d["options"] if "options" in d else {}),
        )


@dataclass
class Import:
    """Class representing the import configuration for discretization and image
    data."""

    discretization: Discretization
    image: Image

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "Import":
        """Creates an Import instance from a dictionary."""
        return Import(
            discretization=Discretization.from_dict(d["discretization"]),
            image=Image.from_dict(d["image"]),
        )


@dataclass
class Smoothing:
    """Class representing the smoothing configuration."""

    area: int = 3
    visualize: bool = False

    @staticmethod
    def from_dict(d: Optional[Dict[str, Any]]) -> Optional["Smoothing"]:
        """Creates a Smoothing instance from a dictionary."""
        if d is None:
            return None
        if d.get("area", 3) <= 0:
            raise ValueError("Smoothing area must be a positive integer.")
        if "smoothing_area" in d:
            raise ValueError(
                "The key 'smoothing_area' is deprecated. "
                "Please use 'area' instead."
            )
        return Smoothing(
            area=d.get("area", 3),
            visualize=d.get("visualize", False),
        )


@dataclass
class Transformation:
    """Class representing the transformation configuration."""

    user_script: Path
    user_function: str
    normalize_values: bool = False
    visualize: bool = False

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "Transformation":
        """Creates a Transformation instance from a dictionary."""
        return Transformation(
            user_script=resolve_and_validate_path(d["user_script"]),
            user_function=d["user_function"],
            normalize_values=d.get("normalize_values", False),
            visualize=d.get("visualize", False),
        )


@dataclass
class NodeScaling:
    """Class representing node scaling configuration for interpolation."""

    surface_node_scaling: float = 1.0
    interior_node_scaling: float = 1.0

    @staticmethod
    def from_dict(
        d: Optional[Dict[str, Any]],
    ) -> "NodeScaling":
        """Creates a NodeScaling instance from a dictionary."""
        if d is None:
            return NodeScaling(
                surface_node_scaling=1.0, interior_node_scaling=1.0
            )
        surface = d.get("surface", 1.0)
        interior = d.get("interior", 1.0)

        if surface < 0 or interior < 0:
            raise ValueError(
                "Node scaling factors (surface and interior) must be "
                "non-negative."
            )
        return NodeScaling(
            surface_node_scaling=surface,
            interior_node_scaling=interior,
        )


@dataclass
class LabelMap:
    """Class representing the label map of the labelmap interpolation."""

    path: Path
    label_field: str = "tu_label"

    @staticmethod
    def from_dict(d: Optional[Dict[str, Any]]) -> "LabelMap":
        """Creates a LabelMap instance from a dictionary."""
        if not d or "path" not in d:
            raise ValueError(
                "The interpolation method 'labelmap' requires the label map "
                "'labelmap: {path: ..., label_field: ...}'."
            )

        label_field = d.get("label_field", "tu_label")
        if not isinstance(label_field, str):
            raise ValueError(
                "'labelmap.label_field' must be the name of the element "
                "field holding the label."
            )

        return LabelMap(
            path=resolve_and_validate_path(d["path"]),
            label_field=label_field,
        )


@dataclass
class Interpolation:
    """Class representing the interpolation configuration."""

    method: str
    node_scaling_factors: NodeScaling
    filter_outliers: bool = False
    idw_power: int = 2
    set_node_value: Optional[float | list[float]] = field(default=None)
    set_ele_value: Optional[float | list[float]] = field(default=None)
    labelmap: Optional[LabelMap] = None

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "Interpolation":
        """Creates an Interpolation instance from a dictionary."""

        method = d["method"]
        try:
            InterpolationType(method)
        except ValueError as error:
            allowed = ", ".join(t.value for t in InterpolationType)
            raise ValueError(
                f"Unsupported interpolation method '{method}'. "
                f"Supported methods are: {allowed}."
            ) from error
        if (
            d.get("set_surface_node_value") is not None
            and d.get("set_surface_element_value") is not None
        ):
            raise ValueError(
                "Both 'set_surface_node_value' "
                "and 'set_surface_element_value' "
                "cannot be set at the same time."
            )
        if d.get("inverse_distance_power", 2) <= 0:
            raise ValueError(
                "Inverse distance power must be a positive integer."
            )
        return Interpolation(
            method=d["method"],
            filter_outliers=d.get("filter_outliers", False),
            set_node_value=d.get("set_surface_node_value"),
            set_ele_value=d.get("set_surface_element_value"),
            node_scaling_factors=NodeScaling.from_dict(
                d.get("node_scaling_factors")
            ),
            idw_power=d.get("inverse_distance_power", 2),
            labelmap=(
                LabelMap.from_dict(d.get("labelmap"))
                if method == InterpolationType.LABELMAP.value
                else None
            ),
        )


@dataclass
class Processing:
    """Class representing the processing configuration."""

    smoothing: Optional[Smoothing]
    transformation: Transformation
    interpolation: Interpolation

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "Processing":
        """Creates a Processing instance from a dictionary."""

        return Processing(
            smoothing=(
                Smoothing.from_dict(d.get("smoothing"))
                if d.get("smoothing") is not None
                else None
            ),
            transformation=Transformation.from_dict(d["transformation"]),
            interpolation=Interpolation.from_dict(d["interpolation"]),
        )


@dataclass
class Export:
    """Class representing the export configuration."""

    folder_path: Path
    file_name: str
    type: str
    output_parameter_name: Optional[str] = None

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "Export":
        """Creates an Export instance from a dictionary."""
        if (
            d.get("output_parameter_name") is not None
            and d["type"] != ExportFormat.JSON.value
        ):
            raise ValueError(
                "'output_parameter_name' is only used for the export type "
                f"'json', but the export type is '{d['type']}'. Remove it "
                "from the export configuration."
            )
        return Export(
            folder_path=resolve_and_validate_path(
                d["folder_path"], must_exist=False
            ),
            file_name=d["file_name"],
            type=d["type"],
            output_parameter_name=d.get("output_parameter_name"),
        )


@dataclass
class I2PPConfig:
    """Class representing the configuration for the I2PP application."""

    import_: Import
    processing: Processing
    export: Export

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "I2PPConfig":
        """Creates an I2PPConfig instance from a dictionary."""
        config = I2PPConfig(
            import_=Import.from_dict(d["import"]),
            processing=Processing.from_dict(d["processing"]),
            export=Export.from_dict(d["export"]),
        )

        discretization_path = config.import_.discretization.path
        if (
            config.export.type == ExportFormat.DISCRETIZATION.value
            and discretization_path.suffix != ".vtu"
        ):
            raise ValueError(
                "The export type 'discretization' is only supported for .vtu "
                f"discretizations, but the discretization is "
                f"{discretization_path.name}."
            )

        return config
