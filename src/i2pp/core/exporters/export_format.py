"""Export format detection and handling."""

from enum import Enum
from typing import Type

from i2pp.core.exporters.exporter import Exporter
from i2pp.core.exporters.json_exporter import JsonExporter
from i2pp.core.exporters.txt_exporter import TxtExporter


class ExportFormat(Enum):
    """ExportFormat (Enum): Represents the supported formats for exporting
    data.

    Attributes:
        JSON: Represents the JSON format for exporting data.
        TXT: Represents the TXT format for exporting data.

    This enum is used to define the format of the exported data and helps
    in determining which exporter class to use for writing the data to a file.
    """

    JSON = "json"
    TXT = "txt"

    def get_exporter(self) -> Type[Exporter]:
        """Returns the appropriate exporter class based on the export format.

        Returns:
            Type[Exporter]: A class that is a subclass of `Exporter`, either
                `JsonExporter` or `TxtExporter`.

        Raises:
            ValueError: If the export format is not supported.
        """
        exporters = {
            ExportFormat.JSON: JsonExporter,
            ExportFormat.TXT: TxtExporter,
        }
        if self not in exporters:
            raise ValueError(f"Unsupported export format: {self}")
        return exporters[self]
